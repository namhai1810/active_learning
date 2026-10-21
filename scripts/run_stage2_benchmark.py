#!/usr/bin/env python3
"""Run one fair Stage 2 method/seed/budget with a frozen HANS test manifest."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import RandomBaseline, UncertaintyBaseline
from src.baselines.literature import (
    LITERATURE_METADATA,
    LITERATURE_STRATEGIES,
    acquire_literature_mnli,
    acquire_literature_synthetic_pairs,
)
from src.benchmark.hans_manifest import FrozenHansManifest, audit_final_test_overlap
from src.benchmark.selection import (
    select_confidence_safe_entropy_novelty_pairs,
    select_complete_pairs,
    select_stratified_utility_pairs,
    select_verification_conditioned_primal_dual_pairs,
)
from src.benchmark.stopping_subset import validate_stopping_subset_manifest
from src.data.labels import MNLI_LABELS
from src.data.hans import HANSExample
from src.data.mnli import MNLIExample, load_mnli
from src.decision.cost_model import CostUsage, compute_cost, cost_rates_from_config
from src.evaluation.hans import evaluate_hans
from src.evaluation.metrics import compute_hans_metrics
from src.evaluation.mnli import evaluate_mnli
from src.intervention.pilot import _start_gpu_timer, _stop_gpu_timer
from src.models.lora_training import (
    LoraTrainingConfig,
    PairwiseLossConfig,
    train_lora,
    train_lora_stateful,
    train_paired_lora,
)
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SyntheticNLIExample
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


MNLI_LITERATURE_METHODS = tuple(
    f"{strategy}_mnli" for strategy in LITERATURE_STRATEGIES
)
SYNTHETIC_LITERATURE_METHODS = tuple(
    f"{strategy}_synthetic" for strategy in LITERATURE_STRATEGIES
)
LITERATURE_METHODS = MNLI_LITERATURE_METHODS + SYNTHETIC_LITERATURE_METHODS
MNLI_METHODS = ("random_mnli", "uncertainty_mnli") + MNLI_LITERATURE_METHODS

METHODS = (
    "zero_shot",
    "random_mnli",
    "uncertainty_mnli",
    "random_synthetic",
    "failure_driven_no_verification",
    "ours_verified",
) + LITERATURE_METHODS


def _literature_strategy(method: str) -> str:
    for suffix in ("_mnli", "_synthetic"):
        if method.endswith(suffix):
            strategy = method[: -len(suffix)]
            if strategy in LITERATURE_STRATEGIES:
                return strategy
    raise ValueError(f"Not a literature baseline method: {method!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", required=True, choices=("A", "B"))
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--per-round-budget", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--ours-objective",
        choices=("pairwise", "ce"),
        default="pairwise",
        help="Training objective for ours_verified; selection remains identical.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _student(model: dict[str, Any], adapter_path: str | None) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model["name_or_path"]),
        adapter_path=adapter_path,
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _format_path(template: Any, *, seed: int) -> Path:
    if not isinstance(template, str) or not template:
        raise ValueError("Synthetic method requires a non-empty artifact path template")
    return Path(template.format(seed=seed)).resolve()


def _release(student: Any) -> None:
    del student
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _optimizer_steps(n: int, training: LoraTrainingConfig) -> int:
    batches = math.ceil(n / training.batch_size)
    return training.num_epochs * math.ceil(
        batches / training.gradient_accumulation_steps
    )


def _verification_certification_scores(
    provenance: dict[str, Any] | None,
) -> dict[str, float]:
    if not isinstance(provenance, dict):
        return {}
    summary = provenance.get("verification_summary")
    if not isinstance(summary, dict):
        return {}
    results = summary.get("results")
    if not isinstance(results, list):
        return {}
    scores: dict[str, float] = {}
    for result in results:
        if not isinstance(result, dict) or result.get("certified") is not True:
            continue
        hypothesis_id = result.get("hypothesis_id")
        lower_bound = result.get("lower_bound")
        if isinstance(hypothesis_id, str) and isinstance(lower_bound, (int, float)):
            scores[hypothesis_id] = float(lower_bound)
    return scores


def _verification_sampling_evidence(
    provenance: dict[str, Any] | None,
    selection_config: dict[str, Any],
    *,
    seed: int,
    include_retained: bool = False,
) -> dict[str, dict[str, Any]]:
    """Join retained verification statistics to target failure labels."""
    if not isinstance(provenance, dict):
        raise ValueError("Verification-conditioned sampling requires provenance")
    summary = provenance.get("verification_summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("results"), list):
        raise ValueError("Verification-conditioned sampling requires verification results")
    probes_path = _format_path(selection_config.get("verification_probes"), seed=seed)
    outcomes_path = _format_path(
        selection_config.get("verification_outcomes"), seed=seed
    )
    probes = read_jsonl(probes_path)
    outcomes = read_jsonl(outcomes_path)
    probe_labels: dict[str, str] = {}
    for probe in probes:
        probe_id = str(probe.get("id", ""))
        label = str(probe.get("expected_label", ""))
        if not probe_id or label not in MNLI_LABELS:
            raise ValueError("Verification probe has an invalid ID or expected label")
        if probe_id in probe_labels:
            raise ValueError(f"Duplicate verification probe ID: {probe_id}")
        probe_labels[probe_id] = label

    outcomes_by_hypothesis: dict[str, list[dict[str, Any]]] = {}
    seen_target_ids: set[str] = set()
    for outcome in outcomes:
        hypothesis_id = str(outcome.get("hypothesis_id", ""))
        target_id = str(outcome.get("target_probe_id", ""))
        if not hypothesis_id or target_id not in probe_labels:
            raise ValueError("Verification outcome references an unknown target probe")
        if target_id in seen_target_ids:
            raise ValueError(f"Duplicate target outcome: {target_id}")
        seen_target_ids.add(target_id)
        outcomes_by_hypothesis.setdefault(hypothesis_id, []).append(outcome)

    evidence: dict[str, dict[str, Any]] = {}
    states_value = provenance.get("hypothesis_verification_states", {})
    retained_value = provenance.get("retained_hypothesis_ids", [])
    states = states_value if isinstance(states_value, dict) else {}
    retained = set(retained_value) if isinstance(retained_value, list) else set()
    for result in summary["results"]:
        if not isinstance(result, dict):
            continue
        hypothesis_id = str(result.get("hypothesis_id", ""))
        if include_retained:
            if hypothesis_id not in retained:
                continue
        elif result.get("certified") is not True:
            continue
        diagnostics = result.get("diagnostics")
        controls = diagnostics.get("controls") if isinstance(diagnostics, dict) else None
        if not hypothesis_id or not isinstance(controls, dict):
            raise ValueError("Certified result lacks control diagnostics")
        counterfactual = controls.get("counterfactual")
        invariant = controls.get("invariant")
        if not isinstance(counterfactual, dict) or not isinstance(invariant, dict):
            raise ValueError("Certified result lacks both control accuracies")
        hypothesis_outcomes = outcomes_by_hypothesis.get(hypothesis_id, [])
        declared_n = int(result.get("n", 0))
        if len(hypothesis_outcomes) != declared_n or declared_n <= 0:
            raise ValueError(
                f"Verification outcome count mismatch for {hypothesis_id!r}: "
                f"declared={declared_n}, observed={len(hypothesis_outcomes)}"
            )
        failures = {label: 0 for label in MNLI_LABELS}
        target_counts = {label: 0 for label in MNLI_LABELS}
        for outcome in hypothesis_outcomes:
            label = probe_labels[str(outcome["target_probe_id"])]
            target_counts[label] += 1
            if outcome.get("target_failure_observed") is True:
                failures[label] += 1
            elif outcome.get("target_failure_observed") is not False:
                raise ValueError("target_failure_observed must be boolean")
        evidence[hypothesis_id] = {
            "state": str(
                states.get(hypothesis_id, {}).get(
                    "state", "certified" if result.get("certified") is True else ""
                )
            ),
            "lower_bound": float(result["lower_bound"]),
            "target_n": declared_n,
            "target_counts_by_label": target_counts,
            "target_failures_by_label": failures,
            "counterfactual_accuracy": float(counterfactual["accuracy"]),
            "invariant_accuracy": float(invariant["accuracy"]),
        }
    if not evidence:
        raise ValueError("Verification-conditioned sampling found no retained hypothesis")
    return evidence


def _paired_gain_confidence(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    delta: float,
    max_looks: int,
) -> dict[str, float | int | str]:
    """Empirical-Bernstein interval for paired accuracy improvements."""
    if not 0.0 < delta < 1.0 or max_looks <= 0:
        raise ValueError("Stopping delta and max_looks must be valid")
    before_index = {str(row["id"]): row for row in before}
    after_index = {str(row["id"]): row for row in after}
    if not before_index or set(before_index) != set(after_index):
        raise ValueError("Paired stopping evaluations must have identical non-empty IDs")
    deltas = [
        int(bool(after_index[item]["correct"]))
        - int(bool(before_index[item]["correct"]))
        for item in sorted(before_index)
    ]
    n = len(deltas)
    mean = sum(deltas) / n
    variance = (
        sum((value - mean) ** 2 for value in deltas) / (n - 1)
        if n > 1
        else 0.0
    )
    # Values lie in [-1, 1] (range=2). The max-look factor predeclares
    # repeated inspection of the same gain_eval partition.
    log_term = math.log(3.0 * max_looks / delta)
    radius = math.sqrt(2.0 * variance * log_term / n) + 6.0 * log_term / n
    return {
        "method": "paired_empirical_bernstein_union_adjusted",
        "n": n,
        "mean_gain": mean,
        "sample_variance": variance,
        "confidence_radius": radius,
        "lower_bound": max(-1.0, mean - radius),
        "upper_bound": min(1.0, mean + radius),
        "delta": delta,
        "max_looks": max_looks,
    }


def _metrics_from_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or any(
        field not in row
        for row in rows
        for field in ("gold", "prediction", "heuristic", "subcase", "correct")
    ):
        raise ValueError("Cached prediction rows are incomplete")
    return compute_hans_metrics(
        [str(row["gold"]) for row in rows],
        [str(row["prediction"]) for row in rows],
        [str(row["heuristic"]) for row in rows],
        [str(row["subcase"]) for row in rows],
    )


def _paired_group_gains(
    before: list[dict[str, Any]], after: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    before_index = {str(row["id"]): row for row in before}
    after_index = {str(row["id"]): row for row in after}
    if not before_index or set(before_index) != set(after_index):
        raise ValueError("Grouped paired gains require identical non-empty IDs")
    result: dict[str, dict[str, float]] = {}
    for output_name, field in (("heuristic", "heuristic"), ("gold", "gold")):
        deltas: dict[str, list[int]] = {}
        for item_id in sorted(before_index):
            before_row = before_index[item_id]
            after_row = after_index[item_id]
            group = str(before_row[field])
            if str(after_row[field]) != group:
                raise ValueError(f"Paired evaluation changed immutable {field}")
            deltas.setdefault(group, []).append(
                int(bool(after_row["correct"])) - int(bool(before_row["correct"]))
            )
        result[output_name] = {
            key: sum(values) / len(values) for key, values in sorted(deltas.items())
        }
    return result


def _adaptive_stop_decision(
    confidence: dict[str, Any],
    *,
    round_index: int,
    consecutive_small: int,
    config: dict[str, Any],
    stage_gpu_seconds: float,
) -> tuple[bool, int, str, dict[str, Any]]:
    decision_rule = str(
        config.get("decision_rule", "paired_empirical_bernstein_ucb_v1")
    )
    minimum_rounds = int(config.get("minimum_rounds", 2))
    patience = int(config.get("patience", 2))
    minimum_gain = float(config.get("minimum_mean_gain", 0.002))
    maximum_small_gain_ucb = float(config.get("maximum_small_gain_ucb", 0.02))
    maximum_heuristic_gain = float(config.get("maximum_heuristic_gain", 1.0))
    maximum_label_gain = float(config.get("maximum_label_gain", 1.0))
    minimum_heuristic_gain = float(config.get("minimum_heuristic_gain", -1.0))
    minimum_label_gain = float(config.get("minimum_label_gain", -1.0))
    minimum_optimistic_gain_per_gpu_hour = float(
        config.get("minimum_optimistic_gain_per_gpu_hour", 0.0)
    )
    if minimum_rounds <= 0 or patience <= 0:
        raise ValueError("Adaptive stopping minimum_rounds and patience must be positive")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (
            minimum_gain,
            maximum_small_gain_ucb,
            minimum_optimistic_gain_per_gpu_hour,
        )
    ):
        raise ValueError("Adaptive stopping thresholds must be finite and non-negative")
    if any(
        value < 0.0 or math.isnan(value)
        for value in (maximum_heuristic_gain, maximum_label_gain)
    ):
        raise ValueError("Adaptive group-gain thresholds must be non-negative")
    if any(
        not math.isfinite(value) or value > 0.0
        for value in (minimum_heuristic_gain, minimum_label_gain)
    ):
        raise ValueError("Adaptive regression limits must be finite and non-positive")
    mean = float(confidence["mean_gain"])
    upper = float(confidence["upper_bound"])
    group_gains = confidence.get("group_gains", {})
    heuristic_gains = group_gains.get("heuristic", {})
    label_gains = group_gains.get("gold", {})
    if decision_rule == "practical_paired_gain_patience_v1":
        if not heuristic_gains or not label_gains:
            raise ValueError("Practical stopping requires paired heuristic/label gains")
        small = (
            mean <= minimum_gain
            and max(float(value) for value in heuristic_gains.values())
            <= maximum_heuristic_gain
            and max(float(value) for value in label_gains.values())
            <= maximum_label_gain
            and min(float(value) for value in heuristic_gains.values())
            >= minimum_heuristic_gain
            and min(float(value) for value in label_gains.values())
            >= minimum_label_gain
        )
    elif decision_rule == "paired_empirical_bernstein_ucb_v1":
        small = mean <= minimum_gain and upper <= maximum_small_gain_ucb
    else:
        raise ValueError(f"Unsupported adaptive stopping decision_rule: {decision_rule}")
    next_consecutive = consecutive_small + 1 if small else 0
    gpu_hours = max(stage_gpu_seconds / 3600.0, 1e-12)
    optimistic_gain_per_gpu_hour = max(0.0, upper) / gpu_hours
    cost_inefficient = (
        minimum_optimistic_gain_per_gpu_hour > 0.0
        and optimistic_gain_per_gpu_hour < minimum_optimistic_gain_per_gpu_hour
    )
    stop = False
    reason = "continue"
    if (
        decision_rule == "paired_empirical_bernstein_ucb_v1"
        and round_index >= minimum_rounds
        and upper <= 0.0
    ):
        stop = True
        reason = "paired_gain_ucb_nonpositive"
    elif round_index >= minimum_rounds and next_consecutive >= patience:
        stop = True
        reason = "diminishing_gain_patience_exhausted"
    elif round_index >= minimum_rounds and cost_inefficient:
        stop = True
        reason = "optimistic_gain_per_gpu_hour_below_threshold"
    return stop, next_consecutive, reason, {
        "small_gain_look": small,
        "decision_rule": decision_rule,
        "consecutive_small_gain_looks": next_consecutive,
        "stage_gpu_seconds": stage_gpu_seconds,
        "optimistic_gain_per_gpu_hour": optimistic_gain_per_gpu_hour,
        "minimum_mean_gain": minimum_gain,
        "maximum_small_gain_ucb": maximum_small_gain_ucb,
        "maximum_heuristic_gain": maximum_heuristic_gain,
        "maximum_label_gain": maximum_label_gain,
        "minimum_heuristic_gain": minimum_heuristic_gain,
        "minimum_label_gain": minimum_label_gain,
        "minimum_optimistic_gain_per_gpu_hour": minimum_optimistic_gain_per_gpu_hour,
    }


def _hans_examples(rows: list[dict[str, Any]]) -> list[HANSExample]:
    """Drop prediction metadata and materialize the immutable HANS fields."""
    return [
        HANSExample(
            id=str(row["id"]),
            premise=str(row["premise"]),
            hypothesis=str(row["hypothesis"]),
            gold=str(row["gold"]),
            heuristic=str(row["heuristic"]),
            subcase=str(row["subcase"]),
        )
        for row in rows
    ]


def _teacher_usage_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return non-duplicated usage records from flat or stage-summary logs.

    Discovery and verification logs contain their actual calls in nested lists;
    the discovery top-level usage duplicates its final hypothesis-generation
    attempt, so it must only be used when no nested call collection exists.
    """
    entries: list[dict[str, Any]] = []
    for key in ("scoring_calls", "hypothesis_discovery_attempts", "calls"):
        values = record.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            usage = value.get("usage")
            if isinstance(usage, dict):
                entries.append(usage)
    if entries:
        return entries
    usage = record.get("usage")
    return [usage] if isinstance(usage, dict) else []


def _synthetic_pool(
    benchmark: dict[str, Any], method: str, seed: int
) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
    artifacts = require_mapping(benchmark, "synthetic_artifacts")
    value = artifacts.get(method)
    if not isinstance(value, dict):
        raise ValueError(f"Missing synthetic_artifacts.{method}")
    accepted_path = _format_path(value.get("accepted"), seed=seed)
    provenance_path = _format_path(value.get("provenance"), seed=seed)
    rows = read_jsonl(accepted_path, allow_empty=True)
    provenance = read_json(provenance_path)
    expected_policy = {
        "random_synthetic": "random_non_failure_targeted",
        "failure_driven_no_verification": "failure_targeted_verification_bypassed",
        "ours_verified": "failure_targeted_verified",
    }[method]
    if provenance.get("stage2_policy") != expected_policy:
        raise ValueError(
            f"Synthetic provenance policy mismatch for {method}: "
            f"{provenance.get('stage2_policy')!r}"
        )
    if provenance.get("binary_hans_labels_converted") is not False:
        raise ValueError("Synthetic benchmark artifacts must use explicit MNLI labels")
    if provenance.get("pair_selection_unit") != "complete_pair":
        raise ValueError("Synthetic benchmark artifacts must preserve complete pairs")
    abstained = provenance.get("status") == "abstained_no_certified_hypotheses"
    if abstained:
        if method != "ours_verified":
            raise ValueError(
                "Only ours_verified may abstain because no hypothesis was certified"
            )
        if (
            rows
            or provenance.get("certified_hypothesis_ids") != []
            or provenance.get("abstention_reason") != "no_certified_hypotheses"
        ):
            raise ValueError(
                "A no-certified-hypotheses abstention must have an empty pool and "
                "no certified hypothesis IDs"
            )
    teacher_input_tokens = 0
    teacher_output_tokens = 0
    calls_value = value.get("teacher_calls_dir")
    if calls_value is not None:
        calls_dir = _format_path(calls_value, seed=seed)
        for path in sorted(calls_dir.glob("*.jsonl")):
            for record in read_jsonl(path, allow_empty=True):
                for usage in _teacher_usage_entries(record):
                    teacher_input_tokens += int(usage.get("input_tokens", 0))
                    teacher_output_tokens += int(usage.get("output_tokens", 0))
    counts = provenance.get("counts", {})
    generated_count = sum(
        int(item.get("generated", 0))
        for item in counts.values()
        if isinstance(item, dict)
    ) if isinstance(counts, dict) else 0
    provenance = {
        **provenance,
        "resource_usage": {
            "accepted_examples": len(rows),
            "generated_examples": generated_count,
            "teacher_input_tokens": teacher_input_tokens,
            "teacher_output_tokens": teacher_output_tokens,
            "pilot_count": 0,
        },
    }
    return [SyntheticNLIExample.from_mapping(row) for row in rows], provenance


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    benchmark = require_mapping(config, "stage2_benchmark")
    allowed_seeds = [int(item) for item in benchmark["seeds"]]
    allowed_budgets = [int(item) for item in benchmark["budgets"]]
    if args.seed not in allowed_seeds or args.budget not in allowed_budgets:
        raise ValueError("Seed/budget is not in the predeclared benchmark grid")
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    set_deterministic_seed(args.seed)
    if args.method == "zero_shot":
        round_budgets: list[int] = []
    else:
        per_round = args.per_round_budget or args.budget
        if per_round <= 0 or per_round * args.rounds != args.budget:
            raise ValueError("rounds * per_round_budget must equal total budget")
        if args.method.endswith("synthetic") or args.method in {
            "failure_driven_no_verification", "ours_verified"
        }:
            if per_round % 2:
                raise ValueError("Synthetic per-round budgets must preserve pairs")
        round_budgets = [per_round] * args.rounds

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    import yaml
    _atomic_text_write(
        output / "config.yaml", yaml.safe_dump(config, sort_keys=False)
    )

    manifest = FrozenHansManifest.from_mapping(
        read_json(Path(str(benchmark["frozen_manifest_path"])))
    )
    shared_predictions = (
        Path(str(benchmark["frozen_manifest_path"])).resolve().parent
        / "predictions.jsonl"
    )
    adaptive_rows = read_jsonl(shared_predictions)
    adaptive_index = {str(row["id"]): row for row in adaptive_rows}
    if set(adaptive_index) != set().union(
        set(manifest.hypothesis_discovery_ids),
        set(manifest.verification_ids),
        set(manifest.intervention_pool_ids),
        set(manifest.gain_eval_ids),
        set(manifest.final_test_ids),
    ):
        raise ValueError("Shared prediction IDs differ from frozen manifest")
    gain_eval_rows = [adaptive_index[item] for item in manifest.gain_eval_ids]
    if any("premise" not in row or "gold" not in row for row in gain_eval_rows):
        raise ValueError("Shared gain-eval rows are incomplete")
    hans_gain_eval = _hans_examples(gain_eval_rows)
    model = require_mapping(config, "model")
    training = LoraTrainingConfig.from_mapping(require_mapping(config, "training"))
    use_pairwise_objective = (
        args.method == "ours_verified" and args.ours_objective == "pairwise"
    )
    pairwise_training = (
        PairwiseLossConfig.from_mapping(
            require_mapping(benchmark, "ours_training_objective")
        )
        if use_pairwise_objective
        else None
    )
    ours_selection_value = benchmark.get("ours_selection", {})
    if not isinstance(ours_selection_value, dict):
        raise ValueError("stage2_benchmark.ours_selection must be a mapping")
    ours_selection_strategy = str(ours_selection_value.get("strategy", ""))
    literature_value = benchmark.get("literature_baselines", {})
    if not isinstance(literature_value, dict):
        raise ValueError("stage2_benchmark.literature_baselines must be a mapping")
    literature_warm_start_rounds = int(
        literature_value.get("warm_start_rounds", 1)
    )
    literature_embedding_batch_size = int(
        literature_value.get("embedding_batch_size", 8)
    )
    literature_cal_neighbors = int(literature_value.get("cal_neighbors", 10))
    if literature_warm_start_rounds < 1:
        raise ValueError("literature warm_start_rounds must be at least one")
    if literature_embedding_batch_size <= 0 or literature_cal_neighbors <= 0:
        raise ValueError("literature embedding batch size and CAL neighbors must be positive")
    if (
        args.method in LITERATURE_METHODS
        and args.rounds <= literature_warm_start_rounds
    ):
        raise ValueError(
            "Literature active-learning runs require at least one acquisition "
            "round after the shared random warm start"
        )
    use_v2_utility_selection = (
        args.method == "ours_verified"
        and ours_selection_strategy == "stratified_student_utility_v1"
    )
    use_v3_utility_selection = (
        args.method == "ours_verified"
        and ours_selection_strategy
        == "verification_conditioned_primal_dual_v1"
    )
    use_confidence_safe_selection = (
        args.method == "ours_verified"
        and ours_selection_strategy == "confidence_safe_entropy_novelty_v1"
    )
    use_planned_utility_selection = (
        use_v3_utility_selection or use_confidence_safe_selection
    )
    adaptive_stopping_value = benchmark.get("ours_adaptive_stopping", {})
    if not isinstance(adaptive_stopping_value, dict):
        raise ValueError("stage2_benchmark.ours_adaptive_stopping must be a mapping")
    # Adaptive stopping belongs exclusively to the planned ours_verified
    # selector. Literature baselines may inherit an ours_verified config to
    # reuse the same model, evaluator, and controlled synthetic pool; they must
    # not inherit this method-specific runtime behavior.
    use_adaptive_stopping = bool(
        args.method == "ours_verified"
        and adaptive_stopping_value.get("enabled", False)
    )
    if use_adaptive_stopping and not use_planned_utility_selection:
        raise ValueError(
            "ours_adaptive_stopping requires a planned ours_verified selector"
        )
    stopping_partition_manifest: dict[str, Any] | None = None
    stopping_eval_rows = gain_eval_rows
    gain_report_rows: list[dict[str, Any]] = []
    curve_evaluation_role = "gain_eval"
    subset_manifest_value = adaptive_stopping_value.get("subset_manifest")
    if use_adaptive_stopping and subset_manifest_value is not None:
        subset_path = Path(str(subset_manifest_value)).resolve()
        stopping_partition_manifest = read_json(subset_path)
        stopping_ids, report_ids = validate_stopping_subset_manifest(
            stopping_partition_manifest,
            gain_eval_rows,
            manifest.gain_eval_ids,
            frozen_manifest_hash=manifest.manifest_hash,
        )
        stopping_eval_rows = [adaptive_index[item] for item in stopping_ids]
        gain_report_rows = [adaptive_index[item] for item in report_ids]
        curve_evaluation_role = "stopping_eval"
    hans_stopping_eval = _hans_examples(stopping_eval_rows)
    hans_gain_report = _hans_examples(gain_report_rows) if gain_report_rows else []
    pool_config = require_mapping(config, "mnli_pool")
    mnli_eval_config = require_mapping(config, "mnli_evaluation")
    batch_size = int(model.get("batch_size", 4))

    mnli_pool: list[MNLIExample] = []
    synthetic_pool: list[SyntheticNLIExample] = []
    provenance: dict[str, Any] | None = None
    if args.method in MNLI_METHODS:
        mnli_pool = load_mnli(
            split=str(pool_config.get("split", "train")),
            dataset_name=str(pool_config.get("name", "nyu-mll/glue")),
            dataset_config=str(pool_config.get("config", "mnli")),
            cache_dir=pool_config.get("cache_dir"),
            max_examples=pool_config.get("max_examples"),
            drop_unlabeled=True,
        )
    elif args.method not in {"zero_shot"}:
        synthetic_source = (
            str(literature_value.get("synthetic_source_method", "ours_verified"))
            if args.method in SYNTHETIC_LITERATURE_METHODS
            else args.method
        )
        synthetic_pool, provenance = _synthetic_pool(
            benchmark, synthetic_source, args.seed
        )
        if args.method in SYNTHETIC_LITERATURE_METHODS:
            provenance = {
                **provenance,
                "controlled_literature_selector": _literature_strategy(args.method),
                "controlled_source_method": synthetic_source,
                "controlled_selection_unit": "complete_pair",
            }

    verified_abstention = bool(
        args.method == "ours_verified"
        and isinstance(provenance, dict)
        and provenance.get("status") == "abstained_no_certified_hypotheses"
    )

    current_adapter = model.get("adapter_path")
    current_optimizer_state: str | None = None
    initial_checkpoint = checkpoint_descriptor(model)
    selected_ids: list[str] = []
    selected_training_rows: list[dict[str, Any]] = []
    selected_mnli_history: list[MNLIExample] = []
    selected_synthetic_history: list[SyntheticNLIExample] = []
    round_records: list[dict[str, Any]] = []
    checkpoint_paths: list[tuple[int, str | None]] = [(0, current_adapter)]
    student = None if args.method == "zero_shot" else _student(model, current_adapter)
    curves: list[dict[str, Any]] = []
    evaluation_gpu_seconds = 0.0
    evaluation_items = 0
    evaluation_runs = 0
    curve_evaluation_gpu_seconds = 0.0
    curve_evaluation_items = 0
    curve_evaluation_runs = 0
    cached_gain_evaluations: dict[
        int, tuple[list[dict[str, Any]], dict[str, Any], float]
    ] = {}
    free_cached_gain_rounds: set[int] = set()
    adaptive_stopped = False
    adaptive_stop_reason: str | None = None
    consecutive_small_gain_looks = 0

    planned_synthetic: list[SyntheticNLIExample] = []
    planned_selection_diagnostics: dict[str, Any] | None = None
    planned_acquisition_gpu_seconds = 0.0
    planned_acquisition_wall_seconds = 0.0
    if use_planned_utility_selection:
        assert student is not None
        acquisition_wall_start = time.perf_counter()
        acquisition_timer = _start_gpu_timer(student)
        student.progress_label = (
            "confidence-safe synthetic probability scoring"
            if use_confidence_safe_selection
            else "ours v3 synthetic utility scoring"
        )
        probabilities = student.predict_proba(
            [item.premise for item in synthetic_pool],
            [item.hypothesis for item in synthetic_pool],
            batch_size=batch_size,
        )
        if len(probabilities) != len(synthetic_pool):
            raise RuntimeError("Student utility score count does not match synthetic pool")
        label_index = {label: index for index, label in enumerate(MNLI_LABELS)}
        utility_scores = {
            item.id: -math.log(
                max(float(probabilities[index][label_index[item.label]]), 1e-12)
            )
            for index, item in enumerate(synthetic_pool)
        }
        if use_confidence_safe_selection:
            entropy_scores = {
                item.id: -sum(
                    float(probability) * math.log(max(float(probability), 1e-12))
                    for probability in probabilities[index]
                )
                / math.log(len(MNLI_LABELS))
                for index, item in enumerate(synthetic_pool)
            }
            student.progress_label = "confidence-safe synthetic embedding extraction"
            embedding_rows = student.encode_pair_embeddings(
                [item.premise for item in synthetic_pool],
                [item.hypothesis for item in synthetic_pool],
                batch_size=int(
                    ours_selection_value.get("embedding_batch_size", batch_size)
                ),
            )
            if len(embedding_rows) != len(synthetic_pool):
                raise RuntimeError(
                    "Student embedding count does not match synthetic pool"
                )
            planned_synthetic, planned_selection_diagnostics = (
                select_confidence_safe_entropy_novelty_pairs(
                    synthetic_pool,
                    budget=args.budget,
                    seed=args.seed,
                    nll_scores=utility_scores,
                    entropy_scores=entropy_scores,
                    embeddings={
                        item.id: embedding_rows[index]
                        for index, item in enumerate(synthetic_pool)
                    },
                    verification_evidence=_verification_sampling_evidence(
                        provenance,
                        ours_selection_value,
                        seed=args.seed,
                        include_retained=True,
                    ),
                    nll_weight=float(ours_selection_value.get("nll_weight", 0.65)),
                    entropy_weight=float(
                        ours_selection_value.get("entropy_weight", 0.35)
                    ),
                    novelty_weight=float(
                        ours_selection_value.get("novelty_weight", 0.50)
                    ),
                    nll_quantile_low=float(
                        ours_selection_value.get("nll_quantile_low", 0.10)
                    ),
                    nll_quantile_high=float(
                        ours_selection_value.get("nll_quantile_high", 0.90)
                    ),
                )
            )
        else:
            planned_synthetic, planned_selection_diagnostics = (
                select_verification_conditioned_primal_dual_pairs(
                    synthetic_pool,
                    budget=args.budget,
                    seed=args.seed,
                    utility_scores=utility_scores,
                    verification_evidence=_verification_sampling_evidence(
                        provenance, ours_selection_value, seed=args.seed
                    ),
                    label_temperature=float(
                        ours_selection_value.get("label_temperature", 1.0)
                    ),
                    hypothesis_temperature=float(
                        ours_selection_value.get("hypothesis_temperature", 0.25)
                    ),
                    utility_quantile_low=float(
                        ours_selection_value.get("utility_quantile_low", 0.10)
                    ),
                    utility_quantile_high=float(
                        ours_selection_value.get("utility_quantile_high", 0.90)
                    ),
                    utility_weight=float(
                        ours_selection_value.get("utility_weight", 1.0)
                    ),
                    diversity_weight=float(
                        ours_selection_value.get("diversity_weight", 0.35)
                    ),
                    specificity_weight=float(
                        ours_selection_value.get("specificity_weight", 0.50)
                    ),
                    label_dual_weight=float(
                        ours_selection_value.get("label_dual_weight", 1.25)
                    ),
                    hypothesis_dual_weight=float(
                        ours_selection_value.get("hypothesis_dual_weight", 0.75)
                    ),
                    pair_type_dual_weight=float(
                        ours_selection_value.get("pair_type_dual_weight", 0.50)
                    ),
                    dual_step=float(ours_selection_value.get("dual_step", 0.50)),
                    dual_clip=float(ours_selection_value.get("dual_clip", 3.0)),
                    sampling_temperature=float(
                        ours_selection_value.get("sampling_temperature", 0.20)
                    ),
                )
            )
        planned_acquisition_gpu_seconds = float(
            _stop_gpu_timer(student, acquisition_timer) or 0.0
        )
        planned_acquisition_wall_seconds = time.perf_counter() - acquisition_wall_start
        if planned_selection_diagnostics.get("shortage") and not verified_abstention:
            raise ValueError(
                "Synthetic pool cannot satisfy the planned complete-pair budget: "
                f"requested={args.budget}, actual={len(planned_synthetic)}"
            )

    if use_adaptive_stopping:
        assert student is not None
        if bool(adaptive_stopping_value.get("reuse_frozen_base_predictions", False)):
            predictions = [dict(row) for row in stopping_eval_rows]
            metrics = _metrics_from_prediction_rows(predictions)
            elapsed = 0.0
            free_cached_gain_rounds.add(0)
        else:
            timer = _start_gpu_timer(student)
            predictions, metrics = evaluate_hans(
                student, hans_stopping_eval, batch_size=batch_size
            )
            elapsed = float(_stop_gpu_timer(student, timer) or 0.0)
            evaluation_gpu_seconds += elapsed
            evaluation_items += len(hans_stopping_eval)
            evaluation_runs += 1
        cached_gain_evaluations[0] = (predictions, metrics, elapsed)

    for round_index, requested in enumerate(round_budgets, start=1):
        assert student is not None
        set_deterministic_seed(args.seed + round_index - 1)
        starting = checkpoint_descriptor({**model, "adapter_path": current_adapter})
        remaining_mnli = [item for item in mnli_pool if item.id not in set(selected_ids)]
        acquisition_wall_start = time.perf_counter()
        acquisition_timer = (
            _start_gpu_timer(student)
            if (
                args.method == "uncertainty_mnli"
                or args.method in LITERATURE_METHODS
                or use_v2_utility_selection
            )
            else None
        )
        acquisition_scored_examples = 0
        if args.method == "random_mnli":
            available = min(requested, len(remaining_mnli))
            acquired_mnli = (
                RandomBaseline().acquire(
                    remaining_mnli, available, args.seed + round_index - 1,
                    student, batch_size,
                )
                if available
                else []
            )
            acquired_synthetic: list[SyntheticNLIExample] = []
        elif args.method == "uncertainty_mnli":
            available = min(requested, len(remaining_mnli))
            acquisition_scored_examples = len(remaining_mnli) if available else 0
            acquired_mnli = (
                UncertaintyBaseline().acquire(
                    remaining_mnli, available, args.seed + round_index - 1,
                    student, batch_size,
                )
                if available
                else []
            )
            acquired_synthetic = []
        elif args.method in MNLI_LITERATURE_METHODS:
            available = min(requested, len(remaining_mnli))
            acquired_mnli, shortage = (
                acquire_literature_mnli(
                    _literature_strategy(args.method),
                    remaining_mnli,
                    budget=available,
                    seed=args.seed + round_index - 1,
                    student=student,
                    probability_batch_size=batch_size,
                    embedding_batch_size=literature_embedding_batch_size,
                    references=selected_mnli_history,
                    warm_start=round_index <= literature_warm_start_rounds,
                    cal_neighbors=literature_cal_neighbors,
                )
                if available
                else ([], {
                    "selection_strategy": _literature_strategy(args.method),
                    "requested_examples": requested,
                    "actual_examples": 0,
                    "candidate_examples_scored": 0,
                    "reference_examples_scored": 0,
                    "shortage": True,
                })
            )
            acquisition_scored_examples = int(
                shortage.get("candidate_examples_scored", 0)
            ) + int(shortage.get("reference_examples_scored", 0))
            acquired_synthetic = []
        elif args.method in SYNTHETIC_LITERATURE_METHODS:
            remaining_synthetic = [
                item for item in synthetic_pool if item.id not in set(selected_ids)
            ]
            acquired_synthetic, shortage = acquire_literature_synthetic_pairs(
                _literature_strategy(args.method),
                remaining_synthetic,
                budget=requested,
                seed=args.seed + round_index - 1,
                student=student,
                probability_batch_size=batch_size,
                embedding_batch_size=literature_embedding_batch_size,
                references=selected_synthetic_history,
                warm_start=round_index <= literature_warm_start_rounds,
                cal_neighbors=literature_cal_neighbors,
            )
            acquisition_scored_examples = int(
                shortage.get("candidate_examples_scored", 0)
            ) + int(shortage.get("reference_examples_scored", 0))
            if shortage.get("shortage"):
                raise ValueError(
                    "Controlled synthetic literature pool cannot satisfy the "
                    f"budget for {args.method}: requested={requested}, "
                    f"actual={len(acquired_synthetic)}"
                )
            acquired_mnli = [item.to_mnli() for item in acquired_synthetic]
        else:
            if use_planned_utility_selection:
                offset = len(selected_ids)
                acquired_synthetic = planned_synthetic[offset : offset + requested]
                stage_label_counts = {
                    label: sum(item.label == label for item in acquired_synthetic)
                    for label in MNLI_LABELS
                }
                shortage = {
                    "selection_strategy": ours_selection_strategy,
                    "requested_examples": requested,
                    "actual_examples": len(acquired_synthetic),
                    "shortage": len(acquired_synthetic) < requested,
                    "planned_offset": offset,
                    "stage_label_counts": stage_label_counts,
                    "full_plan": (
                        planned_selection_diagnostics if round_index == 1 else None
                    ),
                }
                if round_index == 1:
                    acquisition_scored_examples = len(synthetic_pool)
            elif use_v2_utility_selection:
                remaining_synthetic = [
                    item for item in synthetic_pool if item.id not in set(selected_ids)
                ]
                acquisition_scored_examples = len(remaining_synthetic)
                student.progress_label = "ours synthetic utility scoring"
                probabilities = student.predict_proba(
                    [item.premise for item in remaining_synthetic],
                    [item.hypothesis for item in remaining_synthetic],
                    batch_size=batch_size,
                )
                if len(probabilities) != len(remaining_synthetic):
                    raise RuntimeError(
                        "Student utility score count does not match synthetic pool"
                    )
                label_index = {label: index for index, label in enumerate(MNLI_LABELS)}
                utility_scores = {
                    item.id: -math.log(
                        max(
                            float(probabilities[index][label_index[item.label]]),
                            1e-12,
                        )
                    )
                    for index, item in enumerate(remaining_synthetic)
                }
                acquired_synthetic, shortage = select_stratified_utility_pairs(
                    synthetic_pool,
                    budget=requested,
                    seed=args.seed + round_index - 1,
                    excluded_ids=selected_ids,
                    utility_scores=utility_scores,
                    certification_scores=_verification_certification_scores(
                        provenance
                    ),
                    utility_weight=float(
                        ours_selection_value.get("utility_weight", 1.0)
                    ),
                    diversity_weight=float(
                        ours_selection_value.get("diversity_weight", 0.35)
                    ),
                    label_balance_weight=float(
                        ours_selection_value.get("label_balance_weight", 1.0)
                    ),
                    certification_weight=float(
                        ours_selection_value.get("certification_weight", 0.25)
                    ),
                )
            else:
                acquired_synthetic, shortage = select_complete_pairs(
                    synthetic_pool,
                    budget=requested,
                    seed=args.seed + round_index - 1,
                    excluded_ids=selected_ids,
                )
            if shortage.get("shortage") and not verified_abstention:
                raise ValueError(
                    "Synthetic pool cannot satisfy the requested complete-pair "
                    f"budget for {args.method}: requested={requested}, "
                    f"actual={len(acquired_synthetic)}. Rebuild the pool before "
                    "running the benchmark."
                )
            acquired_mnli = [item.to_mnli() for item in acquired_synthetic]
        acquisition_gpu_seconds = float(
            _stop_gpu_timer(student, acquisition_timer) or 0.0
        )
        acquisition_wall_seconds = time.perf_counter() - acquisition_wall_start
        if use_planned_utility_selection and round_index == 1:
            acquisition_gpu_seconds = planned_acquisition_gpu_seconds
            acquisition_wall_seconds = planned_acquisition_wall_seconds
        actual = len(acquired_mnli)
        if args.method in MNLI_METHODS:
            shortage = {
                **(shortage if args.method in MNLI_LITERATURE_METHODS else {}),
                "requested_examples": requested,
                "actual_examples": actual,
                "shortage": actual < requested,
            }
        if not acquired_mnli:
            round_records.append(
                {
                    "round": round_index,
                    "requested_examples": requested,
                    "actual_examples": 0,
                    "shortage": True,
                    "starting_checkpoint": starting,
                    "ending_checkpoint": starting,
                    "optimizer_steps": 0,
                    "acquisition_scored_examples": acquisition_scored_examples,
                    "acquisition_gpu_seconds": acquisition_gpu_seconds,
                    "acquisition_wall_clock_seconds": acquisition_wall_seconds,
                    "gpu_seconds": 0.0,
                    "wall_clock_seconds": 0.0,
                }
            )
            break
        round_dir = output / "rounds" / f"round_{round_index:03d}"
        wall_start = time.perf_counter()
        gpu_timer = _start_gpu_timer(student)
        if use_pairwise_objective:
            assert pairwise_training is not None
            paired_result = train_paired_lora(
                student,
                acquired_synthetic,
                training,
                pairwise_training,
                round_dir / "adapter",
                args.seed + round_index - 1,
            )
            train_loss = paired_result.total_loss
            loss_components: dict[str, Any] | None = asdict(paired_result)
        elif args.method in LITERATURE_METHODS:
            optimizer_state_output = round_dir / "optimizer_state.pt"
            stateful_result = train_lora_stateful(
                student,
                acquired_mnli,
                training,
                round_dir / "adapter",
                args.seed + round_index - 1,
                optimizer_state_input=current_optimizer_state,
                optimizer_state_output=optimizer_state_output,
            )
            train_loss = stateful_result.total_loss
            loss_components = asdict(stateful_result)
            current_optimizer_state = str(optimizer_state_output)
        else:
            train_loss = train_lora(
                student, acquired_mnli, training, round_dir / "adapter",
                args.seed + round_index - 1,
            )
            loss_components = None
        gpu_seconds = _stop_gpu_timer(student, gpu_timer)
        wall_seconds = time.perf_counter() - wall_start
        current_adapter = str(round_dir / "adapter")
        ending = checkpoint_descriptor({**model, "adapter_path": current_adapter})
        ids = [item.id for item in acquired_mnli]
        selected_ids.extend(ids)
        if args.method in MNLI_LITERATURE_METHODS:
            selected_mnli_history.extend(acquired_mnli)
        if args.method in SYNTHETIC_LITERATURE_METHODS:
            selected_synthetic_history.extend(acquired_synthetic)
        if acquired_synthetic:
            selected_training_rows.extend(item.to_dict() for item in acquired_synthetic)
        else:
            selected_training_rows.extend(asdict(item) for item in acquired_mnli)
        checkpoint_paths.append((round_index, current_adapter))
        stopping_record: dict[str, Any] | None = None
        if use_adaptive_stopping:
            timer = _start_gpu_timer(student)
            gain_predictions, gain_metrics = evaluate_hans(
                student, hans_stopping_eval, batch_size=batch_size
            )
            gain_elapsed = float(_stop_gpu_timer(student, timer) or 0.0)
            cached_gain_evaluations[round_index] = (
                gain_predictions,
                gain_metrics,
                gain_elapsed,
            )
            evaluation_gpu_seconds += gain_elapsed
            evaluation_items += len(hans_stopping_eval)
            evaluation_runs += 1
            confidence = _paired_gain_confidence(
                cached_gain_evaluations[round_index - 1][0],
                gain_predictions,
                delta=float(adaptive_stopping_value.get("delta", 0.05)),
                max_looks=len(round_budgets),
            )
            confidence["group_gains"] = _paired_group_gains(
                cached_gain_evaluations[round_index - 1][0],
                gain_predictions,
            )
            should_stop, consecutive_small_gain_looks, stop_reason, cost_stats = (
                _adaptive_stop_decision(
                    confidence,
                    round_index=round_index,
                    consecutive_small=consecutive_small_gain_looks,
                    config=adaptive_stopping_value,
                    stage_gpu_seconds=(
                        float(gpu_seconds or 0.0)
                        + acquisition_gpu_seconds
                        + gain_elapsed
                    ),
                )
            )
            stopping_record = {
                "action": "stop" if should_stop else "continue",
                "reason": stop_reason,
                "paired_gain": confidence,
                "cost_efficiency": cost_stats,
            }
            if should_stop and round_index < len(round_budgets):
                adaptive_stopped = True
                adaptive_stop_reason = stop_reason
            elif round_index == len(round_budgets):
                stopping_record["action"] = "complete"
                stopping_record["reason"] = "maximum_budget_reached"
        round_records.append(
            {
                "round": round_index,
                "requested_examples": requested,
                "actual_examples": actual,
                "shortage": bool(shortage["shortage"]),
                "selected_ids": ids,
                "starting_checkpoint": starting,
                "ending_checkpoint": ending,
                "train_loss": train_loss,
                "loss_components": loss_components,
                "training_objective": (
                    "token_cross_entropy_plus_counterfactual_contrastive_"
                    "plus_invariant_consistency"
                    if use_pairwise_objective
                    else "per_example_causal_lm_cross_entropy"
                ),
                "selection_diagnostics": shortage,
                "optimizer_steps": _optimizer_steps(actual, training),
                "acquisition_scored_examples": acquisition_scored_examples,
                "acquisition_gpu_seconds": acquisition_gpu_seconds,
                "acquisition_wall_clock_seconds": acquisition_wall_seconds,
                "gpu_seconds": gpu_seconds,
                "wall_clock_seconds": wall_seconds,
                "adaptive_stopping": stopping_record,
            }
        )
        if adaptive_stopped:
            break

    if student is not None:
        _release(student)
    # Learning curves use the independent gain-eval partition. The sealed final
    # test remains unopened until the final checkpoint has been selected.
    curve_examples = hans_stopping_eval if use_adaptive_stopping else hans_gain_eval
    for round_index, adapter_path in checkpoint_paths:
        if round_index in cached_gain_evaluations:
            _, metrics, elapsed = cached_gain_evaluations[round_index]
        else:
            eval_student = _student(model, adapter_path)
            timer = _start_gpu_timer(eval_student)
            _, metrics = evaluate_hans(
                eval_student, curve_examples, batch_size=batch_size
            )
            elapsed = float(_stop_gpu_timer(eval_student, timer) or 0.0)
            evaluation_gpu_seconds += elapsed
            evaluation_items += len(curve_examples)
            evaluation_runs += 1
            _release(eval_student)
        if round_index not in free_cached_gain_rounds:
            curve_evaluation_items += len(curve_examples)
            curve_evaluation_runs += 1
        curve_evaluation_gpu_seconds += elapsed
        curves.append(
            {
                "round": round_index,
                "evaluation_role": curve_evaluation_role,
                "cumulative_evaluation_gpu_seconds": curve_evaluation_gpu_seconds,
                "cumulative_evaluation_items": curve_evaluation_items,
                "cumulative_evaluation_runs": curve_evaluation_runs,
                "cumulative_final_training_examples": sum(
                    item["actual_examples"]
                    for item in round_records
                    if item["round"] <= round_index
                ),
                "metrics": metrics,
            }
        )

    # The held-out gain-report partition is opened only after checkpoint
    # selection. It never feeds stopping, acquisition, or training.
    final_student = _student(model, checkpoint_paths[-1][1])
    gain_report_metrics: dict[str, Any] | None = None
    if hans_gain_report:
        timer = _start_gpu_timer(final_student)
        gain_report_predictions, gain_report_metrics = evaluate_hans(
            final_student, hans_gain_report, batch_size=batch_size
        )
        elapsed = float(_stop_gpu_timer(final_student, timer) or 0.0)
        evaluation_gpu_seconds += elapsed
        evaluation_items += len(hans_gain_report)
        evaluation_runs += 1
        write_jsonl(output / "gain_report_predictions.jsonl", gain_report_predictions)

    # Cross the final-test firewall once, for the frozen final checkpoint only.
    final_rows = read_jsonl(Path(str(benchmark["sealed_final_test_path"])))
    if {str(row["id"]) for row in final_rows} != set(manifest.final_test_ids):
        raise ValueError("Sealed HANS final-test IDs differ from frozen manifest")
    hans_final = _hans_examples(final_rows)
    timer = _start_gpu_timer(final_student)
    final_predictions, hans_final_metrics = evaluate_hans(
        final_student, hans_final, batch_size=batch_size
    )
    elapsed = _stop_gpu_timer(final_student, timer)
    evaluation_gpu_seconds += float(elapsed or 0.0)
    evaluation_items += len(hans_final)
    evaluation_runs += 1
    write_jsonl(output / "hans_final_predictions.jsonl", final_predictions)
    mnli_metrics: dict[str, Any] = {}
    for label, split in (
        ("matched", str(mnli_eval_config.get("matched_split", "validation_matched"))),
        ("mismatched", str(mnli_eval_config.get("mismatched_split", "validation_mismatched"))),
    ):
        examples = load_mnli(
            split=split,
            dataset_name=str(mnli_eval_config.get("name", "nyu-mll/glue")),
            dataset_config=str(mnli_eval_config.get("config", "mnli")),
            cache_dir=mnli_eval_config.get("cache_dir"),
            max_examples=mnli_eval_config.get("max_examples"),
            drop_unlabeled=True,
        )
        timer = _start_gpu_timer(final_student)
        _, mnli_metrics[label] = evaluate_mnli(
            final_student, examples, batch_size=batch_size
        )
        elapsed = _stop_gpu_timer(final_student, timer)
        evaluation_gpu_seconds += float(elapsed or 0.0)
        evaluation_items += len(examples)
        evaluation_runs += 1
    _release(final_student)
    leakage = audit_final_test_overlap(
        selected_training_rows,
        final_rows,
        near_duplicate_threshold=float(benchmark.get("near_duplicate_threshold", 0.92)),
    )
    write_json(output / "leakage_audit.json", leakage)
    if leakage["id_overlap"]:
        raise ValueError(
            "Final-test ID overlap detected after frozen acquisition; refusing "
            "to publish an invalid benchmark result"
        )
    write_jsonl(output / "rounds.jsonl", round_records)
    write_json(
        output / "selected_ids.json",
        {"ids": selected_ids, "requested": args.budget, "actual": len(selected_ids)},
    )
    synthetic_resources = (
        provenance.get("resource_usage", {}) if provenance is not None else {}
    )
    measured_usage = CostUsage(
        llm_input_tokens=float(synthetic_resources.get("teacher_input_tokens", 0)),
        llm_output_tokens=float(synthetic_resources.get("teacher_output_tokens", 0)),
        generated_items=float(synthetic_resources.get("generated_examples", 0)),
        verified_items=float(synthetic_resources.get("generated_examples", 0)),
        lora_training_runs=sum(
            int(item["actual_examples"]) > 0 for item in round_records
        ),
        lora_training_gpu_seconds=sum(
            float(item["gpu_seconds"] or 0) for item in round_records
        ),
        evaluation_runs=evaluation_runs + sum(
            int(item["acquisition_scored_examples"]) > 0
            for item in round_records
        ),
        evaluation_items=evaluation_items + sum(
            int(item["acquisition_scored_examples"]) for item in round_records
        ),
        evaluation_gpu_seconds=evaluation_gpu_seconds + sum(
            float(item["acquisition_gpu_seconds"] or 0.0) for item in round_records
        ),
    )
    rates, currency = cost_rates_from_config(require_mapping(config, "cost"))
    assumed_cost = compute_cost(measured_usage, rates, currency=currency)
    for point in curves:
        completed = int(point["round"])
        partial_usage = CostUsage(
            llm_input_tokens=measured_usage.llm_input_tokens,
            llm_output_tokens=measured_usage.llm_output_tokens,
            generated_items=measured_usage.generated_items,
            verified_items=measured_usage.verified_items,
            lora_training_runs=sum(
                item["round"] <= completed and int(item["actual_examples"]) > 0
                for item in round_records
            ),
            lora_training_gpu_seconds=sum(
                float(item["gpu_seconds"] or 0)
                for item in round_records
                if item["round"] <= completed
            ),
            evaluation_runs=(
                int(point["cumulative_evaluation_runs"])
                + sum(
                    item["round"] <= completed
                    and int(item["acquisition_scored_examples"]) > 0
                    for item in round_records
                )
            ),
            evaluation_items=(
                int(point["cumulative_evaluation_items"])
                + sum(
                    int(item["acquisition_scored_examples"])
                    for item in round_records
                    if item["round"] <= completed
                )
            ),
            evaluation_gpu_seconds=(
                float(point["cumulative_evaluation_gpu_seconds"])
                + sum(
                    float(item["acquisition_gpu_seconds"] or 0.0)
                    for item in round_records
                    if item["round"] <= completed
                )
            ),
        )
        point["cumulative_total_cost"] = compute_cost(
            partial_usage, rates, currency=currency
        ).total
        point["cost_axis_excludes_posthoc_test_evaluation"] = True
        point["cost_axis_excludes_gain_report_evaluation"] = True
    curve_filename = (
        "stopping_eval_curve.json"
        if curve_evaluation_role == "stopping_eval"
        else "posthoc_gain_eval_curve.json"
    )
    write_json(output / curve_filename, {"points": curves})
    result = {
        "experiment": args.experiment,
        "method": args.method,
        "seed": args.seed,
        "requested_budget": args.budget,
        "actual_final_training_examples": len(selected_ids),
        "shortage": (
            False
            if args.method == "zero_shot" or adaptive_stopped
            else len(selected_ids) < args.budget
        ),
        "abstained": verified_abstention,
        "run_status": (
            "abstained_no_certified_hypotheses"
            if verified_abstention
            else "complete_adaptive_stop"
            if adaptive_stopped
            else "complete"
        ),
        "abstention_reason": (
            provenance.get("abstention_reason")
            if verified_abstention and isinstance(provenance, dict)
            else None
        ),
        "fixed_budget_comparable": (
            args.method == "zero_shot"
            or (not verified_abstention and len(selected_ids) == args.budget)
        ),
        "final_metrics_reportable": not verified_abstention,
        "adaptive_budget_protocol": use_adaptive_stopping,
        "adaptive_stopped": adaptive_stopped,
        "adaptive_stop_reason": adaptive_stop_reason,
        "adaptive_stopping": {
            "enabled": use_adaptive_stopping,
            "evaluation_role": curve_evaluation_role,
            "evaluation_examples": len(curve_examples),
            "gain_report_examples": len(hans_gain_report),
            "frozen_base_predictions_reused": 0 in free_cached_gain_rounds,
            "configured_maximum_budget": args.budget,
            "completed_rounds": len(checkpoint_paths) - 1,
            "configured_rounds": len(round_budgets),
            "config": adaptive_stopping_value if use_adaptive_stopping else None,
            "last_decision": (
                round_records[-1].get("adaptive_stopping")
                if round_records
                else None
            ),
        },
        "rounds": args.rounds,
        "cross_source_baseline": args.method in MNLI_METHODS,
        "equal_data_access_claimed": False,
        "literature_baseline": (
            {
                **LITERATURE_METADATA[_literature_strategy(args.method)],
                "strategy": _literature_strategy(args.method),
                "pool": (
                    "mnli_train"
                    if args.method in MNLI_LITERATURE_METHODS
                    else str(
                        literature_value.get(
                            "synthetic_source_method", "ours_verified"
                        )
                    )
                ),
                "warm_start_rounds": literature_warm_start_rounds,
                "warm_start_examples": sum(
                    int(item["actual_examples"])
                    for item in round_records[:literature_warm_start_rounds]
                ),
                "cal_neighbors": (
                    literature_cal_neighbors
                    if _literature_strategy(args.method) == "cal"
                    else None
                ),
                "controlled_synthetic_pair_adapter": (
                    args.method in SYNTHETIC_LITERATURE_METHODS
                ),
                "optimizer_state_continuity": "adamw_state_carried_across_rounds",
            }
            if args.method in LITERATURE_METHODS
            else None
        ),
        "training_objective": (
            "none_abstained"
            if verified_abstention
            else "none"
            if args.method == "zero_shot"
            else (
                "token_cross_entropy_plus_counterfactual_contrastive_"
                "plus_invariant_consistency"
                if use_pairwise_objective
                else "per_example_causal_lm_cross_entropy"
            )
        ),
        "pair_structure_used_for_training": (
            use_pairwise_objective and not verified_abstention
        ),
        "pair_structure_used_only_for_selection": args.method in {
            "random_synthetic", "failure_driven_no_verification"
        } or args.method in SYNTHETIC_LITERATURE_METHODS or (
            args.method == "ours_verified" and not use_pairwise_objective
        ),
        "pairwise_loss_config": (
            asdict(pairwise_training) if pairwise_training is not None else None
        ),
        "ours_objective": (
            args.ours_objective if args.method == "ours_verified" else None
        ),
        "ours_selection_strategy": (
            ours_selection_value.get("strategy")
            if args.method == "ours_verified"
            else None
        ),
        "ours_selection_diagnostics": planned_selection_diagnostics,
        "manifest_hash": manifest.manifest_hash,
        "git_commit": _git_commit(),
        "initial_checkpoint": initial_checkpoint,
        "final_checkpoint": checkpoint_descriptor(
            {**model, "adapter_path": checkpoint_paths[-1][1]}
        ),
        "hans_final": hans_final_metrics,
        "final_test_evaluations": 1,
        "learning_curve_evaluation_role": curve_evaluation_role,
        "learning_curve_file": curve_filename,
        "stopping_partition_manifest": stopping_partition_manifest,
        "gain_report": gain_report_metrics,
        "gain_report_evaluations": 1 if gain_report_metrics is not None else 0,
        "mnli": mnli_metrics,
        "optimizer_steps": sum(item["optimizer_steps"] for item in round_records),
        "training_gpu_seconds": sum(float(item["gpu_seconds"] or 0) for item in round_records),
        "acquisition_gpu_seconds": sum(
            float(item["acquisition_gpu_seconds"] or 0) for item in round_records
        ),
        "evaluation_gpu_seconds": evaluation_gpu_seconds,
        "pilot_count": 0,
        "generated_count": int(measured_usage.generated_items),
        "teacher_input_tokens": int(measured_usage.llm_input_tokens),
        "teacher_output_tokens": int(measured_usage.llm_output_tokens),
        "resource_usage": asdict(measured_usage),
        "assumed_cost": assumed_cost.to_dict(),
        "selected_ids_file": "selected_ids.json",
        "leakage_audit": leakage,
        "synthetic_provenance": provenance,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
