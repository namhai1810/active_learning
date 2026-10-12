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
from src.benchmark.hans_manifest import FrozenHansManifest, audit_final_test_overlap
from src.benchmark.selection import select_complete_pairs
from src.data.hans import HANSExample
from src.data.mnli import MNLIExample, load_mnli
from src.decision.cost_model import CostUsage, compute_cost, cost_rates_from_config
from src.evaluation.hans import evaluate_hans
from src.evaluation.mnli import evaluate_mnli
from src.intervention.pilot import _start_gpu_timer, _stop_gpu_timer
from src.models.lora_training import (
    LoraTrainingConfig,
    PairwiseLossConfig,
    train_lora,
    train_paired_lora,
)
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SyntheticNLIExample
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


METHODS = (
    "zero_shot",
    "random_mnli",
    "uncertainty_mnli",
    "random_synthetic",
    "failure_driven_no_verification",
    "ours_verified",
)


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
    teacher_input_tokens = 0
    teacher_output_tokens = 0
    calls_value = value.get("teacher_calls_dir")
    if calls_value is not None:
        calls_dir = _format_path(calls_value, seed=seed)
        for path in sorted(calls_dir.glob("*.jsonl")):
            for record in read_jsonl(path, allow_empty=True):
                usage = record.get("usage")
                if isinstance(usage, dict):
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
    model = require_mapping(config, "model")
    training = LoraTrainingConfig.from_mapping(require_mapping(config, "training"))
    pairwise_training = (
        PairwiseLossConfig.from_mapping(
            require_mapping(benchmark, "ours_training_objective")
        )
        if args.method == "ours_verified"
        else None
    )
    pool_config = require_mapping(config, "mnli_pool")
    mnli_eval_config = require_mapping(config, "mnli_evaluation")
    batch_size = int(model.get("batch_size", 4))

    mnli_pool: list[MNLIExample] = []
    synthetic_pool: list[SyntheticNLIExample] = []
    provenance: dict[str, Any] | None = None
    if args.method in {"random_mnli", "uncertainty_mnli"}:
        mnli_pool = load_mnli(
            split=str(pool_config.get("split", "train")),
            dataset_name=str(pool_config.get("name", "nyu-mll/glue")),
            dataset_config=str(pool_config.get("config", "mnli")),
            cache_dir=pool_config.get("cache_dir"),
            max_examples=pool_config.get("max_examples"),
            drop_unlabeled=True,
        )
    elif args.method not in {"zero_shot"}:
        synthetic_pool, provenance = _synthetic_pool(benchmark, args.method, args.seed)

    current_adapter = model.get("adapter_path")
    initial_checkpoint = checkpoint_descriptor(model)
    selected_ids: list[str] = []
    selected_training_rows: list[dict[str, Any]] = []
    round_records: list[dict[str, Any]] = []
    checkpoint_paths: list[tuple[int, str | None]] = [(0, current_adapter)]
    student = None if args.method == "zero_shot" else _student(model, current_adapter)
    for round_index, requested in enumerate(round_budgets, start=1):
        assert student is not None
        set_deterministic_seed(args.seed + round_index - 1)
        starting = checkpoint_descriptor({**model, "adapter_path": current_adapter})
        remaining_mnli = [item for item in mnli_pool if item.id not in set(selected_ids)]
        acquisition_wall_start = time.perf_counter()
        acquisition_timer = (
            _start_gpu_timer(student)
            if args.method == "uncertainty_mnli"
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
        else:
            acquired_synthetic, shortage = select_complete_pairs(
                synthetic_pool,
                budget=requested,
                seed=args.seed + round_index - 1,
                excluded_ids=selected_ids,
            )
            acquired_mnli = [item.to_mnli() for item in acquired_synthetic]
        acquisition_gpu_seconds = float(
            _stop_gpu_timer(student, acquisition_timer) or 0.0
        )
        acquisition_wall_seconds = time.perf_counter() - acquisition_wall_start
        actual = len(acquired_mnli)
        if args.method in {"random_mnli", "uncertainty_mnli"}:
            shortage = {
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
        if args.method == "ours_verified":
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
        if acquired_synthetic:
            selected_training_rows.extend(item.to_dict() for item in acquired_synthetic)
        else:
            selected_training_rows.extend(asdict(item) for item in acquired_mnli)
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
                    if args.method == "ours_verified"
                    else "per_example_causal_lm_cross_entropy"
                ),
                "optimizer_steps": _optimizer_steps(actual, training),
                "acquisition_scored_examples": acquisition_scored_examples,
                "acquisition_gpu_seconds": acquisition_gpu_seconds,
                "acquisition_wall_clock_seconds": acquisition_wall_seconds,
                "gpu_seconds": gpu_seconds,
                "wall_clock_seconds": wall_seconds,
            }
        )
        checkpoint_paths.append((round_index, current_adapter))

    if student is not None:
        _release(student)
    # Learning curves use the independent gain-eval partition. The sealed final
    # test remains unopened until the final checkpoint has been selected.
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
    curves: list[dict[str, Any]] = []
    evaluation_gpu_seconds = 0.0
    evaluation_items = 0
    evaluation_runs = 0
    curve_evaluation_gpu_seconds = 0.0
    for round_index, adapter_path in checkpoint_paths:
        eval_student = _student(model, adapter_path)
        timer = _start_gpu_timer(eval_student)
        _, metrics = evaluate_hans(
            eval_student, hans_gain_eval, batch_size=batch_size
        )
        elapsed = _stop_gpu_timer(eval_student, timer)
        evaluation_gpu_seconds += float(elapsed or 0.0)
        curve_evaluation_gpu_seconds += float(elapsed or 0.0)
        evaluation_items += len(hans_gain_eval)
        evaluation_runs += 1
        curves.append(
            {
                "round": round_index,
                "evaluation_role": "gain_eval",
                "cumulative_evaluation_gpu_seconds": curve_evaluation_gpu_seconds,
                "cumulative_final_training_examples": sum(
                    item["actual_examples"]
                    for item in round_records
                    if item["round"] <= round_index
                ),
                "metrics": metrics,
            }
        )
        _release(eval_student)

    # Cross the final-test firewall once, for the frozen final checkpoint only.
    final_rows = read_jsonl(Path(str(benchmark["sealed_final_test_path"])))
    if {str(row["id"]) for row in final_rows} != set(manifest.final_test_ids):
        raise ValueError("Sealed HANS final-test IDs differ from frozen manifest")
    hans_final = _hans_examples(final_rows)
    final_student = _student(model, checkpoint_paths[-1][1])
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
        completed_curve_evaluations = sum(
            checkpoint_round <= completed
            for checkpoint_round, _ in checkpoint_paths
        )
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
                completed_curve_evaluations
                + sum(
                    item["round"] <= completed
                    and int(item["acquisition_scored_examples"]) > 0
                    for item in round_records
                )
            ),
            evaluation_items=(
                completed_curve_evaluations * len(hans_gain_eval)
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
    write_json(output / "posthoc_gain_eval_curve.json", {"points": curves})
    result = {
        "experiment": args.experiment,
        "method": args.method,
        "seed": args.seed,
        "requested_budget": args.budget,
        "actual_final_training_examples": len(selected_ids),
        "shortage": False if args.method == "zero_shot" else len(selected_ids) < args.budget,
        "rounds": args.rounds,
        "cross_source_baseline": args.method in {"random_mnli", "uncertainty_mnli"},
        "equal_data_access_claimed": False,
        "training_objective": (
            "none"
            if args.method == "zero_shot"
            else (
                "token_cross_entropy_plus_counterfactual_contrastive_"
                "plus_invariant_consistency"
                if args.method == "ours_verified"
                else "per_example_causal_lm_cross_entropy"
            )
        ),
        "pair_structure_used_for_training": args.method == "ours_verified",
        "pair_structure_used_only_for_selection": args.method in {
            "random_synthetic", "failure_driven_no_verification"
        },
        "pairwise_loss_config": (
            asdict(pairwise_training) if pairwise_training is not None else None
        ),
        "manifest_hash": manifest.manifest_hash,
        "git_commit": _git_commit(),
        "initial_checkpoint": initial_checkpoint,
        "final_checkpoint": checkpoint_descriptor(
            {**model, "adapter_path": checkpoint_paths[-1][1]}
        ),
        "hans_final": hans_final_metrics,
        "final_test_evaluations": 1,
        "learning_curve_evaluation_role": "gain_eval",
        "learning_curve_file": "posthoc_gain_eval_curve.json",
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
