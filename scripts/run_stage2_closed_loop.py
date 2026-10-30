#!/usr/bin/env python3
"""Run fixed-budget Stage-2 training with checkpoint-conditioned verification."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.active_learning.closed_loop import (
    merge_hypothesis_candidates,
    verification_states,
)
from src.benchmark.hans_manifest import FrozenHansManifest, audit_final_test_overlap
from src.benchmark.selection import select_closed_loop_soft_pairs
from src.benchmark.stopping_subset import validate_stopping_subset_manifest
from src.data.hans import HANSExample
from src.data.pipeline_split import manifest_payload, reconstruct_five_way_split
from src.data.labels import MNLI_LABELS
from src.data.mnli import load_mnli
from src.decision.cost_model import CostUsage, compute_cost, cost_rates_from_config
from src.decision.safe_stopping import (
    cumulative_cost_ledger,
    verification_assumption_audit,
)
from src.evaluation.hans import evaluate_hans, evaluate_hans_prediction_rules
from src.evaluation.mnli import evaluate_mnli
from src.models.adapter_continuity import audit_loaded_adapter
from src.models.lora_training import (
    EarlyStoppingConfig,
    LoraTrainingConfig,
    PairedTrainingResult,
    train_lora,
    train_lora_stateful,
    train_lora_stateful_early_stopping,
)
from src.models.qwen_nli import QwenNLIStudent
from src.synthetic.types import SyntheticNLIExample
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import checkpoint_descriptor, hash_local_path
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="Override all experiment RNG seeds while preserving frozen data splits.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume a validated incomplete v7 run without repeating completed rounds.",
    )
    return parser.parse_args()


def _apply_seed_override(config: dict[str, Any], seed: int) -> None:
    """Override stochastic experiment seeds but never the frozen split seed."""
    closed = require_mapping(config, "closed_loop")
    prior_seed = int(closed.get("seed", config.get("seed", 42)))
    delta = seed - prior_seed
    config["seed"] = seed
    closed["seed"] = seed
    for section_name in (
        "hypothesis_generation",
        "verification",
        "synthetic_generation",
        "synthetic_validation",
        "pilot",
        "intervention",
    ):
        section = config.get(section_name)
        if isinstance(section, dict) and "seed" in section:
            section["seed"] = seed
        if (
            section_name == "pilot"
            and isinstance(section, dict)
            and "applicability_seed" in section
        ):
            section["applicability_seed"] = (
                int(section["applicability_seed"]) + delta
            )
    scorer = config.get("hypothesis_generation", {}).get("error_scorer")
    if isinstance(scorer, dict) and "seed" in scorer:
        scorer["seed"] = seed
    verification = config.get("verification")
    if isinstance(verification, dict) and "label_validation_seed" in verification:
        verification["label_validation_seed"] = (
            int(verification["label_validation_seed"]) + delta
        )


def _student(model: dict[str, Any], adapter: str | None) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model["name_or_path"]),
        adapter_path=adapter,
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )


def _clear_cuda_cache() -> None:
    """Return unreferenced model storage to CUDA after the caller deletes it."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _binary_hans_validation(
    probabilities: list[list[float]],
    examples: list[HANSExample],
) -> dict[str, Any]:
    """Compute binary HANS NLL without exposing the sealed final test."""
    if len(probabilities) != len(examples) or not examples:
        raise ValueError("Validation probabilities and HANS examples must align")
    losses: list[float] = []
    predictions: list[str] = []
    for row, example in zip(probabilities, examples, strict=True):
        if len(row) != len(MNLI_LABELS):
            raise ValueError("Validation probability rows must use MNLI label order")
        entailment = float(row[0])
        non_entailment = float(row[1]) + float(row[2])
        target = entailment if example.gold == "entailment" else non_entailment
        losses.append(-math.log(max(target, 1e-12)))
        predictions.append(
            "entailment" if entailment >= non_entailment else "non-entailment"
        )
    metrics = _metrics_from_rows(
        [
            {
                "gold": example.gold,
                "prediction": prediction,
                "heuristic": example.heuristic,
                "subcase": example.subcase,
            }
            for example, prediction in zip(examples, predictions, strict=True)
        ]
    )
    return {
        "loss": sum(losses) / len(losses),
        "accuracy": metrics["overall_accuracy"],
        "entailment_accuracy": metrics["entailment_accuracy"],
        "non_entailment_accuracy": metrics["non_entailment_accuracy"],
        "num_examples": len(examples),
        "source": "predeclared_gain_eval_stopping_subset",
        "sealed_final_test_used": False,
    }


def _teacher_usage(run_dir: Path) -> dict[str, float]:
    """Sum measured teacher usage from canonical teacher-call logs only."""
    totals = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "teacher_wall_clock_seconds": 0.0,
        "calls": 0.0,
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            usage = value.get("usage")
            if isinstance(usage, dict):
                totals["input_tokens"] += float(usage.get("input_tokens", 0.0))
                totals["output_tokens"] += float(usage.get("output_tokens", 0.0))
                totals["teacher_wall_clock_seconds"] += float(
                    usage.get("wall_clock_seconds", 0.0)
                )
                totals["calls"] += 1.0
            for key, item in value.items():
                if key != "usage":
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    calls_dir = run_dir / "teacher_calls"
    if calls_dir.is_dir():
        for call_path in sorted(calls_dir.glob("*.jsonl")):
            for record in read_jsonl(call_path):
                visit(record)
    return totals


def _bypassed_verification_states(
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Retain every candidate while making the bypass explicit in artifacts."""
    return {
        str(item["id"]): {
            "state": "unresolved",
            "certified": False,
            "verification_mode": "off",
            "verification_bypassed": True,
            "lower_bound": None,
            "upper_bound": None,
            "threshold": None,
            "metric": None,
        }
        for item in candidates
    }


def _hans(rows: list[dict[str, Any]]) -> list[HANSExample]:
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


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    import yaml

    _atomic_text_write(path, yaml.safe_dump(value, sort_keys=False))


def _run_stage(script: str, config: Path, run_dir: Path, *extra: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / script),
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def _pair_tokens(group: list[SyntheticNLIExample]) -> frozenset[str]:
    import re

    return frozenset(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                f"{item.premise} {item.hypothesis}" for item in group
            ).lower(),
        )
    )


def _merge_pool(
    current: list[SyntheticNLIExample],
    additions: list[SyntheticNLIExample],
    *,
    threshold: float,
) -> tuple[list[SyntheticNLIExample], dict[str, int]]:
    def groups(rows: list[SyntheticNLIExample]) -> list[list[SyntheticNLIExample]]:
        indexed: dict[str, list[SyntheticNLIExample]] = {}
        for item in rows:
            indexed.setdefault(item.pair_id, []).append(item)
        if any(len(value) != 2 for value in indexed.values()):
            raise ValueError("Closed-loop pool contains an incomplete pair")
        return list(indexed.values())

    merged = list(current)
    known_ids = {item.id for item in current}
    known_pair_ids = {item.pair_id for item in current}
    signatures = [_pair_tokens(group) for group in groups(current)]
    rejected = 0
    accepted_pairs = 0
    for group in groups(additions):
        signature = _pair_tokens(group)
        duplicate = bool(
            known_pair_ids & {item.pair_id for item in group}
            or known_ids & {item.id for item in group}
        )
        if not duplicate:
            for prior in signatures:
                union = signature | prior
                similarity = len(signature & prior) / len(union) if union else 1.0
                if similarity >= threshold:
                    duplicate = True
                    break
        if duplicate:
            rejected += 1
            continue
        merged.extend(group)
        known_ids.update(item.id for item in group)
        known_pair_ids.add(group[0].pair_id)
        signatures.append(signature)
        accepted_pairs += 1
    return merged, {
        "accepted_new_pairs": accepted_pairs,
        "rejected_cross_round_duplicate_pairs": rejected,
        "cumulative_pairs": len(merged) // 2,
    }


def _round_config(
    base: dict[str, Any], *, round_index: int, adapter: str | None
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["model"]["adapter_path"] = adapter
    config["model"]["device_map"] = os.environ.get(
        "CLOSED_LOOP_STUDENT_DEVICE", "cuda:1"
    )
    config["teacher"]["device_map"] = os.environ.get(
        "CLOSED_LOOP_TEACHER_DEVICE", "cuda:0"
    )
    for section, keys in {
        "hypothesis_generation": ("seed",),
        "verification": ("seed", "label_validation_seed"),
        "synthetic_generation": ("seed",),
        "synthetic_validation": ("seed",),
    }.items():
        value = config.get(section)
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    value[key] = int(value[key]) + round_index - 1
    scorer = config.get("hypothesis_generation", {}).get("error_scorer")
    if isinstance(scorer, dict) and "seed" in scorer:
        scorer["seed"] = int(scorer["seed"]) + round_index - 1
    closed = require_mapping(config, "closed_loop")
    if (
        closed.get("training_protocol")
        == "stateful_cost_aware_safe_stopping_v7"
    ):
        maximum = int(closed["max_rounds"])
        active = config.setdefault("active_learning", {})
        active["current_round"] = round_index
        active["max_rounds"] = maximum
        require_mapping(config, "gain")["T"] = maximum
        require_mapping(config, "verification")["max_rounds"] = maximum
        for section in ("pilot", "intervention"):
            value = config.get(section)
            if isinstance(value, dict):
                for key in ("seed", "applicability_seed"):
                    if key in value:
                        value[key] = int(value[key]) + round_index - 1
    return config


def _prepare_current_predictions(
    shared_rows: list[dict[str, Any]],
    manifest: FrozenHansManifest,
    model: dict[str, Any],
    adapter: str | None,
) -> tuple[list[dict[str, Any]], int]:
    indexed = {str(row["id"]): dict(row) for row in shared_rows}
    discovery_rows = [indexed[item] for item in manifest.hypothesis_discovery_ids]
    student = _student(model, adapter)
    student.progress_label = "closed-loop discovery re-evaluation"
    predictions, _ = evaluate_hans(
        student, _hans(discovery_rows), batch_size=int(model.get("batch_size", 4))
    )
    del student
    _clear_cuda_cache()
    for row in predictions:
        indexed[str(row["id"])] = row
    final_ids = set(manifest.final_test_ids)
    output = [
        {"id": str(row["id"])}
        if str(row["id"]) in final_ids
        else indexed[str(row["id"])]
        for row in shared_rows
    ]
    errors = sum(
        not bool(indexed[item]["correct"])
        for item in manifest.hypothesis_discovery_ids
    )
    return output, errors


def _write_round_split_manifest(
    path: Path,
    predictions: list[dict[str, Any]],
    hans_config: dict[str, Any],
) -> None:
    """Write the online five-way manifest expected by Stage-2 subprocesses.

    The benchmark's frozen manifest and the online pipeline manifest describe
    the same partition but deliberately use different schemas.  Refresh rounds
    must materialize the latter even when hypothesis discovery is skipped.
    """
    split = reconstruct_five_way_split(predictions, hans_config)
    write_json(path, manifest_payload(split, hans_config, mode="llm_main"))


def _skipped_discovery_payload(reason: str) -> dict[str, Any]:
    """Preserve discovery provenance when a refresh reuses archived candidates."""
    return {
        "mode": "llm_main",
        "oracle_debug": False,
        "split_role": "hypothesis_discovery",
        "generator": reason,
        "hypotheses": [],
    }


def _reuse_completed_refresh(
    *,
    round_dir: Path,
    round_index: int,
    adapter: str | None,
    archive: list[dict[str, Any]],
    generate_pool: bool,
    optimizer_state: str | None = None,
    validate_optimizer_state: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, dict[str, Any]],
    list[SyntheticNLIExample],
    dict[str, Any],
] | None:
    """Load a fully materialized refresh so resume never repeats teacher work."""
    required = [
        round_dir / "lifecycle.json",
        round_dir / "hypotheses.json",
    ]
    if generate_pool:
        required.extend(
            [
                round_dir / "synthetic" / "accepted.jsonl",
                round_dir / "synthetic" / "manifest.json",
            ]
        )
    if not any(path.exists() for path in required):
        return None
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Cannot safely resume a partially materialized refresh; missing "
            f"artifacts: {missing}"
        )
    lifecycle = read_json(round_dir / "lifecycle.json")
    verification_mode = str(lifecycle.get("verification_mode", "on"))
    if verification_mode not in {"on", "off"}:
        raise ValueError("Cached refresh has an invalid verification mode")
    verification_result = round_dir / "verification" / "results.json"
    if verification_mode == "on" and not verification_result.is_file():
        raise RuntimeError(
            "Cannot resume verification-ON refresh without verification results"
        )
    if verification_mode == "off" and verification_result.exists():
        raise RuntimeError(
            "Verification-OFF refresh unexpectedly contains verification results"
        )
    if int(lifecycle.get("refresh_round", -1)) != round_index:
        raise ValueError("Cached refresh round does not match the requested round")
    cached_adapter = lifecycle.get("checkpoint_adapter")
    if cached_adapter != adapter:
        raise ValueError(
            "Cached refresh was generated from a different checkpoint adapter"
        )
    if (
        validate_optimizer_state
        and lifecycle.get("checkpoint_optimizer_state") != optimizer_state
    ):
        raise ValueError(
            "Cached refresh was generated from a different optimizer state"
        )
    hypothesis_payload = read_json(round_dir / "hypotheses.json")
    candidates = [dict(item) for item in hypothesis_payload["hypotheses"]]
    candidate_ids = [str(item["id"]) for item in candidates]
    if candidate_ids != [str(item) for item in lifecycle["candidate_ids"]]:
        raise ValueError("Cached refresh candidate IDs are inconsistent")
    states = {
        str(key): dict(value) for key, value in lifecycle["states"].items()
    }
    next_active = [str(item) for item in lifecycle["next_active_ids"]]
    expected_active = sorted(
        key
        for key, value in states.items()
        if value["state"] in {"certified", "unresolved"}
    )
    if next_active != expected_active:
        raise ValueError("Cached refresh active IDs are inconsistent with states")
    archive_by_id = {str(item["id"]): dict(item) for item in archive}
    for item in candidates:
        archive_by_id[str(item["id"])] = item
    additions = (
        [
            SyntheticNLIExample.from_mapping(row)
            for row in read_jsonl(round_dir / "synthetic" / "accepted.jsonl")
        ]
        if generate_pool
        else []
    )
    if len(additions) != int(lifecycle["generated_training_examples"]):
        raise ValueError("Cached refresh synthetic count is inconsistent")
    reused = {**lifecycle, "resumed_from_completed_artifacts": True}
    return list(archive_by_id.values()), next_active, states, additions, reused


def _refresh_hypotheses(
    *,
    output: Path,
    base_config: dict[str, Any],
    round_index: int,
    adapter: str | None,
    shared_rows: list[dict[str, Any]],
    manifest: FrozenHansManifest,
    archive: list[dict[str, Any]],
    active_ids: list[str],
    generate_pool: bool,
    reuse_completed: bool = False,
    optimizer_state: str | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, dict[str, Any]],
    list[SyntheticNLIExample],
    dict[str, Any],
]:
    closed = require_mapping(base_config, "closed_loop")
    verification_mode = str(closed.get("verification_mode", "on"))
    if verification_mode not in {"on", "off"}:
        raise ValueError("closed_loop.verification_mode must be on or off")
    round_dir = output / "refresh" / f"round_{round_index:03d}"
    if reuse_completed and round_dir.exists():
        cost_aware = (
            closed.get("training_protocol")
            == "stateful_cost_aware_safe_stopping_v7"
        )
        reused = _reuse_completed_refresh(
            round_dir=round_dir,
            round_index=round_index,
            adapter=adapter,
            archive=archive,
            generate_pool=generate_pool,
            optimizer_state=optimizer_state,
            validate_optimizer_state=cost_aware,
        )
        if reused is not None:
            print(f"[RESUME] Reusing completed refresh artifacts for round {round_index}")
            return reused
    round_dir.mkdir(parents=True)
    config = _round_config(base_config, round_index=round_index, adapter=adapter)
    if (
        require_mapping(config, "closed_loop").get("training_protocol")
        == "stateful_cost_aware_safe_stopping_v7"
    ):
        require_mapping(config, "pilot")["optimizer_state_path"] = optimizer_state
    config_path = round_dir / "config.yaml"
    _write_yaml(config_path, config)
    predictions, discovery_errors = _prepare_current_predictions(
        shared_rows, manifest, config["model"], adapter
    )
    write_jsonl(round_dir / "predictions.jsonl", predictions)

    maximum_candidates = int(closed.get("max_hypotheses_per_round", 5))
    proposal_capacity = max(0, maximum_candidates - len(set(active_ids)))
    novel_search_performed = discovery_errors == 0 or proposal_capacity > 0
    if discovery_errors and proposal_capacity > 0:
        _run_stage("build_hypotheses.py", config_path, round_dir)
        proposal_payload = read_json(round_dir / "hypotheses.json")
        proposals = [dict(item) for item in proposal_payload["hypotheses"]]
    else:
        proposal_payload = _skipped_discovery_payload(
            (
                "skipped_no_discovery_errors"
                if discovery_errors == 0
                else "skipped_active_candidate_capacity_full"
            )
        )
        proposals = []
        _write_round_split_manifest(
            round_dir / "hans_split_v2.json",
            predictions,
            require_mapping(config, "hans"),
        )

    candidates, merge_log = merge_hypothesis_candidates(
        archive,
        proposals,
        active_ids=active_ids,
        maximum=maximum_candidates,
        near_duplicate_threshold=float(
            closed.get("hypothesis_near_duplicate_threshold", 0.78)
        ),
    )
    proposal_payload["hypotheses"] = candidates
    proposal_payload["closed_loop_merge"] = merge_log
    write_json(round_dir / "hypotheses.json", proposal_payload)
    write_json(round_dir / "hypothesis_merge.json", merge_log)
    if not (round_dir / "hans_split_v2.json").is_file():
        raise RuntimeError("Discovery stage did not write its split manifest")

    if candidates and verification_mode == "on":
        _run_stage("verify_hypotheses.py", config_path, round_dir)
        verification = read_json(round_dir / "verification" / "results.json")
        states = verification_states(verification)
    elif candidates:
        states = _bypassed_verification_states(candidates)
    else:
        states = {}
    next_active = sorted(
        hypothesis_id
        for hypothesis_id, value in states.items()
        if value["state"] in {"certified", "unresolved"}
    )
    archive_by_id = {str(item["id"]): dict(item) for item in archive}
    for item in candidates:
        archive_by_id[str(item["id"])] = dict(item)
    next_archive = list(archive_by_id.values())

    additions: list[SyntheticNLIExample] = []
    if generate_pool and next_active:
        _run_stage("generate_synthetic_interventions.py", config_path, round_dir)
        additions = [
            SyntheticNLIExample.from_mapping(row)
            for row in read_jsonl(round_dir / "synthetic" / "accepted.jsonl")
        ]
    lifecycle = {
        "refresh_round": round_index,
        "checkpoint_adapter": adapter,
        "checkpoint_optimizer_state": optimizer_state,
        "discovery_errors": discovery_errors,
        "proposal_capacity_before_verification": proposal_capacity,
        "novel_search_performed": novel_search_performed,
        "previous_active_ids": active_ids,
        "candidate_ids": [str(item["id"]) for item in candidates],
        "novel_hypothesis_ids": merge_log["novel_accepted"],
        "reactivated_hypothesis_ids": merge_log["historical_reactivated"],
        "verification_mode": verification_mode,
        "verification_performed": verification_mode == "on" and bool(candidates),
        "states": states,
        "next_active_ids": next_active,
        "generated_training_examples": len(additions),
        "verification_probes_training_eligible": (
            False if verification_mode == "on" else None
        ),
    }
    write_json(round_dir / "lifecycle.json", lifecycle)
    return next_archive, next_active, states, additions, lifecycle


def _cost_aware_decision(
    *,
    round_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    resume: bool,
    expected_checkpoint: dict[str, Any],
    expected_optimizer_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, Any]]:
    """Freeze or safely reuse pilot-gain and STOP/CONTINUE artifacts."""
    gain_paths = (
        round_dir / "gain" / "gain_estimates.json",
        round_dir / "gain" / "pilot_observations.jsonl",
        round_dir / "gain" / "gain_eval_applicability.json",
    )
    gain_presence = [path.is_file() for path in gain_paths]
    if all(gain_presence):
        if not resume:
            raise FileExistsError(
                f"Cost-aware gain artifacts already exist: {round_dir}"
            )
    elif any(gain_presence):
        raise RuntimeError(
            "Cannot safely resume partially materialized pilot-gain artifacts: "
            f"{[str(path) for path, exists in zip(gain_paths, gain_presence) if not exists]}"
        )
    else:
        _run_stage("run_main_pilot_gain.py", config_path, round_dir)

    decision_paths = (
        round_dir / "decision" / "decision.json",
        round_dir / "decision" / "candidates.json",
        round_dir / "decision" / "round_cost.json",
    )
    decision_presence = [path.is_file() for path in decision_paths]
    if all(decision_presence):
        if not resume:
            raise FileExistsError(f"Cost-aware decision already exists: {round_dir}")
    elif any(decision_presence):
        raise RuntimeError(
            "Cannot safely resume partially frozen cost-aware decision artifacts: "
            f"{[str(path) for path, exists in zip(decision_paths, decision_presence) if not exists]}"
        )
    else:
        _run_stage(
            "run_main_decision.py",
            config_path,
            round_dir,
            "--decision-only",
            "--defer-final-test",
        )

    gain_payload = read_json(gain_paths[0])
    if gain_payload.get("starting_checkpoint") != expected_checkpoint:
        raise ValueError(
            "Pilot-gain checkpoint differs from the current V7 checkpoint"
        )
    if gain_payload.get("starting_optimizer_state") != expected_optimizer_state:
        raise ValueError(
            "Pilot-gain optimizer state differs from the current V7 optimizer"
        )
    if gain_payload.get("main_training_policy_matched") is not True:
        raise ValueError(
            "Pilot gain did not use the stateful V7 early-stopping policy"
        )
    verification_payload = read_json(round_dir / "verification" / "results.json")
    gain_config = require_mapping(config, "gain")
    safe_config = require_mapping(config, "safe_stopping")
    expected_K = int(gain_config["K"])
    expected_T = int(gain_config["T"])
    expected_delta = float(gain_config["delta"])
    gain_bound = gain_payload.get("gain_bound")
    if not isinstance(gain_bound, dict) or any(
        gain_bound.get(key) != value
        for key, value in {
            "K": expected_K,
            "T": expected_T,
            "delta": expected_delta,
        }.items()
    ):
        raise ValueError("Pilot-gain confidence accounting differs from K/T/delta")
    assumption_audit = verification_assumption_audit(
        verification_payload,
        gamma=float(safe_config["verification_margin_gamma"]),
        expected_K=expected_K,
        expected_T=expected_T,
        expected_delta=float(require_mapping(config, "verification")["delta"]),
    )
    if (
        bool(safe_config.get("require_mechanical_assumptions", True))
        and not assumption_audit["mechanical_checks_pass"]
    ):
        raise RuntimeError(
            "Appendix mechanical assumptions failed; refusing to freeze a "
            "safe-stopping decision"
        )
    write_json(
        round_dir / "decision" / "theorem_assumption_audit.json",
        assumption_audit,
    )

    decision = read_json(decision_paths[0])
    candidates = read_json(decision_paths[1])
    cost = read_json(decision_paths[2])
    observations = read_jsonl(gain_paths[1], allow_empty=True)
    selected_hypothesis = decision.get("selected_hypothesis_id")
    excluded_ids = [
        str(item)
        for row in observations
        if row.get("hypothesis_id") == selected_hypothesis
        for item in row.get("train_ids", [])
    ]
    return decision, excluded_ids, candidates, {
        **cost,
        "theorem_assumption_audit": assumption_audit,
        "gain_estimates": gain_payload,
    }


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from src.evaluation.metrics import compute_hans_metrics

    return compute_hans_metrics(
        [str(row["gold"]) for row in rows],
        [str(row["prediction"]) for row in rows],
        [str(row["heuristic"]) for row in rows],
        [str(row["subcase"]) for row in rows],
    )


def _optimizer_steps_for_examples(
    example_count: int, config: LoraTrainingConfig
) -> int:
    batches_per_epoch = math.ceil(example_count / config.batch_size)
    return config.num_epochs * math.ceil(
        batches_per_epoch / config.gradient_accumulation_steps
    )


def _evaluate_checkpoint(
    *,
    model: dict[str, Any],
    adapter: str | None,
    gain_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    mnli_config: dict[str, Any],
    output: Path,
    audit_adapter: bool,
    report_both_hans_rules: bool = False,
) -> dict[str, Any]:
    """Evaluate one final branch and keep its predictions fully separate."""
    output.mkdir(parents=True, exist_ok=True)
    student = _student(model, adapter)
    load_audit = audit_loaded_adapter(student.model, adapter) if audit_adapter else None
    if load_audit is not None:
        write_json(output / "adapter_load_audit.json", load_audit)
    gain_predictions, gain_metrics = evaluate_hans(
        student,
        _hans(gain_rows),
        batch_size=int(model.get("batch_size", 4)),
    )
    write_jsonl(output / "gain_report_predictions.jsonl", gain_predictions)
    prediction_rule_metrics = None
    if report_both_hans_rules:
        final_predictions, prediction_rule_metrics = evaluate_hans_prediction_rules(
            student,
            _hans(final_rows),
            batch_size=int(model.get("batch_size", 4)),
        )
        hans_metrics = prediction_rule_metrics["three_way_to_binary"]
    else:
        final_predictions, hans_metrics = evaluate_hans(
            student,
            _hans(final_rows),
            batch_size=int(model.get("batch_size", 4)),
        )
    write_jsonl(output / "hans_final_predictions.jsonl", final_predictions)
    mnli_metrics: dict[str, Any] = {}
    for name, split in (
        ("matched", str(mnli_config.get("matched_split", "validation_matched"))),
        (
            "mismatched",
            str(mnli_config.get("mismatched_split", "validation_mismatched")),
        ),
    ):
        examples = load_mnli(
            split=split,
            dataset_name=str(mnli_config.get("name", "nyu-mll/glue")),
            dataset_config=str(mnli_config.get("config", "mnli")),
            cache_dir=mnli_config.get("cache_dir"),
            max_examples=mnli_config.get("max_examples"),
            drop_unlabeled=True,
        )
        _, mnli_metrics[name] = evaluate_mnli(
            student, examples, batch_size=int(model.get("batch_size", 4))
        )
    del student
    _clear_cuda_cache()
    return {
        "hans_final": hans_metrics,
        "hans_final_prediction_rules": prediction_rule_metrics,
        "gain_report": gain_metrics,
        "mnli": mnli_metrics,
        "adapter_load_audit": load_audit,
    }


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    if args.seed is not None:
        _apply_seed_override(config, args.seed)
    closed = require_mapping(config, "closed_loop")
    benchmark = require_mapping(config, "stage2_benchmark")
    require_mapping(config, "teacher")
    require_mapping(config, "model")
    require_mapping(config, "training")
    require_mapping(config, "hypothesis_generation")
    require_mapping(config, "verification")
    require_mapping(config, "synthetic_generation")
    require_mapping(config, "synthetic_validation")
    seed = int(closed.get("seed", 42))
    total_budget = int(closed.get("total_budget", 128))
    per_round = int(closed.get("per_round_examples", 32))
    max_rounds = int(closed.get("max_rounds", 4))
    training_protocol = str(
        closed.get("training_protocol", "legacy_independent_rounds_v6")
    )
    if training_protocol not in {
        "legacy_independent_rounds_v6",
        "dual_fair_continuation_and_consolidation_v7",
        "stateful_early_stopping_verification_ablation_v7",
        "stateful_cost_aware_safe_stopping_v7",
    }:
        raise ValueError(f"Unsupported closed-loop training protocol: {training_protocol}")
    dual_v7 = training_protocol == "dual_fair_continuation_and_consolidation_v7"
    fixed_budget_early_stop_v7 = (
        training_protocol == "stateful_early_stopping_verification_ablation_v7"
    )
    cost_aware_v7 = training_protocol == "stateful_cost_aware_safe_stopping_v7"
    early_stop_v7 = fixed_budget_early_stop_v7 or cost_aware_v7
    stateful_training = dual_v7 or early_stop_v7
    verification_mode = str(closed.get("verification_mode", "on"))
    if verification_mode not in {"on", "off"}:
        raise ValueError("closed_loop.verification_mode must be on or off")
    if cost_aware_v7 and verification_mode != "on":
        raise ValueError(
            "Verified-failure safe stopping requires verification_mode: on"
        )
    expected_policy = (
        "failure_targeted_verified"
        if verification_mode == "on"
        else "failure_targeted_verification_bypassed"
    )
    if early_stop_v7 and config.get("stage2_policy") != expected_policy:
        raise ValueError(
            "stage2_policy and closed_loop.verification_mode are inconsistent"
        )
    if total_budget != per_round * max_rounds or per_round % 2:
        raise ValueError("Closed-loop budget must equal max_rounds * even per-round size")
    if cost_aware_v7:
        require_mapping(config, "active_learning")
        require_mapping(config, "pilot")
        require_mapping(config, "gain")
        require_mapping(config, "decision")
        require_mapping(config, "cost")
        require_mapping(config, "intervention")
        require_mapping(config, "safe_stopping")
        if max_rounds != 8 or per_round != 32 or total_budget != 256:
            raise ValueError("Cost-aware V7 protocol is fixed to at most 8 x 32")
    set_deterministic_seed(seed)

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        if args.overwrite:
            shutil.rmtree(output)
        elif args.resume:
            if (output / ".complete").exists():
                raise FileExistsError(f"Refusing to resume completed run {output}")
            saved_config = load_yaml_with_extends(output / "config.yaml")
            if saved_config != config:
                raise ValueError(
                    "Resume config differs from the incomplete run config snapshot"
                )
        else:
            raise FileExistsError(f"Refusing to overwrite {output}")
    elif args.resume:
        raise FileNotFoundError(f"No incomplete output exists to resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        _write_yaml(output / "config.yaml", config)

    manifest = FrozenHansManifest.from_mapping(
        read_json(Path(str(benchmark["frozen_manifest_path"])))
    )
    shared_path = Path(str(closed["shared_predictions_path"]))
    shared_rows = read_jsonl(shared_path)
    shared_index = {str(row["id"]): row for row in shared_rows}
    source_pool = Path(str(closed["initial_pool_dir"]))
    pool = [
        SyntheticNLIExample.from_mapping(row)
        for row in read_jsonl(source_pool / "synthetic" / "accepted.jsonl")
    ]
    initial_hypotheses = read_json(source_pool / "hypotheses.json")["hypotheses"]
    archive = [dict(item) for item in initial_hypotheses]
    initial_manifest = read_json(source_pool / "synthetic" / "manifest.json")
    states = {
        str(key): dict(value)
        for key, value in initial_manifest["hypothesis_verification_states"].items()
    }
    # The v5 initial family was not predeclared as one of v6's repeated looks.
    # Treat all five initial candidates as provisional until their first fresh
    # v6 re-verification; this prevents v6 from inheriting a premature drop.
    active_ids = sorted(str(item["id"]) for item in archive)
    for hypothesis_id in active_ids:
        if states[hypothesis_id]["state"] == "refuted":
            states[hypothesis_id] = {
                **states[hypothesis_id],
                "state": "unresolved",
                "certified": False,
                "v6_initial_status": "provisional_recheck_required",
            }

    subset = read_json(Path(str(closed["stopping_subset_manifest"])))
    gain_rows = [shared_index[item] for item in manifest.gain_eval_ids]
    stopping_ids, report_ids = validate_stopping_subset_manifest(
        subset,
        gain_rows,
        manifest.gain_eval_ids,
        frozen_manifest_hash=manifest.manifest_hash,
    )
    stopping_examples = _hans([shared_index[item] for item in stopping_ids])
    base_stopping_rows = [shared_index[item] for item in stopping_ids]
    learning_curve = [{"round": 0, "metrics": _metrics_from_rows(base_stopping_rows)}]

    model = require_mapping(config, "model")
    model["device_map"] = os.environ.get("CLOSED_LOOP_STUDENT_DEVICE", "cuda:1")
    training_mapping = require_mapping(config, "training")
    training = LoraTrainingConfig.from_mapping(training_mapping)
    early_stopping = (
        EarlyStoppingConfig.from_mapping(training_mapping)
        if early_stop_v7
        else None
    )
    selection = require_mapping(closed, "selection")
    current_adapter = model.get("adapter_path")
    current_optimizer_state: str | None = None
    committed_optimizer_step = 0
    selected_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    cumulative_hypothesis_counts: Counter[str] = Counter()
    round_records: list[dict[str, Any]] = []
    cumulative_modeled_cost = 0.0
    cost_aware_stop_record: dict[str, Any] | None = None
    stop_reason = "maximum_budget_reached"
    start_round = 1

    rounds_path = output / "rounds.jsonl"
    selected_path = output / "selected_training_examples.jsonl"
    resume_log_presence = (rounds_path.is_file(), selected_path.is_file())
    if args.resume and any(resume_log_presence) and not all(resume_log_presence):
        raise RuntimeError(
            "Resume found only one of the round/selected-example logs"
        )
    resume_from_initialization = args.resume and not any(resume_log_presence)
    if resume_from_initialization:
        unexpected = [
            path
            for path in output.iterdir()
            if path.name != "config.yaml"
            and not (cost_aware_v7 and path.name == "refresh")
        ]
        if unexpected:
            raise RuntimeError(
                "Cannot restart a pre-round resume with unexpected artifacts: "
                f"{[str(path) for path in unexpected]}"
            )
        print("[RESUME] Config snapshot exists; restarting safely from round 1")

    if args.resume and not resume_from_initialization:
        round_records = [dict(item) for item in read_jsonl(rounds_path)]
        if not round_records:
            raise RuntimeError("Resume found no completed training round")
        expected_rounds = list(range(1, len(round_records) + 1))
        if [int(item["round"]) for item in round_records] != expected_rounds:
            raise ValueError("Completed round records are not contiguous from round 1")
        if len(round_records) > max_rounds:
            raise ValueError("Resume contains more rounds than the configured maximum")
        selected_rows = [dict(item) for item in read_jsonl(selected_path)]
        selected_ids = [str(item["id"]) for item in selected_rows]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("Resume selected-example log contains duplicate IDs")
        expected_selected = len(round_records) * per_round
        if len(selected_rows) != expected_selected:
            raise ValueError(
                "Resume selected-example count differs from completed rounds"
            )
        last_record = round_records[-1]
        if int(last_record["cumulative_examples"]) != expected_selected:
            raise ValueError("Resume cumulative example metadata is inconsistent")
        current_adapter = str(last_record["ending_adapter"])
        optimizer_record = last_record.get("optimizer_continuity")
        if stateful_training:
            if not isinstance(optimizer_record, dict):
                raise ValueError("Resume round has no optimizer continuity record")
            current_optimizer_state = str(optimizer_record["ending_state"])
            committed_optimizer_step = int(optimizer_record["ending_step"])
            if dual_v7 and committed_optimizer_step != len(round_records):
                raise ValueError("Resume optimizer step does not match completed rounds")
        for required_path in (current_adapter, current_optimizer_state):
            if required_path is None:
                continue
            path = Path(required_path)
            if path.suffix == ".pt":
                exists = path.is_file()
            else:
                exists = (path / "adapter_model.safetensors").is_file()
            if not exists:
                raise FileNotFoundError(f"Resume checkpoint artifact is missing: {path}")
        cumulative_hypothesis_counts.update(
            str(item["hypothesis_id"])
            for item in selected_rows
            if item["pair_member"] == "base"
        )
        learning_curve.extend(
            {
                "round": int(item["round"]),
                "metrics": dict(item["stopping_metrics"]),
            }
            for item in round_records
        )
        if cost_aware_v7:
            last_cost = last_record.get("cost_aware")
            if not isinstance(last_cost, dict):
                raise ValueError("Cost-aware resume round has no cost ledger")
            ledger = last_cost.get("cost_ledger")
            if not isinstance(ledger, dict):
                raise ValueError("Cost-aware resume round has an invalid cost ledger")
            cumulative_modeled_cost = float(ledger["cumulative_after"])
        # Replay only completed refresh metadata to restore the hypothesis and
        # pool state. No teacher/model work is repeated here.
        for replay_round in range(2, len(round_records) + 1):
            replay_dir = output / "refresh" / f"round_{replay_round:03d}"
            reused = _reuse_completed_refresh(
                round_dir=replay_dir,
                round_index=replay_round,
                adapter=str(round_records[replay_round - 2]["ending_adapter"]),
                archive=archive,
                generate_pool=True,
            )
            if reused is None:
                raise RuntimeError(
                    f"Resume is missing completed refresh {replay_round}"
                )
            archive, active_ids, states, additions, _ = reused
            pool, _ = _merge_pool(
                pool,
                additions,
                threshold=float(closed.get("pool_near_duplicate_threshold", 0.92)),
            )
        start_round = len(round_records) + 1
        print(
            f"[RESUME] Restored {len(round_records)} completed round(s), "
            f"{len(selected_ids)} selected examples, optimizer step "
            f"{committed_optimizer_step}"
        )

    for round_index in range(start_round, max_rounds + 1):
        refresh_record: dict[str, Any] | None = None
        cost_decision: dict[str, Any] | None = None
        cost_candidates: dict[str, Any] | None = None
        cost_payload: dict[str, Any] | None = None
        pilot_excluded_ids: list[str] = []
        cost_decision_seconds = 0.0
        refresh_seconds = 0.0
        refresh_teacher_usage = _teacher_usage(output / "refresh" / f"round_{round_index:03d}")
        if round_index > 1 or cost_aware_v7:
            refresh_started = time.perf_counter()
            archive, active_ids, states, additions, refresh_record = _refresh_hypotheses(
                output=output,
                base_config=config,
                round_index=round_index,
                adapter=current_adapter,
                shared_rows=shared_rows,
                manifest=manifest,
                archive=archive,
                active_ids=active_ids,
                generate_pool=True,
                reuse_completed=args.resume,
                optimizer_state=current_optimizer_state,
            )
            pool, pool_merge = _merge_pool(
                pool,
                additions,
                threshold=float(closed.get("pool_near_duplicate_threshold", 0.92)),
            )
            refresh_record["pool_merge"] = pool_merge
            refresh_seconds = time.perf_counter() - refresh_started
            refresh_teacher_usage = _teacher_usage(
                output / "refresh" / f"round_{round_index:03d}"
            )
        if not active_ids:
            stop_reason = (
                "all_tracked_hypotheses_refuted_and_no_novel_hypothesis"
                if refresh_record is None
                or refresh_record.get("novel_search_performed") is True
                else "all_tracked_hypotheses_refuted_novel_search_deferred_by_candidate_cap"
            )
            break

        if cost_aware_v7:
            refresh_dir = output / "refresh" / f"round_{round_index:03d}"
            write_jsonl(
                refresh_dir / "synthetic" / "closed_loop_eligible_pool.jsonl",
                [
                    item.to_dict()
                    for item in pool
                    if item.hypothesis_id in set(active_ids)
                ],
            )
            decision_started = time.perf_counter()
            cost_decision, pilot_excluded_ids, cost_candidates, cost_payload = (
                _cost_aware_decision(
                    round_dir=refresh_dir,
                    config_path=refresh_dir / "config.yaml",
                    config=_round_config(
                        config, round_index=round_index, adapter=current_adapter
                    ),
                    resume=args.resume,
                    expected_checkpoint=checkpoint_descriptor(
                        {**model, "adapter_path": current_adapter}
                    ),
                    expected_optimizer_state={
                        "path": current_optimizer_state,
                        "local_sha256": hash_local_path(current_optimizer_state),
                    },
                )
            )
            cost_decision_seconds = time.perf_counter() - decision_started
            refresh_teacher_usage = _teacher_usage(refresh_dir)
            if cost_decision.get("action") == "stop":
                incurred = float(cost_payload["incurred_exploration_cost"])
                ledger = cumulative_cost_ledger(
                    cumulative_modeled_cost,
                    incurred_exploration=incurred,
                    realized_intervention=0.0,
                )
                cumulative_modeled_cost = ledger["cumulative_after"]
                stop_reason = str(cost_decision.get("reason", "cost_aware_stop"))
                cost_aware_stop_record = {
                    "round": round_index,
                    "cumulative_examples": len(selected_ids),
                    "starting_checkpoint": checkpoint_descriptor(
                        {**model, "adapter_path": current_adapter}
                    ),
                    "decision": cost_decision,
                    "candidates": cost_candidates,
                    "cost": cost_payload,
                    "cost_ledger": ledger,
                    "resource_usage": {
                        "refresh_wall_seconds": refresh_seconds,
                        "pilot_and_decision_wall_seconds": cost_decision_seconds,
                        "teacher": refresh_teacher_usage,
                    },
                    "stop_reason": stop_reason,
                    "checkpoint_unchanged": True,
                }
                write_json(output / "cost_aware_stop.json", cost_aware_stop_record)
                break

        selected_set = set(selected_ids)
        if cost_aware_v7:
            assert cost_decision is not None
            selected_hypothesis = str(cost_decision["selected_hypothesis_id"])
            eligible = [
                item for item in pool if item.hypothesis_id == selected_hypothesis
            ]
            acquisition_excluded_ids = [*selected_ids, *pilot_excluded_ids]
        else:
            eligible = [
                item
                for item in pool
                if item.hypothesis_id in set(active_ids) or item.id in selected_set
            ]
            acquisition_excluded_ids = selected_ids
        starting_adapter = current_adapter
        student = _student(model, current_adapter)
        starting_adapter_audit = (
            audit_loaded_adapter(student.model, current_adapter)
            if stateful_training and current_adapter
            else None
        )
        acquisition_started = time.perf_counter()
        student.progress_label = f"closed-loop round {round_index} probability scoring"
        probabilities = student.predict_proba(
            [item.premise for item in eligible],
            [item.hypothesis for item in eligible],
            batch_size=int(model.get("batch_size", 4)),
        )
        label_index = {label: index for index, label in enumerate(MNLI_LABELS)}
        nll = {
            item.id: -math.log(
                max(float(probabilities[index][label_index[item.label]]), 1e-12)
            )
            for index, item in enumerate(eligible)
        }
        entropy = {
            item.id: -sum(
                float(value) * math.log(max(float(value), 1e-12))
                for value in probabilities[index]
            )
            / math.log(len(MNLI_LABELS))
            for index, item in enumerate(eligible)
        }
        student.progress_label = f"closed-loop round {round_index} embedding extraction"
        vectors = student.encode_pair_embeddings(
            [item.premise for item in eligible],
            [item.hypothesis for item in eligible],
            batch_size=int(selection.get("embedding_batch_size", 8)),
        )
        evidence = {key: states[key] for key in active_ids}
        for item in eligible:
            if item.id in selected_set and item.hypothesis_id not in evidence:
                evidence[item.hypothesis_id] = {"state": "unresolved"}
        acquired, diagnostics = select_closed_loop_soft_pairs(
            eligible,
            budget=per_round,
            seed=seed + round_index - 1,
            nll_scores=nll,
            entropy_scores=entropy,
            embeddings={item.id: vectors[index] for index, item in enumerate(eligible)},
            verification_evidence=evidence,
            excluded_ids=acquisition_excluded_ids,
            prior_hypothesis_pair_counts=cumulative_hypothesis_counts,
            nll_weight=float(selection.get("nll_weight", 0.65)),
            entropy_weight=float(selection.get("entropy_weight", 0.35)),
            novelty_weight=float(selection.get("novelty_weight", 0.50)),
            hypothesis_bonus_weight=float(
                selection.get("hypothesis_bonus_weight", 0.35)
            ),
            new_hypothesis_bonus_weight=float(
                selection.get("new_hypothesis_bonus_weight", 0.50)
            ),
        )
        acquisition_seconds = time.perf_counter() - acquisition_started
        if len(acquired) != per_round:
            del student
            _clear_cuda_cache()
            stop_reason = "insufficient_verified_synthetic_pairs"
            break
        round_dir = output / "training" / f"round_{round_index:03d}"
        training_started = time.perf_counter()
        optimizer_state_output = round_dir / "optimizer_state.pt"
        stateful_result: PairedTrainingResult | None = None
        if early_stop_v7:
            assert early_stopping is not None

            def validation_callback() -> dict[str, Any]:
                student.progress_label = (
                    f"round {round_index} early-stopping validation"
                )
                validation_probabilities = student.predict_proba(
                    [item.premise for item in stopping_examples],
                    [item.hypothesis for item in stopping_examples],
                    batch_size=int(model.get("batch_size", 4)),
                )
                return _binary_hans_validation(
                    validation_probabilities, stopping_examples
                )

            stateful_result = train_lora_stateful_early_stopping(
                student,
                [item.to_mnli() for item in acquired],
                training,
                early_stopping,
                validation_callback,
                round_dir / "adapter",
                seed + round_index - 1,
                optimizer_state_input=current_optimizer_state,
                optimizer_state_output=optimizer_state_output,
            )
            if stateful_result.best_epoch is None:
                raise RuntimeError("Early-stopping training did not select a best epoch")
            steps_per_epoch = math.ceil(
                math.ceil(per_round / training.batch_size)
                / training.gradient_accumulation_steps
            )
            expected_start = committed_optimizer_step
            expected_end = (
                expected_start + stateful_result.best_epoch * steps_per_epoch
            )
            if (
                stateful_result.optimizer_step_start != expected_start
                or stateful_result.optimizer_step_end != expected_end
            ):
                raise RuntimeError(
                    "Early-stopping AdamW continuity/restore check failed: "
                    f"expected {expected_start}->{expected_end}, observed "
                    f"{stateful_result.optimizer_step_start}->"
                    f"{stateful_result.optimizer_step_end}"
                )
            committed_optimizer_step = expected_end
            loss = stateful_result.total_loss
            current_optimizer_state = str(optimizer_state_output)
        elif dual_v7:
            stateful_result = train_lora_stateful(
                student,
                [item.to_mnli() for item in acquired],
                training,
                round_dir / "adapter",
                seed + round_index - 1,
                optimizer_state_input=current_optimizer_state,
                optimizer_state_output=optimizer_state_output,
            )
            expected_start = (round_index - 1) * _optimizer_steps_for_examples(
                per_round, training
            )
            expected_end = round_index * _optimizer_steps_for_examples(
                per_round, training
            )
            if (
                stateful_result.optimizer_step_start != expected_start
                or stateful_result.optimizer_step_end != expected_end
            ):
                raise RuntimeError(
                    "Closed-loop AdamW continuity check failed: "
                    f"expected {expected_start}->{expected_end}, observed "
                    f"{stateful_result.optimizer_step_start}->"
                    f"{stateful_result.optimizer_step_end}"
                )
            committed_optimizer_step = expected_end
            loss = stateful_result.total_loss
            current_optimizer_state = str(optimizer_state_output)
        else:
            loss = train_lora(
                student,
                [item.to_mnli() for item in acquired],
                training,
                round_dir / "adapter",
                seed + round_index - 1,
            )
        training_seconds = time.perf_counter() - training_started
        current_adapter = str(round_dir / "adapter")
        ending_adapter_audit = (
            audit_loaded_adapter(student.model, current_adapter)
            if stateful_training
            else None
        )
        student.progress_label = f"closed-loop round {round_index} stopping accuracy"
        stopping_started = time.perf_counter()
        stopping_predictions, stopping_metrics = evaluate_hans(
            student,
            stopping_examples,
            batch_size=int(model.get("batch_size", 4)),
        )
        stopping_seconds = time.perf_counter() - stopping_started
        elapsed = training_seconds + stopping_seconds
        del student
        _clear_cuda_cache()
        write_jsonl(round_dir / "stopping_predictions.jsonl", stopping_predictions)
        ids = [item.id for item in acquired]
        selected_ids.extend(ids)
        selected_rows.extend(item.to_dict() for item in acquired)
        cumulative_hypothesis_counts.update(
            item.hypothesis_id for item in acquired if item.pair_member == "base"
        )
        learning_curve.append({"round": round_index, "metrics": stopping_metrics})
        cost_aware_round: dict[str, Any] | None = None
        if cost_aware_v7:
            assert cost_decision is not None
            assert cost_candidates is not None
            assert cost_payload is not None
            assert stateful_result is not None
            rates, currency = cost_rates_from_config(
                require_mapping(config, "cost")
            )
            validation_passes = len(stateful_result.epoch_history)
            realized_intervention = compute_cost(
                CostUsage(
                    lora_training_runs=1,
                    lora_training_gpu_seconds=max(
                        0.0, training_seconds - stateful_result.validation_seconds
                    ),
                    evaluation_runs=validation_passes + 3,
                    evaluation_items=(
                        2 * len(eligible)
                        + (validation_passes + 1) * len(stopping_examples)
                    ),
                    evaluation_gpu_seconds=(
                        acquisition_seconds
                        + stateful_result.validation_seconds
                        + stopping_seconds
                    ),
                ),
                rates,
                currency=currency,
            ).to_dict()
            ledger = cumulative_cost_ledger(
                cumulative_modeled_cost,
                incurred_exploration=float(
                    cost_payload["incurred_exploration_cost"]
                ),
                realized_intervention=float(realized_intervention["total"]),
            )
            cumulative_modeled_cost = ledger["cumulative_after"]
            cost_aware_round = {
                "decision": cost_decision,
                "candidates": cost_candidates,
                "gain_estimates": cost_payload["gain_estimates"],
                "theorem_assumption_audit": cost_payload[
                    "theorem_assumption_audit"
                ],
                "cost_at_decision": {
                    key: value
                    for key, value in cost_payload.items()
                    if key not in {"gain_estimates", "theorem_assumption_audit"}
                },
                "realized_intervention_cost": realized_intervention,
                "cost_ledger": ledger,
                "pilot_training_ids_excluded_from_main_acquisition": (
                    pilot_excluded_ids
                ),
                "selected_hypothesis_id": cost_decision[
                    "selected_hypothesis_id"
                ],
            }
        record = {
            "round": round_index,
            "requested_examples": per_round,
            "actual_examples": len(acquired),
            "cumulative_examples": len(selected_ids),
            "starting_adapter": starting_adapter,
            "ending_adapter": current_adapter,
            "train_loss": loss,
            "adapter_continuity": {
                "starting_adapter_load": starting_adapter_audit,
                "ending_adapter_save": ending_adapter_audit,
            }
            if stateful_training
            else None,
            "optimizer_continuity": {
                "state_loaded": stateful_result.optimizer_state_loaded,
                "starting_step": stateful_result.optimizer_step_start,
                "ending_step": stateful_result.optimizer_step_end,
                "last_attempted_step": stateful_result.optimizer_step_last_attempted,
                "ending_state": current_optimizer_state,
                "restore_to_best_verified": (
                    stateful_result.optimizer_step_end
                    <= stateful_result.optimizer_step_last_attempted
                ),
            }
            if stateful_result is not None
            else None,
            "training": (
                {
                    "effective_batch_size": (
                        training.batch_size * training.gradient_accumulation_steps
                    ),
                    "micro_batch_size": training.batch_size,
                    "gradient_accumulation_steps": (
                        training.gradient_accumulation_steps
                    ),
                    "max_epochs": (
                        early_stopping.max_epochs
                        if early_stopping is not None
                        else training.num_epochs
                    ),
                    "epochs_completed": stateful_result.epochs_completed,
                    "best_epoch": stateful_result.best_epoch,
                    "best_validation_loss": stateful_result.best_validation_loss,
                    "stopped_early": stateful_result.stopped_early,
                    "epoch_history": list(stateful_result.epoch_history),
                }
                if stateful_result is not None
                else None
            ),
            "resource_usage": {
                "refresh_wall_seconds": refresh_seconds,
                "acquisition_gpu_wall_seconds": acquisition_seconds,
                "training_gpu_wall_seconds": max(
                    0.0,
                    training_seconds
                    - (
                        stateful_result.validation_seconds
                        if stateful_result is not None
                        else 0.0
                    ),
                ),
                "validation_gpu_wall_seconds": (
                    stateful_result.validation_seconds
                    if stateful_result is not None
                    else 0.0
                ),
                "stopping_evaluation_gpu_wall_seconds": stopping_seconds,
                "cost_aware_pilot_and_decision_wall_seconds": (
                    cost_decision_seconds
                ),
                "teacher": refresh_teacher_usage,
            },
            "wall_clock_seconds_training_and_stopping_eval": elapsed,
            "active_hypothesis_ids": active_ids,
            "hypothesis_states": {
                key: states[key] for key in active_ids if key in states
            },
            "verification_refresh": refresh_record,
            "selection": diagnostics,
            "stopping_metrics": stopping_metrics,
            "cost_aware": cost_aware_round,
        }
        round_records.append(record)
        if stateful_result is not None:
            write_json(
                round_dir / "training_history.json",
                {
                    "round": round_index,
                    "best_epoch": stateful_result.best_epoch,
                    "best_validation_loss": stateful_result.best_validation_loss,
                    "stopped_early": stateful_result.stopped_early,
                    "optimizer_step_start": stateful_result.optimizer_step_start,
                    "optimizer_step_end": stateful_result.optimizer_step_end,
                    "optimizer_step_last_attempted": (
                        stateful_result.optimizer_step_last_attempted
                    ),
                    "epochs": list(stateful_result.epoch_history),
                },
            )
        write_jsonl(output / "rounds.jsonl", round_records)
        write_jsonl(output / "selected_training_examples.jsonl", selected_rows)

    if fixed_budget_early_stop_v7 and (
        len(round_records) != max_rounds or len(selected_ids) != total_budget
    ):
        raise RuntimeError(
            "Fixed-budget ablation is incomplete: expected "
            f"{max_rounds} rounds/{total_budget} examples, observed "
            f"{len(round_records)} rounds/{len(selected_ids)} examples. "
            "No final metrics or completion marker were written."
        )

    final_audit = None
    final_audit_seconds = 0.0
    final_audit_teacher_usage = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "teacher_wall_clock_seconds": 0.0,
        "calls": 0.0,
    }
    if (
        not cost_aware_v7
        and bool(closed.get("final_hypothesis_audit", True))
        and current_adapter
    ):
        final_audit_started = time.perf_counter()
        archive, final_active, final_states, _, final_audit = _refresh_hypotheses(
            output=output,
            base_config=config,
            round_index=max_rounds + 1,
            adapter=current_adapter,
            shared_rows=shared_rows,
            manifest=manifest,
            archive=archive,
            active_ids=active_ids,
            generate_pool=False,
            reuse_completed=args.resume,
        )
        final_audit_seconds = time.perf_counter() - final_audit_started
        final_audit_teacher_usage = _teacher_usage(
            output / "refresh" / f"round_{max_rounds + 1:03d}"
        )
        if not final_active:
            stop_reason = (
                "all_tracked_hypotheses_refuted_and_no_novel_hypothesis"
                if final_audit.get("novel_search_performed") is True
                else "all_tracked_hypotheses_refuted_novel_search_deferred_by_candidate_cap"
            )
        else:
            stop_reason = "maximum_budget_reached_with_remaining_hypotheses"

    gain_report_rows = [shared_index[item] for item in report_ids]
    final_rows = read_jsonl(Path(str(benchmark["sealed_final_test_path"])))
    mnli_config = require_mapping(config, "mnli_evaluation")
    leakage = audit_final_test_overlap(
        selected_rows,
        final_rows,
        near_duplicate_threshold=float(benchmark.get("near_duplicate_threshold", 0.92)),
    )
    write_json(output / "leakage_audit.json", leakage)
    if leakage["id_overlap"]:
        raise ValueError("Final-test ID overlap detected")
    write_json(output / "hypothesis_archive.json", {"hypotheses": archive})

    if early_stop_v7:
        if fixed_budget_early_stop_v7 and (
            current_adapter is None or not round_records
        ):
            raise RuntimeError("Early-stopping v7 produced no continuation checkpoint")
        if cost_aware_v7 and cost_aware_stop_record is None:
            stop_reason = "maximum_rounds_reached"
        evaluation_started = time.perf_counter()
        evaluation = _evaluate_checkpoint(
            model=model,
            adapter=current_adapter,
            gain_rows=gain_report_rows,
            final_rows=final_rows,
            mnli_config=mnli_config,
            output=output,
            audit_adapter=current_adapter is not None,
            report_both_hans_rules=cost_aware_v7,
        )
        final_evaluation_seconds = time.perf_counter() - evaluation_started
        teacher_totals = dict(final_audit_teacher_usage)
        resource_totals = {
            "refresh_wall_seconds": final_audit_seconds,
            "acquisition_gpu_wall_seconds": 0.0,
            "training_gpu_wall_seconds": 0.0,
            "validation_gpu_wall_seconds": 0.0,
            "stopping_evaluation_gpu_wall_seconds": 0.0,
            "cost_aware_pilot_and_decision_wall_seconds": 0.0,
            "final_evaluation_gpu_wall_seconds": final_evaluation_seconds,
        }
        for row in round_records:
            usage = row["resource_usage"]
            for key in (
                "refresh_wall_seconds",
                "acquisition_gpu_wall_seconds",
                "training_gpu_wall_seconds",
                "validation_gpu_wall_seconds",
                "stopping_evaluation_gpu_wall_seconds",
                "cost_aware_pilot_and_decision_wall_seconds",
            ):
                resource_totals[key] += float(usage[key])
            for key in teacher_totals:
                teacher_totals[key] += float(usage["teacher"][key])
        if cost_aware_stop_record is not None:
            stop_usage = cost_aware_stop_record["resource_usage"]
            resource_totals["refresh_wall_seconds"] += float(
                stop_usage["refresh_wall_seconds"]
            )
            resource_totals[
                "cost_aware_pilot_and_decision_wall_seconds"
            ] += float(stop_usage["pilot_and_decision_wall_seconds"])
            for key in teacher_totals:
                teacher_totals[key] += float(stop_usage["teacher"][key])
        resource_totals["teacher"] = teacher_totals
        resource_totals["scope"] = (
            "current_ablation_run_only_shared_initial_pool_cost_excluded"
        )
        resource_totals["gpu_fields"] = (
            "measured_wall_clock_proxies_not_hardware_utilization_integrals"
        )
        fixed_budget_comparable = len(selected_ids) == total_budget
        result = {
            "method": (
                "ours_v7_cost_aware_safe_stopping"
                if cost_aware_v7
                else f"ours_v7_early_stop_verification_{verification_mode}"
            ),
            "experiment": (
                "v7_cost_aware_safe_stopping"
                if cost_aware_v7
                else "verification_on_off_ablation"
            ),
            "seed": seed,
            "budget": total_budget,
            "requested_budget": total_budget,
            "rounds": len(round_records),
            "rounds_attempted": (
                len(round_records)
                + (1 if cost_aware_stop_record is not None else 0)
            ),
            "per_round_examples": per_round,
            "actual_training_examples": len(selected_ids),
            "fixed_budget_comparable": fixed_budget_comparable,
            "final_metrics_reportable": (
                True if cost_aware_v7 else fixed_budget_comparable
            ),
            "verification_mode": verification_mode,
            "verification_is_only_ablation_variable": not cost_aware_v7,
            "synthetic_qc_enabled": True,
            "acquisition_stopping": (
                "gain_ucb_minus_lambda_cost_up_to_8x32"
                if cost_aware_v7
                else "disabled_fixed_4x32"
            ),
            "cost_aware_stopping": cost_aware_v7,
            "training_objective": "per_example_causal_lm_cross_entropy",
            "training_policy": {
                "stateful_lora": True,
                "stateful_adamw": True,
                "micro_batch_size": training.batch_size,
                "gradient_accumulation_steps": training.gradient_accumulation_steps,
                "effective_batch_size": (
                    training.batch_size * training.gradient_accumulation_steps
                ),
                "max_epochs": early_stopping.max_epochs,
                "patience": early_stopping.patience,
                "min_delta": early_stopping.min_delta,
                "validation_metric": "binary_hans_negative_log_likelihood",
                "validation_source": "predeclared_gain_eval_stopping_subset",
                "restore_best_checkpoint": True,
            },
            "protocol": (
                "discover_verify_generate_qc_pilot_gain_ucb_cost_decide_"
                "select_stateful_train_early_stop_refresh_v7"
                if cost_aware_v7
                else "discover_verify_or_bypass_generate_qc_select_"
                "stateful_train_early_stop_refresh_v7"
            ),
            "stop_reason": stop_reason,
            "optimizer_steps": committed_optimizer_step,
            "optimizer_continuity_verified": all(
                bool(row["adapter_continuity"]["ending_adapter_save"]["matched"])
                and bool(row["optimizer_continuity"]["restore_to_best_verified"])
                and (
                    row["optimizer_continuity"]["starting_step"]
                    == (
                        0
                        if index == 0
                        else round_records[index - 1]["optimizer_continuity"][
                            "ending_step"
                        ]
                    )
                )
                for index, row in enumerate(round_records)
            ),
            "selected_example_ids": selected_ids,
            "selected_hypothesis_pair_counts": dict(
                sorted(cumulative_hypothesis_counts.items())
            ),
            "final_checkpoint": checkpoint_descriptor(
                {**model, "adapter_path": current_adapter}
            ),
            "hans_final": evaluation["hans_final"],
            "hans_final_prediction_rules": evaluation[
                "hans_final_prediction_rules"
            ],
            "gain_report": evaluation["gain_report"],
            "mnli": evaluation["mnli"],
            "learning_curve": learning_curve,
            "training_history": [
                {
                    "round": row["round"],
                    **row["training"],
                    "optimizer_continuity": row["optimizer_continuity"],
                }
                for row in round_records
            ],
            "hypothesis_history": [
                {
                    "round": row["round"],
                    "active_hypothesis_ids": row["active_hypothesis_ids"],
                    "states": row["hypothesis_states"],
                    "refresh": row["verification_refresh"],
                }
                for row in round_records
            ],
            "final_hypothesis_audit": final_audit,
            "cost_aware_rounds": (
                [row["cost_aware"] for row in round_records]
                if cost_aware_v7
                else None
            ),
            "cost_aware_stop_record": cost_aware_stop_record,
            "total_modeled_adaptive_cost": (
                cumulative_modeled_cost if cost_aware_v7 else None
            ),
            "resource_usage": resource_totals,
            "sealed_final_test_policy": {
                "used_for_early_stopping": False,
                "used_for_tuning": False,
                "evaluated_after_training": True,
            },
            "verification_probes_used_for_training": False,
            "leakage_audit": leakage,
        }
        write_json(output / "result.json", result)
        rule_metrics = evaluation["hans_final_prediction_rules"]
        lines = [
            (
                "# V7 cost-aware safe stopping"
                if cost_aware_v7
                else "# V7 training early stopping + verification ablation"
            ),
            "",
            f"- Verification: {verification_mode.upper()}",
            f"- Seed: {seed}",
            f"- Acquisition: {len(round_records)} x {per_round} = {len(selected_ids)} examples (maximum {total_budget})",
            f"- Effective batch: {training.batch_size * training.gradient_accumulation_steps}",
            f"- Committed AdamW steps: {committed_optimizer_step}",
            f"- HANS final accuracy: {float(evaluation['hans_final']['overall_accuracy']):.6f}",
            *(
                [
                    "- HANS Direct Binary accuracy: "
                    f"{float(rule_metrics['direct_binary']['overall_accuracy']):.6f}",
                    f"- Modeled adaptive cost: {cumulative_modeled_cost:.6f}",
                    f"- Stop reason: {stop_reason}",
                ]
                if cost_aware_v7 and rule_metrics is not None
                else []
            ),
            f"- MNLI matched/mismatched: {float(evaluation['mnli']['matched']['accuracy']):.6f} / {float(evaluation['mnli']['mismatched']['accuracy']):.6f}",
            "",
            "Training early stopping uses only the predeclared gain-eval stopping subset.",
            "The sealed HANS final test and MNLI validation are evaluated only after training.",
        ]
        _atomic_text_write(
            output
            / (
                "V7_COST_AWARE_STOPPING.md"
                if cost_aware_v7
                else "V7_EARLY_STOP_ABLATION.md"
            ),
            "\n".join(lines) + "\n",
        )
        _atomic_text_write(
            output / ".complete",
            (
                "v7 cost-aware safe stopping complete\n"
                if cost_aware_v7
                else f"v7 early-stop verification-{verification_mode} complete\n"
            ),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if not dual_v7:
        evaluation = _evaluate_checkpoint(
            model=model,
            adapter=str(current_adapter),
            gain_rows=gain_report_rows,
            final_rows=final_rows,
            mnli_config=mnli_config,
            output=output,
            audit_adapter=False,
        )
        result = {
            "method": "ours_verified_closed_loop_v1",
            "seed": seed,
            "budget": total_budget,
            "actual_training_examples": len(selected_ids),
            "training_objective": "per_example_causal_lm_cross_entropy",
            "protocol": "checkpoint_conditioned_discover_verify_generate_select_train",
            "stop_reason": stop_reason,
            "final_checkpoint": checkpoint_descriptor(
                {**model, "adapter_path": current_adapter}
            ),
            "hans_final": evaluation["hans_final"],
            "gain_report": evaluation["gain_report"],
            "mnli": evaluation["mnli"],
            "learning_curve": learning_curve,
            "final_hypothesis_audit": final_audit,
            "selected_hypothesis_pair_counts": dict(
                sorted(cumulative_hypothesis_counts.items())
            ),
            "verification_probes_used_for_training": False,
        }
        write_json(output / "result.json", result)
        lines = [
            "# Stage 2 closed-loop verification report",
            "",
            f"- Seed: `{seed}`",
            f"- Fixed maximum budget: `{total_budget}` examples",
            f"- Actual trained examples: `{len(selected_ids)}`",
            f"- Completed rounds: `{len(round_records)}/{max_rounds}`",
            f"- Stop reason: `{stop_reason}`",
            f"- HANS final accuracy: `{float(evaluation['hans_final']['overall_accuracy']):.6f}`",
            f"- Gain-report accuracy: `{float(evaluation['gain_report']['overall_accuracy']):.6f}`",
            f"- MNLI matched accuracy: `{float(evaluation['mnli']['matched']['accuracy']):.6f}`",
            f"- MNLI mismatched accuracy: `{float(evaluation['mnli']['mismatched']['accuracy']):.6f}`",
            "",
            "Verification probes were used only for certification/diagnosis and never for training.",
            "A hypothesis is resolved only when its fresh checkpoint-conditioned Hoeffding UCB is below tau_v.",
        ]
        _atomic_text_write(output / "CLOSED_LOOP_REPORT.md", "\n".join(lines) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if current_adapter is None or not round_records:
        raise RuntimeError("v7 produced no continuation checkpoint")
    fixed_budget_comparable = len(selected_ids) == total_budget
    continuation_dir = output / "branches" / "continuation"
    continuation_evaluation = _evaluate_checkpoint(
        model=model,
        adapter=str(current_adapter),
        gain_rows=gain_report_rows,
        final_rows=final_rows,
        mnli_config=mnli_config,
        output=continuation_dir,
        audit_adapter=True,
    )

    consolidation_dir = output / "branches" / "consolidation"
    consolidation_adapter = consolidation_dir / "adapter"
    consolidation_optimizer = consolidation_dir / "optimizer_state.pt"
    set_deterministic_seed(seed)
    consolidation_student = _student(model, model.get("adapter_path"))
    consolidation_parent_audit = (
        audit_loaded_adapter(consolidation_student.model, str(model["adapter_path"]))
        if model.get("adapter_path")
        else None
    )
    consolidation_start = time.perf_counter()
    consolidation_training = train_lora_stateful(
        consolidation_student,
        [SyntheticNLIExample.from_mapping(row).to_mnli() for row in selected_rows],
        training,
        consolidation_adapter,
        seed,
        optimizer_state_input=None,
        optimizer_state_output=consolidation_optimizer,
    )
    consolidation_seconds = time.perf_counter() - consolidation_start
    expected_consolidation_steps = _optimizer_steps_for_examples(
        len(selected_rows), training
    )
    if (
        consolidation_training.optimizer_step_start != 0
        or consolidation_training.optimizer_step_end != expected_consolidation_steps
    ):
        raise RuntimeError(
            "Consolidation optimizer-step audit failed: expected "
            f"0->{expected_consolidation_steps}, observed "
            f"{consolidation_training.optimizer_step_start}->"
            f"{consolidation_training.optimizer_step_end}"
        )
    consolidation_save_audit = audit_loaded_adapter(
        consolidation_student.model, consolidation_adapter
    )
    del consolidation_student
    _clear_cuda_cache()
    consolidation_evaluation = _evaluate_checkpoint(
        model=model,
        adapter=str(consolidation_adapter),
        gain_rows=gain_report_rows,
        final_rows=final_rows,
        mnli_config=mnli_config,
        output=consolidation_dir,
        audit_adapter=True,
    )

    common = {
        "experiment": "B",
        "seed": seed,
        "budget": total_budget,
        "requested_budget": total_budget,
        "actual_training_examples": len(selected_ids),
        "actual_final_training_examples": len(selected_ids),
        "fixed_budget_comparable": fixed_budget_comparable,
        "final_metrics_reportable": fixed_budget_comparable,
        "shortage": not fixed_budget_comparable,
        "manifest_hash": manifest.manifest_hash,
        "cross_source_baseline": False,
        "training_objective": "per_example_causal_lm_cross_entropy",
        "closed_loop_acquisition_trace": "../../rounds.jsonl",
        "selected_training_examples": "../../selected_training_examples.jsonl",
        "selected_example_ids": selected_ids,
        "stop_reason": stop_reason,
        "final_hypothesis_audit": final_audit,
        "selected_hypothesis_pair_counts": dict(
            sorted(cumulative_hypothesis_counts.items())
        ),
        "verification_probes_used_for_training": False,
        "leakage_audit": leakage,
    }
    continuation_result = {
        **common,
        "method": "ours_verified_v7_continuation",
        "protocol": "closed_loop_stateful_lora_and_adamw_continuation_v7",
        "closed_loop_controller": "this_checkpoint",
        "optimizer_steps": round_records[-1]["optimizer_continuity"]["ending_step"],
        "optimizer_continuity_verified": all(
            bool(row["adapter_continuity"]["ending_adapter_save"]["matched"])
            and row["optimizer_continuity"]["ending_step"]
            == row["round"] * _optimizer_steps_for_examples(per_round, training)
            for row in round_records
        ),
        "final_checkpoint": checkpoint_descriptor(
            {**model, "adapter_path": current_adapter}
        ),
        "hans_final": continuation_evaluation["hans_final"],
        "gain_report": continuation_evaluation["gain_report"],
        "mnli": continuation_evaluation["mnli"],
        "learning_curve": learning_curve,
    }
    consolidation_result = {
        **common,
        "method": "ours_verified_v7_consolidation",
        "protocol": "closed_loop_acquisition_then_fresh_single_pass_consolidation_v7",
        "closed_loop_controller": "shared_continuation_checkpoint",
        "optimizer_steps": consolidation_training.optimizer_step_end,
        "optimizer_continuity_verified": True,
        "consolidation_train_loss": consolidation_training.total_loss,
        "consolidation_training_seconds": consolidation_seconds,
        "parent_adapter_load_audit": consolidation_parent_audit,
        "adapter_save_audit": consolidation_save_audit,
        "final_checkpoint": checkpoint_descriptor(
            {**model, "adapter_path": str(consolidation_adapter)}
        ),
        "hans_final": consolidation_evaluation["hans_final"],
        "gain_report": consolidation_evaluation["gain_report"],
        "mnli": consolidation_evaluation["mnli"],
        "learning_curve": [learning_curve[0]],
    }
    write_json(continuation_dir / "result.json", continuation_result)
    write_json(consolidation_dir / "result.json", consolidation_result)
    comparison = {
        "protocol": training_protocol,
        "seed": seed,
        "budget": total_budget,
        "shared_selected_example_ids": selected_ids,
        "identical_acquisition_trace": True,
        "continuation": continuation_result,
        "consolidation": consolidation_result,
    }
    write_json(output / "comparison.json", comparison)
    lines = [
        "# Stage 2 v7 fair closed-loop comparison",
        "",
        "Both branches use the exact same checkpoint-conditioned closed-loop acquisition trace and selected examples.",
        "",
        f"- Seed: `{seed}`",
        f"- Requested/actual examples: `{total_budget}/{len(selected_ids)}`",
        f"- Completed rounds: `{len(round_records)}/{max_rounds}`",
        f"- Stop reason: `{stop_reason}`",
        f"- Continuation AdamW steps: `{continuation_result['optimizer_steps']}`",
        f"- Consolidation AdamW steps: `{consolidation_result['optimizer_steps']}`",
        f"- Continuation HANS accuracy: `{float(continuation_evaluation['hans_final']['overall_accuracy']):.6f}`",
        f"- Consolidation HANS accuracy: `{float(consolidation_evaluation['hans_final']['overall_accuracy']):.6f}`",
        f"- Continuation MNLI matched/mismatched: `{float(continuation_evaluation['mnli']['matched']['accuracy']):.6f}` / `{float(continuation_evaluation['mnli']['mismatched']['accuracy']):.6f}`",
        f"- Consolidation MNLI matched/mismatched: `{float(consolidation_evaluation['mnli']['matched']['accuracy']):.6f}` / `{float(consolidation_evaluation['mnli']['mismatched']['accuracy']):.6f}`",
        "",
        "Continuation carries both LoRA tensors and AdamW moments across rounds. Consolidation starts from the original base checkpoint and trains once on all selected examples.",
        "Verification probes are used only for certification/diagnosis and never for training.",
    ]
    _atomic_text_write(output / "V7_COMPARISON.md", "\n".join(lines) + "\n")
    _atomic_text_write(output / ".complete", "v7 dual fair closed-loop complete\n")
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
