#!/usr/bin/env python3
"""Replay V7 seed artifacts and run only the new m=32 Cost-Aware pilots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stage2_closed_loop import (  # noqa: E402
    _cost_aware_stop_category,
    _evaluate_checkpoint,
    _finalize_mixed_policy_round_cost,
    _hans,
    _mixed_policy_round_cost,
    _run_v7_mixed_policy_pilots,
    _teacher_usage,
)
from src.benchmark.hans_manifest import FrozenHansManifest  # noqa: E402
from src.benchmark.stopping_subset import (  # noqa: E402
    validate_stopping_subset_manifest,
)
from src.decision.v7_minimal_cost_aware import (  # noqa: E402
    MIXED_POLICY_ID,
    V7MinimalCostAwareConfig,
    decide_mixed_policy_round,
    predeclare_pilot_eval_plan,
)
from src.decision.cost_model import CostUsage  # noqa: E402
from src.gain.estimation import estimate_gain  # noqa: E402
from src.models.lora_training import (  # noqa: E402
    EarlyStoppingConfig,
    LoraTrainingConfig,
)
from src.synthetic.types import SyntheticNLIExample  # noqa: E402
from src.utils.artifacts import read_json, read_jsonl  # noqa: E402
from src.utils.checkpoint import checkpoint_descriptor, hash_local_path  # noqa: E402
from src.utils.config import load_yaml_with_extends, require_mapping  # noqa: E402
from src.utils.io import write_json, write_jsonl  # noqa: E402
from src.utils.seed import set_deterministic_seed  # noqa: E402


DEFAULT_SOURCE = "outputs/stage2_runs_v7_minimal_cost_aware/seed_42"
DEFAULT_PILOT_CACHE = "outputs/stage2_runs_v7_minimal_cost_aware_m32/seed_42"
DEFAULT_OUTPUT = (
    "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_42"
)
CORE_CONFIG_SECTIONS = (
    "model",
    "training",
    "stage2_benchmark",
    "hypothesis_generation",
    "synthetic_generation",
    "synthetic_validation",
    "mnli_evaluation",
)
INITIAL_REPLAY_FILES = (
    "hypotheses.json",
    "synthetic/accepted.jsonl",
    "synthetic/manifest.json",
)
REFRESH_REPLAY_FILES = (
    "config.yaml",
    "hypotheses.json",
    "hypothesis_merge.json",
    "lifecycle.json",
    "predictions.jsonl",
    "synthetic/accepted.jsonl",
    "synthetic/generated.jsonl",
    "synthetic/manifest.json",
    "synthetic/rejected.jsonl",
    "teacher_calls/synthetic.jsonl",
    "teacher_calls/verification.jsonl",
    "verification/generated_probes.jsonl",
    "verification/probe_set_outcomes.jsonl",
    "verification/probes.jsonl",
    "verification/rejected_probes.jsonl",
    "verification/results.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/stage2_v7_minimal_cost_aware.yaml"
    )
    parser.add_argument("--source-run", default=DEFAULT_SOURCE)
    parser.add_argument("--pilot-cache-source", default=DEFAULT_PILOT_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--decision-only",
        action="store_true",
        help="Replay the next cached decision and exit before new GPU work.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate immutable source/config/data/checkpoint hashes and exit on CPU.",
    )
    parser.add_argument(
        "--benchmark-one-pilot",
        action="store_true",
        help="Run/cache only pilot 1 of round 2, write timing, and exit.",
    )
    return parser.parse_args()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required replay artifact is missing: {path}")


def _validate_source(
    *,
    source: Path,
    config: dict[str, Any],
    source_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    for name in (".complete", "result.json", "rounds.jsonl", "selected_training_examples.jsonl"):
        _assert_file(source / name)
    source_result = read_json(source / "result.json")
    if int(source_result.get("seed", -1)) != 42:
        raise ValueError("Replay source must be completed seed 42")
    if source_result.get("method") != "ours_v7_minimal_cost_aware":
        raise ValueError("Replay source is not the completed V7 minimal Cost-Aware run")

    for section in CORE_CONFIG_SECTIONS:
        if config.get(section) != source_config.get(section):
            raise ValueError(f"Replay changes preserved V7 config section: {section}")
    for key in ("initial_pool_dir", "selection", "per_round_examples"):
        if config["closed_loop"].get(key) != source_config["closed_loop"].get(key):
            raise ValueError(f"Replay changes preserved closed_loop.{key}")
    current_verification = copy.deepcopy(config["verification"])
    old_verification = copy.deepcopy(source_config["verification"])
    if current_verification != old_verification:
        raise ValueError("Replay must preserve the complete Adaptive Verification config")

    rounds = [dict(item) for item in read_jsonl(source / "rounds.jsonl")]
    selected = [dict(item) for item in read_jsonl(source / "selected_training_examples.jsonl")]
    if [int(item["round"]) for item in rounds] != list(range(1, 9)):
        raise ValueError("Replay source must contain contiguous rounds 1-8")
    if len(selected) != 256 or len({str(item["id"]) for item in selected}) != 256:
        raise ValueError("Replay source selected trajectory must contain 256 unique IDs")

    previous_adapter: str | None = None
    previous_step = 0
    checkpoint_audit: list[dict[str, Any]] = []
    for round_index, record in enumerate(rounds, start=1):
        if record.get("starting_adapter") != previous_adapter:
            raise ValueError(f"Adapter continuity mismatch at source round {round_index}")
        optimizer = record.get("optimizer_continuity") or {}
        if int(optimizer.get("starting_step", -1)) != previous_step:
            raise ValueError(f"AdamW continuity mismatch at source round {round_index}")
        adapter = str(record["ending_adapter"])
        optimizer_path = str(optimizer["ending_state"])
        if not (Path(adapter) / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"Source adapter is missing: {adapter}")
        if not Path(optimizer_path).is_file():
            raise FileNotFoundError(f"Source AdamW state is missing: {optimizer_path}")
        checkpoint_audit.append(
            {
                "round": round_index,
                "adapter": adapter,
                "adapter_sha256": hash_local_path(adapter),
                "optimizer_state": optimizer_path,
                "optimizer_sha256": hash_local_path(optimizer_path),
                "optimizer_step": int(optimizer["ending_step"]),
            }
        )
        previous_adapter = adapter
        previous_step = int(optimizer["ending_step"])

    initial_pool = Path(str(config["closed_loop"]["initial_pool_dir"]))
    initial_artifact_audit: dict[str, str] = {}
    for relative in INITIAL_REPLAY_FILES:
        path = initial_pool / relative
        _assert_file(path)
        initial_artifact_audit[relative] = hash_local_path(str(path))

    refresh_artifact_audit: list[dict[str, Any]] = []
    for round_index in range(2, 9):
        refresh = source / "refresh" / f"round_{round_index:03d}"
        hashes: dict[str, str] = {}
        for relative in REFRESH_REPLAY_FILES:
            path = refresh / relative
            _assert_file(path)
            hashes[relative] = hash_local_path(str(path))
        refresh_artifact_audit.append({"round": round_index, "files": hashes})

    for round_index in range(2, 9):
        expected = [str(item["id"]) for item in selected[(round_index-1)*32:round_index*32]]
        source_decision = rounds[round_index - 1].get("cost_aware") or {}
        if source_decision.get("selected_action_example_ids") != expected:
            raise ValueError(f"Selected batch ID mismatch at source round {round_index}")

    replay_manifest = {
        "schema_version": 1,
        "source_run": str(source),
        "source_result_sha256": hash_local_path(str(source / "result.json")),
        "source_config_sha256": _sha256_json(source_config),
        "replay_config_sha256": _sha256_json(config),
        "selected_ids_sha256": _sha256_json([str(item["id"]) for item in selected]),
        "source_round_checkpoints": checkpoint_audit,
        "initial_pool_artifact_hashes": initial_artifact_audit,
        "refresh_artifact_hashes": refresh_artifact_audit,
        "source_artifacts_are_read_only": True,
        "teacher_compute_reused": True,
        "main_training_compute_reused": True,
    }
    return rounds, selected, {"result": source_result, "manifest": replay_manifest}


def _prepare_data(
    config: dict[str, Any], policy: V7MinimalCostAwareConfig
) -> tuple[list[Any], dict[str, Any], dict[int, list[list[Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    closed = require_mapping(config, "closed_loop")
    benchmark = require_mapping(config, "stage2_benchmark")
    manifest = FrozenHansManifest.from_mapping(
        read_json(Path(str(benchmark["frozen_manifest_path"])))
    )
    shared_rows = read_jsonl(Path(str(closed["shared_predictions_path"])))
    shared_index = {str(row["id"]): row for row in shared_rows}
    gain_rows = [shared_index[item] for item in manifest.gain_eval_ids]
    stopping_ids, report_ids = validate_stopping_subset_manifest(
        read_json(Path(str(closed["stopping_subset_manifest"]))),
        gain_rows,
        manifest.gain_eval_ids,
        frozen_manifest_hash=manifest.manifest_hash,
    )
    plan = predeclare_pilot_eval_plan(
        report_ids,
        decision_rounds=range(2, 9),
        pilots_per_round=policy.required_pilots,
        eval_size=policy.pilot_eval_size,
        seed=policy.pilot_seed,
        shared_eval_shard_within_round=policy.shared_eval_shard_within_round,
    )
    stopping_examples = _hans([shared_index[item] for item in stopping_ids])
    round_shards: dict[int, list[list[Any]]] = {}
    for key, shards in plan["round_shards"].items():
        if len(shards) != 1:
            raise ValueError("m=32 replay requires one fixed eval shard per round")
        shard = _hans([shared_index[item] for item in shards[0]])
        round_shards[int(key)] = [shard] * policy.required_pilots
    heldout_rows = [shared_index[item] for item in plan["held_out_gain_report_ids"]]
    final_rows = read_jsonl(Path(str(benchmark["sealed_final_test_path"])))
    if set(plan["pilot_ids"]) & set(stopping_ids):
        raise ValueError("Pilot eval IDs overlap the early-stopping subset")
    if set(plan["pilot_ids"]) & {str(item["id"]) for item in final_rows}:
        raise ValueError("Pilot eval IDs overlap sealed final HANS")
    if set(plan["pilot_ids"]) & set(plan["held_out_gain_report_ids"]):
        raise ValueError("Pilot eval IDs overlap held-out gain report")
    return stopping_examples, plan, round_shards, heldout_rows, final_rows


def _eligible_count_from_source(record: dict[str, Any]) -> int:
    cost = (record.get("cost_aware") or {}).get("decision_cost") or {}
    usage = (((cost.get("phases") or {}).get("incurred_exploration") or {}).get("usage") or {})
    items = int(float(usage.get("evaluation_items", 0)))
    if items <= 0 or items % 2:
        raise ValueError("Cannot reconstruct source eligible-example count")
    return items // 2


def _write_once_or_validate(path: Path, payload: Any) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise ValueError(f"Frozen replay artifact differs: {path}")
        return
    write_json(path, payload)


def _validate_pilot_cache(
    *,
    pilot_source: Path,
    rounds: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    plan: dict[str, Any],
    pilot_seed: int,
) -> dict[str, Any]:
    """Validate external pilot observations before importing them read-only."""
    _assert_file(pilot_source / ".complete")
    _assert_file(pilot_source / "result.json")
    entries: list[dict[str, Any]] = []
    for round_index in range(2, 9):
        root = pilot_source / "cost_aware" / f"round_{round_index:03d}" / "pilots"
        paths = sorted(root.glob("pilot_*/observation.json")) if root.is_dir() else []
        expected_train_ids = [
            str(item["id"])
            for item in selected_rows[(round_index - 1) * 32 : round_index * 32]
        ]
        expected_eval_ids = list(plan["round_shards"][str(round_index)][0])
        previous = rounds[round_index - 2]
        expected_adapter_hash = hash_local_path(str(previous["ending_adapter"]))
        expected_optimizer_hash = hash_local_path(
            str(previous["optimizer_continuity"]["ending_state"])
        )
        for offset, path in enumerate(paths):
            payload = read_json(path)
            pilot_index = offset + 1
            checks = {
                "policy": payload.get("hypothesis_id") == MIXED_POLICY_ID,
                "pilot_index": int(payload.get("pilot_index", -1)) == pilot_index,
                "pilot_seed": int(payload.get("pilot_seed", -1))
                == pilot_seed + round_index * 1000 + offset,
                "train_ids": payload.get("train_ids") == expected_train_ids,
                "eval_ids": payload.get("eval_ids") == expected_eval_ids,
                "adapter_hash": (
                    (payload.get("starting_checkpoint") or {}).get(
                        "adapter_local_sha256"
                    )
                    == expected_adapter_hash
                ),
                "optimizer_hash": (
                    (((payload.get("training_config") or {}).get(
                        "optimizer_state_input"
                    ) or {}).get("local_sha256"))
                    == expected_optimizer_hash
                ),
                "exact_v7_action": (
                    (payload.get("training_config") or {}).get("action")
                    == "exact_v7_mixed_soft_pair_batch"
                ),
            }
            if not all(checks.values()):
                raise ValueError(
                    f"External pilot cache mismatch at round {round_index}, "
                    f"pilot {pilot_index}: {checks}"
                )
            entries.append(
                {
                    "round": round_index,
                    "pilot": pilot_index,
                    "path": str(path),
                    "sha256": hash_local_path(str(path)),
                }
            )
    if len([item for item in entries if item["round"] == 2]) != 32:
        raise ValueError("Seed-42 decision replay requires all 32 round-2 pilots")
    return {
        "source": str(pilot_source),
        "source_result_sha256": hash_local_path(str(pilot_source / "result.json")),
        "observations": entries,
        "source_is_read_only": True,
    }


def _import_pilot_cache(
    *, output: Path, pilot_manifest: dict[str, Any]
) -> int:
    imported = 0
    for entry in pilot_manifest["observations"]:
        payload = read_json(Path(entry["path"]))
        target = (
            output
            / "cost_aware"
            / f"round_{int(entry['round']):03d}"
            / "pilots"
            / f"pilot_{int(entry['pilot']):03d}"
            / "observation.json"
        )
        _write_once_or_validate(target, payload)
        imported += 1
    return imported


def main() -> None:
    args = parse_args()
    source = Path(args.source_run).resolve()
    pilot_source = Path(args.pilot_cache_source).resolve()
    output = Path(args.output_dir).resolve()
    config = load_yaml_with_extends(args.config)
    source_config = load_yaml_with_extends(source / "config.yaml")
    policy = V7MinimalCostAwareConfig.from_mapping(config.get("cost_aware"))
    if not policy.enabled or policy.required_pilots != 32:
        raise ValueError("Replay requires enabled Cost-Aware with exactly 32 pilots")
    if not policy.shared_eval_shard_within_round:
        raise ValueError("m=32 replay requires fixed shared eval shard per round")
    if int(config.get("seed", 42)) != 42:
        raise ValueError("Artifact replay is only valid for seed 42")

    rounds, selected_rows, source_payload = _validate_source(
        source=source,
        config=config,
        source_config=source_config,
    )
    stopping_examples, plan, round_shards, heldout_rows, final_rows = _prepare_data(
        config, policy
    )
    source_payload["manifest"]["pilot_eval_plan_sha256"] = _sha256_json(plan)
    pilot_manifest = _validate_pilot_cache(
        pilot_source=pilot_source,
        rounds=rounds,
        selected_rows=selected_rows,
        plan=plan,
        pilot_seed=policy.pilot_seed,
    )
    source_payload["manifest"]["external_pilot_cache"] = pilot_manifest

    if args.audit_only:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "source_validation": source_payload["manifest"],
                    "pilot_eval": {
                        "pilots_per_round": policy.required_pilots,
                        "unique_eval_ids": len(plan["pilot_ids"]),
                        "held_out_gain_report_ids": len(
                            plan["held_out_gain_report_ids"]
                        ),
                        "independence_unit": plan["independence_unit"],
                        "shared_examples_counted_as_independent": False,
                    },
                    "gpu_or_teacher_work_run": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Replay output exists; rerun with --resume: {output}")
    if args.resume and not output.exists():
        raise FileNotFoundError(f"No replay output exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_once_or_validate(output / "config_snapshot.json", config)
    _write_once_or_validate(output / "replay_manifest.json", source_payload["manifest"])
    _write_once_or_validate(output / "cost_aware" / "pilot_eval_plan.json", plan)
    imported_pilots = _import_pilot_cache(
        output=output, pilot_manifest=pilot_manifest
    )

    model = require_mapping(config, "model")
    model["device_map"] = os.environ.get("CLOSED_LOOP_STUDENT_DEVICE", "cuda:1")
    training_mapping = require_mapping(config, "training")
    training = LoraTrainingConfig.from_mapping(training_mapping)
    early = EarlyStoppingConfig.from_mapping(training_mapping)
    set_deterministic_seed(42)

    trace: list[dict[str, Any]] = [
        {
            "round": 1,
            "cumulative_examples_before_decision": 0,
            "decision": {
                "action": "continue",
                "reason": "round_1_preserved_and_reused_from_v7",
                "gain_cost_stop_evaluated": False,
                "appendix_theorem_claimed": False,
            },
            "gain_estimate": None,
            "decision_cost": None,
            "pilot_observations": [],
            "reused_source_round": 1,
        }
    ]
    stop_record: dict[str, Any] | None = None
    cumulative_decision_cost = 0.0
    cumulative_incurred_cost = 0.0
    cumulative_total_pipeline_cost = 0.0
    completed_main_rounds = 1

    for round_index in range(2, 9):
        decision_path = output / "cost_aware" / f"round_{round_index:03d}" / "decision.json"
        if decision_path.is_file():
            if not args.resume:
                raise FileExistsError(f"Replay decision already exists: {decision_path}")
            replay_round = read_json(decision_path)
            trace.append(replay_round)
            cumulative_decision_cost += float(
                replay_round["decision_cost"]["decision_cost"]
            )
            cumulative_incurred_cost += float(
                replay_round["decision_cost"]["incurred_exploration_cost"]
            )
            cumulative_total_pipeline_cost += float(
                replay_round["decision_cost"]["total_pipeline_cost"]
            )
            if args.decision_only:
                print(json.dumps(replay_round, indent=2, sort_keys=True))
                return
            if replay_round["decision"]["action"] == "stop":
                stop_record = replay_round
                break
            completed_main_rounds = round_index
            continue

        source_record = rounds[round_index - 1]
        previous = rounds[round_index - 2]
        selected_slice = selected_rows[(round_index - 1) * 32 : round_index * 32]
        acquired = [SyntheticNLIExample.from_mapping(item) for item in selected_slice]
        states = {
            str(key): dict(value)
            for key, value in source_record["hypothesis_states"].items()
        }
        certified = [key for key, value in states.items() if value.get("state") == "certified"]
        observations: list[dict[str, Any]] = []
        if certified:
            requested_shards = round_shards[round_index]
            if args.benchmark_one_pilot:
                requested_shards = requested_shards[:1]
            started = time.perf_counter()
            observations = _run_v7_mixed_policy_pilots(
                output=output,
                round_index=round_index,
                acquired=acquired,
                eval_shards=requested_shards,
                model=model,
                current_adapter=str(previous["ending_adapter"]),
                current_optimizer_state=str(previous["optimizer_continuity"]["ending_state"]),
                committed_optimizer_step=int(previous["optimizer_continuity"]["ending_step"]),
                training=training,
                early_stopping=early,
                stopping_examples=stopping_examples,
                pilot_seed=policy.pilot_seed,
                gain_bound_M=policy.M,
                preserve_adapters=policy.preserve_pilot_adapters,
                resume=(args.resume or imported_pilots > 0),
            )
            if args.benchmark_one_pilot:
                benchmark = {
                    "round": round_index,
                    "pilot": observations[0],
                    "elapsed_seconds": time.perf_counter() - started,
                    "estimated_32_pilot_serial_seconds": 32
                    * float(observations[0]["wall_clock_seconds"]),
                    "teacher_compute_run": False,
                    "source_artifacts_reused": True,
                }
                write_json(output / "pilot_benchmark.json", benchmark)
                print(json.dumps(benchmark, indent=2, sort_keys=True))
                return

        estimate = estimate_gain(
            MIXED_POLICY_ID,
            [float(item["gain"]) for item in observations],
            M=policy.M,
            delta=policy.delta,
            K=policy.K,
            T=policy.T,
        )
        refresh_dir = source / "refresh" / f"round_{round_index:03d}"
        decision_cost = _mixed_policy_round_cost(
            refresh_dir=refresh_dir,
            teacher_usage=_teacher_usage(refresh_dir),
            observations=observations,
            eligible_examples=_eligible_count_from_source(source_record),
            acquired_examples=len(acquired),
            stopping_examples=len(stopping_examples),
            early_stopping=early,
            cost_config=dict(config["cost_aware"]["cost"]),
        )
        decision = decide_mixed_policy_round(
            states=states,
            estimate=estimate,
            required_pilots=policy.required_pilots,
            cost=float(decision_cost["decision_cost"]),
            lambda_cost=policy.lambda_cost,
        )
        actual_main_usage = None
        if decision.action == "continue":
            source_usage = source_record["resource_usage"]
            epochs_completed = int(source_record["training"]["epochs_completed"])
            actual_main_usage = CostUsage(
                lora_training_runs=1,
                lora_training_gpu_seconds=float(
                    source_usage["training_gpu_wall_seconds"]
                ),
                evaluation_runs=epochs_completed + 1,
                evaluation_items=(epochs_completed + 1) * len(stopping_examples),
                evaluation_gpu_seconds=(
                    float(source_usage["validation_gpu_wall_seconds"])
                    + float(source_usage["stopping_evaluation_gpu_wall_seconds"])
                ),
            )
        decision_cost = _finalize_mixed_policy_round_cost(
            decision_cost,
            action=decision.action,
            actual_main_update_usage=actual_main_usage,
        )
        replay_round = {
            "round": round_index,
            "cumulative_examples_before_decision": (round_index - 1) * 32,
            "selected_action_example_ids": [item.id for item in acquired],
            "gain_estimate": estimate.to_dict(),
            "decision_cost": decision_cost,
            "decision": decision.to_dict(),
            "pilot_observations": observations,
            "reused_source_round": round_index,
            "reused_starting_adapter_sha256": hash_local_path(str(previous["ending_adapter"])),
            "reused_optimizer_state_sha256": hash_local_path(
                str(previous["optimizer_continuity"]["ending_state"])
            ),
            "teacher_compute_run": False,
            "main_training_compute_run": False,
            "shared_eval_examples_counted_as_independent": False,
            "pilot_independence_unit": plan["independence_unit"],
            "appendix_hypothesis_specific_theorem_claimed": False,
        }
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(decision_path.parent / "gain_estimate.json", estimate.to_dict())
        write_json(decision_path.parent / "cost.json", decision_cost)
        write_json(decision_path, replay_round)
        trace.append(replay_round)
        cumulative_decision_cost += float(decision_cost["decision_cost"])
        cumulative_incurred_cost += float(
            decision_cost["incurred_exploration_cost"]
        )
        cumulative_total_pipeline_cost += float(
            decision_cost["total_pipeline_cost"]
        )
        if args.decision_only:
            write_json(
                output / "decision_replay_summary.json",
                {
                    "protocol_version": "v7_m32_marginal_continue_cost_v2",
                    "round": round_index,
                    "gain_estimate": estimate.to_dict(),
                    "decision_cost": decision_cost,
                    "decision": decision.to_dict(),
                    "new_gpu_teacher_or_verification_work": False,
                    "imported_pilot_observations": len(observations),
                },
            )
            print(json.dumps(replay_round, indent=2, sort_keys=True))
            return
        if decision.action == "stop":
            stop_record = replay_round
            write_json(output / "cost_aware_stop.json", replay_round)
            break
        completed_main_rounds = round_index

    final_source_record = rounds[completed_main_rounds - 1]
    final_adapter = str(final_source_record["ending_adapter"])
    source_result = source_payload["result"]
    source_final_adapter = str(source_result["final_checkpoint"]["adapter_path"])
    evaluation_reused = (
        completed_main_rounds == 8
        and hash_local_path(final_adapter) == hash_local_path(source_final_adapter)
    )
    evaluation_seconds = 0.0
    if evaluation_reused:
        evaluation = {
            "hans_final": source_result["hans_final"],
            "hans_prediction_rules": source_result["hans_prediction_rules"],
            "mnli": source_result["mnli"],
            "gain_report": None,
        }
    else:
        started = time.perf_counter()
        evaluation = _evaluate_checkpoint(
            model=model,
            adapter=final_adapter,
            gain_rows=heldout_rows,
            final_rows=final_rows,
            mnli_config=require_mapping(config, "mnli_evaluation"),
            output=output,
            audit_adapter=True,
            report_hans_prediction_rules=True,
        )
        evaluation_seconds = time.perf_counter() - started

    stop_reason = (
        str(stop_record["decision"]["reason"])
        if stop_record is not None
        else "maximum_budget_reached_with_remaining_hypotheses"
    )
    continued_decisions = [
        item for item in trace if item.get("decision", {}).get("action") != "stop"
    ]
    pilot_wall = sum(
        float(obs.get("wall_clock_seconds", 0.0))
        for item in trace
        for obs in item.get("pilot_observations", [])
    )
    result = {
        "method": "ours_v7_minimal_cost_aware",
        "experiment": "v7_minimal_cost_aware_m32_marginal_cost_v2_replay",
        "cost_aware_protocol_version": "v7_m32_marginal_continue_cost_v2",
        "seed": 42,
        "budget": 256,
        "requested_budget": 256,
        "rounds": completed_main_rounds,
        "per_round_examples": 32,
        "actual_training_examples": completed_main_rounds * 32,
        "stop_reason": stop_reason,
        "stop_reason_category": _cost_aware_stop_category(stop_reason),
        "optimizer_steps": int(final_source_record["optimizer_continuity"]["ending_step"]),
        "optimizer_continuity_verified": True,
        "final_checkpoint": checkpoint_descriptor(
            {**model, "adapter_path": final_adapter}
        ),
        "hans_final": evaluation["hans_final"],
        "hans_prediction_rules": evaluation["hans_prediction_rules"],
        "mnli": evaluation["mnli"],
        "gain_report": evaluation.get("gain_report"),
        "learning_curve": source_result["learning_curve"][: completed_main_rounds + 1],
        "training_history": source_result["training_history"][:completed_main_rounds],
        "hypothesis_history": source_result["hypothesis_history"][:completed_main_rounds],
        "cost_aware_rounds": continued_decisions,
        "cost_aware_stop_record": stop_record,
        "total_modeled_decision_cost_rounds_2_plus": cumulative_decision_cost,
        "total_incurred_exploration_cost_rounds_2_plus": (
            cumulative_incurred_cost
        ),
        "total_pipeline_cost_rounds_2_plus": cumulative_total_pipeline_cost,
        "pilot_eval_plan": {
            key: value
            for key, value in plan.items()
            if key not in {"source_ids", "pilot_ids", "round_shards"}
        },
        "replay": {
            **source_payload["manifest"],
            "evaluation_reused_by_checkpoint_hash": evaluation_reused,
            "later_source_rounds_ignored_after_stop": stop_record is not None,
            "new_teacher_compute": False,
            "new_main_training_compute": False,
        },
        "resource_usage": {
            "teacher": {
                "calls": 0.0,
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "teacher_wall_clock_seconds": 0.0,
            },
            "cost_aware_pilot_wall_seconds": pilot_wall,
            "final_evaluation_gpu_wall_seconds": evaluation_seconds,
            "refresh_wall_seconds": 0.0,
            "training_gpu_wall_seconds": 0.0,
            "validation_gpu_wall_seconds": 0.0,
            "stopping_evaluation_gpu_wall_seconds": 0.0,
            "acquisition_gpu_wall_seconds": 0.0,
            "scope": "incremental_replay_compute_only",
            "decision_cost_includes_incurred_exploration": False,
        },
        "reused_source_resource_usage_full_8_round_historical": source_result[
            "resource_usage"
        ],
        "sealed_final_test_policy": {
            "used_for_stopping": False,
            "used_for_lambda_tuning": False,
            "evaluation_reused_only_if_checkpoint_hash_matches": True,
        },
        "cost_aware_theory_scope": {
            "appendix_hypothesis_specific_theorem_claimed": False,
            "pilot_gain_target": "expected_fixed-shard_empirical-risk_reduction_over_training_randomization",
            "conditional_independence_empirically_proven": False,
            "shared_eval_examples_are_independent_observations": False,
        },
    }
    write_jsonl(output / "replay_trace.jsonl", trace)
    write_json(output / "result.json", result)
    (output / ".complete").write_text("v7 minimal cost-aware m32 replay complete\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
