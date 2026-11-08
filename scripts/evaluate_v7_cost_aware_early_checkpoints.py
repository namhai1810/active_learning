#!/usr/bin/env python3
"""Evaluate frozen V7 Cost-Aware early checkpoints on full HANS.

Diagnostic only: this script loads existing adapters and the frozen 15,000-row
sealed HANS artifact.  It never trains, calls a teacher, verifies hypotheses,
runs pilots, or feeds final-test results back into an online stopping policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.hans_manifest import FrozenHansManifest
from src.data.hans import HANSExample
from src.data.labels import MNLI_LABELS, mnli_to_hans
from src.evaluation.metrics import compute_hans_metrics
from src.models.adapter_continuity import (
    adapter_artifact_sha256,
    audit_loaded_adapter,
)
from src.models.qwen_nli import QwenNLIStudent
from src.utils.artifacts import read_json, read_jsonl
from src.utils.checkpoint import hash_local_path
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


TARGETS: dict[int, dict[str, Any]] = {
    42: {
        "round": 3,
        "budget": 96,
        "config": "outputs/stage2_runs_v7_minimal_cost_aware/seed_42/config.yaml",
        "adapter": "outputs/stage2_runs_v7_minimal_cost_aware/seed_42/training/round_003/adapter",
        "replay_result": "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_42/result.json",
        "r8_result": "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_42/result.json",
    },
    43: {
        "round": 4,
        "budget": 128,
        "config": "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_43/config.yaml",
        "adapter": "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_43/training/round_004/adapter",
        "replay_result": None,
        "r8_result": "outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2/seed_43/result.json",
    },
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument(
        "--output-dir",
        default="outputs/stage2_diagnostics/v7_correct_stopping_full_hans_early_checkpoints",
    )
    parser.add_argument(
        "--report",
        default="reports/v7_minimal_cost_aware/EARLY_CHECKPOINT_FULL_HANS.md",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _expected_replay_hash(result_path: Path, round_index: int) -> str | None:
    payload = read_json(result_path)
    replay = payload.get("replay")
    if not isinstance(replay, dict):
        raise ValueError(f"Replay metadata missing from {result_path}")
    matches = [
        item
        for item in replay.get("source_round_checkpoints", [])
        if isinstance(item, dict) and item.get("round") == round_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one replay checkpoint for round {round_index}, found {len(matches)}"
        )
    value = matches[0].get("adapter_sha256")
    return str(value) if value else None


def _validate_target(seed: int, target: Mapping[str, Any]) -> dict[str, Any]:
    config_path = (PROJECT_ROOT / str(target["config"])).resolve()
    adapter_path = (PROJECT_ROOT / str(target["adapter"])).resolve()
    r8_result_path = (PROJECT_ROOT / str(target["r8_result"])).resolve()
    required = [
        config_path,
        adapter_path / "adapter_config.json",
        adapter_path / "adapter_model.safetensors",
        r8_result_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Seed {seed} checkpoint is incomplete; missing: {missing}"
        )
    # Replay manifests use the full local-directory fingerprint, while the
    # load audit fingerprints only serialized config+weights. Record both and
    # compare like with like.
    actual_hash = hash_local_path(adapter_path)
    serialized_hash = adapter_artifact_sha256(adapter_path)
    replay_result = target.get("replay_result")
    expected_hash = None
    if replay_result:
        replay_path = (PROJECT_ROOT / str(replay_result)).resolve()
        if not replay_path.is_file():
            raise FileNotFoundError(f"Replay result is missing: {replay_path}")
        expected_hash = _expected_replay_hash(replay_path, int(target["round"]))
        if expected_hash != actual_hash:
            raise ValueError(
                f"Seed {seed} adapter hash differs from replay manifest: "
                f"expected={expected_hash}, actual={actual_hash}"
            )
    return {
        "config_path": config_path,
        "adapter_path": adapter_path,
        "adapter_sha256": actual_hash,
        "serialized_adapter_sha256": serialized_hash,
        "expected_replay_adapter_sha256": expected_hash,
        "r8_result_path": r8_result_path,
    }


def _frozen_examples(config: Mapping[str, Any]) -> tuple[list[HANSExample], dict[str, Any]]:
    benchmark = require_mapping(dict(config), "stage2_benchmark")
    final_path = (PROJECT_ROOT / str(benchmark["sealed_final_test_path"])).resolve()
    manifest_path = (PROJECT_ROOT / str(benchmark["frozen_manifest_path"])).resolve()
    rows = read_jsonl(final_path)
    manifest = FrozenHansManifest.from_mapping(read_json(manifest_path))
    ids = tuple(str(row.get("id", "")) for row in rows)
    if len(rows) != 15_000:
        raise ValueError(f"Expected full HANS 15,000 rows, observed {len(rows)}")
    if ids != manifest.final_test_ids:
        raise ValueError("Sealed HANS rows do not match frozen final-test IDs/order")
    examples = [
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
    return examples, {
        "sealed_final_test_path": str(final_path),
        "sealed_final_test_sha256": hash_local_path(final_path),
        "frozen_manifest_path": str(manifest_path),
        "frozen_manifest_hash": manifest.manifest_hash,
        "num_examples": len(examples),
    }


def _evaluate_rules(
    student: QwenNLIStudent,
    examples: list[HANSExample],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probabilities = student.predict_proba(
        [item.premise for item in examples],
        [item.hypothesis for item in examples],
        batch_size=batch_size,
    )
    if len(probabilities) != len(examples):
        raise RuntimeError("HANS probabilities and examples do not align")
    records: list[dict[str, Any]] = []
    three_way_predictions: list[str] = []
    direct_predictions: list[str] = []
    for example, row in zip(examples, probabilities, strict=True):
        values = [float(value) for value in row]
        if len(values) != len(MNLI_LABELS) or any(
            not math.isfinite(value) or value < 0.0 for value in values
        ):
            raise ValueError("Invalid MNLI probability row")
        mnli_prediction = MNLI_LABELS[
            max(range(len(values)), key=values.__getitem__)
        ]
        three_way = mnli_to_hans(mnli_prediction)
        direct = (
            "entailment"
            if values[0] >= values[1] + values[2]
            else "non-entailment"
        )
        three_way_predictions.append(three_way)
        direct_predictions.append(direct)
        records.append(
            {
                "id": example.id,
                "gold": example.gold,
                "prediction": three_way,
                "direct_binary_prediction": direct,
                "three_way_correct": three_way == example.gold,
                "direct_binary_correct": direct == example.gold,
                "prediction_rules_disagree": three_way != direct,
                "heuristic": example.heuristic,
                "subcase": example.subcase,
                "mnli_prediction": mnli_prediction,
                "mnli_probabilities": dict(zip(MNLI_LABELS, values, strict=True)),
            }
        )

    def metrics(predictions: list[str]) -> dict[str, Any]:
        return compute_hans_metrics(
            [item.gold for item in examples],
            predictions,
            [item.heuristic for item in examples],
            [item.subcase for item in examples],
        )

    disagreement = sum(
        left != right
        for left, right in zip(three_way_predictions, direct_predictions, strict=True)
    )
    return records, {
        "three_way_argmax_to_binary": metrics(three_way_predictions),
        "direct_binary_probability_aggregation": metrics(direct_predictions),
        "disagreement": {
            "count": disagreement,
            "rate": disagreement / len(examples),
        },
        "mnli_label_order": list(MNLI_LABELS),
        "sealed_final_test_used_for_decision": False,
    }


def _clear_cuda(student: QwenNLIStudent) -> None:
    torch = student._torch
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_target(
    seed: int,
    target: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    output_root: Path,
    device: str,
    batch_size_override: int | None,
    resume: bool,
) -> Path:
    round_index = int(target["round"])
    output = output_root / f"seed_{seed}" / f"round_{round_index:03d}"
    metrics_path = output / "metrics.json"
    provenance_path = output / "provenance.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not metrics_path.is_file() or not provenance_path.is_file():
            raise FileExistsError(f"Diagnostic output exists; use --resume: {output}")
        provenance = read_json(provenance_path)
        if provenance.get("adapter_sha256") != validated["adapter_sha256"]:
            raise RuntimeError(f"Resume adapter hash mismatch: {output}")
        print(f"[REUSE] seed={seed} round={round_index}: {metrics_path}")
        return metrics_path

    config = load_yaml_with_extends(validated["config_path"])
    model = require_mapping(config, "model")
    examples, frozen = _frozen_examples(config)
    batch_size = int(batch_size_override or model.get("batch_size", 4))
    set_deterministic_seed(seed)
    output.mkdir(parents=True, exist_ok=resume)
    started = time.perf_counter()
    student = QwenNLIStudent(
        name_or_path=str(model["name_or_path"]),
        adapter_path=str(validated["adapter_path"]),
        device_map=device,
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )
    student.progress_label = f"full HANS seed {seed} round {round_index}"
    load_audit = audit_loaded_adapter(student.model, validated["adapter_path"])
    records, metrics = _evaluate_rules(student, examples, batch_size=batch_size)
    wall_seconds = time.perf_counter() - started
    write_jsonl(output / "predictions.jsonl", records)
    write_json(metrics_path, metrics)
    write_json(output / "adapter_load_audit.json", load_audit)
    write_json(
        provenance_path,
        {
            "schema": "v7_early_checkpoint_full_hans_v1",
            "diagnostic_only": True,
            "used_for_online_stopping_or_tuning": False,
            "training_teacher_verification_pilots_run": False,
            "seed": seed,
            "round": round_index,
            "budget": int(target["budget"]),
            "adapter_path": str(validated["adapter_path"]),
            "adapter_sha256": validated["adapter_sha256"],
            "serialized_adapter_sha256": validated[
                "serialized_adapter_sha256"
            ],
            "expected_replay_adapter_sha256": validated[
                "expected_replay_adapter_sha256"
            ],
            "base_model": str(model["name_or_path"]),
            "device": device,
            "dtype": str(model.get("dtype", "bfloat16")),
            "batch_size": batch_size,
            "wall_seconds": wall_seconds,
            "git_commit": _git_commit(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            **frozen,
        },
    )
    _clear_cuda(student)
    print(f"[DONE] seed={seed} round={round_index}: {metrics_path}")
    return metrics_path


def _r8_rules(result_path: Path) -> Mapping[str, Any]:
    payload = read_json(result_path)
    rules = payload.get("hans_prediction_rules")
    if not isinstance(rules, dict):
        raise ValueError(f"R8 prediction-rule metrics missing from {result_path}")
    return rules


def _write_report(
    report_path: Path,
    *,
    seeds: list[int],
    output_root: Path,
    epsilon: float = 0.01,
) -> None:
    lines = [
        "# V7 Cost-Aware early-checkpoint full-HANS diagnostic",
        "",
        "> Retrospective diagnostic only. Full-HANS outcomes were not used to make or tune online stopping decisions.",
        "",
        f"Pairwise regret is `max(A_early, A_R8) - A_early`; epsilon-optimal uses `epsilon={epsilon:.2f}`.",
        "`R8 - early` is also reported so improvements over R8 remain visible.",
        "",
        "## Early-checkpoint full-HANS breakdown",
        "",
        "| Seed | Checkpoint | Rule | Overall | Entailment | Non-entailment | Rule disagreements |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        target = TARGETS[seed]
        round_index = int(target["round"])
        metrics_path = output_root / f"seed_{seed}" / f"round_{round_index:03d}" / "metrics.json"
        if not metrics_path.is_file():
            continue
        early = read_json(metrics_path)
        disagreement = early["disagreement"]
        disagreement_text = (
            f"{int(disagreement['count'])} ({100*float(disagreement['rate']):.4f}%)"
        )
        for key, label in (
            ("three_way_argmax_to_binary", "Three-way → binary"),
            ("direct_binary_probability_aggregation", "Direct Binary"),
        ):
            metric = early[key]
            lines.append(
                f"| {seed} | R{round_index} | {label} | "
                f"{100*float(metric['overall_accuracy']):.4f}% | "
                f"{100*float(metric['entailment_accuracy']):.4f}% | "
                f"{100*float(metric['non_entailment_accuracy']):.4f}% | "
                f"{disagreement_text} |"
            )
    lines.extend(
        [
        "",
        "## Early checkpoint versus R8",
        "",
        "| Seed | Checkpoint | Budget | Rule | Early accuracy | R8 accuracy | R8 − early | Pairwise regret | ε-optimal |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    comparison: list[dict[str, Any]] = []
    for seed in seeds:
        target = TARGETS[seed]
        round_index = int(target["round"])
        metrics_path = output_root / f"seed_{seed}" / f"round_{round_index:03d}" / "metrics.json"
        if not metrics_path.is_file():
            continue
        early = read_json(metrics_path)
        r8 = _r8_rules((PROJECT_ROOT / str(target["r8_result"])).resolve())
        for key, label in (
            ("three_way_argmax_to_binary", "Three-way → binary"),
            ("direct_binary_probability_aggregation", "Direct Binary"),
        ):
            early_acc = float(early[key]["overall_accuracy"])
            r8_acc = float(r8[key]["overall_accuracy"])
            delta = r8_acc - early_acc
            regret = max(early_acc, r8_acc) - early_acc
            epsilon_optimal = regret <= epsilon
            row = {
                "seed": seed,
                "round": round_index,
                "budget": int(target["budget"]),
                "rule": key,
                "early_accuracy": early_acc,
                "r8_accuracy": r8_acc,
                "r8_minus_early": delta,
                "pairwise_regret": regret,
                "epsilon": epsilon,
                "epsilon_optimal": epsilon_optimal,
            }
            comparison.append(row)
            lines.append(
                f"| {seed} | R{round_index} | {target['budget']} | {label} | "
                f"{100*early_acc:.4f}% | {100*r8_acc:.4f}% | "
                f"{100*delta:+.4f} pp | {100*regret:.4f} pp | "
                f"{'yes' if epsilon_optimal else 'no'} |"
            )
    lines.extend(
        [
            "",
            "This is a two-checkpoint comparison, not a complete `A_star=max_t A_t` oracle over all eight rounds. It cannot establish a population-level or online stopping guarantee.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_root / "comparison.json", {"epsilon": epsilon, "rows": comparison})


def main() -> None:
    args = _args()
    seeds = list(dict.fromkeys(args.seeds))
    unsupported = sorted(set(seeds) - set(TARGETS))
    if unsupported:
        raise ValueError(f"No predeclared early checkpoint target for seeds: {unsupported}")
    output_root = (PROJECT_ROOT / args.output_dir).resolve()
    validated = {seed: _validate_target(seed, TARGETS[seed]) for seed in seeds}
    print(
        json.dumps(
            {
                str(seed): {
                    "round": TARGETS[seed]["round"],
                    "adapter": str(value["adapter_path"]),
                    "adapter_sha256": value["adapter_sha256"],
                }
                for seed, value in validated.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    for seed in seeds:
        _evaluate_target(
            seed,
            TARGETS[seed],
            validated[seed],
            output_root=output_root,
            device=args.device,
            batch_size_override=args.batch_size,
            resume=args.resume,
        )
    _write_report(
        (PROJECT_ROOT / args.report).resolve(),
        seeds=seeds,
        output_root=output_root,
    )
    print(f"[DONE] report: {(PROJECT_ROOT / args.report).resolve()}")


if __name__ == "__main__":
    main()
