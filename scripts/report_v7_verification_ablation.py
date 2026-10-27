#!/usr/bin/env python3
"""Report the matched v7 training/verification ON-OFF ablation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OLD_V7_RESULT = (
    PROJECT_ROOT
    / "outputs"
    / "stage2_runs_ours_verified_v7_dual"
    / "experiment_b"
    / "ours_verified"
    / "budget_128"
    / "seed_42"
    / "branches"
    / "continuation"
    / "result.json"
)
MODES = ("on", "off")
SEEDS = (42, 43, 44)
GPU_TIME_FIELDS = (
    "acquisition_gpu_wall_seconds",
    "training_gpu_wall_seconds",
    "validation_gpu_wall_seconds",
    "stopping_evaluation_gpu_wall_seconds",
    "final_evaluation_gpu_wall_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _nested(result: dict[str, Any], *keys: str) -> float:
    value: Any = result
    for key in keys:
        value = value[key]
    return float(value)


def _summary(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def _display(values: list[float], *, percent: bool = True) -> str:
    mean, std = _summary(values)
    scale = 100.0 if percent else 1.0
    return f"{mean * scale:.4f} ± {std * scale:.4f}"


def _load_runs(root: Path) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for seed in SEEDS:
            path = root / f"verification_{mode}" / f"seed_{seed}" / "result.json"
            if not path.is_file():
                continue
            result = _read_json(path)
            if result.get("verification_mode") != mode:
                raise ValueError(f"Verification mode mismatch: {path}")
            if int(result.get("seed", -1)) != seed:
                raise ValueError(f"Seed mismatch: {path}")
            if result.get("acquisition_stopping") != "disabled_fixed_4x32":
                raise ValueError(f"Adaptive acquisition unexpectedly enabled: {path}")
            if int(result.get("requested_budget", -1)) != 128:
                raise ValueError(f"Unexpected acquisition budget: {path}")
            if (
                result.get("fixed_budget_comparable") is not True
                or int(result.get("actual_training_examples", -1)) != 128
                or int(result.get("rounds", -1)) != 4
            ):
                raise ValueError(f"Incomplete fixed-budget run cannot be reported: {path}")
            runs[mode].append(result)
    policies = {
        json.dumps(result["training_policy"], sort_keys=True)
        for mode_runs in runs.values()
        for result in mode_runs
    }
    if len(policies) > 1:
        raise ValueError("ON/OFF runs do not share one training policy")
    return runs


def main() -> None:
    args = parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(root)
    completed = sum(len(items) for items in runs.values())
    if not completed:
        raise FileNotFoundError(f"No completed ablation results under {root}")

    metric_rows: list[dict[str, str]] = []
    for mode in MODES:
        for result in runs[mode]:
            metric_rows.append(
                {
                    "verification": mode,
                    "seed": str(result["seed"]),
                    "hans_accuracy": str(
                        _nested(result, "hans_final", "overall_accuracy")
                    ),
                    "hans_non_entailment_accuracy": str(
                        _nested(result, "hans_final", "non_entailment_accuracy")
                    ),
                    "mnli_matched_accuracy": str(
                        _nested(result, "mnli", "matched", "accuracy")
                    ),
                    "mnli_mismatched_accuracy": str(
                        _nested(result, "mnli", "mismatched", "accuracy")
                    ),
                    "optimizer_steps": str(result["optimizer_steps"]),
                }
            )
    csv_path = output / "metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    lines = [
        "# V7 Training Early Stopping + Verification ON/OFF",
        "",
        "The two arms share the same initial pool, 4x32 acquisition budget, "
        "selector, stateful LoRA+AdamW training policy, seed schedule, and "
        "synthetic QC. Only hypothesis verification differs.",
        "",
        "Training early stopping uses binary NLL on the predeclared gain-eval "
        "stopping subset. The sealed final test is evaluated only after training.",
        "",
        "## Final metrics",
        "",
        "| Verification | Seeds complete | HANS | HANS non-ent. | MNLI-m | MNLI-mm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        items = runs[mode]
        if not items:
            lines.append(f"| {mode.upper()} | 0/3 | pending | pending | pending | pending |")
            continue
        lines.append(
            f"| {mode.upper()} | {len(items)}/3 | "
            f"{_display([_nested(x, 'hans_final', 'overall_accuracy') for x in items])} | "
            f"{_display([_nested(x, 'hans_final', 'non_entailment_accuracy') for x in items])} | "
            f"{_display([_nested(x, 'mnli', 'matched', 'accuracy') for x in items])} | "
            f"{_display([_nested(x, 'mnli', 'mismatched', 'accuracy') for x in items])} |"
        )

    if OLD_V7_RESULT.is_file():
        old_v7 = _read_json(OLD_V7_RESULT)
        lines.extend(
            [
                "",
                "Historical old-v7 reference (different training policy): "
                f"HANS {_nested(old_v7, 'hans_final', 'overall_accuracy') * 100:.4f}%, "
                f"MNLI-m {_nested(old_v7, 'mnli', 'matched', 'accuracy') * 100:.4f}%, "
                f"MNLI-mm {_nested(old_v7, 'mnli', 'mismatched', 'accuracy') * 100:.4f}%. "
                "It is context only, not the controlled ON/OFF contrast.",
            ]
        )

    lines.extend(
        [
            "",
            "Mean ± sample standard deviation is reported when at least two seeds "
            "are complete; a single completed seed has standard deviation 0.",
            "",
            "## Learning curves",
            "",
            "| Verification | Round | Seeds | Stopping accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        by_round: dict[int, list[float]] = {}
        for result in runs[mode]:
            for point in result["learning_curve"]:
                by_round.setdefault(int(point["round"]), []).append(
                    float(point["metrics"]["overall_accuracy"])
                )
        for round_index in sorted(by_round):
            values = by_round[round_index]
            lines.append(
                f"| {mode.upper()} | {round_index} | {len(values)} | "
                f"{_display(values)} |"
            )

    lines.extend(
        [
            "",
            "## Training and resource summary",
            "",
            "| Run | Committed AdamW steps | Best epochs per round | "
            "GPU proxy seconds | Teacher seconds | Teacher input/output tokens |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for result in runs[mode]:
            epochs = ",".join(
                str(item["best_epoch"]) for item in result["training_history"]
            )
            resources = result["resource_usage"]
            teacher = resources["teacher"]
            gpu_seconds = sum(float(resources.get(key, 0.0)) for key in GPU_TIME_FIELDS)
            lines.append(
                f"| {mode.upper()} seed {result['seed']} | "
                f"{result['optimizer_steps']} | {epochs} | "
                f"{gpu_seconds:.2f} | "
                f"{float(teacher.get('teacher_wall_clock_seconds', 0.0)):.2f} | "
                f"{int(teacher['input_tokens'])}/{int(teacher['output_tokens'])} |"
            )

    missing = [
        f"{mode}:{seed}"
        for mode in MODES
        for seed in SEEDS
        if not (
            root / f"verification_{mode}" / f"seed_{seed}" / "result.json"
        ).is_file()
    ]
    lines.extend(
        [
            "",
            "## Completion",
            "",
            f"- Completed runs: {completed}/6",
            f"- Missing runs: {', '.join(missing) if missing else 'none'}",
            "",
        ]
    )
    report_path = output / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
