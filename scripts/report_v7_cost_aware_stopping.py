#!/usr/bin/env python3
"""Aggregate completed V7 cost-aware safe-stopping seeds without inference."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.artifacts import read_json
from src.utils.io import _atomic_text_write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default="outputs/stage2_runs_v7_cost_aware_safe_stopping",
    )
    parser.add_argument(
        "--output",
        default="reports/stage2_v7_cost_aware_safe_stopping/REPORT.md",
    )
    return parser.parse_args()


def _mean_std(values: list[float], *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    mean = statistics.mean(values) * scale
    std = statistics.stdev(values) * scale if len(values) > 1 else 0.0
    suffix = "%" if percent else ""
    return f"{mean:.4f} +/- {std:.4f}{suffix}"


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    rows: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        run = root / f"seed_{seed}"
        if not (run / ".complete").is_file() or not (run / "result.json").is_file():
            continue
        result = read_json(run / "result.json")
        rules = result.get("hans_final_prediction_rules")
        if not isinstance(rules, dict):
            raise ValueError(f"Seed {seed} has no dual-rule HANS metrics")
        teacher = result["resource_usage"]["teacher"]
        rows.append(
            {
                "seed": seed,
                "rounds": int(result.get("rounds_attempted", result["rounds"])),
                "examples": int(result["actual_training_examples"]),
                "stop_reason": str(result["stop_reason"]),
                "three": float(rules["three_way_to_binary"]["overall_accuracy"]),
                "three_e": float(
                    rules["three_way_to_binary"]["entailment_accuracy"]
                ),
                "three_ne": float(
                    rules["three_way_to_binary"]["non_entailment_accuracy"]
                ),
                "binary": float(rules["direct_binary"]["overall_accuracy"]),
                "binary_e": float(rules["direct_binary"]["entailment_accuracy"]),
                "binary_ne": float(
                    rules["direct_binary"]["non_entailment_accuracy"]
                ),
                "cost": float(result["total_modeled_adaptive_cost"]),
                "input_tokens": float(teacher["input_tokens"]),
                "output_tokens": float(teacher["output_tokens"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed cost-aware runs under {root}")

    lines = [
        "# V7 Cost-Aware Safe-Stopping Report",
        "",
        "Only completed saved artifacts are aggregated; this script performs no inference.",
        "",
        "| Seed | Rounds | Examples | Stop reason | Three-way O/E/NE | Direct Binary O/E/NE | Modeled adaptive cost | Teacher input/output tokens |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['rounds']} | {row['examples']} | "
            f"`{row['stop_reason']}` | {100 * row['three']:.4f}% / "
            f"{100 * row['three_e']:.4f}% / {100 * row['three_ne']:.4f}% | "
            f"{100 * row['binary']:.4f}% / {100 * row['binary_e']:.4f}% / "
            f"{100 * row['binary_ne']:.4f}% | {row['cost']:.6f} | "
            f"{row['input_tokens']:.0f} / {row['output_tokens']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Mean +/- sample standard deviation",
            "",
            f"- HANS Three-way: {_mean_std([row['three'] for row in rows], percent=True)}",
            f"- HANS Direct Binary: {_mean_std([row['binary'] for row in rows], percent=True)}",
            f"- Modeled adaptive cost: {_mean_std([row['cost'] for row in rows])}",
            f"- Teacher input tokens: {_mean_std([row['input_tokens'] for row in rows])}",
            f"- Teacher output tokens: {_mean_std([row['output_tokens'] for row in rows])}",
            "",
            "The sealed HANS final test is evaluated only after the adaptive decisions are frozen. Cost projections are decision-time assumptions; the cumulative ledger adds incurred exploration and realized intervention phases once.",
        ]
    )
    output = Path(args.output)
    _atomic_text_write(output, "\n".join(lines) + "\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
