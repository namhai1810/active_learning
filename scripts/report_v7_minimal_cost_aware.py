#!/usr/bin/env python3
"""Build an artifact-only three-seed V7 minimal Cost-Aware report."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.artifacts import read_json  # noqa: E402
from src.utils.io import _atomic_text_write  # noqa: E402


SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", default="outputs/stage2_runs_v7_minimal_cost_aware"
    )
    parser.add_argument(
        "--output", default="reports/v7_minimal_cost_aware/REPORT.md"
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def _mean_std(values: list[float], *, percent: bool = False) -> str:
    if not values:
        return "n/a"
    scale = 100.0 if percent else 1.0
    mean = statistics.mean(values) * scale
    std = statistics.stdev(values) * scale if len(values) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}{'%' if percent else ''}"


def load_runs(root: Path, *, allow_incomplete: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in SEEDS:
        run = root / f"seed_{seed}"
        if not (run / ".complete").is_file():
            missing.append(seed)
            continue
        result = read_json(run / "result.json")
        if result.get("method") != "ours_v7_minimal_cost_aware":
            raise ValueError(f"Unexpected method in {run}")
        rows.append(result)
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Missing completed seeds: {missing}")
    return rows


def build_report(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# V7 Minimal Cost-Aware Report",
        "",
        "> Artifact-only report; no model, teacher, training, or inference is run.",
        "",
        "The controller evaluates the exact mixed-hypothesis V7 soft-pair action. "
        "It does not claim the appendix theorem for hypothesis-specific interventions.",
        "",
        "## Final results",
        "",
        "| Seed | Rounds | Examples | Stop category | Stop reason | HANS | MNLI m/mm | Modeled decision cost (r2+) | Teacher tokens |",
        "|---:|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for result in sorted(rows, key=lambda item: int(item["seed"])):
        teacher = result.get("resource_usage", {}).get("teacher", {})
        tokens = float(teacher.get("input_tokens", 0.0)) + float(
            teacher.get("output_tokens", 0.0)
        )
        lines.append(
            f"| {result['seed']} | {result['rounds']} | "
            f"{result['actual_training_examples']} | "
            f"`{result['stop_reason_category']}` | `{result['stop_reason']}` | "
            f"{100*float(result['hans_final']['overall_accuracy']):.4f}% | "
            f"{100*float(result['mnli']['matched']['accuracy']):.4f}% / "
            f"{100*float(result['mnli']['mismatched']['accuracy']):.4f}% | "
            f"{float(result['total_modeled_decision_cost_rounds_2_plus']):.6f} | "
            f"{tokens:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Mean ± sample standard deviation",
            "",
            f"- HANS: {_mean_std([float(r['hans_final']['overall_accuracy']) for r in rows], percent=True)}",
            f"- MNLI matched: {_mean_std([float(r['mnli']['matched']['accuracy']) for r in rows], percent=True)}",
            f"- MNLI mismatched: {_mean_std([float(r['mnli']['mismatched']['accuracy']) for r in rows], percent=True)}",
            f"- Acquired examples: {_mean_std([float(r['actual_training_examples']) for r in rows])}",
            f"- Completed rounds: {_mean_std([float(r['rounds']) for r in rows])}",
            f"- Modeled decision cost (rounds 2+): {_mean_std([float(r['total_modeled_decision_cost_rounds_2_plus']) for r in rows])}",
            "",
            "## Decision traces",
            "",
            "| Seed | Round | Budget before | Certified | Unresolved | Pilots | Gain | Radius | Cost | UCB net | Action | Reason |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for result in sorted(rows, key=lambda item: int(item["seed"])):
        decisions = list(result.get("cost_aware_rounds") or [])
        stop = result.get("cost_aware_stop_record")
        if isinstance(stop, dict):
            decisions.append(stop)
        for item in decisions:
            decision = item["decision"]
            estimate = item.get("gain_estimate") or {}
            lines.append(
                f"| {result['seed']} | {item['round']} | "
                f"{item['cumulative_examples_before_decision']} | "
                f"{len(decision.get('certified_hypotheses', []))} | "
                f"{len(decision.get('unresolved_hypotheses', []))} | "
                f"{estimate.get('m', 0)} | "
                f"{estimate.get('delta_hat', 'n/a')} | "
                f"{estimate.get('radius', 'n/a')} | "
                f"{decision.get('cost', 'n/a')} | "
                f"{decision.get('optimistic_net_value', 'n/a')} | "
                f"`{decision['action']}` | `{decision['reason']}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Unresolved hypotheses force CONTINUE; they are never treated as refuted.",
            "- Pilot gain uses predeclared, disjoint gain-eval shards and never sealed HANS/MNLI.",
            "- Conditional independence/unbiasedness of related LoRA pilots is not empirically proven.",
            "- With m=4, K=1, T=7, δ=.05 and M=1, the radius is about 1.6785; λ=.001 is likely to make STOP rare.",
            "- Round-1 and the frozen initial-pool sunk cost are outside the round-2+ decision-cost aggregate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = load_runs(Path(args.input_root), allow_incomplete=args.allow_incomplete)
    output = Path(args.output)
    _atomic_text_write(output, build_report(rows))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

