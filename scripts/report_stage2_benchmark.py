#!/usr/bin/env python3
"""Aggregate completed Stage 2 runs into JSON, CSV, and a paper-ready report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.artifacts import read_json
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _metric(result: dict[str, Any], path: str) -> float | None:
    value: Any = result
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "normal_95ci_half_width": None}
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": std,
        "normal_95ci_half_width": 1.96 * std / math.sqrt(len(values)),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_entries = [
        (path, read_json(path)) for path in sorted(root.rglob("result.json"))
    ]
    if not result_entries:
        raise FileNotFoundError(f"No Stage 2 result.json files beneath {root}")
    results = [result for _, result in result_entries]
    hashes = {result.get("manifest_hash") for result in results}
    if len(hashes) != 1 or None in hashes:
        raise ValueError("All Stage 2 runs must use exactly one frozen manifest hash")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[
            (
                str(result["experiment"]),
                str(result["method"]),
                int(result["requested_budget"]),
            )
        ].append(result)
    subcases = sorted(
        {
            subcase
            for result in results
            for subcase in result.get("hans_final", {})
            .get("accuracy_by_subcase", {})
        }
    )
    metric_paths = (
        "hans_final.overall_accuracy",
        "hans_final.non_entailment_accuracy",
        "hans_final.accuracy_by_heuristic.lexical_overlap",
        "hans_final.accuracy_by_heuristic.subsequence",
        "hans_final.accuracy_by_heuristic.constituent",
        "mnli.matched.accuracy",
        "mnli.mismatched.accuracy",
        "actual_final_training_examples",
        "optimizer_steps",
        "pilot_count",
        "resource_usage.llm_input_tokens",
        "resource_usage.llm_output_tokens",
        "resource_usage.generated_items",
        "training_gpu_seconds",
        "acquisition_gpu_seconds",
        "evaluation_gpu_seconds",
        "assumed_cost.total",
        *(f"hans_final.accuracy_by_subcase.{item}" for item in subcases),
    )
    summaries: list[dict[str, Any]] = []
    for (experiment, method, budget), group in sorted(grouped.items()):
        row: dict[str, Any] = {
            "experiment": experiment,
            "method": method,
            "budget": budget,
            "seeds": sorted(int(item["seed"]) for item in group),
            "cross_source_baseline": group[0]["cross_source_baseline"],
            "any_shortage": any(bool(item["shortage"]) for item in group),
            "any_leakage_audit_failure": any(
                not bool(item.get("leakage_audit", {}).get("passed", False))
                for item in group
            ),
        }
        for path in metric_paths:
            row[path] = _summary(
                [value for item in group if (value := _metric(item, path)) is not None]
            )
        summaries.append(row)

    curve_groups: dict[
        tuple[str, str, int, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for result_path, result in result_entries:
        curve_name = str(result.get("learning_curve_file", ""))
        if not curve_name:
            raise ValueError(f"Run has no learning curve reference: {result_path}")
        curve = read_json(result_path.parent / curve_name)
        points = curve.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"Run has no learning curve points: {result_path}")
        for point in points:
            if not isinstance(point, dict) or point.get("evaluation_role") != "gain_eval":
                raise ValueError(f"Learning curve must use gain_eval: {result_path}")
            key = (
                str(result["experiment"]),
                str(result["method"]),
                int(result["requested_budget"]),
                int(point["round"]),
                int(point["cumulative_final_training_examples"]),
            )
            curve_groups[key].append(point)
    curve_summaries: list[dict[str, Any]] = []
    for (experiment, method, budget, round_index, examples), points in sorted(
        curve_groups.items()
    ):
        curve_summaries.append(
            {
                "experiment": experiment,
                "method": method,
                "budget": budget,
                "round": round_index,
                "cumulative_final_training_examples": examples,
                "gain_eval_overall_accuracy": _summary(
                    [float(point["metrics"]["overall_accuracy"]) for point in points]
                ),
                "gain_eval_non_entailment_accuracy": _summary(
                    [
                        float(point["metrics"]["non_entailment_accuracy"])
                        for point in points
                    ]
                ),
                "cumulative_total_cost": _summary(
                    [float(point["cumulative_total_cost"]) for point in points]
                ),
            }
        )
    write_json(
        output / "stage2_summary.json",
        {
            "manifest_hash": next(iter(hashes)),
            "uncertainty": (
                "Mean, sample standard deviation, and normal-approximation 95% CI "
                "across completed seeds; n is reported and no missing seed is imputed."
            ),
            "groups": summaries,
            "learning_curve_role": "gain_eval",
            "learning_curves": curve_summaries,
        },
    )
    csv_path = output / "stage2_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "experiment", "method", "budget", "seeds",
            "cross_source_baseline", "any_shortage",
            "any_leakage_audit_failure",
        ]
        for path in metric_paths:
            fields.extend([f"{path}.mean", f"{path}.std", f"{path}.ci95", f"{path}.n"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            flat: dict[str, Any] = {key: item[key] for key in fields[:7]}
            flat["seeds"] = ",".join(map(str, item["seeds"]))
            for path in metric_paths:
                flat[f"{path}.mean"] = item[path]["mean"]
                flat[f"{path}.std"] = item[path]["std"]
                flat[f"{path}.ci95"] = item[path]["normal_95ci_half_width"]
                flat[f"{path}.n"] = item[path]["n"]
            writer.writerow(flat)
    curve_csv_path = output / "stage2_learning_curves.csv"
    with curve_csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "experiment", "method", "budget", "round",
            "cumulative_final_training_examples",
            "gain_eval_overall_accuracy.mean", "gain_eval_overall_accuracy.std",
            "gain_eval_non_entailment_accuracy.mean",
            "gain_eval_non_entailment_accuracy.std",
            "cumulative_total_cost.mean", "cumulative_total_cost.std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in curve_summaries:
            writer.writerow(
                {
                    "experiment": item["experiment"],
                    "method": item["method"],
                    "budget": item["budget"],
                    "round": item["round"],
                    "cumulative_final_training_examples": item[
                        "cumulative_final_training_examples"
                    ],
                    "gain_eval_overall_accuracy.mean": item[
                        "gain_eval_overall_accuracy"
                    ]["mean"],
                    "gain_eval_overall_accuracy.std": item[
                        "gain_eval_overall_accuracy"
                    ]["std"],
                    "gain_eval_non_entailment_accuracy.mean": item[
                        "gain_eval_non_entailment_accuracy"
                    ]["mean"],
                    "gain_eval_non_entailment_accuracy.std": item[
                        "gain_eval_non_entailment_accuracy"
                    ]["std"],
                    "cumulative_total_cost.mean": item["cumulative_total_cost"][
                        "mean"
                    ],
                    "cumulative_total_cost.std": item["cumulative_total_cost"][
                        "std"
                    ],
                }
            )
    lines = [
        "# Stage 2 benchmark report",
        "",
        f"Frozen HANS manifest: `{next(iter(hashes))}`",
        "",
        "Random/Uncertainty are MNLI cross-source baselines; equal data access with "
        "HANS-guided synthetic methods is not claimed. Synthetic pair structure is "
        "used for selection only; training uses per-example cross-entropy.",
        "",
        "| Exp. | Method | Budget | Seeds | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    def fmt(summary: dict[str, Any]) -> str:
        if summary["mean"] is None:
            return "NA"
        return f"{summary['mean']:.4f} ± {summary['std']:.4f}"
    for item in summaries:
        lines.append(
            "| {experiment} | {method} | {budget} | {seeds} | {hans} | {nonent} | {matched} / {mismatch} | {actual} | {cost} | {shortage} | {leakage} |".format(
                experiment=item["experiment"], method=item["method"], budget=item["budget"],
                seeds=",".join(map(str, item["seeds"])),
                hans=fmt(item["hans_final.overall_accuracy"]),
                nonent=fmt(item["hans_final.non_entailment_accuracy"]),
                matched=fmt(item["mnli.matched.accuracy"]),
                mismatch=fmt(item["mnli.mismatched.accuracy"]),
                actual=fmt(item["actual_final_training_examples"]),
                cost=fmt(item["assumed_cost.total"]),
                shortage="yes" if item["any_shortage"] else "no",
                leakage=(
                    "FAIL" if item["any_leakage_audit_failure"] else "pass"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Post-hoc learning curves (`gain_eval`)",
            "",
            "| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in curve_summaries:
        lines.append(
            "| {experiment} | {method} | {budget} | {round} | {examples} | {accuracy} | {cost} |".format(
                experiment=item["experiment"],
                method=item["method"],
                budget=item["budget"],
                round=item["round"],
                examples=item["cumulative_final_training_examples"],
                accuracy=fmt(item["gain_eval_overall_accuracy"]),
                cost=fmt(item["cumulative_total_cost"]),
            )
        )
    lines.extend(
        [
            "",
            "The learning curve uses `gain_eval`; the sealed HANS final-test is "
            "evaluated once on the final checkpoint only.",
            "",
            "Intervals are descriptive across seeds and use a normal approximation; "
            "with three seeds they should not be interpreted as strong coverage guarantees.",
        ]
    )
    (output / "STAGE2_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(results), "groups": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
