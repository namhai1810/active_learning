#!/usr/bin/env python3
"""Aggregate completed Stage 2 runs into JSON, CSV, and a paper-ready report."""

from __future__ import annotations

import argparse
import csv
import html
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
from src.utils.io import _atomic_text_write, write_json


METHOD_ORDER = (
    "zero_shot",
    "random_mnli",
    "uncertainty_mnli",
    "random_synthetic",
    "failure_driven_no_verification",
    "ours_verified",
)
METHOD_COLORS = {
    "zero_shot": "#64748B",
    "random_mnli": "#3B82F6",
    "uncertainty_mnli": "#8B5CF6",
    "random_synthetic": "#F59E0B",
    "failure_driven_no_verification": "#F97316",
    "ours_verified": "#10B981",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_ORDER,
        help="Include only these methods (default: all completed methods).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Include only these seeds (default: all completed seeds).",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        help="Include only these budgets (default: all completed budgets).",
    )
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


def _svg(width: int, height: int, title: str, body: list[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{html.escape(title)}</title>
<rect width="100%" height="100%" fill="#FFFFFF" rx="16"/>
<style>
text {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; fill: #334155; }}
.title {{ font-size: 22px; font-weight: 700; fill: #0F172A; }}
.axis {{ font-size: 11px; fill: #64748B; }}
.label {{ font-size: 12px; font-weight: 600; }}
</style>
{chr(10).join(body)}
</svg>
'''


def _write_final_accuracy_figure(
    summaries: list[dict[str, Any]], figures: Path
) -> None:
    rows = [
        row for row in summaries
        if row["experiment"] == "A"
        and not row.get("any_abstained", False)
        and not row.get("any_shortage", False)
        and row["hans_final.overall_accuracy"]["mean"] is not None
    ]
    rows.sort(
        key=lambda row: (
            METHOD_ORDER.index(row["method"]),
            int(row["budget"]),
        )
    )
    width, row_height = 1120, 34
    left, right, top, bottom = 310, 55, 90, 55
    height = top + bottom + row_height * len(rows)
    chart_width = width - left - right
    body = [
        '<text x="34" y="40" class="title">Experiment A · sealed HANS final accuracy</text>',
        '<text x="34" y="64" class="axis">Mean across completed seeds; error bars are descriptive 95% normal intervals</text>',
    ]
    for tick in range(0, 101, 20):
        x = left + chart_width * tick / 100
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+24}" text-anchor="middle" class="axis">{tick}%</text>',
            ]
        )
    for index, row in enumerate(rows):
        y = top + index * row_height
        mean = 100.0 * float(row["hans_final.overall_accuracy"]["mean"])
        ci = 100.0 * float(
            row["hans_final.overall_accuracy"]["normal_95ci_half_width"] or 0.0
        )
        end = left + chart_width * mean / 100.0
        low = left + chart_width * max(0.0, mean - ci) / 100.0
        high = left + chart_width * min(100.0, mean + ci) / 100.0
        label = html.escape(f'{row["method"]} · budget {row["budget"]}')
        color = METHOD_COLORS.get(row["method"], "#334155")
        body.extend(
            [
                f'<text x="{left-12}" y="{y+19}" text-anchor="end" class="label">{label}</text>',
                f'<rect x="{left}" y="{y+6}" width="{max(1.0, end-left):.1f}" height="16" rx="4" fill="{color}" opacity="0.88"/>',
                f'<line x1="{low:.1f}" y1="{y+14}" x2="{high:.1f}" y2="{y+14}" stroke="#0F172A" stroke-width="1.5"/>',
                f'<text x="{min(width-right, end+7):.1f}" y="{y+19}" class="axis">{mean:.2f}%</text>',
            ]
        )
    _atomic_text_write(
        figures / "experiment_a_final_hans_accuracy.svg",
        _svg(width, height, "Experiment A final HANS accuracy", body),
    )


def _write_learning_curve_figure(
    curves: list[dict[str, Any]], figures: Path
) -> None:
    aggregated: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in curves:
        mean = row["gain_eval_overall_accuracy"]["mean"]
        if row["experiment"] == "A" and mean is not None:
            aggregated[
                (row["method"], int(row["cumulative_final_training_examples"]))
            ].append(float(mean))
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (method, examples), values in aggregated.items():
        series[method].append((examples, statistics.mean(values)))
    width, height = 1050, 620
    left, right, top, bottom = 90, 250, 90, 80
    chart_width, chart_height = width - left - right, height - top - bottom
    max_examples = max(
        1,
        max((x for points in series.values() for x, _ in points), default=1),
    )
    body = [
        '<text x="34" y="40" class="title">Experiment A · post-hoc gain-eval learning curve</text>',
        '<text x="34" y="64" class="axis">The sealed final-test is not used for these intermediate curves</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_height * (1.0 - tick / 100.0)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}%</text>',
            ]
        )
    for tick in (0, 32, 64, 96, 128):
        if tick > max_examples:
            continue
        x = left + chart_width * tick / max_examples
        body.append(
            f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" class="axis">{tick}</text>'
        )
    for legend_index, method in enumerate(METHOD_ORDER):
        points = sorted(series.get(method, []))
        if not points:
            continue
        color = METHOD_COLORS[method]
        coordinates = [
            (
                left + chart_width * examples / max_examples,
                top + chart_height * (1.0 - accuracy),
            )
            for examples, accuracy in points
        ]
        body.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="3"/>'.format(
                " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates), color
            )
        )
        for x, y in coordinates:
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = top + legend_index * 28
        body.extend(
            [
                f'<rect x="{width-right+35}" y="{legend_y-9}" width="18" height="5" fill="{color}"/>',
                f'<text x="{width-right+62}" y="{legend_y}" class="label">{html.escape(method)}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-22}" text-anchor="middle" class="axis">Cumulative final-training examples</text>',
            f'<text x="22" y="{top+chart_height/2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top+chart_height/2:.1f})">gain_eval accuracy</text>',
        ]
    )
    _atomic_text_write(
        figures / "experiment_a_gain_eval_learning_curve.svg",
        _svg(width, height, "Experiment A gain-eval learning curve", body),
    )


def _write_cost_accuracy_figure(
    summaries: list[dict[str, Any]], figures: Path
) -> None:
    rows = [
        row for row in summaries
        if row["experiment"] == "A"
        and not row.get("any_abstained", False)
        and not row.get("any_shortage", False)
        and row["hans_final.overall_accuracy"]["mean"] is not None
        and row["assumed_cost.total"]["mean"] is not None
    ]
    width, height = 1100, 650
    left, right, top, bottom = 95, 280, 90, 85
    chart_width, chart_height = width - left - right, height - top - bottom
    max_cost = max((float(row["assumed_cost.total"]["mean"]) for row in rows), default=1.0)
    max_cost = max(max_cost, 1e-12)
    body = [
        '<text x="34" y="40" class="title">Experiment A · final accuracy versus modeled cost</text>',
        '<text x="34" y="64" class="axis">Cost uses measured quantities and configured rates saved by each run</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_height * (1.0 - tick / 100.0)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}%</text>',
            ]
        )
    for index in range(6):
        value = max_cost * index / 5
        x = left + chart_width * index / 5
        body.append(
            f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" class="axis">{value:.3g}</text>'
        )
    for row in rows:
        cost = float(row["assumed_cost.total"]["mean"])
        accuracy = float(row["hans_final.overall_accuracy"]["mean"])
        x = left + chart_width * cost / max_cost
        y = top + chart_height * (1.0 - accuracy)
        color = METHOD_COLORS.get(row["method"], "#334155")
        label = html.escape(f'{row["method"]} · b{row["budget"]}')
        body.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{label}: cost={cost:.6g}, accuracy={accuracy:.4f}</title></circle>',
                f'<text x="{width-right+30}" y="{top+16*rows.index(row):.1f}" class="axis" fill="{color}">{label}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-23}" text-anchor="middle" class="axis">Modeled total cost</text>',
            f'<text x="22" y="{top+chart_height/2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top+chart_height/2:.1f})">Sealed HANS final accuracy</text>',
        ]
    )
    _atomic_text_write(
        figures / "experiment_a_accuracy_vs_cost.svg",
        _svg(width, height, "Experiment A accuracy versus cost", body),
    )


def main() -> None:
    args = parse_args()
    root = Path(args.runs_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    discovered_entries = [
        (path, read_json(path)) for path in sorted(root.rglob("result.json"))
    ]
    selected_methods = set(args.methods) if args.methods else None
    selected_seeds = set(args.seeds) if args.seeds else None
    selected_budgets = set(args.budgets) if args.budgets else None
    result_entries = [
        (path, result)
        for path, result in discovered_entries
        if (selected_methods is None or str(result.get("method")) in selected_methods)
        and (selected_seeds is None or int(result.get("seed", -1)) in selected_seeds)
        and (
            selected_budgets is None
            or int(result.get("requested_budget", -1)) in selected_budgets
        )
    ]
    if not result_entries:
        raise FileNotFoundError(
            f"No Stage 2 result.json files beneath {root} match the requested filters"
        )
    results = [result for _, result in result_entries]
    for result in results:
        if not bool(result.get("abstained", False)):
            continue
        if (
            result.get("run_status") != "abstained_no_certified_hypotheses"
            or result.get("abstention_reason") != "no_certified_hypotheses"
            or int(result.get("actual_final_training_examples", -1)) != 0
            or result.get("shortage") is not True
            or result.get("fixed_budget_comparable") is not False
        ):
            raise ValueError("Malformed Stage 2 certification-abstention result")
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
        abstained_seeds = sorted(
            int(item["seed"]) for item in group if bool(item.get("abstained", False))
        )
        comparable_seeds = sorted(
            int(item["seed"])
            for item in group
            if bool(item.get("fixed_budget_comparable", True))
        )
        row: dict[str, Any] = {
            "experiment": experiment,
            "method": method,
            "budget": budget,
            "seeds": sorted(int(item["seed"]) for item in group),
            "cross_source_baseline": group[0]["cross_source_baseline"],
            "any_shortage": any(bool(item["shortage"]) for item in group),
            "any_abstained": bool(abstained_seeds),
            "abstained_seeds": abstained_seeds,
            "comparable_seeds": comparable_seeds,
            "all_fixed_budget_comparable": len(comparable_seeds) == len(group),
            "any_leakage_audit_failure": any(
                not bool(item.get("leakage_audit", {}).get("passed", False))
                for item in group
            ),
        }
        for path in metric_paths:
            metric_group = (
                [
                    item
                    for item in group
                    if bool(item.get("fixed_budget_comparable", True))
                ]
                if path.startswith(("hans_final.", "mnli."))
                else group
            )
            row[path] = _summary(
                [
                    value
                    for item in metric_group
                    if (value := _metric(item, path)) is not None
                ]
            )
        summaries.append(row)

    curve_groups: dict[
        tuple[str, str, int, int, int], list[dict[str, Any]]
    ] = defaultdict(list)
    for result_path, result in result_entries:
        if not bool(result.get("fixed_budget_comparable", True)):
            continue
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
            "cross_source_baseline", "any_shortage", "any_abstained",
            "abstained_seeds", "comparable_seeds",
            "all_fixed_budget_comparable",
            "any_leakage_audit_failure",
        ]
        for path in metric_paths:
            fields.extend([f"{path}.mean", f"{path}.std", f"{path}.ci95", f"{path}.n"])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summaries:
            flat: dict[str, Any] = {key: item[key] for key in fields[:11]}
            flat["seeds"] = ",".join(map(str, item["seeds"]))
            flat["abstained_seeds"] = ",".join(
                map(str, item["abstained_seeds"])
            )
            flat["comparable_seeds"] = ",".join(
                map(str, item["comparable_seeds"])
            )
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
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_final_accuracy_figure(summaries, figures)
    _write_learning_curve_figure(curve_summaries, figures)
    _write_cost_accuracy_figure(summaries, figures)
    lines = [
        "# Stage 2 benchmark report",
        "",
        f"Frozen HANS manifest: `{next(iter(hashes))}`",
        "",
        "Random/Uncertainty are MNLI cross-source baselines; equal data access with "
        "HANS-guided synthetic methods is not claimed. Random-synthetic and the "
        "no-verification ablation use pairs for selection but train with per-example "
        "cross-entropy. Ours uses token cross-entropy plus counterfactual contrastive "
        "and invariant consistency losses.",
        "Runs that abstain because no failure hypothesis is certified are retained "
        "as protocol outcomes and costs, but are excluded from equal-budget "
        "accuracy figures and learning-curve comparisons.",
        "",
        "![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)",
        "",
        "![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)",
        "",
        "![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)",
        "",
        "| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    def fmt(summary: dict[str, Any]) -> str:
        if summary["mean"] is None:
            return "NA"
        return f"{summary['mean']:.4f} ± {summary['std']:.4f}"
    for item in summaries:
        lines.append(
            "| {experiment} | {method} | {budget} | {seeds} | {status} | {hans} | {nonent} | {matched} / {mismatch} | {actual} | {cost} | {shortage} | {leakage} |".format(
                experiment=item["experiment"], method=item["method"], budget=item["budget"],
                seeds=",".join(map(str, item["seeds"])),
                status=(
                    "ABSTAINED"
                    if item["any_abstained"]
                    else "complete"
                ),
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
    _atomic_text_write(output / "STAGE2_REPORT.md", "\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "runs": len(results),
                "groups": len(summaries),
                "figures": 3,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
