#!/usr/bin/env python3
"""Build one professor/paper-facing seed-42 Stage-2 overview and SVG figures."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import _atomic_text_write


@dataclass(frozen=True)
class RunSpec:
    label: str
    relative_path: str
    group: str
    status: str
    rankable: bool = True


CORE_SPECS = (
    RunSpec(
        "zero_shot",
        "outputs/stage2_runs/experiment_a/zero_shot/budget_128/seed_42/result.json",
        "reference",
        "reference; no training",
    ),
    RunSpec(
        "random_synthetic",
        "outputs/stage2_runs/experiment_a/random_synthetic/budget_128/seed_42/result.json",
        "synthetic baseline",
        "complete synthetic baseline",
    ),
    RunSpec(
        "random_mnli",
        "outputs/stage2_runs/experiment_a/random_mnli/budget_128/seed_42/result.json",
        "MNLI baseline",
        "complete cross-source baseline",
    ),
    RunSpec(
        "uncertainty_mnli",
        "outputs/stage2_runs/experiment_a/uncertainty_mnli/budget_128/seed_42/result.json",
        "MNLI baseline",
        "complete cross-source baseline",
    ),
    RunSpec(
        "failure_driven_no_verification",
        "outputs/stage2_runs/experiment_a/failure_driven_no_verification/budget_128/seed_42/result.json",
        "ablation",
        "complete verification ablation",
    ),
    RunSpec(
        "ours_verified_strict_initial",
        "outputs/stage2_runs_ours_verified/experiment_a/ours_verified/budget_128/seed_42/result.json",
        "verified",
        "abstained; zero-shot fallback",
        rankable=False,
    ),
    RunSpec(
        "ours_verified_target_only_early",
        "outputs/stage2_runs_ours_verified_target_only/experiment_a/ours_verified/budget_128/seed_42/result.json",
        "verified",
        "complete early target-only protocol",
    ),
    RunSpec(
        "ours_verified_v2_ce",
        "outputs/stage2_runs_ours_verified_v2_ce/experiment_a/ours_verified/budget_128/seed_42/result.json",
        "verified",
        "complete objective ablation",
    ),
    RunSpec(
        "ours_verified_v2_pairwise",
        "outputs/stage2_runs_ours_verified_v2_pairwise/experiment_a/ours_verified/budget_128/seed_42/result.json",
        "verified",
        "complete objective ablation",
    ),
    RunSpec(
        "ours_verified_v3_ce",
        "outputs/stage2_runs_ours_verified_v3_ce/experiment_b/ours_verified/budget_128/seed_42/result.json",
        "verified",
        "complete",
    ),
    RunSpec(
        "ours_verified_v5_ce",
        "outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified/budget_128/seed_42/result.json",
        "verified",
        "complete",
    ),
    RunSpec(
        "ours_verified_v6_closed_loop",
        "outputs/preserved_checkpoints/ours_verified_v6_closed_loop/seed_42_budget_128_diagnostic_20261019/result.json",
        "diagnostic",
        "preserved diagnostic; checkpoint continuity failed",
        rankable=False,
    ),
    RunSpec(
        "ours_verified_v7_continuation",
        "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/continuation/result.json",
        "verified",
        "complete stateful closed-loop continuation",
    ),
    RunSpec(
        "ours_verified_v7_consolidation",
        "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/consolidation/result.json",
        "verified",
        "complete fresh one-pass consolidation",
    ),
)

EXPECTED_LITERATURE = (
    "margin_mnli",
    "coreset_mnli",
    "badge_mnli",
    "cal_mnli",
    "margin_synthetic",
    "coreset_synthetic",
    "badge_synthetic",
    "cal_synthetic",
)

GROUP_COLORS = {
    "reference": "#64748B",
    "MNLI baseline": "#2563EB",
    "synthetic baseline": "#D97706",
    "ablation": "#EA580C",
    "verified": "#059669",
    "published synthetic": "#7C3AED",
    "published MNLI": "#DB2777",
    "diagnostic": "#94A3B8",
}

VENUE_ABBREVIATIONS = {
    "Journal of Machine Learning Research (JMLR)": "JMLR",
    "International Conference on Learning Representations (ICLR)": "ICLR",
    "Conference on Empirical Methods in Natural Language Processing (EMNLP)": "EMNLP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/stage2_seed42_overview",
        help="Destination for the overview, CSV, JSON, and SVG figures.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def metric(result: dict[str, Any], *keys: str) -> float:
    value: Any = result
    for key in keys:
        value = value[key]
    return float(value)


def objective_name(result: dict[str, Any]) -> str:
    raw = str(result.get("training_objective", "unknown"))
    if raw == "none" or raw == "none_abstained":
        return "none"
    if "contrastive" in raw or "consistency" in raw:
        return "pairwise"
    return "CE"


def build_rows() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for spec in CORE_SPECS:
        path = PROJECT_ROOT / spec.relative_path
        if not path.is_file():
            continue
        result = read_json(path)
        rows.append(row_from_result(spec, result, path))

    literature_root = PROJECT_ROOT / "outputs" / "stage2_runs_literature" / "experiment_b"
    completed_literature: set[str] = set()
    for method in EXPECTED_LITERATURE:
        path = literature_root / method / "budget_128" / "seed_42" / "result.json"
        if not path.is_file():
            continue
        result = read_json(path)
        if result.get("run_status") != "complete":
            continue
        completed_literature.add(method)
        group = "published synthetic" if method.endswith("_synthetic") else "published MNLI"
        metadata = result.get("literature_baseline", {})
        venue = str(metadata.get("venue", "published baseline"))
        spec = RunSpec(
            method,
            str(path.relative_to(PROJECT_ROOT)),
            group,
            f"complete; {venue}",
        )
        rows.append(row_from_result(spec, result, path))
    pending = [method for method in EXPECTED_LITERATURE if method not in completed_literature]
    return rows, pending


def row_from_result(
    spec: RunSpec, result: dict[str, Any], path: Path
) -> dict[str, Any]:
    metadata = result.get("literature_baseline")
    return {
        "method": spec.label,
        "group": spec.group,
        "objective": objective_name(result),
        "hans_accuracy": metric(result, "hans_final", "overall_accuracy"),
        "hans_entailment": metric(result, "hans_final", "entailment_accuracy"),
        "hans_non_entailment": metric(
            result, "hans_final", "non_entailment_accuracy"
        ),
        "mnli_matched": metric(result, "mnli", "matched", "accuracy"),
        "mnli_mismatched": metric(result, "mnli", "mismatched", "accuracy"),
        "actual_training_examples": int(
            result.get("actual_final_training_examples", 0) or 0
        ),
        "rankable": spec.rankable,
        "status": spec.status,
        "result_path": str(path.relative_to(PROJECT_ROOT)),
        "acquisition_gpu_seconds": (
            None
            if result.get("acquisition_gpu_seconds") is None
            else float(result["acquisition_gpu_seconds"])
        ),
        "paper": metadata.get("paper") if isinstance(metadata, dict) else None,
        "venue": metadata.get("venue") if isinstance(metadata, dict) else None,
        "year": metadata.get("year") if isinstance(metadata, dict) else None,
        "implementation": (
            metadata.get("implementation") if isinstance(metadata, dict) else None
        ),
    }


def svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{html.escape(title)}</title>
<rect width="100%" height="100%" fill="#FFFFFF"/>
<style>
text {{ font-family: Inter, Arial, sans-serif; fill: #334155; }}
.title {{ font-size: 24px; font-weight: 700; fill: #0F172A; }}
.subtitle {{ font-size: 12px; fill: #64748B; }}
.label {{ font-size: 12px; font-weight: 600; }}
.citation {{ font-size: 12px; font-style: italic; font-weight: 500; fill: #475569; }}
.tick {{ font-size: 11px; fill: #64748B; }}
.value {{ font-size: 11px; font-weight: 700; fill: #0F172A; }}
</style>
{chr(10).join(body)}
</svg>
'''


def uses_human_annotated_mnli(row: dict[str, Any]) -> bool:
    """Return whether training examples come from the human-annotated MNLI pool."""
    return row["group"] in {"MNLI baseline", "published MNLI"}


def published_citation(row: dict[str, Any]) -> str | None:
    """Build the compact paper · venue · year label used in Figure 1."""
    paper = row.get("paper")
    venue = row.get("venue")
    year = row.get("year")
    if not paper or not venue or not year:
        return None
    short_venue = VENUE_ABBREVIATIONS.get(str(venue), str(venue))
    return f"{paper} · {short_venue} · {year}"


def write_accuracy_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    rankable = [row for row in rows if row["rankable"]]
    non_mnli = sorted(
        (row for row in rankable if not uses_human_annotated_mnli(row)),
        key=lambda row: row["hans_accuracy"],
    )
    mnli_trained = sorted(
        (row for row in rankable if uses_human_annotated_mnli(row)),
        key=lambda row: row["hans_accuracy"],
    )
    plotted = non_mnli + mnli_trained
    width, row_height = 1560, 48
    separator_height = 30 if mnli_trained else 0
    left, right, top, bottom = 610, 110, 100, 65
    height = top + bottom + row_height * len(plotted) + separator_height
    x_min, x_max = 0.62, 0.68
    chart_width = width - left - right
    body = [
        '<text x="36" y="42" class="title">Stage 2 seed 42 · sealed HANS final accuracy</text>',
        '<text x="36" y="67" class="subtitle">Published selectors include paper · venue · year; MNLI-trained runs grouped below divider; diagnostic and abstained runs excluded</text>',
    ]
    for tick in (0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68):
        x = left + chart_width * (tick - x_min) / (x_max - x_min)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+25}" text-anchor="middle" class="tick">{100*tick:.0f}%</text>',
            ]
        )
    zero = next(row["hans_accuracy"] for row in rows if row["method"] == "zero_shot")
    zero_x = left + chart_width * (zero - x_min) / (x_max - x_min)
    body.append(
        f'<line x1="{zero_x:.1f}" y1="{top-14}" x2="{zero_x:.1f}" y2="{height-bottom}" stroke="#475569" stroke-width="2" stroke-dasharray="5 4"/>'
    )
    if mnli_trained:
        separator_y = top + len(non_mnli) * row_height + 5
        body.extend(
            [
                f'<text x="{left:.1f}" y="{separator_y-7:.1f}" class="subtitle" font-weight="700">Human-annotated MNLI training data</text>',
                f'<line x1="{left:.1f}" y1="{separator_y:.1f}" x2="{width-right:.1f}" y2="{separator_y:.1f}" stroke="#2563EB" stroke-width="2" stroke-dasharray="8 5"/>',
            ]
        )
    for index, row in enumerate(plotted):
        y = top + index * row_height
        if index >= len(non_mnli):
            y += separator_height
        value = row["hans_accuracy"]
        end = left + chart_width * (value - x_min) / (x_max - x_min)
        color = GROUP_COLORS[row["group"]]
        citation = published_citation(row)
        method_y = y + (17 if citation else 27)
        citation_element = (
            f'<text x="{left-14}" y="{y+35}" text-anchor="end" class="citation">({html.escape(citation)})</text>'
            if citation
            else ""
        )
        body.extend(
            [
                f'<text x="{left-14}" y="{method_y}" text-anchor="end" class="label">{html.escape(row["method"])}</text>',
                citation_element,
                f'<rect x="{left}" y="{y+11}" width="{max(2.0, end-left):.1f}" height="20" rx="3" fill="{color}" opacity="0.88"/>',
                f'<text x="{end+8:.1f}" y="{y+26}" class="value">{100*value:.3f}%</text>',
            ]
        )
    _atomic_text_write(
        figures / "figure_1_hans_accuracy.svg",
        svg_document(width, height, "Stage 2 HANS final accuracy", body),
    )


def write_tradeoff_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    plotted = [row for row in rows if row["rankable"]]
    width, height = 1320, 760
    left, right, top, bottom = 105, 445, 125, 90
    chart_width, chart_height = width - left - right, height - top - bottom
    x_min, x_max = 0.57, 0.62
    y_min, y_max = 0.32, 0.43
    body = [
        '<text x="36" y="42" class="title">General NLI retention versus HANS non-entailment</text>',
        '<text x="36" y="67" class="subtitle">x: MNLI matched accuracy · y: HANS non-entailment accuracy · higher is better on both axes</text>',
        '<text x="36" y="87" class="subtitle">Outlined markers were trained on human-annotated MNLI examples; MNLI on the x-axis is an evaluation metric for every method.</text>',
    ]
    for tick in (0.57, 0.58, 0.59, 0.60, 0.61, 0.62):
        x = left + chart_width * (tick - x_min) / (x_max - x_min)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" class="tick">{100*tick:.0f}%</text>',
            ]
        )
    for tick in (0.32, 0.34, 0.36, 0.38, 0.40, 0.42):
        y = top + chart_height * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{100*tick:.0f}%</text>',
            ]
        )
    ordered = sorted(plotted, key=lambda row: row["hans_accuracy"], reverse=True)
    for index, row in enumerate(ordered, start=1):
        x = left + chart_width * (row["mnli_matched"] - x_min) / (x_max - x_min)
        y = top + chart_height * (y_max - row["hans_non_entailment"]) / (y_max - y_min)
        color = GROUP_COLORS[row["group"]]
        source_outline = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="14" fill="none" stroke="#0F172A" stroke-width="2"/>'
            if uses_human_annotated_mnli(row)
            else ""
        )
        legend_outline = (
            f'<circle cx="{width-right+32}" cy="{top+24*(index-1):.1f}" r="9" fill="none" stroke="#0F172A" stroke-width="1.5"/>'
            if uses_human_annotated_mnli(row)
            else ""
        )
        body.extend(
            [
                source_outline,
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" opacity="0.9"/>',
                f'<text x="{x:.1f}" y="{y+4:.1f}" text-anchor="middle" font-size="9" font-weight="700" fill="#FFFFFF">{index}</text>',
                legend_outline,
                f'<circle cx="{width-right+32}" cy="{top+24*(index-1):.1f}" r="6" fill="{color}"/>',
                f'<text x="{width-right+47}" y="{top+4+24*(index-1):.1f}" class="tick">{index}. {html.escape(row["method"])}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-27}" text-anchor="middle" class="label">MNLI matched accuracy</text>',
        ]
    )
    _atomic_text_write(
        figures / "figure_2_nli_hans_tradeoff.svg",
        svg_document(width, height, "MNLI and HANS non-entailment trade-off", body),
    )


def curve_points(row: dict[str, Any]) -> list[dict[str, Any]]:
    result_path = PROJECT_ROOT / row["result_path"]
    if row["method"] == "ours_verified_v7_continuation":
        result = read_json(result_path)
        points = result.get("learning_curve", [])
        return [
            {
                "cumulative_final_training_examples": 32 * int(point["round"]),
                "metrics": point["metrics"],
            }
            for point in points
            if isinstance(point, dict)
        ]
    curve_path = result_path.parent / "posthoc_gain_eval_curve.json"
    if not curve_path.is_file():
        return []
    value = read_json(curve_path)
    points = value.get("points", [])
    return [point for point in points if isinstance(point, dict)]


def write_learning_curve_figure(rows: list[dict[str, Any]], figures: Path) -> None:
    plotted = [
        row
        for row in rows
        if (
            row["group"] in {"published synthetic", "published MNLI"}
            or row["method"]
            in {
                "failure_driven_no_verification",
                "ours_verified_v7_continuation",
            }
        )
        and curve_points(row)
    ]
    if not plotted:
        return
    curve_colors = {
        "badge_synthetic": "#0D9488",
        "cal_synthetic": "#E11D48",
        "coreset_synthetic": "#65A30D",
        "margin_synthetic": "#9333EA",
        "failure_driven_no_verification": "#EA580C",
        "ours_verified_v7_continuation": "#2563EB",
    }
    width, height = 1320, 680
    left, right, top, bottom = 95, 490, 115, 100
    chart_width, chart_height = width - left - right, height - top - bottom
    curve_accuracies = [
        float(point["metrics"]["overall_accuracy"])
        for row in plotted
        for point in curve_points(row)
    ]
    tick_step = 0.005
    y_min = tick_step * math.floor((min(curve_accuracies) - 0.002) / tick_step)
    y_max = tick_step * math.ceil((max(curve_accuracies) + 0.002) / tick_step)
    body = [
        '<text x="36" y="42" class="title">Active-learning curves · selectors, no-verification ablation, and verified v7</text>',
        '<text x="36" y="67" class="subtitle">Solid: post-hoc gain-eval (published baselines + one-shot ablation) · dashed: verified v7 on 900-example stopping-eval</text>',
        '<text x="36" y="87" class="subtitle">Different evaluation subsets: compare trends, not absolute vertical position · sealed final-test is never used for selection</text>',
    ]
    tick_count = int(round((y_max - y_min) / tick_step))
    for tick_index in range(tick_count + 1):
        tick = y_min + tick_step * tick_index
        y = top + chart_height * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{100*tick:.1f}%</text>',
            ]
        )
    for examples in (0, 32, 64, 96, 128):
        x = left + chart_width * examples / 128
        body.append(
            f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" class="tick">{examples}</text>'
        )
    for index, row in enumerate(sorted(plotted, key=lambda value: value["method"])):
        points = curve_points(row)
        coordinates: list[tuple[float, float]] = []
        for point in points:
            examples = float(point["cumulative_final_training_examples"])
            accuracy = float(point["metrics"]["overall_accuracy"])
            x = left + chart_width * examples / 128
            y = top + chart_height * (y_max - accuracy) / (y_max - y_min)
            coordinates.append((x, y))
        color = curve_colors.get(row["method"], GROUP_COLORS[row["group"]])
        is_v7 = row["method"] == "ours_verified_v7_continuation"
        if is_v7:
            dash = ' stroke-dasharray="10 6"'
        elif row["group"] == "published MNLI":
            dash = ' stroke-dasharray="3 4"'
        else:
            dash = ""
        stroke_width = 4 if is_v7 else 3
        body.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)}" fill="none" stroke="{color}" stroke-width="{stroke_width}"{dash}/>'
        )
        marker_outline = (
            ' stroke="#0F172A" stroke-width="1"' if is_v7 else ""
        )
        for x, y in coordinates:
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5 if is_v7 else 4}" '
                f'fill="{color}"{marker_outline}/>'
            )
        legend_y = top + index * 29
        if is_v7:
            legend_label = "ours_verified_v7_continuation (stopping-eval*)"
        elif row["method"] == "failure_driven_no_verification":
            legend_label = "failure_driven_no_verification (gain-eval; one-shot)"
        else:
            legend_label = row["method"]
        body.extend(
            [
                f'<line x1="{width-right+30}" y1="{legend_y}" x2="{width-right+54}" y2="{legend_y}" stroke="{color}" stroke-width="4"{dash}/>',
                f'<text x="{width-right+64}" y="{legend_y+4}" class="label">{html.escape(legend_label)}</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-45}" text-anchor="middle" class="label">Cumulative training examples</text>',
            f'<text x="{left}" y="{height-17}" class="tick">* v7 values use its predeclared stopping subset and are not directly comparable to baseline gain-eval values.</text>',
        ]
    )
    _atomic_text_write(
        figures / "figure_3_literature_learning_curves.svg",
        svg_document(width, height, "Published active-learning learning curves", body),
    )


def verification_stopping_evidence() -> dict[str, Any]:
    result_path = (
        PROJECT_ROOT
        / "outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified"
        / "budget_128/seed_42/result.json"
    )
    result = read_json(result_path)
    summary = result["synthetic_provenance"]["verification_summary"]
    results: list[dict[str, Any]] = []
    for item in summary["results"]:
        probe_design = item["probe_design"]
        stop_reason = str(probe_design["stop_reason"])
        if stop_reason == "hoeffding_lcb_certified":
            outcome = "certified"
        elif stop_reason == "hoeffding_ucb_below_threshold":
            outcome = "rejected"
        else:
            outcome = "unresolved"
        results.append(
            {
                "hypothesis": str(item["hypothesis_name"]),
                "used": int(probe_design["attempted_probe_sets"]),
                "maximum": int(max(probe_design["look_schedule"])),
                "outcome": outcome,
                "stop_reason": stop_reason,
            }
        )
    actual = sum(item["used"] for item in results)
    maximum = sum(item["maximum"] for item in results)
    return {
        "results": results,
        "actual_probe_sets": actual,
        "maximum_probe_sets": maximum,
        "avoided_probe_sets": maximum - actual,
        "avoided_fraction": (maximum - actual) / maximum,
    }


def write_verification_stopping_figure(
    evidence: dict[str, Any], figures: Path
) -> None:
    results = evidence["results"]
    width, row_height = 1250, 64
    left, right, top, bottom = 430, 230, 115, 100
    height = top + bottom + row_height * len(results)
    chart_width = width - left - right
    maximum = max(item["maximum"] for item in results)
    colors = {
        "certified": "#059669",
        "rejected": "#DC2626",
        "unresolved": "#D97706",
    }
    body = [
        '<text x="36" y="42" class="title">Adaptive Hoeffding verification stopping</text>',
        '<text x="36" y="68" class="subtitle">Colored bars: probe sets actually generated · gray background: fixed 60-set maximum per hypothesis</text>',
    ]
    for tick in range(0, maximum + 1, 10):
        x = left + chart_width * tick / maximum
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+26}" text-anchor="middle" class="tick">{tick}</text>',
            ]
        )
    for index, item in enumerate(results):
        y = top + index * row_height
        used_width = chart_width * item["used"] / maximum
        color = colors[item["outcome"]]
        body.extend(
            [
                f'<text x="{left-16}" y="{y+22}" text-anchor="end" class="label">{html.escape(item["hypothesis"])}</text>',
                f'<rect x="{left}" y="{y+7}" width="{chart_width:.1f}" height="22" rx="4" fill="#E2E8F0"/>',
                f'<rect x="{left}" y="{y+7}" width="{used_width:.1f}" height="22" rx="4" fill="{color}" opacity="0.9"/>',
                f'<text x="{left+used_width+9:.1f}" y="{y+23}" class="value">{item["used"]}/{item["maximum"]} · {item["outcome"]}</text>',
            ]
        )
    saved = 100.0 * evidence["avoided_fraction"]
    body.append(
        f'<text x="{left}" y="{height-25}" class="label">Total: {evidence["actual_probe_sets"]}/{evidence["maximum_probe_sets"]} probe sets; {evidence["avoided_probe_sets"]} avoided ({saved:.1f}%) versus always using the maximum.</text>'
    )
    _atomic_text_write(
        figures / "figure_4_adaptive_verification_stopping.svg",
        svg_document(width, height, "Adaptive Hoeffding verification stopping", body),
    )


def controlled_cost_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = ["ours_verified_v5_ce"] + sorted(
        row["method"] for row in rows if row["group"] == "published synthetic"
    )
    by_method = {row["method"]: row for row in rows}
    cost_rows: list[dict[str, Any]] = []
    for method in methods:
        row = by_method[method]
        result = read_json(PROJECT_ROOT / row["result_path"])
        assumed = result.get("assumed_cost")
        cost_rows.append(
            {
                "method": method,
                "acquisition_gpu_seconds": float(result["acquisition_gpu_seconds"]),
                "training_gpu_seconds": float(result["training_gpu_seconds"]),
                "evaluation_gpu_seconds": float(result["evaluation_gpu_seconds"]),
                "modeled_total_usd": (
                    None
                    if not isinstance(assumed, dict) or assumed.get("total") is None
                    else float(assumed["total"])
                ),
            }
        )
    return cost_rows


def write_acquisition_cost_figure(
    cost_rows: list[dict[str, Any]], figures: Path
) -> None:
    ordered = sorted(cost_rows, key=lambda row: row["acquisition_gpu_seconds"])
    width, row_height = 1160, 58
    left, right, top, bottom = 310, 120, 115, 95
    height = top + bottom + row_height * len(ordered)
    chart_width = width - left - right
    maximum = max(row["acquisition_gpu_seconds"] for row in ordered)
    axis_max = 10.0 * math.ceil(maximum / 10.0)
    body = [
        '<text x="36" y="42" class="title">Controlled synthetic-pool acquisition compute</text>',
        '<text x="36" y="68" class="subtitle">GPU time for acquisition only · excludes generation, verification, training, and evaluation</text>',
    ]
    for tick in range(0, int(axis_max) + 1, 10):
        x = left + chart_width * tick / axis_max
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+26}" text-anchor="middle" class="tick">{tick}s</text>',
            ]
        )
    for index, row in enumerate(ordered):
        y = top + index * row_height
        bar_width = chart_width * row["acquisition_gpu_seconds"] / axis_max
        color = "#059669" if row["method"] == "ours_verified_v5_ce" else "#7C3AED"
        body.extend(
            [
                f'<text x="{left-15}" y="{y+21}" text-anchor="end" class="label">{html.escape(row["method"])}</text>',
                f'<rect x="{left}" y="{y+7}" width="{bar_width:.1f}" height="21" rx="4" fill="{color}" opacity="0.9"/>',
                f'<text x="{left+bar_width+9:.1f}" y="{y+22}" class="value">{row["acquisition_gpu_seconds"]:.1f} GPU-s</text>',
            ]
        )
    body.append(
        f'<text x="{left}" y="{height-24}" class="label">This component alone does not represent end-to-end cost.</text>'
    )
    _atomic_text_write(
        figures / "figure_5_controlled_acquisition_cost.svg",
        svg_document(width, height, "Controlled acquisition GPU time", body),
    )


def percentage(value: float) -> str:
    return f"{100.0 * value:.4f}%"


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    fields = (
        "method",
        "group",
        "objective",
        "hans_accuracy",
        "hans_entailment",
        "hans_non_entailment",
        "mnli_matched",
        "mnli_mismatched",
        "actual_training_examples",
        "rankable",
        "status",
        "paper",
        "venue",
        "year",
        "implementation",
        "result_path",
    )
    path = output / "stage2_overview.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_report(
    rows: list[dict[str, Any]],
    pending: list[str],
    stopping: dict[str, Any],
    cost_rows: list[dict[str, Any]],
    output: Path,
) -> None:
    ranked = sorted(
        (row for row in rows if row["rankable"]),
        key=lambda row: row["hans_accuracy"],
        reverse=True,
    )
    by_method = {row["method"]: row for row in rows}
    best = ranked[0]
    v7 = by_method["ours_verified_v7_continuation"]
    random_synthetic = by_method["random_synthetic"]
    published_synthetic = sorted(
        (row for row in rows if row["group"] == "published synthetic"),
        key=lambda row: row["hans_accuracy"],
        reverse=True,
    )
    lines = [
        "# Stage 2 comprehensive overview — seed 42, budget 128",
        "",
        "This is the single consolidated report for discussion with the supervisor and for preparing the experimental section. It combines the original baselines, verified-flow versions v2--v7, ablations, and every completed published active-learning baseline. All reported final HANS metrics use the same frozen 15,000-example final-test manifest.",
        "",
        "> **Scope.** These are development results from one seed. They support debugging and experiment selection, not a paper-level significance or state-of-the-art claim. Multi-seed confirmation remains required.",
        "",
        "## Executive summary",
        "",
        f"- The strongest completed result is `{best['method']}` at **{percentage(best['hans_accuracy'])}** HANS accuracy. It is a {best['group'].lower()} and therefore is not an equal-data-access comparison with the verified synthetic flow.",
        f"- The strongest valid verified result is `ours_verified_v7_continuation` at **{percentage(v7['hans_accuracy'])}**.",
    ]
    if published_synthetic:
        top_published = published_synthetic[0]
        difference = 100.0 * (top_published["hans_accuracy"] - v7["hans_accuracy"])
        additional = round(
            15000 * (top_published["hans_accuracy"] - v7["hans_accuracy"])
        )
        lines.extend(
            [
                f"- On the controlled static verified-v5 pool, `{top_published['method']}` is numerically best at **{percentage(top_published['hans_accuracy'])}**, only {difference:+.4f} percentage points ({additional:+d} predictions out of 15,000) relative to v7 continuation. This is an effective tie at one seed.",
                f"- Every completed published synthetic selector improves over `random_synthetic` ({percentage(random_synthetic['hans_accuracy'])}), showing that acquisition matters on the verified pool.",
            ]
        )
    if pending:
        lines.append(
            "- The following published MNLI runs are still pending and are not estimated from partial checkpoints: "
            + ", ".join(f"`{method}`" for method in pending)
            + "."
        )
    lines.extend(
        [
            "",
            "## Paper-ready figures",
            "",
            "> **Training-data source note.** Methods named `*_mnli` train on human-annotated MNLI examples and are cross-source baselines, not equal-data-access comparisons with synthetic-data methods. Figure 1 places them below a labeled divider; Figure 2 uses outlined markers. MNLI accuracy on Figure 2's x-axis is an evaluation metric for every method and does not imply MNLI training.",
            "",
            "![Sealed HANS final accuracy](figures/figure_1_hans_accuracy.svg)",
            "",
            "**Figure 1.** Sealed HANS final accuracy. Published selectors are annotated with paper title, venue, and publication year. Runs trained on human-annotated MNLI examples are grouped below the labeled blue divider. The horizontal axis is explicitly zoomed to make small differences visible; abstained and invalid diagnostic runs are excluded.",
            "",
            "![MNLI–HANS trade-off](figures/figure_2_nli_hans_tradeoff.svg)",
            "",
            "**Figure 2.** MNLI matched accuracy versus HANS non-entailment accuracy. Outlined markers identify runs trained on human-annotated MNLI examples; all other markers use no MNLI training data. This exposes whether a method improves robustness while preserving general NLI performance.",
            "",
            "![Published baseline learning curves](figures/figure_3_literature_learning_curves.svg)",
            "",
            "**Figure 3.** Learning curves for the completed published synthetic acquisition baselines, the `failure_driven_no_verification` ablation, and `ours_verified_v7_continuation`. Solid curves use the same 3,750-example post-hoc gain-eval set. The no-verification ablation has only endpoints because it trains once on all 128 examples. The dashed v7 curve uses its predeclared 900-example stopping-eval subset; therefore compare its trend, not its absolute vertical position, with the solid curves. All plotted runs use synthetic training data; the human-annotated MNLI variants are pending. The sealed final-test is evaluated only after training and is never used for acquisition.",
            "",
            "![Adaptive verification stopping](figures/figure_4_adaptive_verification_stopping.svg)",
            "",
            "**Figure 4.** Probe sets used by adaptive Hoeffding verification compared with a fixed 60-set maximum for every hypothesis. This figure measures verification-stage savings, not total active-learning budget savings.",
            "",
            "![Controlled acquisition compute](figures/figure_5_controlled_acquisition_cost.svg)",
            "",
            "**Figure 5.** Acquisition-only GPU time on the controlled synthetic setting. It excludes teacher generation, verification, training, and evaluation and therefore must not be presented as total cost.",
            "",
            "All figures are SVG vector graphics and can be converted to PDF during paper typesetting without rasterization.",
            "",
            "## Complete result table",
            "",
            "| Rank | Method | Group | Objective | Train n | HANS | HANS non-ent. | MNLI m/mm | Status |",
            "|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    ranks = {row["method"]: index for index, row in enumerate(ranked, start=1)}
    for row in sorted(rows, key=lambda value: value["hans_accuracy"], reverse=True):
        rank = str(ranks[row["method"]]) if row["rankable"] else "—"
        lines.append(
            f"| {rank} | `{row['method']}` | {row['group']} | {row['objective']} | "
            f"{row['actual_training_examples']} | {percentage(row['hans_accuracy'])} | "
            f"{percentage(row['hans_non_entailment'])} | "
            f"{percentage(row['mnli_matched'])} / {percentage(row['mnli_mismatched'])} | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Controlled published acquisition baselines",
            "",
            "The literature methods are not new selectors from this project. They are protocol-compatible reimplementations of published acquisition rules, run with the shared Qwen3-1.7B/LoRA trainer and sealed evaluator. Synthetic variants use the immutable verified-v5 candidate pool; MNLI variants use the original MNLI train pool.",
            "",
            "| Method | Publication | Venue | Year | HANS | Acquisition GPU time | Local implementation |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    literature_rows = sorted(
        (row for row in rows if row["paper"] is not None),
        key=lambda row: row["method"],
    )
    for row in literature_rows:
        cost = row["acquisition_gpu_seconds"]
        cost_text = "n/a" if cost is None else f"{cost:.1f} s"
        lines.append(
            f"| `{row['method']}` | {row['paper']} | {row['venue']} | {row['year']} | "
            f"{percentage(row['hans_accuracy'])} | {cost_text} | `{row['implementation']}` |"
        )
    lines.extend(
        [
            "",
            "The four completed synthetic methods share the same 32-example random warm start; rounds 2--4 use the named published selector. LoRA weights and AdamW state continue across all rounds. Core-set and CAL mainly improve non-entailment accuracy while slightly reducing entailment accuracy relative to v7 continuation.",
            "",
            "This static-pool comparison isolates acquisition behavior but does **not** reproduce v7's dynamic candidate trace. The v7 closed loop may retire or add hypotheses and generate fresh candidates after refresh; the published selectors cannot.",
            "",
            "## Fairness and interpretation",
            "",
            "- `random_mnli`, `uncertainty_mnli`, and published `*_mnli` methods have access to original human-written MNLI candidates. They are cross-source baselines relative to LLM-synthetic methods.",
            "- Published `*_synthetic` methods and `random_synthetic` operate on synthetic candidates. The completed published methods use the same static verified-v5 pool, permitting a controlled selector comparison among themselves.",
            "- `failure_driven_no_verification` is an ablation of the proposed verification component, not an external published method.",
            "- `ours_verified_v7_continuation` is the valid online closed-loop result. `ours_verified_v7_consolidation` retrains once from the base model on the same frozen 128 selected examples.",
            "- `ours_verified_v6_closed_loop` is preserved for auditability but checkpoint continuity failed; it is excluded from ranking. The strict initial run abstained and is also excluded.",
            "- The teacher-token cost stored in each static-pool result is inherited provenance. Because the same pool is reused, that generation cost must not be summed once per selector.",
            "",
            "## Budget and stopping evidence",
            "",
            "### What is already demonstrated",
            "",
            f"Adaptive Hoeffding verification used **{stopping['actual_probe_sets']}/{stopping['maximum_probe_sets']} probe sets**. It therefore avoided **{stopping['avoided_probe_sets']} sets ({100.0 * stopping['avoided_fraction']:.1f}%)** relative to always generating the fixed maximum of 60 sets for each of five hypotheses.",
            "",
            "| Hypothesis | Used / maximum | Decision |",
            "|---|---:|---|",
        ]
    )
    for item in stopping["results"]:
        lines.append(
            f"| {item['hypothesis']} | {item['used']} / {item['maximum']} | "
            f"{item['outcome']} (`{item['stop_reason']}`) |"
        )
    lines.extend(
        [
            "",
            "This supports the narrow claim that confidence-bound verification can stop collecting probes once a hypothesis is statistically certified or rejected. It does not by itself prove that the final decision is optimal; correctness still requires repeated seeds and held-out validation.",
            "",
            "### What is not yet demonstrated",
            "",
            "All valid v3, v5, v7, and published-selector runs trained on the full **128-example** budget. Therefore the current experiment does **not** demonstrate that the complete method uses fewer training examples than its baselines or knows the optimal point at which to stop the whole active-learning loop.",
            "",
            "The v5 stopping rule reached the maximum budget. Its final look had negative mean gain and counted as only the first small-gain look, while the predeclared patience was two consecutive looks. Thus stopping at 128 would be a maximum-budget termination, not an adaptive early-stop success.",
            "",
            "The modeled v5 cost is lower than the current published synthetic runs partly because v5 evaluates stopping on a reused 900-example subset whereas those runs repeatedly evaluate all 3,750 gain-eval examples. That protocol difference prevents using the current cost totals as evidence that the proposed algorithm itself is cheaper.",
            "",
            "To establish an end-to-end stopping claim, predeclare a larger maximum budget (for example 256), allow at least two additional looks after 128, compare adaptive and fixed-budget arms under identical evaluation frequency, and evaluate the sealed final-test only after stopping. Report stopped budget, accuracy regret relative to the best fixed checkpoint, GPU-hours, teacher tokens, and stopping frequency across seeds.",
            "",
            "## Cost comparison currently supported",
            "",
            "The only reasonably controlled compute comparison currently available is acquisition GPU time on the static verified-synthetic setting:",
            "",
            "| Method | Acquisition GPU-s | Training GPU-s | Evaluation GPU-s | Stored modeled total |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(cost_rows, key=lambda value: value["acquisition_gpu_seconds"]):
        total = item["modeled_total_usd"]
        total_text = "n/a" if total is None else f"${total:.3f}"
        lines.append(
            f"| `{item['method']}` | {item['acquisition_gpu_seconds']:.1f} | "
            f"{item['training_gpu_seconds']:.1f} | {item['evaluation_gpu_seconds']:.1f} | "
            f"{total_text} |"
        )
    lines.extend(
        [
            "",
            "On acquisition compute, v5 is cheaper than CAL, approximately tied with BADGE, and more expensive than Margin and Core-set. Training cost is effectively the same for all methods because the backbone, LoRA recipe, budget, and optimizer-step count are controlled.",
            "",
            "The stored modeled totals are shown for auditability but are **not directly comparable**: v5 uses the 900-example stopping subset and frozen base predictions, whereas the published runs repeatedly evaluate the 3,750-example gain-eval set; published synthetic runs also reuse the already generated v5 pool. Accordingly, the current artifacts do not establish an end-to-end cost advantage over other methods.",
            "",
            "There is also no cross-model cost evidence. Every row in the published acquisition comparison uses the same Qwen3-1.7B backbone. No RoBERTa, DeBERTa, BERT, Llama, or other-backbone run has been performed under this protocol, so the report must not claim lower cost than other models.",
            "",
            "A valid end-to-end cost comparison requires identical evaluation schedules and separate reporting of: one-time pool construction, verification teacher tokens, acquisition GPU time, training GPU time, evaluation GPU time, peak VRAM, wall time, and amortized versus non-amortized cost. A cross-model claim additionally requires running the same method/budget grid on each backbone.",
            "",
            "## Recommended statement for the paper",
            "",
            "A defensible current statement is: *Under a controlled static verified-synthetic pool, published acquisition functions and the v7 verified closed loop achieve approximately 65% HANS accuracy at seed 42; Core-set, CAL, and v7 are effectively tied. The verified method provides an auditable hypothesis-certification and closed-loop data-generation mechanism, but the present single-seed experiment does not establish an accuracy advantage over standard acquisition.*",
            "",
            "Do not currently claim that v7 is uniformly best. The full-pool published MNLI runs and the remaining predeclared seeds are required before finalizing the experimental claim.",
            "",
            "## Reproduction and source artifacts",
            "",
            "Machine-readable table: `stage2_overview.csv`",
            "",
            "| Method | Source result |",
            "|---|---|",
        ]
    )
    for row in sorted(rows, key=lambda value: value["method"]):
        lines.append(f"| `{row['method']}` | `{row['result_path']}` |")
    lines.extend(
        [
            "",
            "Regenerate this report after new runs complete:",
            "",
            "```bash",
            "python scripts/report_stage2_overview.py",
            "```",
            "",
        ]
    )
    _atomic_text_write(output / "STAGE2_OVERVIEW.md", "\n".join(lines))


def main() -> None:
    args = parse_args()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, pending = build_rows()
    stopping = verification_stopping_evidence()
    cost_rows = controlled_cost_evidence(rows)
    write_csv(rows, output)
    _atomic_text_write(
        output / "stage2_overview.json",
        json.dumps(
            {
                "rows": rows,
                "pending_literature": pending,
                "verification_stopping": stopping,
                "controlled_cost_comparison": cost_rows,
            },
            indent=2,
        )
        + "\n",
    )
    write_accuracy_figure(rows, figures)
    write_tradeoff_figure(rows, figures)
    write_learning_curve_figure(rows, figures)
    write_verification_stopping_figure(stopping, figures)
    write_acquisition_cost_figure(cost_rows, figures)
    write_report(rows, pending, stopping, cost_rows, output)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pending_literature": pending,
                "report": str(output / "STAGE2_OVERVIEW.md"),
                "figures": 5,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
