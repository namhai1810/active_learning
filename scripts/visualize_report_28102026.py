#!/usr/bin/env python3
"""Create self-contained SVG figures for reports/stage2_seed42_overview/28102026.md."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "human": "#2563EB",
    "static": "#7C3AED",
    "dynamic": "#059669",
    "ablation": "#EA580C",
    "reference": "#64748B",
    "margin": "#2563EB",
    "badge": "#7C3AED",
    "coreset": "#0D9488",
    "cal": "#E11D48",
    "uncertainty": "#9333EA",
    "random": "#64748B",
}


RESULT_PATHS = {
    "uncertainty_mnli": "outputs/stage2_runs/experiment_a/uncertainty_mnli/budget_128/seed_42/result.json",
    "random_mnli": "outputs/stage2_runs/experiment_a/random_mnli/budget_128/seed_42/result.json",
    "failure_driven_no_verification": "outputs/stage2_runs/experiment_a/failure_driven_no_verification/budget_128/seed_42/result.json",
    "random_synthetic": "outputs/stage2_runs/experiment_a/random_synthetic/budget_128/seed_42/result.json",
    "zero_shot": "outputs/stage2_runs/experiment_a/zero_shot/budget_128/seed_42/result.json",
    "margin_mnli": "outputs/stage2_runs_literature/experiment_b/margin_mnli/budget_128/seed_42/result.json",
    "coreset_mnli": "outputs/stage2_runs_literature/experiment_b/coreset_mnli/budget_128/seed_42/result.json",
    "badge_mnli": "outputs/stage2_runs_literature/experiment_b/badge_mnli/budget_128/seed_42/result.json",
    "cal_mnli": "outputs/stage2_runs_literature/experiment_b/cal_mnli/budget_128/seed_42/result.json",
    "margin_synthetic": "outputs/stage2_runs_literature/experiment_b/margin_synthetic/budget_128/seed_42/result.json",
    "coreset_synthetic": "outputs/stage2_runs_literature/experiment_b/coreset_synthetic/budget_128/seed_42/result.json",
    "badge_synthetic": "outputs/stage2_runs_literature/experiment_b/badge_synthetic/budget_128/seed_42/result.json",
    "cal_synthetic": "outputs/stage2_runs_literature/experiment_b/cal_synthetic/budget_128/seed_42/result.json",
    "ours_verified_v2_ce": "outputs/stage2_runs_ours_verified_v2_ce/experiment_a/ours_verified/budget_128/seed_42/result.json",
    "ours_verified_v3_ce": "outputs/stage2_runs_ours_verified_v3_ce/experiment_b/ours_verified/budget_128/seed_42/result.json",
    "ours_verified_v5_ce": "outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified/budget_128/seed_42/result.json",
    "ours_verified_v7_continuation": "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/continuation/result.json",
    "ours_verified_v7_consolidation": "outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/consolidation/result.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/stage2_seed42_overview/figures_28102026",
    )
    return parser.parse_args()


def read_json(relative_path: str) -> dict[str, Any]:
    with (PROJECT_ROOT / relative_path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {relative_path}")
    return value


def load_results() -> dict[str, dict[str, Any]]:
    return {method: read_json(path) for method, path in RESULT_PATHS.items()}


def accuracy(results: dict[str, dict[str, Any]], method: str) -> float:
    return 100.0 * float(results[method]["hans_final"]["overall_accuracy"])


def svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{html.escape(title)}</title>
<rect width="100%" height="100%" fill="#FFFFFF" rx="18"/>
<style>
text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill: #334155; }}
.title {{ font-size: 24px; font-weight: 750; fill: #0F172A; }}
.subtitle {{ font-size: 13px; fill: #64748B; }}
.section {{ font-size: 14px; font-weight: 750; fill: #0F172A; }}
.label {{ font-size: 13px; font-weight: 650; }}
.tick {{ font-size: 11px; fill: #64748B; }}
.value {{ font-size: 12px; font-weight: 750; fill: #0F172A; }}
.note {{ font-size: 11px; fill: #475569; }}
</style>
{chr(10).join(body)}
</svg>
'''


def write_svg(output: Path, width: int, height: int, title: str, body: list[str]) -> None:
    output.write_text(svg_document(width, height, title, body), encoding="utf-8")


def write_hans_by_source(results: dict[str, dict[str, Any]], output: Path) -> None:
    groups = [
        (
            "Human-labelled MNLI acquisition (cross-source reference)",
            "human",
            [
                "uncertainty_mnli",
                "margin_mnli",
                "badge_mnli",
                "random_mnli",
                "coreset_mnli",
                "cal_mnli",
            ],
        ),
        (
            "Static verified-synthetic pool (controlled among these selectors)",
            "static",
            [
                "coreset_synthetic",
                "cal_synthetic",
                "margin_synthetic",
                "badge_synthetic",
                "random_synthetic",
            ],
        ),
        (
            "Dynamic synthetic flows and ablation (different candidate traces)",
            "dynamic",
            [
                "failure_driven_no_verification",
                "ours_verified_v7_continuation",
                "ours_verified_v5_ce",
                "ours_verified_v3_ce",
                "zero_shot",
            ],
        ),
    ]
    width, height = 1440, 1010
    left, right, top, bottom = 430, 95, 118, 65
    x_min, x_max = 63.5, 67.2
    chart_width = width - left - right
    body = [
        '<text x="38" y="43" class="title">Sealed HANS accuracy — separate the data sources before comparing</text>',
        '<text x="38" y="69" class="subtitle">Seed 42 · budget 128 · bars share the same final-test, but groups do not share the same acquisition pool</text>',
    ]
    for tick in (63.5, 64.0, 64.5, 65.0, 65.5, 66.0, 66.5, 67.0):
        x = left + chart_width * (tick - x_min) / (x_max - x_min)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+26}" text-anchor="middle" class="tick">{tick:.1f}%</text>',
            ]
        )
    v7_x = left + chart_width * (65.0 - x_min) / (x_max - x_min)
    body.append(
        f'<line x1="{v7_x:.1f}" y1="{top-10}" x2="{v7_x:.1f}" y2="{height-bottom}" stroke="#059669" stroke-width="2" stroke-dasharray="7 5"/>'
    )
    body.append(
        f'<text x="{v7_x+7:.1f}" y="{top+4}" class="note" fill="#047857">v7 = 65.000%</text>'
    )
    y = top + 26
    for group_index, (group_title, group_key, methods) in enumerate(groups):
        if group_index:
            y += 24
        body.append(
            f'<rect x="36" y="{y-22}" width="{width-72}" height="31" rx="8" fill="{COLORS[group_key]}" opacity="0.10"/>'
        )
        body.append(
            f'<text x="48" y="{y}" class="section">{html.escape(group_title)}</text>'
        )
        y += 25
        for method in methods:
            value = accuracy(results, method)
            end = left + chart_width * (value - x_min) / (x_max - x_min)
            color_key = "ablation" if method == "failure_driven_no_verification" else group_key
            body.extend(
                [
                    f'<text x="{left-14}" y="{y+18}" text-anchor="end" class="label">{html.escape(method)}</text>',
                    f'<rect x="{left}" y="{y+5}" width="{max(2.0, end-left):.1f}" height="19" rx="4" fill="{COLORS[color_key]}" opacity="0.88"/>',
                    f'<text x="{end+8:.1f}" y="{y+19}" class="value">{value:.3f}%</text>',
                ]
            )
            y += 38
    body.append(
        f'<text x="38" y="{height-18}" class="note">Read within a colored section for the cleanest comparison; cross-section gaps mix data-source and pipeline effects.</text>'
    )
    write_svg(output, width, height, "HANS accuracy separated by acquisition source", body)


def learning_curve(result: dict[str, Any], method: str) -> list[tuple[int, float]]:
    path = (
        PROJECT_ROOT
        / "outputs/stage2_runs_literature/experiment_b"
        / method
        / "budget_128/seed_42/posthoc_gain_eval_curve.json"
    )
    del result
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return [
        (
            int(point["cumulative_final_training_examples"]),
            100.0 * float(point["metrics"]["overall_accuracy"]),
        )
        for point in value["points"]
    ]


def write_mnli_learning_curves(
    results: dict[str, dict[str, Any]], output: Path
) -> None:
    methods = ["margin_mnli", "badge_mnli", "coreset_mnli", "cal_mnli"]
    width, height = 1240, 680
    left, right, top, bottom = 100, 255, 105, 90
    chart_width, chart_height = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 128.0
    y_min, y_max = 62.4, 65.1
    body = [
        '<text x="36" y="42" class="title">MNLI literature baselines — post-hoc gain-eval learning curves</text>',
        '<text x="36" y="68" class="subtitle">Same base and same random 32-example warm start; named selectors begin after budget 32</text>',
    ]
    for tick in (62.5, 63.0, 63.5, 64.0, 64.5, 65.0):
        y = top + chart_height * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}%</text>',
            ]
        )
    for tick in (0, 32, 64, 96, 128):
        x = left + chart_width * (tick - x_min) / (x_max - x_min)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#F1F5F9"/>',
                f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" class="tick">{tick}</text>',
            ]
        )
    for index, method in enumerate(methods):
        points = learning_curve(results[method], method)
        coordinates = [
            (
                left + chart_width * (examples - x_min) / (x_max - x_min),
                top + chart_height * (y_max - value) / (y_max - y_min),
            )
            for examples, value in points
        ]
        selector = method.split("_")[0]
        color = COLORS[selector]
        body.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)}" fill="none" stroke="{color}" stroke-width="4"/>'
        )
        for x, y in coordinates:
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>'
            )
        legend_y = top + 34 * index
        body.extend(
            [
                f'<line x1="{width-right+35}" y1="{legend_y}" x2="{width-right+65}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>',
                f'<text x="{width-right+75}" y="{legend_y+4}" class="label">{html.escape(method)}</text>',
                f'<text x="{width-right+75}" y="{legend_y+20}" class="note">final {points[-1][1]:.3f}%</text>',
            ]
        )
    body.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-32}" text-anchor="middle" class="label">Cumulative training examples</text>',
            f'<text x="36" y="{height-15}" class="note">This is the shared 3,750-example gain-eval view, not sealed HANS final-test.</text>',
        ]
    )
    write_svg(output, width, height, "MNLI selector learning curves", body)


def write_verified_evolution(results: dict[str, dict[str, Any]], output: Path) -> None:
    main = [
        ("Zero-shot", "zero_shot"),
        ("v2 CE", "ours_verified_v2_ce"),
        ("v3 CE", "ours_verified_v3_ce"),
        ("v5 CE", "ours_verified_v5_ce"),
        ("v7 cont.", "ours_verified_v7_continuation"),
    ]
    width, height = 1120, 610
    # Keep room for the two v7 comparison-branch labels.
    left, right, top, bottom = 105, 230, 105, 105
    chart_width, chart_height = width - left - right, height - top - bottom
    y_min, y_max = 63.6, 65.7
    body = [
        '<text x="36" y="42" class="title">Verified-flow evolution — small but consistent gain through v7</text>',
        '<text x="36" y="68" class="subtitle">Main line follows the valid CE progression; consolidation and no-verification are comparison branches</text>',
    ]
    for tick in (63.8, 64.0, 64.4, 64.8, 65.2, 65.6):
        y = top + chart_height * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}%</text>',
            ]
        )
    coordinates: list[tuple[float, float]] = []
    for index, (label, method) in enumerate(main):
        x = left + chart_width * index / (len(main) - 1)
        value = accuracy(results, method)
        y = top + chart_height * (y_max - value) / (y_max - y_min)
        coordinates.append((x, y))
        body.extend(
            [
                f'<text x="{x:.1f}" y="{height-bottom+29}" text-anchor="middle" class="label">{html.escape(label)}</text>',
                f'<text x="{x:.1f}" y="{height-bottom+47}" text-anchor="middle" class="note">{value:.3f}%</text>',
            ]
        )
    body.append(
        f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)}" fill="none" stroke="{COLORS["dynamic"]}" stroke-width="5"/>'
    )
    for x, y in coordinates:
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#FFFFFF" stroke="{COLORS["dynamic"]}" stroke-width="4"/>'
        )
    v7_x, _ = coordinates[-1]
    for label, method, offset, color in (
        ("v7 consolidation", "ours_verified_v7_consolidation", -95, COLORS["static"]),
        ("no verification", "failure_driven_no_verification", 95, COLORS["ablation"]),
    ):
        value = accuracy(results, method)
        y = top + chart_height * (y_max - value) / (y_max - y_min)
        x = v7_x + offset
        body.extend(
            [
                f'<line x1="{v7_x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2" stroke-dasharray="5 4"/>',
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>',
                f'<text x="{x:.1f}" y="{y-14:.1f}" text-anchor="middle" class="label">{html.escape(label)}</text>',
                f'<text x="{x:.1f}" y="{y+25:.1f}" text-anchor="middle" class="value">{value:.3f}%</text>',
            ]
        )
    body.append(
        f'<text x="36" y="{height-18}" class="note">The no-verification branch is an ablation signal, not part of the verified-flow progression.</text>'
    )
    write_svg(output, width, height, "Verified flow evolution", body)


def write_compute_tradeoff(results: dict[str, dict[str, Any]], output: Path) -> None:
    methods = [
        "random_mnli",
        "uncertainty_mnli",
        "coreset_mnli",
        "margin_mnli",
        "cal_mnli",
        "badge_mnli",
    ]
    width, height = 1180, 700
    left, right, top, bottom = 105, 285, 105, 95
    chart_width, chart_height = width - left - right, height - top - bottom
    x_min, x_max = 0.0, 46.0
    y_min, y_max = 64.5, 67.2
    body = [
        '<text x="36" y="42" class="title">MNLI acquisition trade-off — accuracy versus selector GPU time</text>',
        '<text x="36" y="68" class="subtitle">Lower-right is expensive; upper-left is preferred. Training and final evaluation time are excluded from x.</text>',
    ]
    for tick in (0, 10, 20, 30, 40):
        x = left + chart_width * (tick - x_min) / (x_max - x_min)
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#E2E8F0"/>',
                f'<text x="{x:.1f}" y="{height-bottom+27}" text-anchor="middle" class="tick">{tick} h</text>',
            ]
        )
    for tick in (64.5, 65.0, 65.5, 66.0, 66.5, 67.0):
        y = top + chart_height * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.1f}%</text>',
            ]
        )
    point_coordinates: dict[str, tuple[float, float]] = {}
    for index, method in enumerate(methods):
        acquisition_hours = float(results[method].get("acquisition_gpu_seconds", 0.0)) / 3600.0
        value = accuracy(results, method)
        x = left + chart_width * (acquisition_hours - x_min) / (x_max - x_min)
        y = top + chart_height * (y_max - value) / (y_max - y_min)
        point_coordinates[method] = (x, y)
        selector = method.split("_")[0]
        color = COLORS.get(selector, COLORS["human"])
        body.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{color}" stroke="#FFFFFF" stroke-width="3"/>',
                f'<text x="{x+13:.1f}" y="{y-8:.1f}" class="label">{html.escape(method)}</text>',
                f'<text x="{x+13:.1f}" y="{y+9:.1f}" class="note">{value:.3f}% · {acquisition_hours:.2f} h</text>',
            ]
        )
    random_x, random_y = point_coordinates["random_mnli"]
    uncertainty_x, uncertainty_y = point_coordinates["uncertainty_mnli"]
    body.append(
        f'<polyline points="{random_x:.1f},{random_y:.1f} {uncertainty_x:.1f},{uncertainty_y:.1f}" fill="none" stroke="#0F172A" stroke-width="2" stroke-dasharray="6 5" opacity="0.7"/>'
    )
    body.extend(
        [
            f'<text x="{left+chart_width/2:.1f}" y="{height-35}" text-anchor="middle" class="label">Acquisition GPU-hours</text>',
            f'<text x="{width-right+22}" y="{top+30}" class="section">Seed-42 takeaway</text>',
            f'<text x="{width-right+22}" y="{top+57}" class="note">Uncertainty has the highest</text>',
            f'<text x="{width-right+22}" y="{top+75}" class="note">accuracy and lower acquisition</text>',
            f'<text x="{width-right+22}" y="{top+93}" class="note">time than all 4 new selectors.</text>',
            f'<text x="36" y="{height-17}" class="note">Protocol caveat: Random/Uncertainty are old one-shot runs; the four literature methods use stateful four-round acquisition.</text>',
        ]
    )
    write_svg(output, width, height, "MNLI accuracy and acquisition compute trade-off", body)


def main() -> None:
    args = parse_args()
    output = (PROJECT_ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = load_results()
    write_hans_by_source(results, output / "01_hans_by_data_source.svg")
    write_mnli_learning_curves(results, output / "02_mnli_learning_curves.svg")
    write_verified_evolution(results, output / "03_verified_flow_evolution.svg")
    write_compute_tradeoff(results, output / "04_mnli_accuracy_compute.svg")
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "figures": sorted(path.name for path in output.glob("*.svg")),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
