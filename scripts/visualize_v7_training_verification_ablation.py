#!/usr/bin/env python3
"""Generate SVG figures for the V7 verification ON/OFF seed-42 report."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ON_PATH = "outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_42/result.json"
OFF_PATH = "outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_42/result.json"
ON_R3_PATH = (
    "outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_42/"
    "posthoc/round_003_stopping_accuracy_71/result.json"
)
ON_ROUNDS_PATH = (
    "outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_42/rounds.jsonl"
)

HUMAN_PATHS = {
    "Uncertainty": "outputs/stage2_runs/experiment_a/uncertainty_mnli/budget_128/seed_42/result.json",
    "Margin": "outputs/stage2_runs_literature/experiment_b/margin_mnli/budget_128/seed_42/result.json",
    "BADGE": "outputs/stage2_runs_literature/experiment_b/badge_mnli/budget_128/seed_42/result.json",
    "Random": "outputs/stage2_runs/experiment_a/random_mnli/budget_128/seed_42/result.json",
    "Core-set": "outputs/stage2_runs_literature/experiment_b/coreset_mnli/budget_128/seed_42/result.json",
    "CAL": "outputs/stage2_runs_literature/experiment_b/cal_mnli/budget_128/seed_42/result.json",
}

COLORS = {
    "on_r3": "#059669",
    "on_r4": "#2563EB",
    "off_r4": "#EA580C",
    "human": "#64748B",
    "grid": "#E2E8F0",
    "ink": "#0F172A",
    "muted": "#64748B",
}

SEEDS = (42, 43, 44)
LOOP_GPU_FIELDS = (
    "acquisition_gpu_wall_seconds",
    "training_gpu_wall_seconds",
    "validation_gpu_wall_seconds",
    "stopping_evaluation_gpu_wall_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/stage2_v7_training_verification_ablation/figures",
    )
    parser.add_argument(
        "--skip-png",
        action="store_true",
        help="Write SVG only instead of also rendering PNG with ImageMagick.",
    )
    return parser.parse_args()


def read_json(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {relative}")
    return value


def read_jsonl(relative: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pct(value: float) -> float:
    return 100.0 * float(value)


def svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{html.escape(title)}</title>
<rect width="100%" height="100%" rx="18" fill="#FFFFFF"/>
<style>
text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif; fill: #334155; }}
.title {{ font-size: 25px; font-weight: 760; fill: #0F172A; }}
.subtitle {{ font-size: 13px; fill: #64748B; }}
.label {{ font-size: 13px; font-weight: 650; }}
.tick {{ font-size: 11px; fill: #64748B; }}
.value {{ font-size: 12px; font-weight: 760; fill: #0F172A; }}
.note {{ font-size: 11px; fill: #475569; }}
.panel {{ font-size: 16px; font-weight: 760; fill: #0F172A; }}
</style>
{chr(10).join(body)}
</svg>
'''


def write_svg(path: Path, width: int, height: int, title: str, body: list[str]) -> None:
    path.write_text(svg_document(width, height, title, body), encoding="utf-8")


def metric(result: dict[str, Any], key: str) -> float:
    if key == "hans":
        return pct(result["hans_final"]["overall_accuracy"])
    if key == "entailment":
        return pct(result["hans_final"]["entailment_accuracy"])
    if key == "non_entailment":
        return pct(result["hans_final"]["non_entailment_accuracy"])
    if key == "mnli_m":
        return pct(result["mnli"]["matched"]["accuracy"])
    if key == "mnli_mm":
        return pct(result["mnli"]["mismatched"]["accuracy"])
    raise KeyError(key)


def write_learning_curve(on: dict[str, Any], off: dict[str, Any], output: Path) -> None:
    width, height = 1240, 650
    left, right, top, bottom = 95, 70, 105, 105
    chart_w, chart_h = width - left - right, height - top - bottom
    y_min, y_max = 62.0, 72.0
    body = [
        '<text x="36" y="42" class="title">Verification ON/OFF learning curves</text>',
        '<text x="36" y="68" class="subtitle">Seed 42 · predeclared 900-example stopping subset · fixed acquisition rounds of 32 examples</text>',
    ]
    for tick in range(62, 73, 2):
        y = top + chart_h * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}%</text>',
            ]
        )
    for round_index in range(5):
        x = left + chart_w * round_index / 4
        body.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height-bottom}" stroke="#F1F5F9"/>',
                f'<text x="{x:.1f}" y="{height-bottom+28}" text-anchor="middle" class="tick">Round {round_index}</text>',
                f'<text x="{x:.1f}" y="{height-bottom+46}" text-anchor="middle" class="note">{round_index*32} examples</text>',
            ]
        )

    def points(result: dict[str, Any]) -> list[tuple[float, float, float]]:
        output_points = []
        for item in result["learning_curve"]:
            round_index = int(item["round"])
            value = pct(item["metrics"]["overall_accuracy"])
            x = left + chart_w * round_index / 4
            y = top + chart_h * (y_max - value) / (y_max - y_min)
            output_points.append((x, y, value))
        return output_points

    for name, result, color in (
        ("Verification ON", on, COLORS["on_r3"]),
        ("Verification OFF", off, COLORS["off_r4"]),
    ):
        curve = points(result)
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in curve)
        body.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>'
        )
        for x, y, value in curve:
            body.extend(
                [
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>',
                    f'<text x="{x:.1f}" y="{y-12:.1f}" text-anchor="middle" class="value">{value:.2f}%</text>',
                ]
            )
        legend_x = 875 if name.endswith("ON") else 1045
        body.extend(
            [
                f'<line x1="{legend_x}" y1="86" x2="{legend_x+30}" y2="86" stroke="{color}" stroke-width="4"/>',
                f'<text x="{legend_x+38}" y="90" class="label">{name}</text>',
            ]
        )

    on_round3 = points(on)[3]
    body.extend(
        [
            '<rect x="410" y="390" width="310" height="43" rx="8" fill="#F8FAFC" stroke="#CBD5E1"/>',
            '<text x="426" y="408" class="label">ON and OFF overlap through round 2</text>',
            '<text x="426" y="424" class="note">Same initial state and selections before divergence</text>',
            f'<circle cx="{on_round3[0]:.1f}" cy="{on_round3[1]:.1f}" r="14" fill="none" stroke="#047857" stroke-width="3" stroke-dasharray="4 3"/>',
            f'<rect x="{on_round3[0]-185:.1f}" y="{on_round3[1]+25:.1f}" width="250" height="48" rx="8" fill="#ECFDF5" stroke="#A7F3D0"/>',
            f'<text x="{on_round3[0]-173:.1f}" y="{on_round3[1]+45:.1f}" class="label" fill="#047857">Post-hoc candidate: ON round 3</text>',
            f'<text x="{on_round3[0]-173:.1f}" y="{on_round3[1]+62:.1f}" class="note">71.00% stopping accuracy · 96 examples</text>',
            f'<text x="36" y="{height-23}" class="note">The fourth ON round lowers stopping accuracy to 66.67%; the curve motivates, but does not predeclare, round-3 selection.</text>',
        ]
    )
    write_svg(output, width, height, "Verification ON/OFF learning curves", body)


def grouped_panel(
    body: list[str],
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    metrics: list[tuple[str, str]],
    series: list[tuple[str, dict[str, Any], str]],
    y_min: float,
    y_max: float,
) -> None:
    body.append(f'<text x="{x0}" y="{y0-28}" class="panel">{html.escape(title)}</text>')
    for tick_index in range(5):
        tick = y_min + (y_max - y_min) * tick_index / 4
        y = y0 + height * (y_max - tick) / (y_max - y_min)
        body.extend(
            [
                f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+width}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
                f'<text x="{x0-9}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick:.0f}%</text>',
            ]
        )
    group_w = width / len(metrics)
    bar_w = min(44.0, group_w / (len(series) + 1))
    for metric_index, (metric_label, metric_key) in enumerate(metrics):
        center = x0 + group_w * (metric_index + 0.5)
        start = center - bar_w * len(series) / 2
        for series_index, (_, result, color) in enumerate(series):
            value = metric(result, metric_key)
            bar_h = height * (value - y_min) / (y_max - y_min)
            x = start + series_index * bar_w
            y = y0 + height - bar_h
            body.extend(
                [
                    f'<rect x="{x+3:.1f}" y="{y:.1f}" width="{bar_w-6:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}" opacity="0.90"/>',
                    f'<text x="{x+bar_w/2:.1f}" y="{y-7:.1f}" text-anchor="middle" class="value">{value:.2f}</text>',
                ]
            )
        body.append(
            f'<text x="{center:.1f}" y="{y0+height+25:.1f}" text-anchor="middle" class="label">{html.escape(metric_label)}</text>'
        )


def write_checkpoint_comparison(
    on: dict[str, Any], off: dict[str, Any], on_r3: dict[str, Any], output: Path
) -> None:
    width, height = 1500, 720
    body = [
        '<text x="36" y="42" class="title">Checkpoint comparison: ON round 3 versus final ON/OFF</text>',
        '<text x="36" y="68" class="subtitle">Seed 42 · full HANS 15,000 examples · MNLI matched/mismatched evaluation</text>',
    ]
    series = [
        ("ON round 3 (post-hoc)", on_r3, COLORS["on_r3"]),
        ("ON round 4", on, COLORS["on_r4"]),
        ("OFF round 4", off, COLORS["off_r4"]),
    ]
    grouped_panel(
        body,
        95,
        145,
        620,
        420,
        "HANS robustness",
        [("Overall", "hans"), ("Non-entailment", "non_entailment")],
        series,
        45,
        75,
    )
    grouped_panel(
        body,
        825,
        145,
        590,
        420,
        "MNLI retention",
        [("Matched", "mnli_m"), ("Mismatched", "mnli_mm")],
        series,
        55,
        65,
    )
    for index, (label, _, color) in enumerate(series):
        x = 455 + index * 235
        body.extend(
            [
                f'<rect x="{x}" y="625" width="18" height="18" rx="3" fill="{color}"/>',
                f'<text x="{x+27}" y="639" class="label">{html.escape(label)}</text>',
            ]
        )
    body.append(
        '<text x="36" y="692" class="note">ON round 3 is post-hoc and uses 96 examples; ON/OFF round 4 are the predeclared 128-example final checkpoints.</text>'
    )
    write_svg(output, width, height, "Checkpoint metric comparison", body)


def loop_gpu_seconds(result: dict[str, Any]) -> float:
    usage = result["resource_usage"]
    return sum(
        float(usage[key])
        for key in (
            "acquisition_gpu_wall_seconds",
            "stopping_evaluation_gpu_wall_seconds",
            "training_gpu_wall_seconds",
            "validation_gpu_wall_seconds",
        )
    )


def on_round3_cost(rounds: list[dict[str, Any]]) -> dict[str, float]:
    rows = rounds[:3]
    teacher = [row["resource_usage"]["teacher"] for row in rows]
    return {
        "tokens": sum(float(row["input_tokens"] + row["output_tokens"]) for row in teacher),
        "calls": sum(float(row["calls"]) for row in teacher),
        "gpu_minutes": sum(
            sum(
                float(row["resource_usage"][key])
                for key in (
                    "acquisition_gpu_wall_seconds",
                    "stopping_evaluation_gpu_wall_seconds",
                    "training_gpu_wall_seconds",
                    "validation_gpu_wall_seconds",
                )
            )
            for row in rows
        )
        / 60.0,
    }


def write_cost_comparison(
    on: dict[str, Any], off: dict[str, Any], rounds: list[dict[str, Any]], output: Path
) -> None:
    width, height = 1520, 680
    on_r3 = on_round3_cost(rounds)
    data = [
        (
            "ON full",
            {
                "tokens": float(on["resource_usage"]["teacher"]["input_tokens"])
                + float(on["resource_usage"]["teacher"]["output_tokens"]),
                "calls": float(on["resource_usage"]["teacher"]["calls"]),
                "gpu_minutes": loop_gpu_seconds(on) / 60.0,
            },
            COLORS["on_r4"],
        ),
        ("ON round 3", on_r3, COLORS["on_r3"]),
        (
            "OFF full",
            {
                "tokens": float(off["resource_usage"]["teacher"]["input_tokens"])
                + float(off["resource_usage"]["teacher"]["output_tokens"]),
                "calls": float(off["resource_usage"]["teacher"]["calls"]),
                "gpu_minutes": loop_gpu_seconds(off) / 60.0,
            },
            COLORS["off_r4"],
        ),
    ]
    body = [
        '<text x="36" y="42" class="title">Cost trade-off: cutting ON after round 3</text>',
        '<text x="36" y="68" class="subtitle">Seed 42 · shared initial-pool and archived failed-attempt costs excluded · final evaluation excluded from loop GPU proxy</text>',
    ]
    panels = [
        ("Teacher tokens", "tokens", 1_100_000.0, lambda v: f"{v/1000:.0f}k"),
        ("Teacher calls", "calls", 520.0, lambda v: f"{v:.0f}"),
        ("Closed-loop GPU proxy", "gpu_minutes", 40.0, lambda v: f"{v:.1f} min"),
    ]
    for panel_index, (title, key, maximum, formatter) in enumerate(panels):
        x0 = 55 + panel_index * 495
        y0, panel_w, panel_h = 135, 430, 390
        body.append(f'<text x="{x0}" y="108" class="panel">{title}</text>')
        for tick_index in range(5):
            tick = maximum * tick_index / 4
            y = y0 + panel_h * (maximum - tick) / maximum
            body.extend(
                [
                    f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
                    f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" class="tick">{formatter(tick)}</text>',
                ]
            )
        bar_w, gap = 86, 43
        for index, (label, values, color) in enumerate(data):
            value = values[key]
            bar_h = panel_h * value / maximum
            x = x0 + 40 + index * (bar_w + gap)
            y = y0 + panel_h - bar_h
            body.extend(
                [
                    f'<rect x="{x}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" rx="6" fill="{color}" opacity="0.90"/>',
                    f'<text x="{x+bar_w/2:.1f}" y="{y-8:.1f}" text-anchor="middle" class="value">{formatter(value)}</text>',
                    f'<text x="{x+bar_w/2:.1f}" y="{y0+panel_h+25:.1f}" text-anchor="middle" class="label">{label}</text>',
                ]
            )
    body.extend(
        [
            '<rect x="70" y="588" width="630" height="48" rx="9" fill="#ECFDF5" stroke="#A7F3D0"/>',
            '<text x="86" y="608" class="label" fill="#047857">Cutting ON at round 3 saves 45.8% of teacher tokens versus full ON.</text>',
            '<text x="86" y="625" class="note">But ON-r3 still uses 2.54x the teacher tokens of OFF-full.</text>',
            '<text x="745" y="608" class="note">GPU values are measured wall-clock proxies, not hardware-utilization integrals.</text>',
            '<text x="745" y="625" class="note">Teacher and refresh wall times overlap and are not summed in this figure.</text>',
        ]
    )
    write_svg(output, width, height, "Cost comparison for ON round-3 stopping", body)


def load_multiseed_cost_points() -> list[dict[str, float | int | str]]:
    points: list[dict[str, float | int | str]] = []
    for mode in ("on", "off"):
        for seed in SEEDS:
            base = (
                "outputs/stage2_runs_v7_training_verification_ablation/"
                f"verification_{mode}/seed_{seed}"
            )
            result = read_json(f"{base}/result.json")
            records = read_jsonl(f"{base}/rounds.jsonl")
            accuracy_by_round = {
                int(item["round"]): pct(item["metrics"]["overall_accuracy"])
                for item in result["learning_curve"]
            }
            cumulative_tokens = 0.0
            cumulative_gpu_seconds = 0.0
            cumulative_wall_seconds = 0.0
            for record in records:
                round_index = int(record["round"])
                usage = record["resource_usage"]
                teacher = usage["teacher"]
                cumulative_tokens += float(teacher["input_tokens"]) + float(
                    teacher["output_tokens"]
                )
                gpu_seconds = sum(float(usage[key]) for key in LOOP_GPU_FIELDS)
                cumulative_gpu_seconds += gpu_seconds
                cumulative_wall_seconds += float(usage["refresh_wall_seconds"]) + gpu_seconds
                points.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "round": round_index,
                        "budget": round_index * 32,
                        "accuracy": accuracy_by_round[round_index],
                        "teacher_tokens": cumulative_tokens,
                        "gpu_minutes": cumulative_gpu_seconds / 60.0,
                        "total_wall_hours": cumulative_wall_seconds / 3600.0,
                    }
                )
    return points


def summarize_multiseed_cost_points(
    points: list[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    for mode in ("on", "off"):
        for round_index in range(1, 5):
            rows = [
                row
                for row in points
                if row["mode"] == mode and row["round"] == round_index
            ]
            if len(rows) != len(SEEDS):
                raise ValueError(f"Incomplete cost curve for {mode} round {round_index}")

            def mean(key: str) -> float:
                return statistics.mean(float(row[key]) for row in rows)

            accuracies = [float(row["accuracy"]) for row in rows]
            summary.append(
                {
                    "mode": mode,
                    "round": round_index,
                    "budget": round_index * 32,
                    "accuracy": statistics.mean(accuracies),
                    "accuracy_std": statistics.stdev(accuracies),
                    "teacher_tokens": mean("teacher_tokens"),
                    "gpu_minutes": mean("gpu_minutes"),
                    "total_wall_hours": mean("total_wall_hours"),
                }
            )
    return summary


def write_performance_cost_curves(
    points: list[dict[str, float | int | str]], output: Path
) -> None:
    summary = summarize_multiseed_cost_points(points)
    width, height = 1600, 1040
    panel_w, panel_h = 650, 305
    y_min, y_max = 62.0, 74.5
    body = [
        '<text x="36" y="42" class="title">Performance–cost curves across seeds 42–44</text>',
        '<text x="36" y="68" class="subtitle">Mean validation accuracy on the predeclared 900-example stopping subset · vertical bars: sample standard deviation</text>',
    ]
    panels = [
        ("Accuracy vs acquisition budget", "budget", "examples", lambda value: f"{value:.0f}"),
        ("Accuracy vs cumulative teacher tokens", "teacher_tokens", "tokens", lambda value: f"{value/1000:.0f}k"),
        ("Accuracy vs cumulative GPU proxy", "gpu_minutes", "minutes", lambda value: f"{value:.0f}"),
        ("Accuracy vs cumulative total wall cost", "total_wall_hours", "hours", lambda value: f"{value:.1f}"),
    ]
    origins = ((90, 145), (855, 145), (90, 590), (855, 590))

    for (title, key, unit, formatter), (x0, y0) in zip(panels, origins):
        values = [float(row[key]) for row in summary]
        if key == "budget":
            x_min, x_max = 24.0, 136.0
        else:
            x_min, x_max = 0.0, max(values) * 1.08
        body.extend(
            [
                f'<text x="{x0}" y="{y0-32}" class="panel">{html.escape(title)}</text>',
                f'<text x="{x0+panel_w}" y="{y0+panel_h+48}" text-anchor="end" class="note">{html.escape(unit)}</text>',
            ]
        )
        for tick in (62, 64, 66, 68, 70, 72, 74):
            y = y0 + panel_h * (y_max - tick) / (y_max - y_min)
            body.extend(
                [
                    f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>',
                    f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{tick}%</text>',
                ]
            )
        for tick_index in range(5):
            value = x_min + (x_max - x_min) * tick_index / 4
            x = x0 + panel_w * tick_index / 4
            body.extend(
                [
                    f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+panel_h}" stroke="#F1F5F9"/>',
                    f'<text x="{x:.1f}" y="{y0+panel_h+24}" text-anchor="middle" class="tick">{formatter(value)}</text>',
                ]
            )

        for mode, label, color in (
            ("on", "Verification ON", COLORS["on_r4"]),
            ("off", "Verification OFF", COLORS["off_r4"]),
        ):
            rows = [row for row in summary if row["mode"] == mode]
            coordinates: list[tuple[float, float, dict[str, float | int | str]]] = []
            for row in rows:
                x_value = float(row[key])
                accuracy = float(row["accuracy"])
                x = x0 + panel_w * (x_value - x_min) / (x_max - x_min)
                y = y0 + panel_h * (y_max - accuracy) / (y_max - y_min)
                coordinates.append((x, y, row))
            body.append(
                f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coordinates)}" fill="none" stroke="{color}" stroke-width="4" stroke-linejoin="round"/>'
            )
            for x, y, row in coordinates:
                error = panel_h * float(row["accuracy_std"]) / (y_max - y_min)
                label_y = y - 9 if mode == "on" else y + 20
                body.extend(
                    [
                        f'<line x1="{x:.1f}" y1="{y-error:.1f}" x2="{x:.1f}" y2="{y+error:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<line x1="{x-5:.1f}" y1="{y-error:.1f}" x2="{x+5:.1f}" y2="{y-error:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<line x1="{x-5:.1f}" y1="{y+error:.1f}" x2="{x+5:.1f}" y2="{y+error:.1f}" stroke="{color}" stroke-width="2"/>',
                        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#FFFFFF" stroke="{color}" stroke-width="3"/>',
                        f'<text x="{x+8:.1f}" y="{label_y:.1f}" class="value">B{int(row["budget"])}</text>',
                    ]
                )

    body.extend(
        [
            f'<line x1="1130" y1="995" x2="1162" y2="995" stroke="{COLORS["on_r4"]}" stroke-width="4"/>',
            '<text x="1170" y="999" class="label">Verification ON</text>',
            f'<line x1="1330" y1="995" x2="1362" y2="995" stroke="{COLORS["off_r4"]}" stroke-width="4"/>',
            '<text x="1370" y="999" class="label">Verification OFF</text>',
            '<text x="36" y="999" class="note">Total wall cost = refresh wall time + closed-loop GPU proxy; teacher time is contained in refresh. Shared initial pool, final audit/evaluation, and retry overhead are excluded.</text>',
        ]
    )
    write_svg(output, width, height, "V7 multi-seed performance-cost curves", body)


def write_human_comparison(on_r3: dict[str, Any], output: Path) -> None:
    methods: list[tuple[str, dict[str, Any], str]] = [
        ("ON round 3 · verified synthetic", on_r3, COLORS["on_r3"])
    ]
    for name, path in HUMAN_PATHS.items():
        methods.append((f"{name} · human MNLI", read_json(path), COLORS["human"]))
    width, height = 1540, 780
    body = [
        '<text x="36" y="42" class="title">Post-hoc ON round 3 versus human-labelled MNLI baselines</text>',
        '<text x="36" y="68" class="subtitle">Seed 42 · shared final evaluators · cross-source and pipeline-level comparison, not equal-data-access</text>',
    ]

    def horizontal_panel(
        x0: float,
        width_: float,
        title: str,
        minimum: float,
        maximum: float,
        value_fn: Any,
    ) -> None:
        top, row_h = 130, 70
        body.append(f'<text x="{x0}" y="105" class="panel">{title}</text>')
        for tick in range(int(minimum), int(maximum) + 1, 2):
            x = x0 + width_ * (tick - minimum) / (maximum - minimum)
            body.extend(
                [
                    f'<line x1="{x:.1f}" y1="{top-18}" x2="{x:.1f}" y2="{top+row_h*len(methods)-12}" stroke="{COLORS["grid"]}"/>',
                    f'<text x="{x:.1f}" y="{top+row_h*len(methods)+10}" text-anchor="middle" class="tick">{tick}%</text>',
                ]
            )
        for index, (label, result, color) in enumerate(methods):
            value = float(value_fn(result))
            y = top + index * row_h
            end = x0 + width_ * (value - minimum) / (maximum - minimum)
            body.extend(
                [
                    f'<text x="{x0}" y="{y-7}" class="label">{html.escape(label)}</text>',
                    f'<rect x="{x0}" y="{y+3}" width="{max(3.0, end-x0):.1f}" height="23" rx="5" fill="{color}" opacity="0.90"/>',
                    f'<text x="{end+8:.1f}" y="{y+20}" class="value">{value:.2f}%</text>',
                ]
            )

    horizontal_panel(65, 620, "Full HANS accuracy", 64, 72, lambda r: metric(r, "hans"))
    horizontal_panel(
        825,
        600,
        "MNLI retention (mean of matched/mismatched)",
        58,
        64,
        lambda r: (metric(r, "mnli_m") + metric(r, "mnli_mm")) / 2,
    )
    body.extend(
        [
            '<rect x="58" y="674" width="1395" height="65" rx="10" fill="#FFF7ED" stroke="#FED7AA"/>',
            '<text x="76" y="697" class="label" fill="#9A3412">Numerically, ON round 3 leads the completed human-MNLI baselines: +4.26 pp HANS over Uncertainty.</text>',
            '<text x="76" y="718" class="note">Caution: ON-r3 is post-hoc, uses a different multi-step training policy and teacher-generated verified data, and currently has only seed 42.</text>',
        ]
    )
    write_svg(output, width, height, "ON round 3 versus human MNLI baselines", body)


def render_pngs(output: Path) -> None:
    convert = shutil.which("convert")
    if convert is None:
        raise RuntimeError(
            "ImageMagick 'convert' is required for PNG rendering; use --skip-png "
            "to generate SVG only."
        )
    for svg in sorted(output.glob("*.svg")):
        subprocess.run(
            [
                convert,
                "-background",
                "white",
                "-density",
                "144",
                str(svg),
                str(svg.with_suffix(".png")),
            ],
            check=True,
        )


def main() -> None:
    args = parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    on = read_json(ON_PATH)
    off = read_json(OFF_PATH)
    on_r3 = read_json(ON_R3_PATH)
    rounds = read_jsonl(ON_ROUNDS_PATH)
    write_learning_curve(on, off, output / "01_learning_curve.svg")
    write_checkpoint_comparison(on, off, on_r3, output / "02_checkpoint_comparison.svg")
    write_cost_comparison(on, off, rounds, output / "03_cost_comparison.svg")
    write_human_comparison(on_r3, output / "04_human_mnli_comparison.svg")
    write_performance_cost_curves(
        load_multiseed_cost_points(), output / "05_performance_cost_curves.svg"
    )
    if not args.skip_png:
        render_pngs(output)
    formats = "SVG" if args.skip_png else "SVG and PNG"
    print(f"Wrote 5 figures in {formats} format to {output}")


if __name__ == "__main__":
    main()
