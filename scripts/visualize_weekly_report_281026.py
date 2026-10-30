#!/usr/bin/env python3
"""Generate self-contained figures for the 21--28 October 2026 weekly report."""

from __future__ import annotations

import html
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_281026_figures"

NAVY = "#17324D"
BLUE = "#377EB8"
TEAL = "#2A9D8F"
ORANGE = "#F4A261"
RED = "#D95F59"
PURPLE = "#7B61A8"
GREY = "#6B7280"
LIGHT = "#F4F7FA"
GREEN = "#4C956C"


def _svg_text(
    x: float,
    y: float,
    value: str,
    css_class: str = "label",
    anchor: str = "middle",
    line_height: int = 18,
) -> str:
    lines = value.split("\n")
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        spans.append(
            f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'class="{css_class}">{"".join(spans)}</text>'
    )


def _svg_box(
    x: float,
    y: float,
    width: float,
    height: float,
    value: str,
    color: str,
    *,
    light: bool = False,
    font_class: str = "boxtext",
) -> str:
    text_class = "darkboxtext" if light else font_class
    line_count = len(value.split("\n"))
    text_y = y + height / 2 - (line_count - 1) * 9 + 5
    return "\n".join(
        [
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="14" '
            f'fill="{color}" stroke="{NAVY if light else color}" stroke-width="{1 if light else 0}"/>',
            _svg_text(x + width / 2, text_y, value, text_class),
        ]
    )


def _svg_arrow(x1: float, y1: float, x2: float, y2: float, color: str = GREY) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>'
    )


def _document(width: int, height: int, title: str, body: list[str]) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
<title>{html.escape(title)}</title>
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="#6B7280"/>
  </marker>
</defs>
<rect width="100%" height="100%" rx="18" fill="#FFFFFF"/>
<style>
text {{ font-family: Inter, DejaVu Sans, Arial, sans-serif; fill: #334155; }}
.title {{ font-size: 28px; font-weight: 760; fill: #17324D; }}
.subtitle {{ font-size: 15px; fill: #64748B; }}
.heading {{ font-size: 15px; font-weight: 700; fill: #64748B; }}
.boxtext {{ font-size: 14px; font-weight: 720; fill: #FFFFFF; }}
.darkboxtext {{ font-size: 13px; font-weight: 700; fill: #17324D; }}
.label {{ font-size: 14px; font-weight: 650; fill: #334155; }}
.small {{ font-size: 12px; fill: #64748B; }}
.value {{ font-size: 13px; font-weight: 760; fill: #17324D; }}
.panel {{ font-size: 19px; font-weight: 760; fill: #17324D; }}
</style>
{chr(10).join(body)}
</svg>
'''


def _write_figure(stem: str, width: int, height: int, title: str, body: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{stem}.svg"
    png_path = OUT / f"{stem}.png"
    svg_path.write_text(_document(width, height, title, body), encoding="utf-8")
    converter = shutil.which("convert") or shutil.which("magick")
    if converter:
        subprocess.run(
            [converter, "-density", "170", str(svg_path), str(png_path)],
            check=True,
        )


def _pure_protocol_map() -> None:
    body = [
        _svg_text(700, 48, "Experimental map: source, protocol, and interpretation", "title"),
        _svg_text(700, 76, "All branches share the frozen 15,000-example HANS final view", "subtitle"),
    ]
    headings = [(145, "Candidate source"), (480, "Acquisition and training"), (825, "Training set"), (1160, "Interpretation")]
    for x, label in headings:
        body.append(_svg_text(x, 118, label, "heading"))
    rows = [
        (150, BLUE, "Human-labelled MNLI\npool (~392k)", "Random / Uncertainty / Margin\nCore-set / BADGE / CAL", "128 human-labelled\nexamples", "Cross-source\nreference"),
        (365, ORANGE, "Static verified-v5\nsynthetic pool", "Margin / Core-set\nBADGE / CAL", "128 synthetic\nexamples", "Controlled selector\ncomparison"),
        (580, TEAL, "Dynamic V7\nhypothesis archive", "Discover -> Verify ON/OFF -> Generate + QC\nSelect 32 -> Train -> Refresh (x4)", "96 / 128 targeted\nsynthetic examples", "Full closed-loop\ncomparison"),
    ]
    for y, color, source, method, train, meaning in rows:
        body.extend(
            [
                _svg_box(35, y, 220, 125, source, color),
                _svg_box(310, y, 340, 125, method, color),
                _svg_box(710, y, 225, 125, train, color),
                _svg_box(995, y, 330, 125, meaning, LIGHT, light=True),
                _svg_arrow(255, y + 62, 305, y + 62, color),
                _svg_arrow(650, y + 62, 705, y + 62, color),
                _svg_arrow(935, y + 62, 990, y + 62, color),
            ]
        )
    body.append(_svg_text(700, 755, "Do not interpret differences between rows as equal-data-access selector effects.", "label"))
    _write_figure("01_experiment_protocol_map", 1400, 790, "Experimental protocol map", body)


def _pure_v7_flow() -> None:
    body = [
        _svg_text(700, 48, "V7 early-stopping verification ablation", "title"),
        _svg_text(700, 76, "Acquisition remains fixed: 4 rounds x 32 = 128 examples", "subtitle"),
    ]
    nodes = [
        (30, "Discover", BLUE),
        (245, "Verify\nON / OFF", PURPLE),
        (460, "Generate\n+ synthetic QC", ORANGE),
        (675, "Select\n32 examples", TEAL),
        (890, "Train LoRA\n+ AdamW", RED),
        (1105, "Refresh\nhypotheses", BLUE),
    ]
    for x, label, color in nodes:
        body.append(_svg_box(x, 170, 175, 125, label, color))
    for index in range(len(nodes) - 1):
        body.append(_svg_arrow(nodes[index][0] + 175, 232, nodes[index + 1][0] - 5, 232))
    body.extend(
        [
            '<path d="M1192 300 C1192 390 118 390 118 305" fill="none" stroke="#6B7280" stroke-width="3" marker-end="url(#arrow)"/>',
            _svg_text(700, 405, "repeat until round 4", "heading"),
            _svg_box(190, 490, 430, 145, "Validation policy\n900-example stopping subset\nbinary NLL; sealed HANS is excluded", LIGHT, light=True),
            _svg_box(780, 490, 430, 145, "Training policy\neffective batch 8; max 8 epochs\npatience 2; min_delta 0.001", LIGHT, light=True),
            _svg_arrow(978, 295, 995, 485, RED),
            _svg_arrow(780, 562, 625, 562),
            _svg_text(700, 690, "Best epoch restores both LoRA tensors and complete AdamW state; continuity is audited on resume.", "label"),
            _svg_text(700, 730, "Only intended ablation variable: hypothesis verification. Selector, QC, initial state, seeds, budget, and training policy are shared.", "small"),
        ]
    )
    _write_figure("02_v7_early_stopping_flow", 1400, 770, "V7 early-stopping flow", body)


def _bar(
    body: list[str],
    x: float,
    y_base: float,
    width: float,
    value: float,
    minimum: float,
    maximum: float,
    scale_height: float,
    color: str,
    label: str,
) -> None:
    height = scale_height * (value - minimum) / (maximum - minimum)
    body.append(f'<rect x="{x}" y="{y_base-height:.1f}" width="{width}" height="{height:.1f}" rx="5" fill="{color}"/>')
    body.append(_svg_text(x + width / 2, y_base - height - 9, f"{value:.2f}", "value"))
    body.append(_svg_text(x + width / 2, y_base + 23, label, "small"))


def _pure_result_landscape() -> None:
    body = [
        _svg_text(750, 44, "Result landscape for 21--28 October 2026", "title"),
        _svg_text(370, 95, "Key results across different protocols", "panel"),
        _svg_text(1120, 95, "Matched ON/OFF rule diagnostic", "panel"),
    ]
    left = [
        ("Old V7", 65.0000, TEAL),
        ("Core-set syn.", 65.0533, ORANGE),
        ("Uncertainty", 66.9933, BLUE),
        ("Margin", 66.1267, BLUE),
        ("BADGE", 65.7000, BLUE),
        ("ON r3", 71.2533, PURPLE),
    ]
    for index, (label, value, color) in enumerate(left):
        _bar(body, 55 + index * 105, 550, 72, value, 63.5, 72.0, 390, color, label)
    body.extend(
        [
            _svg_text(365, 615, "Human MNLI, static synthetic, and dynamic V7 are shown together only as numeric context.", "small"),
            '<line x1="748" y1="110" x2="748" y2="640" stroke="#CBD5E1" stroke-width="2"/>',
        ]
    )
    checkpoints = [
        ("ON r3\n96", 71.2533, 71.2533),
        ("OFF r3\n96", 64.6867, 64.6867),
        ("ON r4\n128", 68.6333, 71.8533),
        ("OFF r4\n128", 70.5067, 71.1533),
    ]
    for index, (label, current, direct) in enumerate(checkpoints):
        x = 800 + index * 165
        _bar(body, x, 550, 55, current, 62.0, 73.0, 390, GREY, "")
        _bar(body, x + 60, 550, 55, direct, 62.0, 73.0, 390, TEAL, "")
        body.append(_svg_text(x + 57, 585, label, "small"))
    body.extend(
        [
            '<rect x="905" y="620" width="18" height="18" rx="3" fill="#6B7280"/>',
            _svg_text(935, 634, "3-way argmax -> binary", "small", anchor="start"),
            '<rect x="1115" y="620" width="18" height="18" rx="3" fill="#2A9D8F"/>',
            _svg_text(1145, 634, "Direct binary N+C", "small", anchor="start"),
            _svg_text(1120, 680, "Final-test diagnostic only; not a rule/checkpoint selection procedure.", "small"),
        ]
    )
    _write_figure("03_weekly_result_landscape", 1500, 720, "Weekly result landscape", body)


def _pure_rule_diagnostic() -> None:
    body = [
        _svg_text(700, 46, "Prediction-rule sensitivity is concentrated in ON round 4", "title"),
        _svg_text(360, 100, "Predictions changed between rules", "panel"),
        _svg_text(1050, 100, "Net correct: direct binary minus current", "panel"),
    ]
    names = ["ON r3", "OFF r3", "ON r4", "OFF r4"]
    disagreement = [0.0, 0.0, 6.4333, 0.94]
    net = [0, 0, 483, 97]
    colors = [PURPLE, GREY, PURPLE, GREY]
    for index, (name, value, color) in enumerate(zip(names, disagreement, colors, strict=True)):
        x = 80 + index * 145
        height = value / 7.0 * 350
        body.extend(
            [
                f'<rect x="{x}" y="{520-height:.1f}" width="78" height="{height:.1f}" rx="6" fill="{color}"/>',
                _svg_text(x + 39, 505 - height, f"{value:.2f}%", "value"),
                _svg_text(x + 39, 550, name, "label"),
            ]
        )
    body.append('<line x1="700" y1="110" x2="700" y2="610" stroke="#CBD5E1" stroke-width="2"/>')
    for index, (name, value, color) in enumerate(zip(names, net, colors, strict=True)):
        x = 770 + index * 145
        height = value / 520.0 * 350
        body.extend(
            [
                f'<rect x="{x}" y="{520-height:.1f}" width="78" height="{height:.1f}" rx="6" fill="{color}"/>',
                _svg_text(x + 39, 505 - height, f"+{value}", "value"),
                _svg_text(x + 39, 550, name, "label"),
            ]
        )
    body.extend(
        [
            _svg_text(360, 600, "ON r4: 965 / 15,000 changed (6.43%)", "small"),
            _svg_text(1050, 600, "ON r4 gains 483 correct; OFF r4 gains 97 correct", "small"),
            _svg_text(700, 660, "Rounds 3 are identical under both rules; ranking at round 4 changes with the binary mapping.", "label"),
        ]
    )
    _write_figure("04_hans_prediction_rule_diagnostic", 1400, 700, "HANS prediction-rule diagnostic", body)


def main() -> None:
    _pure_protocol_map()
    _pure_v7_flow()
    _pure_result_landscape()
    _pure_rule_diagnostic()
    print(f"Wrote weekly-report SVG/PNG figures to {OUT}")


if __name__ == "__main__":
    main()
