#!/usr/bin/env python3
"""Build paper-ready HANS/baseline tables, data exports, figures, and Markdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import _atomic_text_write, write_json


@dataclass(frozen=True)
class RunSpec:
    name: str
    run_name: str
    method: str


ZERO_SHOT_RUN = RunSpec("HANS zero-shot", "hans_qwen3_1_7b_base", "zero_shot")
BASELINE_RUNS = (
    RunSpec("Base", "baseline_base_b32_seed42", "base"),
    RunSpec("Random", "baseline_random_b32_seed42", "random"),
    RunSpec("Uncertainty", "baseline_uncertainty_b32_seed42", "uncertainty"),
)

COLORS = ("#3B82F6", "#64748B", "#F59E0B", "#10B981")
LABEL_ORDER = ("entailment", "non-entailment")
HEURISTIC_ORDER = ("lexical_overlap", "subsequence", "constituent")
SUMMARY_FIELDS = (
    "name",
    "run_name",
    "method",
    "seed",
    "student_model",
    "device",
    "gpu_name",
    "git_commit",
    "prediction_id_sha256",
    "num_examples",
    "num_correct",
    "num_acquired",
    "acquisition_budget",
    "overall_accuracy",
    "entailment_accuracy",
    "non_entailment_accuracy",
    "lexical_overlap_accuracy",
    "subsequence_accuracy",
    "constituent_accuracy",
    "train_loss",
    "gpu_seconds",
    "wall_clock_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs", help="Experiment root")
    parser.add_argument(
        "--report-dir",
        default="reports/hans_baselines_seed42",
        help="Destination for Markdown, data exports, and figures",
    )
    parser.add_argument(
        "--expected-num-examples",
        type=int,
        default=30_000,
        help="Required prediction count per run (default: full 30,000-row HANS)",
    )
    parser.add_argument(
        "--include-standalone-zero-shot",
        action="store_true",
        help=(
            "Also require outputs/hans_qwen3_1_7b_base. By default the report "
            "uses only Base, Random, and Uncertainty so each input is produced "
            "by scripts/run_baseline.py."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Artifact has no records: {path}")
    return rows


def _last_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    rows = _read_jsonl(path)
    return rows[-1]


def _mean(values: Iterable[bool]) -> tuple[int, int, float | None]:
    materialized = list(values)
    if not materialized:
        return 0, 0, None
    correct = sum(materialized)
    return len(materialized), correct, correct / len(materialized)


def _group_rows(
    records: list[dict[str, Any]], spec: RunSpec
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for record in records:
        correct = record.get("correct")
        if not isinstance(correct, bool):
            raise ValueError(
                f"{spec.run_name} has a non-boolean or missing 'correct' value"
            )
        groups[("overall", "all")].append(correct)
        groups[("label", str(record.get("gold")))].append(correct)
        groups[("heuristic", str(record.get("heuristic")))].append(correct)
        groups[("subcase", str(record.get("subcase")))].append(correct)

    rows = []
    for (group_type, group), correctness in groups.items():
        count, num_correct, accuracy = _mean(correctness)
        rows.append(
            {
                "name": spec.name,
                "run_name": spec.run_name,
                "method": spec.method,
                "group_type": group_type,
                "group": group,
                "num_examples": count,
                "num_correct": num_correct,
                "accuracy": accuracy,
            }
        )
    return rows


def _accuracy_lookup(
    group_rows: list[dict[str, Any]], group_type: str, group: str
) -> float | None:
    matches = [
        row["accuracy"]
        for row in group_rows
        if row["group_type"] == group_type and row["group"] == group
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Expected one {group_type}/{group} aggregate")
    return matches[0]


def _assert_close(actual: Any, expected: float | None, label: str) -> None:
    if actual is None and expected is None:
        return
    if not isinstance(actual, (int, float)) or expected is None:
        raise ValueError(f"Metric mismatch for {label}: artifact={actual}, derived={expected}")
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Metric mismatch for {label}: artifact={actual}, derived={expected}")


def collect_run(
    outputs_root: Path, spec: RunSpec
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = outputs_root / spec.run_name
    metrics = _read_json(run_dir / "metrics.json")
    records = _read_jsonl(run_dir / "predictions.jsonl")
    ids = [str(record.get("id")) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate prediction IDs in {run_dir}")
    prediction_id_sha256 = hashlib.sha256(
        "\n".join(sorted(ids)).encode("utf-8")
    ).hexdigest()

    groups = _group_rows(records, spec)
    overall = _accuracy_lookup(groups, "overall", "all")
    entailment = _accuracy_lookup(groups, "label", "entailment")
    non_entailment = _accuracy_lookup(groups, "label", "non-entailment")
    heuristic_values = {
        name: _accuracy_lookup(groups, "heuristic", name)
        for name in HEURISTIC_ORDER
    }

    if spec.method == "zero_shot":
        artifact_values = {
            "overall": metrics.get("overall_accuracy"),
            "entailment": metrics.get("entailment_accuracy"),
            "non_entailment": metrics.get("non_entailment_accuracy"),
            **dict(metrics.get("accuracy_by_heuristic", {})),
        }
    else:
        artifact_values = {
            "overall": metrics.get("hans_accuracy"),
            "entailment": metrics.get("hans_entailment_accuracy"),
            "non_entailment": metrics.get("hans_non_entailment_accuracy"),
            "lexical_overlap": metrics.get("lexical_overlap_accuracy"),
            "subsequence": metrics.get("subsequence_accuracy"),
            "constituent": metrics.get("constituent_accuracy"),
        }
    for key, expected in {
        "overall": overall,
        "entailment": entailment,
        "non_entailment": non_entailment,
        **heuristic_values,
    }.items():
        _assert_close(artifact_values.get(key), expected, f"{spec.run_name}/{key}")

    run = _read_json(run_dir / "run.json") if (run_dir / "run.json").is_file() else {}
    round_record = _last_jsonl(run_dir / "rounds.jsonl")
    summary = {
        "name": spec.name,
        "run_name": spec.run_name,
        "method": spec.method,
        "seed": run.get("seed", metrics.get("seed")),
        "student_model": run.get("student_model"),
        "device": run.get("device"),
        "gpu_name": run.get("gpu_name"),
        "git_commit": run.get("git_commit"),
        "prediction_id_sha256": prediction_id_sha256,
        "num_examples": len(records),
        "num_correct": sum(bool(record["correct"]) for record in records),
        "num_acquired": metrics.get("num_acquired", 0),
        "acquisition_budget": metrics.get("acquisition_budget", 0),
        "overall_accuracy": overall,
        "entailment_accuracy": entailment,
        "non_entailment_accuracy": non_entailment,
        "lexical_overlap_accuracy": heuristic_values["lexical_overlap"],
        "subsequence_accuracy": heuristic_values["subsequence"],
        "constituent_accuracy": heuristic_values["constituent"],
        "train_loss": round_record.get("train_loss"),
        "gpu_seconds": run.get("gpu_seconds", round_record.get("gpu_seconds")),
        "wall_clock_seconds": run.get(
            "wall_clock_seconds", round_record.get("wall_clock_seconds")
        ),
    }
    return summary, groups


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _svg_document(width: int, height: int, body: str, title: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">{html.escape(title)}</title>
<desc id="desc">Generated from full HANS experiment artifacts.</desc>
<rect width="100%" height="100%" fill="#FFFFFF" rx="18"/>
<style>
  text {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .title {{ fill: #0F172A; font-size: 24px; font-weight: 700; }}
  .subtitle {{ fill: #64748B; font-size: 13px; }}
  .axis {{ fill: #475569; font-size: 12px; }}
  .label {{ fill: #334155; font-size: 13px; font-weight: 600; }}
  .value {{ fill: #0F172A; font-size: 13px; font-weight: 700; }}
</style>
{body}
</svg>
"""


def _write_svg(figures_dir: Path, stem: str, content: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text_write(figures_dir / f"{stem}.svg", content)


def _plot_overall(summaries: list[dict[str, Any]], figures_dir: Path) -> None:
    values = [100.0 * row["overall_accuracy"] for row in summaries]
    width, height = 1000, 570
    left, right, top, bottom = 90, 35, 105, 105
    chart_width = width - left - right
    chart_height = height - top - bottom
    parts = [
        '<text x="42" y="43" class="title">HANS accuracy by acquisition strategy</text>',
        '<text x="42" y="69" class="subtitle">Full HANS evaluation · single seed · higher is better</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_height * (1.0 - tick / 100.0)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>'
        )
        parts.append(
            f'<text x="{left-13}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}</text>'
        )
    slot = chart_width / len(summaries)
    bar_width = min(130.0, slot * 0.58)
    colors = COLORS[: len(summaries)]
    for index, (row, value, color) in enumerate(
        zip(summaries, values, colors, strict=True)
    ):
        x = left + slot * index + (slot - bar_width) / 2
        bar_height = chart_height * value / 100.0
        y = top + chart_height - bar_height
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="8" fill="{color}"/>',
                f'<text x="{x+bar_width/2:.1f}" y="{max(top-8, y-10):.1f}" text-anchor="middle" class="value">{value:.2f}%</text>',
                f'<text x="{x+bar_width/2:.1f}" y="{top+chart_height+29:.1f}" text-anchor="middle" class="label">{html.escape(row["name"])}</text>',
            ]
        )
    parts.append(
        f'<text x="22" y="{top+chart_height/2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top+chart_height/2:.1f})">Accuracy (%)</text>'
    )
    svg = _svg_document(width, height, "\n".join(parts), "HANS overall accuracy")
    _write_svg(figures_dir, "overall_accuracy", svg)


def _plot_grouped(summaries: list[dict[str, Any]], figures_dir: Path) -> None:
    metrics = (
        ("entailment_accuracy", "Entailment"),
        ("non_entailment_accuracy", "Non-entailment"),
        ("lexical_overlap_accuracy", "Lexical overlap"),
        ("subsequence_accuracy", "Subsequence"),
        ("constituent_accuracy", "Constituent"),
    )
    width, height = 1320, 650
    left, right, top, bottom = 85, 35, 115, 150
    chart_width = width - left - right
    chart_height = height - top - bottom
    group_slot = chart_width / len(metrics)
    gap = 4.0
    bar_width = min(46.0, (group_slot - 24.0) / len(summaries) - gap)
    cluster_width = len(summaries) * (bar_width + gap) - gap
    colors = COLORS[: len(summaries)]
    parts = [
        '<text x="42" y="43" class="title">Accuracy decomposition</text>',
        '<text x="42" y="69" class="subtitle">Labels and HANS heuristics · full benchmark</text>',
    ]
    for tick in range(0, 101, 20):
        y = top + chart_height * (1.0 - tick / 100.0)
        parts.extend(
            [
                f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#E2E8F0"/>',
                f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="axis">{tick}</text>',
            ]
        )
    for metric_index, (field, label) in enumerate(metrics):
        center = left + group_slot * (metric_index + 0.5)
        start = center - cluster_width / 2
        for run_index, (row, color) in enumerate(
            zip(summaries, colors, strict=True)
        ):
            x = start + run_index * (bar_width + gap)
            if row[field] is None:
                parts.append(
                    f'<text x="{x+bar_width/2:.1f}" y="{top+chart_height-8:.1f}" text-anchor="middle" class="axis">NA</text>'
                )
                continue
            value = 100.0 * row[field]
            bar_height = chart_height * value / 100.0
            y = top + chart_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="4" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{center:.1f}" y="{top+chart_height+31:.1f}" text-anchor="middle" class="label">{html.escape(label)}</text>'
        )
    legend_y = height - 52
    legend_width = 235
    legend_start = (width - legend_width * len(summaries)) / 2
    for index, (row, color) in enumerate(
        zip(summaries, colors, strict=True)
    ):
        x = legend_start + index * legend_width
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{legend_y-12}" width="18" height="18" rx="4" fill="{color}"/>',
                f'<text x="{x+27:.1f}" y="{legend_y+2}" class="axis">{html.escape(row["name"])}</text>',
            ]
        )
    parts.append(
        f'<text x="22" y="{top+chart_height/2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 22 {top+chart_height/2:.1f})">Accuracy (%)</text>'
    )
    svg = _svg_document(width, height, "\n".join(parts), "HANS label and heuristic accuracy")
    _write_svg(figures_dir, "label_heuristic_accuracy", svg)


def _heat_color(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "#E2E8F0"
    stops = (
        (0.0, (68, 1, 84)),
        (25.0, (59, 82, 139)),
        (50.0, (33, 145, 140)),
        (75.0, (94, 201, 98)),
        (100.0, (253, 231, 37)),
    )
    bounded = min(100.0, max(0.0, value))
    for (lower, low_rgb), (upper, high_rgb) in zip(stops, stops[1:], strict=True):
        if bounded <= upper:
            ratio = (bounded - lower) / (upper - lower)
            rgb = tuple(round(a + ratio * (b - a)) for a, b in zip(low_rgb, high_rgb, strict=True))
            return "#{:02X}{:02X}{:02X}".format(*rgb)
    return "#FDE725"


def _plot_subcases(
    summaries: list[dict[str, Any]], groups: list[dict[str, Any]], figures_dir: Path
) -> None:
    subcases = sorted(
        {row["group"] for row in groups if row["group_type"] == "subcase"}
    )
    lookup = {
        (row["run_name"], row["group"]): row["accuracy"]
        for row in groups
        if row["group_type"] == "subcase"
    }
    cell_width, cell_height = 45, 52
    left, top, right, bottom = 180, 105, 75, 195
    width = max(1000, left + len(subcases) * cell_width + right)
    height = top + len(summaries) * cell_height + bottom
    parts = [
        '<text x="42" y="43" class="title">HANS subcase accuracy heatmap</text>',
        '<text x="42" y="69" class="subtitle">Fine-grained failure profile · 0–100% accuracy</text>',
    ]
    for row_index, summary in enumerate(summaries):
        y = top + row_index * cell_height
        parts.append(
            f'<text x="{left-14}" y="{y+cell_height/2+5:.1f}" text-anchor="end" class="label">{html.escape(summary["name"])}</text>'
        )
        for column_index, subcase in enumerate(subcases):
            raw = lookup.get((summary["run_name"], subcase))
            value = None if raw is None else 100.0 * raw
            x = left + column_index * cell_width
            parts.append(
                f'<rect x="{x+1}" y="{y+1}" width="{cell_width-3}" height="{cell_height-3}" rx="5" fill="{_heat_color(value)}"><title>{html.escape(summary["name"])} · {html.escape(subcase)}: {"NA" if value is None else f"{value:.2f}%"}</title></rect>'
            )
    label_y = top + len(summaries) * cell_height + 13
    for column_index, subcase in enumerate(subcases):
        x = left + column_index * cell_width + cell_width / 2
        label = html.escape(subcase.replace("_", " "))
        parts.append(
            f'<text x="{x:.1f}" y="{label_y}" class="axis" text-anchor="end" transform="rotate(-55 {x:.1f} {label_y})">{label}</text>'
        )
    legend_x = width - 250
    legend_y = 55
    for index in range(101):
        parts.append(
            f'<rect x="{legend_x+index*1.8:.1f}" y="{legend_y}" width="2" height="12" fill="{_heat_color(float(index))}"/>'
        )
    parts.extend(
        [
            f'<text x="{legend_x}" y="{legend_y-7}" class="axis">0%</text>',
            f'<text x="{legend_x+182}" y="{legend_y-7}" text-anchor="end" class="axis">100%</text>',
        ]
    )
    svg = _svg_document(width, height, "\n".join(parts), "HANS subcase accuracy heatmap")
    _write_svg(figures_dir, "subcase_accuracy_heatmap", svg)


def _percent(value: Any) -> str:
    return "—" if value is None else f"{100.0 * float(value):.2f}%"


def _number(value: Any, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def _markdown_table(summaries: list[dict[str, Any]]) -> str:
    reference = next(
        (row for row in summaries if row["method"] == "zero_shot"),
        next(row for row in summaries if row["method"] == "base"),
    )
    reference_accuracy = reference["overall_accuracy"]
    header = (
        f"| Method | N | Acquired | Overall | Δ vs {reference['name']} | Entail. | "
        "Non-entail. | Lexical | Subsequence | Constituent | Wall time (min) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for row in summaries:
        delta = 100.0 * (row["overall_accuracy"] - reference_accuracy)
        rows.append(
            "| {name} | {num_examples:,} | {num_acquired:,} | {overall} | {delta:+.2f} pp | "
            "{entailment} | {non_entailment} | {lexical} | {subsequence} | "
            "{constituent} | {wall} |".format(
                name=row["name"],
                num_examples=row["num_examples"],
                num_acquired=int(row["num_acquired"] or 0),
                overall=_percent(row["overall_accuracy"]),
                delta=delta,
                entailment=_percent(row["entailment_accuracy"]),
                non_entailment=_percent(row["non_entailment_accuracy"]),
                lexical=_percent(row["lexical_overlap_accuracy"]),
                subsequence=_percent(row["subsequence_accuracy"]),
                constituent=_percent(row["constituent_accuracy"]),
                wall=_number(
                    None
                    if row["wall_clock_seconds"] is None
                    else row["wall_clock_seconds"] / 60.0
                ),
            )
        )
    return "\n".join(rows)


def _build_markdown(summaries: list[dict[str, Any]], generated_at: str) -> str:
    best = max(summaries, key=lambda row: row["overall_accuracy"])
    by_method = {row["method"]: row for row in summaries}
    reproducibility_note = ""
    if "zero_shot" in by_method:
        base_gap = 100.0 * (
            by_method["base"]["overall_accuracy"]
            - by_method["zero_shot"]["overall_accuracy"]
        )
        reproducibility_note = (
            " Base differs from the standalone zero-shot evaluation by "
            f"**{base_gap:+.4f} percentage points**; this is a reproducibility "
            "check because both use the same unadapted checkpoint."
        )
    metadata_rows = []
    for row in summaries:
        commit = row["git_commit"][:12] if row["git_commit"] else "—"
        metadata_rows.append(
            f"| {row['name']} | `{row['run_name']}` | {row['seed']} | "
            f"`{row['student_model'] or '—'}` | `{row['device'] or '—'}` | "
            f"{row['gpu_name'] or '—'} | `{commit}` |"
        )
    return f"""# HANS acquisition baselines

Generated automatically at `{generated_at}`. This report contains full HANS runs only; smoke artifacts are excluded.

## Main comparison

{_markdown_table(summaries)}

The highest observed single-seed HANS accuracy is **{best['name']}** at **{_percent(best['overall_accuracy'])}**.{reproducibility_note}

![Overall HANS accuracy](figures/overall_accuracy.svg)

## Breakdown by label and heuristic

![Accuracy by label and heuristic](figures/label_heuristic_accuracy.svg)

## Breakdown by HANS subcase

![HANS subcase heatmap](figures/subcase_accuracy_heatmap.svg)

## Run provenance

| Method | Run directory | Seed | Student | Device | GPU | Git commit |
|---|---|---:|---|---|---|---|
{chr(10).join(metadata_rows)}

## Reusable artifacts

- `data/run_summary.csv`: one row per method, suitable for the paper's main table.
- `data/group_metrics_long.csv`: tidy overall/label/heuristic/subcase metrics with counts.
- `data/results.json`: lossless combined summary and provenance.
- `figures/*.svg`: dependency-free, editable vector figures suitable for the paper.
- Source predictions and original metrics remain under `outputs/<run_name>/`.

## Interpretation guardrails

- These are single-seed results (`seed=42`), so the report intentionally shows no confidence intervals.
- Base is the unadapted checkpoint and is the default reference for Random and Uncertainty. When `--include-standalone-zero-shot` is used, its agreement with Base is an additional reproducibility check.
- Random and Uncertainty both use acquisition budget 32 and the same LoRA recipe.
- Do not replace missing values with zero. The reporter fails on incomplete artifacts or disagreement between `metrics.json` and recomputed prediction-level metrics.
"""


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs).resolve()
    report_dir = Path(args.report_dir).resolve()
    run_specs = (
        (ZERO_SHOT_RUN, *BASELINE_RUNS)
        if args.include_standalone_zero_shot
        else BASELINE_RUNS
    )
    summaries: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for spec in run_specs:
        summary, groups = collect_run(outputs_root, spec)
        summaries.append(summary)
        group_rows.extend(groups)

    invalid_counts = [
        (row["run_name"], row["num_examples"])
        for row in summaries
        if row["num_examples"] != args.expected_num_examples
    ]
    if invalid_counts:
        raise ValueError(
            f"Expected {args.expected_num_examples} HANS examples per run; "
            f"observed {invalid_counts}"
        )
    id_digests = {row["prediction_id_sha256"] for row in summaries}
    if len(id_digests) != 1:
        raise ValueError("Runs do not evaluate the same set of HANS example IDs")

    report_dir.mkdir(parents=True, exist_ok=True)
    data_dir = report_dir / "data"
    figures_dir = report_dir / "figures"
    _write_csv(data_dir / "run_summary.csv", summaries, SUMMARY_FIELDS)
    group_fields = (
        "name",
        "run_name",
        "method",
        "group_type",
        "group",
        "num_examples",
        "num_correct",
        "accuracy",
    )
    _write_csv(data_dir / "group_metrics_long.csv", group_rows, group_fields)
    generated_at = datetime.now(timezone.utc).isoformat()
    write_json(
        data_dir / "results.json",
        {
            "generated_at_utc": generated_at,
            "runs": summaries,
            "group_metrics": group_rows,
            "run_specification": [asdict(spec) for spec in run_specs],
        },
    )
    _plot_overall(summaries, figures_dir)
    _plot_grouped(summaries, figures_dir)
    _plot_subcases(summaries, group_rows, figures_dir)
    report_path = report_dir / "HANS_BASELINE_REPORT.md"
    _atomic_text_write(report_path, _build_markdown(summaries, generated_at))
    print(f"Wrote report: {report_path}")
    print(f"Wrote reusable data: {data_dir}")
    print(f"Wrote figures: {figures_dir}")


if __name__ == "__main__":
    main()
