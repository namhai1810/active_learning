#!/usr/bin/env python3
"""Compare CE and pairwise ours_verified runs on exactly the same selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.artifacts import read_json
from src.utils.io import _atomic_text_write, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-result", required=True)
    parser.add_argument("--ce-result", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _selected_ids(result_path: Path, result: dict[str, Any]) -> list[str]:
    selected_path = result_path.parent / str(
        result.get("selected_ids_file", "selected_ids.json")
    )
    payload = read_json(selected_path)
    ids = payload.get("ids")
    if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
        raise ValueError(f"Invalid selected IDs artifact: {selected_path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Selected IDs are not unique: {selected_path}")
    return ids


def _metric(result: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = result
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Missing metric {'.'.join(path)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Metric {'.'.join(path)} is not numeric")
    return float(value)


def main() -> None:
    args = parse_args()
    paths = {
        "pairwise": Path(args.pairwise_result).resolve(),
        "ce": Path(args.ce_result).resolve(),
    }
    results = {name: read_json(path) for name, path in paths.items()}
    for name, result in results.items():
        if result.get("method") != "ours_verified":
            raise ValueError(f"{name} result is not ours_verified")
        if result.get("ours_objective") != name:
            raise ValueError(
                f"Expected {name} objective, got {result.get('ours_objective')!r}"
            )
        if result.get("run_status") != "complete":
            raise ValueError(f"{name} run is not complete")

    comparability_fields = ("seed", "requested_budget", "manifest_hash")
    for field in comparability_fields:
        if results["pairwise"].get(field) != results["ce"].get(field):
            raise ValueError(f"Objective runs differ on {field}")
    selected = {
        name: _selected_ids(paths[name], result)
        for name, result in results.items()
    }
    same_ordered_ids = selected["pairwise"] == selected["ce"]
    if not same_ordered_ids:
        raise ValueError(
            "CE and pairwise runs did not train on the same selected IDs in the "
            "same order; refusing to report this as a controlled ablation"
        )

    metric_paths = {
        "HANS overall": ("hans_final", "overall_accuracy"),
        "HANS entailment": ("hans_final", "entailment_accuracy"),
        "HANS non-entailment": ("hans_final", "non_entailment_accuracy"),
        "MNLI matched": ("mnli", "matched", "accuracy"),
        "MNLI mismatched": ("mnli", "mismatched", "accuracy"),
    }
    rows = []
    for label, metric_path in metric_paths.items():
        pairwise = _metric(results["pairwise"], metric_path)
        ce = _metric(results["ce"], metric_path)
        rows.append(
            {
                "metric": label,
                "pairwise": pairwise,
                "ce": ce,
                "pairwise_minus_ce": pairwise - ce,
            }
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": results["pairwise"]["seed"],
        "budget": results["pairwise"]["requested_budget"],
        "manifest_hash": results["pairwise"]["manifest_hash"],
        "same_selected_ids_in_same_order": same_ordered_ids,
        "selected_example_count": len(selected["pairwise"]),
        "pairwise_result": str(paths["pairwise"]),
        "ce_result": str(paths["ce"]),
        "metrics": rows,
    }
    write_json(output / "objective_ablation.json", payload)
    lines = [
        "# Ours verified: training-objective ablation",
        "",
        f"Seed: `{payload['seed']}`; budget: `{payload['budget']}`.",
        "",
        "Controlled comparison check: **pass** — CE and pairwise training used "
        f"the same {payload['selected_example_count']} example IDs in the same order.",
        "",
        "| Metric | Pairwise | CE | Pairwise − CE |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {pairwise:.4%} | {ce:.4%} | {delta:+.4%} |".format(
                metric=row["metric"],
                pairwise=row["pairwise"],
                ce=row["ce"],
                delta=row["pairwise_minus_ce"],
            )
        )
    lines.extend(
        [
            "",
            "Positive deltas favor pairwise training; negative deltas favor CE. "
            "This report isolates the training objective, not pool generation or "
            "acquisition selection.",
        ]
    )
    _atomic_text_write(output / "OBJECTIVE_ABLATION.md", "\n".join(lines) + "\n")
    print(f"Wrote objective ablation report to {output}")


if __name__ == "__main__":
    main()
