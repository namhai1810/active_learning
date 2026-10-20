#!/usr/bin/env python3
"""Append both fair v7 branches to the preserved seed-42 comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_COMPARISON = (
    PROJECT_ROOT
    / "reports"
    / "stage2_seed42_comparison_with_v6"
    / "stage2_comparison.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-output", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _metric(result: dict, *keys: str) -> float:
    value = result
    for key in keys:
        value = value[key]
    return float(value)


def main() -> None:
    args = parse_args()
    v7_output = Path(args.v7_output).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with BASE_COMPARISON.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not any(row["method"] == "ours_verified_v6_closed_loop" for row in rows):
        raise ValueError("The preserved v6 diagnostic row is missing")

    branches = (
        ("continuation", "complete_fair_stateful_continuation"),
        ("consolidation", "complete_fair_fresh_consolidation_primary"),
    )
    for branch, status in branches:
        result = _read_json(v7_output / "branches" / branch / "result.json")
        if result.get("selected_example_ids") != _read_json(
            v7_output / "comparison.json"
        ).get("shared_selected_example_ids"):
            raise ValueError(f"{branch} does not use the shared selected IDs")
        rows.append(
            {
                "method": str(result["method"]),
                "seed": str(result["seed"]),
                "configured_budget": str(result["requested_budget"]),
                "hans_accuracy": str(_metric(result, "hans_final", "overall_accuracy")),
                "hans_non_entailment_accuracy": str(
                    _metric(result, "hans_final", "non_entailment_accuracy")
                ),
                "mnli_matched_accuracy": str(
                    _metric(result, "mnli", "matched", "accuracy")
                ),
                "mnli_mismatched_accuracy": str(
                    _metric(result, "mnli", "mismatched", "accuracy")
                ),
                "fixed_budget_comparable": str(
                    bool(result["fixed_budget_comparable"])
                ).lower(),
                "status": status,
            }
        )

    fieldnames = list(rows[0])
    csv_path = output / "stage2_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def display(row: dict) -> str:
        interpretation = row["status"].replace("_", " ")
        return (
            f"| {row['method']} | {float(row['hans_accuracy']):.6f} | "
            f"{float(row['hans_non_entailment_accuracy']):.6f} | "
            f"{float(row['mnli_matched_accuracy']):.6f} | "
            f"{float(row['mnli_mismatched_accuracy']):.6f} | {interpretation} |"
        )

    report = [
        "# Stage 2 seed-42 comparison including fair v7 closed loop",
        "",
        "All trained rows use budget 128. Both v7 rows share exactly the same closed-loop acquisition trace and 128 selected examples.",
        "",
        "| Method | HANS accuracy | HANS non-entailment | MNLI matched | MNLI mismatched | Status |",
        "|---|---:|---:|---:|---:|---|",
        *(display(row) for row in rows),
        "",
        "## Fairness interpretation",
        "",
        "- `ours_verified_v7_continuation` carries LoRA weights and AdamW moments through four 32-example rounds (four optimizer steps total).",
        "- `ours_verified_v7_consolidation` starts from the original base checkpoint and trains once on all 128 selected examples (four optimizer steps total). This is the primary apples-to-apples acquisition comparison against the old one-pass baselines.",
        "- The continuation checkpoint controls discovery and verification in the shared closed loop; consolidation is trained only after acquisition is frozen.",
        "- `ours_verified_v6_closed_loop` remains in the table as a preserved diagnostic and must not be ranked as a valid fixed-budget result.",
        "- Random/uncertainty MNLI are cross-source baselines and do not have the same data access as HANS-guided synthetic methods.",
        "",
    ]
    (output / "STAGE2_COMPARISON.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output / 'STAGE2_COMPARISON.md'}")


if __name__ == "__main__":
    main()

