#!/usr/bin/env python3
"""Print a compact table from local baseline metrics files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs", help="Run root directory")
    return parser.parse_args()


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    root = Path(parse_args().outputs)
    rows = []
    for path in sorted(root.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if "method" not in metrics:
            continue
        rows.append(
            [
                metrics.get("method"),
                metrics.get("seed"),
                metrics.get("acquisition_budget"),
                metrics.get("hans_accuracy"),
                metrics.get("lexical_overlap_accuracy"),
                metrics.get("subsequence_accuracy"),
                metrics.get("constituent_accuracy"),
            ]
        )

    headers = ["method", "seed", "budget", "HANS", "lexical", "subsequence", "constituent"]
    table = [headers] + [[_format(value) for value in row] for row in rows]
    widths = [max(len(row[index]) for row in table) for index in range(len(headers))]
    for row_index, row in enumerate(table):
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
        if row_index == 0:
            print("-+-".join("-" * width for width in widths))


if __name__ == "__main__":
    main()
