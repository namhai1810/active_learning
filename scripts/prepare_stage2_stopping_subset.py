#!/usr/bin/env python3
"""Freeze a prediction-blind stopping subset inside Stage 2 gain_eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.hans_manifest import FrozenHansManifest
from src.benchmark.stopping_subset import build_stopping_subset_manifest
from src.utils.artifacts import read_json, read_jsonl
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples-per-subcase", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    benchmark = require_mapping(config, "stage2_benchmark")
    frozen_path = Path(str(benchmark["frozen_manifest_path"])).resolve()
    frozen = FrozenHansManifest.from_mapping(read_json(frozen_path))
    prediction_rows = read_jsonl(frozen_path.parent / "predictions.jsonl")
    index = {str(row["id"]): row for row in prediction_rows}
    gain_rows = [index[item] for item in frozen.gain_eval_ids]
    if any("subcase" not in row or "gold" not in row for row in gain_rows):
        raise ValueError("Shared predictions do not contain complete gain_eval metadata")
    manifest = build_stopping_subset_manifest(
        gain_rows,
        frozen.gain_eval_ids,
        frozen_manifest_hash=frozen.manifest_hash,
        examples_per_subcase=args.examples_per_subcase,
    )
    output = Path(args.output).resolve()
    if output.is_file():
        existing = read_json(output)
        if existing != manifest:
            raise ValueError(
                f"Refusing to replace a different frozen stopping subset: {output}"
            )
        print(f"[SKIP] Frozen stopping subset already matches: {output}")
        return
    write_json(output, manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "stopping_eval": len(manifest["stopping_eval_ids"]),
                "gain_report": len(manifest["gain_report_ids"]),
                "subcases": manifest["num_subcases"],
                "partition_hash": manifest["partition_hash"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
