#!/usr/bin/env python3
"""Freeze HANS roles before inference and evaluate only adaptive HANS roles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.hans_manifest import (
    FrozenHansManifest,
    build_frozen_hans_manifest,
    validate_frozen_hans_manifest,
)
from src.data.hans import load_hans
from src.evaluation.hans import evaluate_hans
from src.models.qwen_nli import QwenNLIStudent
from src.utils.artifacts import read_json
from src.utils.checkpoint import checkpoint_descriptor
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.io import write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    config = load_yaml_with_extends(args.config)
    benchmark = require_mapping(config, "stage2_benchmark")
    hans = require_mapping(config, "hans")
    model = require_mapping(config, "model")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output / "hans_frozen_manifest.json",
        "predictions": output / "predictions.jsonl",
        "sealed": output / "sealed_final_test.jsonl",
        "metadata": output / "preparation.json",
    }
    if any(path.exists() for path in paths.values()) and not args.overwrite:
        raise FileExistsError("Stage 2 preparation artifacts already exist")

    examples = load_hans(
        split=str(hans.get("split", "validation")),
        dataset_name=str(hans.get("name", "hans")),
        data_file=hans.get("data_file"),
        cache_dir=hans.get("cache_dir"),
        max_examples=_optional_int(hans.get("max_examples")),
    )
    rows = [asdict(item) for item in examples]
    manifest = build_frozen_hans_manifest(
        rows,
        split_seed=int(benchmark["manifest_seed"]),
        discovery_fraction=float(hans["discovery_fraction"]),
        verification_fraction=float(hans["verification_fraction"]),
        intervention_fraction=float(hans["intervention_fraction"]),
        gain_eval_fraction=float(hans["gain_eval_fraction"]),
    )
    if paths["manifest"].exists():
        existing = FrozenHansManifest.from_mapping(read_json(paths["manifest"]))
        if existing != manifest:
            raise ValueError("Refusing to replace a different frozen HANS manifest")
    final_ids = set(manifest.final_test_ids)
    adaptive = [item for item in examples if item.id not in final_ids]
    final = [item for item in examples if item.id in final_ids]
    if len(adaptive) + len(final) != len(examples):
        raise RuntimeError("Frozen split did not partition HANS")

    set_deterministic_seed(int(benchmark.get("preparation_seed", 42)))
    student = QwenNLIStudent(
        name_or_path=str(model["name_or_path"]),
        adapter_path=model.get("adapter_path"),
        device_map=model.get("device_map", "auto"),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        local_files_only=bool(model.get("local_files_only", False)),
        max_length=int(model.get("max_length", 512)),
    )
    predictions, _ = evaluate_hans(
        student, adaptive, batch_size=int(model.get("batch_size", 4))
    )
    predicted = {str(row["id"]): row for row in predictions}
    # Preserve the complete ID universe for deterministic split reconstruction,
    # but do not expose final-test inputs, labels, predictions, or metrics.
    source = [
        {"id": item.id} if item.id in final_ids else predicted[item.id]
        for item in examples
    ]
    validate_frozen_hans_manifest(manifest, rows)
    write_json(paths["manifest"], manifest.to_dict())
    write_jsonl(paths["predictions"], source)
    write_jsonl(paths["sealed"], [asdict(item) for item in final])
    write_json(
        paths["metadata"],
        {
            "git_commit": _git_commit(),
            "initial_checkpoint": checkpoint_descriptor(model),
            "manifest_hash": manifest.manifest_hash,
            "num_adaptive": len(adaptive),
            "num_final_test": len(final),
            "final_test_evaluated": False,
            "final_test_fields_in_predictions_artifact": ["id"],
        },
    )
    print(json.dumps(read_json(paths["metadata"]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
