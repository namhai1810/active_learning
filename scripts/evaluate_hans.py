#!/usr/bin/env python3
"""Evaluate a Qwen NLI student on HANS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.hans import load_hans
from src.evaluation.hans import evaluate_hans
from src.models.qwen_nli import QwenNLIStudent
from src.utils.config import load_yaml, require_mapping
from src.utils.io import _atomic_text_write, write_json, write_jsonl
from src.utils.seed import set_deterministic_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    return parser.parse_args()


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _device_metadata(student: QwenNLIStudent) -> tuple[str, str | None]:
    device = student.input_device
    gpu_name = None
    if device.type == "cuda":
        try:
            gpu_name = student._torch.cuda.get_device_name(device)
        except (RuntimeError, ValueError):
            gpu_name = None
    return str(device), gpu_name


def _start_gpu_timer(student: QwenNLIStudent) -> tuple[Any, Any] | None:
    device = student.input_device
    if device.type != "cuda":
        return None
    torch = student._torch
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    return start, end


def _stop_gpu_timer(
    student: QwenNLIStudent, timer: tuple[Any, Any] | None
) -> float | None:
    if timer is None:
        return None
    start, end = timer
    torch = student._torch
    with torch.cuda.device(student.input_device):
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / 1000.0


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_config = require_mapping(config, "dataset")
    model_config = require_mapping(config, "model")

    run_name = str(config.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("run_name must be a non-empty string")
    output_root = Path(str(config.get("output_dir", "outputs")))
    run_directory = output_root / run_name
    predictions_path = run_directory / "predictions.jsonl"
    metrics_path = run_directory / "metrics.json"
    overwrite = bool(config.get("overwrite", False))
    existing = [path for path in (predictions_path, metrics_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing artifacts: {joined}. "
            "Set overwrite: true explicitly to replace them."
        )

    seed = int(config.get("seed", 42))
    set_deterministic_seed(seed)
    examples = load_hans(
        split=str(dataset_config.get("split", "validation")),
        dataset_name=str(dataset_config.get("name", "hans")),
        data_file=dataset_config.get("data_file"),
        cache_dir=dataset_config.get("cache_dir"),
        max_examples=_optional_int(
            dataset_config.get("max_examples"), "dataset.max_examples"
        ),
    )
    student = QwenNLIStudent(
        name_or_path=str(model_config.get("name_or_path", "Qwen/Qwen3-1.7B")),
        adapter_path=model_config.get("adapter_path"),
        device_map=model_config.get("device_map", "auto"),
        dtype=str(model_config.get("dtype", "bfloat16")),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        local_files_only=bool(model_config.get("local_files_only", False)),
        max_length=int(model_config.get("max_length", 512)),
    )
    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    gpu_timer = _start_gpu_timer(student)
    records, metrics = evaluate_hans(
        student,
        examples,
        batch_size=int(model_config.get("batch_size", 4)),
    )
    gpu_seconds = _stop_gpu_timer(student, gpu_timer)
    wall_clock_seconds = time.perf_counter() - wall_start
    device, gpu_name = _device_metadata(student)

    _atomic_text_write(
        run_directory / "config.yaml",
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    )
    write_jsonl(predictions_path, records)
    write_json(metrics_path, metrics)
    write_json(
        run_directory / "run.json",
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_file": str(Path(args.config).resolve()),
            "device": device,
            "git_commit": _git_commit(),
            "gpu_name": gpu_name,
            "gpu_seconds": gpu_seconds,
            "method": "zero_shot",
            "num_examples": len(records),
            "run_name": run_name,
            "seed": seed,
            "started_at_utc": started_at.isoformat(),
            "student_model": str(model_config.get("name_or_path")),
            "task": "hans_zero_shot",
            "wall_clock_seconds": wall_clock_seconds,
        },
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote {predictions_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
