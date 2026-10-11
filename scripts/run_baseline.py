#!/usr/bin/env python3
"""Run one simple acquisition baseline and evaluate it on HANS."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import BaseBaseline, RandomBaseline, UncertaintyBaseline
from src.baselines.base import Baseline
from src.data.hans import load_hans
from src.data.mnli import load_mnli
from src.evaluation.hans import evaluate_hans
from src.models.lora_training import LoraTrainingConfig, train_lora
from src.models.qwen_nli import QwenNLIStudent
from src.utils.config import load_yaml_with_extends, require_mapping
from src.utils.experiment_logger import (
    ExperimentLogger,
    make_final_metrics,
    make_round_record,
)
from src.utils.seed import set_deterministic_seed

BASELINES: dict[str, type[Baseline]] = {
    "base": BaseBaseline,
    "random": RandomBaseline,
    "uncertainty": UncertaintyBaseline,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(BASELINES))
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace logger artifacts in an existing run directory",
    )
    return parser.parse_args()


def _optional_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer or null")
    return value


def _build_student(model_config: dict[str, Any]) -> QwenNLIStudent:
    return QwenNLIStudent(
        name_or_path=str(model_config.get("name_or_path", "Qwen/Qwen3-1.7B")),
        adapter_path=model_config.get("adapter_path"),
        device_map=model_config.get("device_map", "auto"),
        dtype=str(model_config.get("dtype", "bfloat16")),
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        local_files_only=bool(model_config.get("local_files_only", False)),
        max_length=int(model_config.get("max_length", 512)),
    )


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
    config = load_yaml_with_extends(args.config)
    configured_method = str(config.get("method", ""))
    if configured_method != args.method:
        raise ValueError(
            f"CLI method {args.method!r} does not match config method "
            f"{configured_method!r}"
        )

    run_name = str(config.get("run_name", "")).strip()
    if not run_name:
        raise ValueError("run_name must be a non-empty string")
    seed = int(config.get("seed", 42))
    budget = int(config.get("acquisition_budget", 0))
    if budget <= 0:
        raise ValueError("acquisition_budget must be positive")

    model_config = require_mapping(config, "model")
    pool_config = require_mapping(config, "pool")
    hans_config = require_mapping(config, "hans")
    training_mapping = require_mapping(config, "training")
    training_config = LoraTrainingConfig.from_mapping(training_mapping)

    output_root = Path(str(config.get("output_dir", "outputs")))
    run_directory = output_root / run_name
    logger = ExperimentLogger(
        run_directory,
        config,
        overwrite=args.overwrite,
    )
    logger.save_config()
    logger.initialize_rounds()

    set_deterministic_seed(seed)
    baseline = BASELINES[args.method]()
    student = _build_student(model_config)
    device, gpu_name = _device_metadata(student)
    logger.save_run_metadata(
        {
            "run_name": run_name,
            "method": args.method,
            "seed": seed,
            "student_model": str(model_config.get("name_or_path")),
            "starting_checkpoint": (
                model_config.get("adapter_path")
                or model_config.get("name_or_path")
            ),
            "acquisition_budget": budget,
            "config_file": str(Path(args.config).resolve()),
            "git_commit": _git_commit(),
            "device": device,
            "gpu_name": gpu_name,
        }
    )

    pool = []
    if baseline.trains_after_acquisition:
        pool = load_mnli(
            split=str(pool_config.get("split", "train")),
            dataset_name=str(pool_config.get("name", "nyu-mll/glue")),
            dataset_config=str(pool_config.get("config", "mnli")),
            cache_dir=pool_config.get("cache_dir"),
            max_examples=_optional_int(
                pool_config.get("max_examples"), "pool.max_examples"
            ),
            drop_unlabeled=True,
        )

    wall_start = time.perf_counter()
    gpu_timer = _start_gpu_timer(student)
    student.progress_label = f"{args.method}: acquisition scoring"
    selected = baseline.acquire(
        pool=pool,
        budget=budget,
        seed=seed,
        student=student,
        batch_size=int(model_config.get("batch_size", 4)),
    )
    selected_ids = [example.id for example in selected]
    logger.save_selected_ids(args.method, budget, selected_ids)

    train_loss = None
    if baseline.trains_after_acquisition:
        train_loss = train_lora(
            student,
            selected,
            training_config,
            adapter_output_dir=run_directory / "adapter",
            seed=seed,
        )

    hans_examples = load_hans(
        split=str(hans_config.get("split", "validation")),
        dataset_name=str(hans_config.get("name", "hans")),
        data_file=hans_config.get("data_file"),
        cache_dir=hans_config.get("cache_dir"),
        max_examples=_optional_int(
            hans_config.get("max_examples"), "hans.max_examples"
        ),
    )
    student.progress_label = f"{args.method}: HANS evaluation"
    records, metrics = evaluate_hans(
        student,
        hans_examples,
        batch_size=int(model_config.get("batch_size", 4)),
    )
    gpu_seconds = _stop_gpu_timer(student, gpu_timer)
    wall_clock_seconds = time.perf_counter() - wall_start

    final_metrics = make_final_metrics(
        method=args.method,
        seed=seed,
        acquisition_budget=budget,
        num_acquired=len(selected),
        evaluation_metrics=metrics,
    )
    round_record = make_round_record(
        round_number=0 if args.method == "base" else 1,
        method=args.method,
        seed=seed,
        acquisition_budget=budget,
        num_acquired=len(selected),
        train_loss=train_loss,
        train_examples=len(selected) if baseline.trains_after_acquisition else None,
        evaluation_metrics=metrics,
        gpu_seconds=gpu_seconds,
        wall_clock_seconds=wall_clock_seconds,
    )
    logger.save_predictions(records)
    logger.save_metrics(final_metrics)
    logger.log_round(round_record)

    print(json.dumps(final_metrics, indent=2, sort_keys=True))
    print(f"Wrote baseline artifacts to {run_directory}")


if __name__ == "__main__":
    main()
