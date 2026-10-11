"""Independent targeted pilot execution with injectable model callbacks."""

from __future__ import annotations

import gc
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.gain.estimation import risk_reduction
from src.gain.types import PilotGainObservation
from src.hypotheses.types import FailureHypothesis
from src.intervention.hans_oracle_targeting import (
    pilot_seed,
    select_gain_eval_batches,
    select_intervention_examples,
    to_mnli_training_examples,
)


@dataclass(frozen=True)
class PilotRunConfig:
    m: int
    pilot_train_size: int
    gain_eval_size: int
    seed: int
    M: float = 1.0
    tolerance: float = 1e-12
    non_entailment_label: str = "contradiction"

    def __post_init__(self) -> None:
        if self.m <= 0 or self.pilot_train_size <= 0 or self.gain_eval_size <= 0:
            raise ValueError("m, pilot_train_size and gain_eval_size must be positive")
        if not math.isfinite(self.M) or self.M <= 0:
            raise ValueError("M must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        if self.non_entailment_label not in {"neutral", "contradiction"}:
            raise ValueError("non_entailment_label must be neutral or contradiction")


StudentFactory = Callable[[int], Any]
TrainCallback = Callable[[Any, list[Any], Path, int], float]
RiskCallback = Callable[[Any, Sequence[Mapping[str, Any]]], float]


def _start_gpu_timer(student: Any) -> tuple[Any, Any] | None:
    torch = getattr(student, "_torch", None)
    device = getattr(student, "input_device", None)
    if torch is None or device is None or getattr(device, "type", None) != "cuda":
        return None
    with torch.cuda.device(device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    return start, end


def _stop_gpu_timer(student: Any, timer: tuple[Any, Any] | None) -> float | None:
    if timer is None:
        return None
    start, end = timer
    torch = student._torch
    with torch.cuda.device(student.input_device):
        end.record()
        end.synchronize()
    return start.elapsed_time(end) / 1000.0


def run_targeted_pilots(
    *,
    hypothesis: FailureHypothesis,
    intervention_records: Sequence[Mapping[str, Any]],
    gain_eval_records: Sequence[Mapping[str, Any]],
    config: PilotRunConfig,
    starting_checkpoint: Mapping[str, Any],
    training_config: Mapping[str, Any],
    pilot_root: str | Path,
    student_factory: StudentFactory,
    train_callback: TrainCallback,
    risk_callback: RiskCallback,
) -> list[PilotGainObservation]:
    """Run pilots independently; every loop calls the same checkpoint factory."""
    eval_batches = select_gain_eval_batches(
        hypothesis,
        gain_eval_records,
        m=config.m,
        gain_eval_size=config.gain_eval_size,
        seed=config.seed,
    )
    observations = []
    for index in range(config.m):
        seed = pilot_seed(config.seed, hypothesis.id, index)
        train_records = select_intervention_examples(
            hypothesis,
            intervention_records,
            config.pilot_train_size,
            seed,
        )
        eval_records = eval_batches[index]
        if not train_records or not eval_records:
            continue

        pilot_dir = Path(pilot_root) / hypothesis.id / f"pilot_{index:03d}"
        adapter_dir = pilot_dir / "adapter"
        student = student_factory(seed)
        wall_start = time.perf_counter()
        gpu_timer = _start_gpu_timer(student)
        risk_before = risk_callback(student, eval_records)
        train_examples = to_mnli_training_examples(
            train_records, config.non_entailment_label
        )
        train_loss = train_callback(student, train_examples, adapter_dir, seed)
        risk_after = risk_callback(student, eval_records)
        gpu_seconds = _stop_gpu_timer(student, gpu_timer)
        wall_seconds = time.perf_counter() - wall_start
        gain = risk_reduction(
            risk_before,
            risk_after,
            M=config.M,
            tolerance=config.tolerance,
        )
        observations.append(
            PilotGainObservation(
                hypothesis_id=hypothesis.id,
                pilot_index=index,
                pilot_seed=seed,
                starting_checkpoint=dict(starting_checkpoint),
                train_ids=[str(record["id"]) for record in train_records],
                eval_ids=[str(record["id"]) for record in eval_records],
                training_config=dict(training_config),
                risk_before=risk_before,
                risk_after=risk_after,
                gain=gain,
                train_loss=train_loss,
                wall_clock_seconds=wall_seconds,
                gpu_seconds=gpu_seconds,
                adapter_path=str(adapter_dir),
            )
        )
        torch = getattr(student, "_torch", None)
        del student
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return observations
