"""Expensive offline gain-reference repetitions from a fixed checkpoint."""

from __future__ import annotations

import gc
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.gain.estimation import risk_reduction
from src.hypotheses.types import FailureHypothesis
from src.intervention.hans_oracle_targeting import (
    select_intervention_examples,
    to_mnli_training_examples,
)
from src.intervention.pilot import _start_gpu_timer, _stop_gpu_timer


StudentFactory = Callable[[int], Any]
TrainCallback = Callable[[Any, list[Any], Path, int], float]
RiskCallback = Callable[[Any, Sequence[Mapping[str, Any]]], float]


def _reference_seed(base_seed: int, hypothesis_id: str, repetition: int) -> int:
    digest = hashlib.sha256(
        f"gain-reference:{base_seed}:{hypothesis_id}:{repetition}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _matching_reference(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for record in records:
        if record.get("split_role") != "gain_reference":
            raise ValueError("Reference gain evaluation requires gain_reference role")
        example_id = str(record.get("id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate gain-reference ID: {example_id!r}")
        seen.add(example_id)
        if record.get("heuristic") == hypothesis.heuristic:
            selected.append(dict(record))
    return sorted(selected, key=lambda record: str(record["id"]))


def run_reference_gain_repetitions(
    *,
    hypothesis: FailureHypothesis,
    intervention_records: Sequence[Mapping[str, Any]],
    gain_reference_records: Sequence[Mapping[str, Any]],
    repetitions: int,
    pilot_train_size: int,
    seed: int,
    non_entailment_label: str,
    M: float,
    tolerance: float,
    starting_checkpoint: Mapping[str, Any],
    training_config: Mapping[str, Any],
    output_dir: str | Path,
    student_factory: StudentFactory,
    train_callback: TrainCallback,
    risk_callback: RiskCallback,
) -> list[dict[str, Any]]:
    if repetitions <= 0 or pilot_train_size <= 0:
        raise ValueError("repetitions and pilot_train_size must be positive")
    eval_records = _matching_reference(hypothesis, gain_reference_records)
    if not eval_records:
        return []
    observations = []
    for repetition in range(repetitions):
        repetition_seed = _reference_seed(seed, hypothesis.id, repetition)
        train_records = select_intervention_examples(
            hypothesis,
            intervention_records,
            pilot_train_size,
            repetition_seed,
        )
        if not train_records:
            continue
        adapter_dir = (
            Path(output_dir)
            / hypothesis.id
            / f"reference_{repetition:03d}"
            / "adapter"
        )
        student = student_factory(repetition_seed)
        wall_start = time.perf_counter()
        gpu_timer = _start_gpu_timer(student)
        risk_before = risk_callback(student, eval_records)
        train_loss = train_callback(
            student,
            to_mnli_training_examples(train_records, non_entailment_label),
            adapter_dir,
            repetition_seed,
        )
        risk_after = risk_callback(student, eval_records)
        gpu_seconds = _stop_gpu_timer(student, gpu_timer)
        gain = risk_reduction(risk_before, risk_after, M, tolerance)
        observations.append(
            {
                "hypothesis_id": hypothesis.id,
                "repetition": repetition,
                "seed": repetition_seed,
                "starting_checkpoint": dict(starting_checkpoint),
                "train_ids": [str(record["id"]) for record in train_records],
                "eval_ids": [str(record["id"]) for record in eval_records],
                "training_config": dict(training_config),
                "risk_before": risk_before,
                "risk_after": risk_after,
                "gain": gain,
                "train_loss": train_loss,
                "wall_clock_seconds": time.perf_counter() - wall_start,
                "gpu_seconds": gpu_seconds,
                "adapter_path": str(adapter_dir),
                "evaluation_role": "gain_reference",
            }
        )
        torch = getattr(student, "_torch", None)
        del student
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return observations
