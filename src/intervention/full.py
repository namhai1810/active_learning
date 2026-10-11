"""One fresh full intervention after a frozen cost-aware decision."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.decision.types import InterventionDecision
from src.gain.estimation import risk_reduction
from src.hypotheses.types import FailureHypothesis
from src.intervention.hans_oracle_targeting import (
    select_intervention_examples,
    select_matching_gain_eval_examples,
    to_mnli_training_examples,
)
from src.intervention.pilot import _start_gpu_timer, _stop_gpu_timer


@dataclass(frozen=True)
class FullInterventionResult:
    hypothesis_id: str
    seed: int
    starting_checkpoint: dict[str, Any]
    train_ids: list[str]
    eval_ids: list[str]
    training_config: dict[str, Any]
    cost: int
    risk_before: float
    risk_after: float
    observed_full_gain: float
    train_loss: float | None
    wall_clock_seconds: float
    gpu_seconds: float | None
    adapter_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


StudentFactory = Callable[[int], Any]
TrainCallback = Callable[[Any, list[Any], Path, int], float]
RiskCallback = Callable[[Any, Sequence[Mapping[str, Any]]], float]


def run_full_intervention(
    *,
    decision: InterventionDecision,
    hypothesis: FailureHypothesis,
    intervention_records: Sequence[Mapping[str, Any]],
    gain_eval_records: Sequence[Mapping[str, Any]],
    full_intervention_size: int,
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
) -> FullInterventionResult:
    """Train exactly one selected hypothesis from the original checkpoint."""
    if decision.action != "continue":
        raise ValueError("A full intervention requires a continue decision")
    if decision.selected_hypothesis_id != hypothesis.id:
        raise ValueError("Decision and intervention hypothesis do not match")
    if full_intervention_size <= 0:
        raise ValueError("full_intervention_size must be positive")

    train_records = select_intervention_examples(
        hypothesis,
        intervention_records,
        full_intervention_size,
        seed,
    )
    eval_records = select_matching_gain_eval_examples(hypothesis, gain_eval_records)
    if not train_records:
        raise ValueError("Selected hypothesis has no eligible intervention examples")
    if not eval_records:
        raise ValueError("Selected hypothesis has no matching gain-eval examples")
    if decision.cost != float(len(train_records)):
        raise ValueError("Saved decision cost does not match actual acquired examples")

    adapter_dir = Path(output_dir) / hypothesis.id / "adapter"
    student = student_factory(seed)
    wall_start = time.perf_counter()
    gpu_timer = _start_gpu_timer(student)
    risk_before = risk_callback(student, eval_records)
    training_examples = to_mnli_training_examples(
        train_records, non_entailment_label
    )
    train_loss = train_callback(
        student, training_examples, adapter_dir, seed
    )
    risk_after = risk_callback(student, eval_records)
    gpu_seconds = _stop_gpu_timer(student, gpu_timer)
    wall_clock_seconds = time.perf_counter() - wall_start
    observed_gain = risk_reduction(
        risk_before,
        risk_after,
        M=M,
        tolerance=tolerance,
    )
    result = FullInterventionResult(
        hypothesis_id=hypothesis.id,
        seed=seed,
        starting_checkpoint=dict(starting_checkpoint),
        train_ids=[str(record["id"]) for record in train_records],
        eval_ids=[str(record["id"]) for record in eval_records],
        training_config=dict(training_config),
        cost=len(train_records),
        risk_before=risk_before,
        risk_after=risk_after,
        observed_full_gain=observed_gain,
        train_loss=train_loss,
        wall_clock_seconds=wall_clock_seconds,
        gpu_seconds=gpu_seconds,
        adapter_path=str(adapter_dir),
    )
    torch = getattr(student, "_torch", None)
    del student
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
