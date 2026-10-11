"""Pilot and full interventions trained only on validated synthetic NLI data."""

from __future__ import annotations

import gc
import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.decision.types import InterventionDecision
from src.gain.estimation import risk_reduction
from src.gain.types import PilotGainObservation
from src.intervention.full import FullInterventionResult
from src.intervention.pilot import _start_gpu_timer, _stop_gpu_timer
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample


@dataclass(frozen=True)
class SyntheticPilotConfig:
    m: int
    pilot_train_size: int
    gain_eval_size: int
    seed: int
    M: float = 1.0
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.m <= 0 or self.pilot_train_size <= 0 or self.gain_eval_size <= 0:
            raise ValueError("m and pilot/evaluation sizes must be positive")
        if self.pilot_train_size % 2:
            raise ValueError("pilot_train_size must be even to preserve complete pairs")
        if not math.isfinite(self.M) or self.M <= 0:
            raise ValueError("M must be finite and positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")


StudentFactory = Callable[[int], Any]
TrainCallback = Callable[[Any, list[Any], Path, int], float]
RiskCallback = Callable[[Any, Sequence[Mapping[str, Any]]], float]


def _rank_id(example_id: str, seed: int, purpose: str) -> tuple[bytes, str]:
    return (
        hashlib.sha256(f"{seed}:{purpose}:{example_id}".encode("utf-8")).digest(),
        example_id,
    )


def _complete_pair_groups(
    examples: Sequence[SyntheticNLIExample], *, hypothesis_id: str
) -> list[list[SyntheticNLIExample]]:
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    ids: set[str] = set()
    for item in examples:
        if item.hypothesis_id != hypothesis_id:
            raise ValueError("Synthetic training data contains another hypothesis")
        if item.id in ids:
            raise ValueError(f"Duplicate synthetic training ID: {item.id}")
        ids.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)
    result = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        if len({member.pair_type for member in members}) != 1:
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        pair_type = members[0].pair_type
        if {member.pair_member for member in members} != set(
            SYNTHETIC_PAIR_MEMBERS[pair_type]
        ):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid member roles")
        result.append(
            sorted(
                members,
                key=lambda member: SYNTHETIC_PAIR_MEMBERS[pair_type].index(
                    member.pair_member
                ),
            )
        )
    return result


def _release_student(student: Any) -> None:
    torch = getattr(student, "_torch", None)
    del student
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_synthetic_pilots(
    *,
    hypothesis_id: str,
    synthetic_examples: Sequence[SyntheticNLIExample],
    applicable_gain_eval: Sequence[Mapping[str, Any]],
    config: SyntheticPilotConfig,
    starting_checkpoint: Mapping[str, Any],
    training_config: Mapping[str, Any],
    pilot_root: str | Path,
    student_factory: StudentFactory,
    train_callback: TrainCallback,
    risk_callback: RiskCallback,
) -> list[PilotGainObservation]:
    """Run disjoint small interventions from the same original checkpoint."""
    if not hypothesis_id:
        raise ValueError("hypothesis_id must be non-empty")
    pair_groups = _complete_pair_groups(
        synthetic_examples, hypothesis_id=hypothesis_id
    )
    train_ids = [item.id for group in pair_groups for item in group]
    eval_ids = [str(record.get("id", "")) for record in applicable_gain_eval]
    if len(train_ids) != len(set(train_ids)) or len(eval_ids) != len(set(eval_ids)):
        raise ValueError("Pilot training and evaluation IDs must be unique")
    ranked_pairs = sorted(
        pair_groups,
        key=lambda group: _rank_id(
            group[0].pair_id, config.seed, f"pilot-train-pair:{hypothesis_id}"
        ),
    )
    ranked_eval = sorted(
        applicable_gain_eval,
        key=lambda record: _rank_id(
            str(record["id"]), config.seed, f"pilot-eval:{hypothesis_id}"
        ),
    )
    feasible = min(
        config.m,
        len(ranked_pairs) // (config.pilot_train_size // 2),
        len(ranked_eval) // config.gain_eval_size,
    )
    observations = []
    for index in range(feasible):
        seed = int.from_bytes(
            hashlib.sha256(
                f"{config.seed}:{hypothesis_id}:pilot:{index}".encode("utf-8")
            ).digest()[:4],
            "big",
        )
        pairs_per_pilot = config.pilot_train_size // 2
        selected_pairs = ranked_pairs[
            index * pairs_per_pilot : (index + 1) * pairs_per_pilot
        ]
        train_batch = [item for group in selected_pairs for item in group]
        eval_batch = ranked_eval[
            index * config.gain_eval_size : (index + 1) * config.gain_eval_size
        ]
        pilot_dir = Path(pilot_root) / hypothesis_id / f"pilot_{index:03d}"
        adapter_dir = pilot_dir / "adapter"
        student = student_factory(seed)
        wall_start = time.perf_counter()
        gpu_timer = _start_gpu_timer(student)
        risk_before = risk_callback(student, eval_batch)
        train_loss = train_callback(
            student,
            [item.to_mnli() for item in train_batch],
            adapter_dir,
            seed,
        )
        risk_after = risk_callback(student, eval_batch)
        gpu_seconds = _stop_gpu_timer(student, gpu_timer)
        wall_seconds = time.perf_counter() - wall_start
        gain = risk_reduction(
            risk_before, risk_after, M=config.M, tolerance=config.tolerance
        )
        observations.append(
            PilotGainObservation(
                hypothesis_id=hypothesis_id,
                pilot_index=index,
                pilot_seed=seed,
                starting_checkpoint=dict(starting_checkpoint),
                train_ids=[item.id for item in train_batch],
                eval_ids=[str(record["id"]) for record in eval_batch],
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
        _release_student(student)
    return observations


def select_full_synthetic_examples(
    examples: Sequence[SyntheticNLIExample],
    *,
    hypothesis_id: str,
    excluded_ids: Sequence[str],
    full_size: int,
    seed: int,
) -> list[SyntheticNLIExample]:
    if full_size <= 0:
        raise ValueError("full_size must be positive")
    if full_size % 2:
        raise ValueError("full_size must be even to preserve complete pairs")
    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    relevant = [item for item in examples if item.hypothesis_id == hypothesis_id]
    pair_groups = _complete_pair_groups(relevant, hypothesis_id=hypothesis_id)
    eligible_pairs = [
        group for group in pair_groups if not excluded & {item.id for item in group}
    ]
    ranked_pairs = sorted(
        eligible_pairs,
        key=lambda group: _rank_id(
            group[0].pair_id, seed, f"full-train-pair:{hypothesis_id}"
        ),
    )[: full_size // 2]
    return [item for group in ranked_pairs for item in group]


def run_synthetic_full_intervention(
    *,
    decision: InterventionDecision,
    training_examples: Sequence[SyntheticNLIExample],
    applicable_gain_eval: Sequence[Mapping[str, Any]],
    seed: int,
    M: float,
    tolerance: float,
    starting_checkpoint: Mapping[str, Any],
    training_config: Mapping[str, Any],
    output_dir: str | Path,
    student_factory: StudentFactory,
    train_callback: TrainCallback,
    risk_callback: RiskCallback,
    expected_training_examples: int | None = None,
) -> FullInterventionResult:
    """Execute one fresh full run only after a frozen CONTINUE decision."""
    if decision.action != "continue" or decision.selected_hypothesis_id is None:
        raise ValueError("Full synthetic intervention requires a CONTINUE decision")
    hypothesis_id = decision.selected_hypothesis_id
    if not training_examples or not applicable_gain_eval:
        raise ValueError("Full intervention requires training and gain-eval examples")
    pair_groups = _complete_pair_groups(
        training_examples, hypothesis_id=hypothesis_id
    )
    if sum(len(group) for group in pair_groups) != len(training_examples):
        raise ValueError("Full intervention requires complete synthetic pairs")
    if expected_training_examples is None:
        # Compatibility for the historical acquired-example-count cost mode.
        if decision.cost != float(len(training_examples)):
            raise ValueError("Frozen decision cost differs from actual full data count")
    elif expected_training_examples != len(training_examples):
        raise ValueError("Selected training count differs from the frozen candidate plan")
    adapter_dir = Path(output_dir) / hypothesis_id / "adapter"
    student = student_factory(seed)
    wall_start = time.perf_counter()
    gpu_timer = _start_gpu_timer(student)
    risk_before = risk_callback(student, applicable_gain_eval)
    train_loss = train_callback(
        student,
        [item.to_mnli() for item in training_examples],
        adapter_dir,
        seed,
    )
    risk_after = risk_callback(student, applicable_gain_eval)
    gpu_seconds = _stop_gpu_timer(student, gpu_timer)
    wall_seconds = time.perf_counter() - wall_start
    observed_gain = risk_reduction(
        risk_before, risk_after, M=M, tolerance=tolerance
    )
    result = FullInterventionResult(
        hypothesis_id=hypothesis_id,
        seed=seed,
        starting_checkpoint=dict(starting_checkpoint),
        train_ids=[item.id for item in training_examples],
        eval_ids=[str(record["id"]) for record in applicable_gain_eval],
        training_config=dict(training_config),
        cost=len(training_examples),
        risk_before=risk_before,
        risk_after=risk_after,
        observed_full_gain=observed_gain,
        train_loss=train_loss,
        wall_clock_seconds=wall_seconds,
        gpu_seconds=gpu_seconds,
        adapter_path=str(adapter_dir),
    )
    _release_student(student)
    return result
