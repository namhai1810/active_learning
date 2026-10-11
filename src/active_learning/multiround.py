"""Reusable controller for adaptive, cost-aware active-learning rounds."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.decision.cost_aware import build_decision_candidates, select_intervention
from src.decision.types import DecisionCandidate, InterventionDecision


Checkpoint = Mapping[str, Any]


@dataclass(frozen=True)
class ActiveLearningConfig:
    max_rounds: int
    max_budget: float | None
    lambda_cost: float

    def __post_init__(self) -> None:
        if isinstance(self.max_rounds, bool) or self.max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if (
            isinstance(self.lambda_cost, bool)
            or not isinstance(self.lambda_cost, (int, float))
            or not math.isfinite(self.lambda_cost)
            or self.lambda_cost < 0
        ):
            raise ValueError("lambda_cost must be finite and non-negative")
        if self.max_budget is not None and (
            isinstance(self.max_budget, bool)
            or not isinstance(self.max_budget, (int, float))
            or not math.isfinite(self.max_budget)
            or self.max_budget < 0
        ):
            raise ValueError("max_budget must be finite and non-negative")


@dataclass(frozen=True)
class ActiveLearningCallbacks:
    """The six concrete stages required in every attempted round.

    Each callback receives only earlier-round state plus outputs from earlier
    stages in the same round. In particular, no callback receives final-test
    data. ``update_student`` is invoked only for a frozen CONTINUE decision.
    """

    discover: Callable[[int, Checkpoint], Sequence[Mapping[str, Any]]]
    verify: Callable[
        [int, Checkpoint, Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ]
    generate: Callable[
        [int, Checkpoint, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
        Any,
    ]
    estimate_gains: Callable[
        [int, Checkpoint, Sequence[Mapping[str, Any]], Any],
        Sequence[Mapping[str, Any]],
    ]
    estimate_costs: Callable[
        [
            int,
            Checkpoint,
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
            Any,
        ],
        tuple[Mapping[str, float], Mapping[str, Mapping[str, Any]]],
    ]
    update_student: Callable[
        [int, Checkpoint, InterventionDecision, Any], Mapping[str, Any]
    ]


@dataclass(frozen=True)
class ActiveLearningRound:
    round_index: int
    starting_checkpoint: dict[str, Any]
    ending_checkpoint: dict[str, Any]
    num_hypotheses: int
    num_certified: int
    candidates: tuple[DecisionCandidate, ...]
    decision: InterventionDecision
    spent_before: float
    spent_after: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["candidates"] = [item.to_dict() for item in self.candidates]
        value["decision"] = self.decision.to_dict()
        return value


@dataclass(frozen=True)
class ActiveLearningResult:
    rounds: tuple[ActiveLearningRound, ...]
    final_checkpoint: dict[str, Any]
    total_cost: float
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": [item.to_dict() for item in self.rounds],
            "final_checkpoint": dict(self.final_checkpoint),
            "total_cost": self.total_cost,
            "stop_reason": self.stop_reason,
        }


def partition_ids_by_round(
    ids: Sequence[str], max_rounds: int, *, seed: int, namespace: str
) -> tuple[tuple[str, ...], ...]:
    """Deterministically create disjoint adaptive-evaluation shards."""
    if isinstance(max_rounds, bool) or max_rounds <= 0:
        raise ValueError("max_rounds must be a positive integer")
    normalized = [str(item) for item in ids]
    if not namespace:
        raise ValueError("namespace must be non-empty")
    if not all(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("ids must be non-empty and unique")
    ranked = sorted(
        normalized,
        key=lambda item: hashlib.sha256(
            f"{seed}:{namespace}:{item}".encode("utf-8")
        ).digest(),
    )
    shards: list[list[str]] = [[] for _ in range(max_rounds)]
    for index, item in enumerate(ranked):
        shards[index % max_rounds].append(item)
    return tuple(tuple(shard) for shard in shards)


def run_active_learning_loop(
    *,
    config: ActiveLearningConfig,
    initial_checkpoint: Mapping[str, Any],
    callbacks: ActiveLearningCallbacks,
) -> ActiveLearningResult:
    """Run adaptive rounds until UCB stopping, budget, or round exhaustion.

    The callback boundary is intentionally final-test-free. Confidence validity
    is not created by this controller; callers must supply fresh conditionally
    valid verification/pilot observations and configure bounds for the declared
    maximum number of hypotheses and rounds.
    """
    current = dict(initial_checkpoint)
    if not current:
        raise ValueError("initial_checkpoint must be non-empty")
    spent = 0.0
    history: list[ActiveLearningRound] = []
    stop_reason = "max_rounds_reached"

    for round_index in range(1, config.max_rounds + 1):
        starting = dict(current)
        hypotheses = tuple(callbacks.discover(round_index, starting))
        verification = tuple(callbacks.verify(round_index, starting, hypotheses))
        generated = callbacks.generate(
            round_index, starting, hypotheses, verification
        )
        gains = tuple(
            callbacks.estimate_gains(round_index, starting, hypotheses, generated)
        )
        costs, breakdowns = callbacks.estimate_costs(
            round_index, starting, hypotheses, verification, generated
        )
        candidates = tuple(
            build_decision_candidates(
                verification,
                gains,
                costs,
                config.lambda_cost,
                cost_breakdowns=breakdowns,
            )
        )
        remaining = (
            None if config.max_budget is None else max(0.0, config.max_budget - spent)
        )
        decision = select_intervention(
            candidates, config.lambda_cost, max_cost=remaining
        )
        ending = starting
        spent_after = spent
        if decision.action == "continue":
            if decision.cost is None:
                raise RuntimeError("CONTINUE decision is missing its cost")
            updated = dict(
                callbacks.update_student(round_index, starting, decision, generated)
            )
            if not updated:
                raise ValueError("update_student returned an empty checkpoint")
            ending = updated
            spent_after = spent + decision.cost
        history.append(
            ActiveLearningRound(
                round_index=round_index,
                starting_checkpoint=starting,
                ending_checkpoint=ending,
                num_hypotheses=len(hypotheses),
                num_certified=sum(
                    item.get("certified") is True for item in verification
                ),
                candidates=candidates,
                decision=decision,
                spent_before=spent,
                spent_after=spent_after,
            )
        )
        spent = spent_after
        current = ending
        if decision.action == "stop":
            stop_reason = decision.reason
            break
        if config.max_budget is not None and spent >= config.max_budget:
            stop_reason = "max_budget_reached"
            break

    return ActiveLearningResult(
        rounds=tuple(history),
        final_checkpoint=current,
        total_cost=spent,
        stop_reason=stop_reason,
    )
