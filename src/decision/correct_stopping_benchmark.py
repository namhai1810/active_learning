"""Pure CPU-side retrospective Correct Stopping evaluation.

This module never loads a model and never decides an online V7 action.  It
evaluates already-recorded policy traces against a retrospective quality
oracle.  Sealed/future quality is used only inside ``evaluate_policy`` and
``oracle_round`` after a stop trace has been frozen from online evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


ONLINE_STOP_CATEGORIES = {
    "gain_cost_stop",
    "validation_patience_stop",
}


@dataclass(frozen=True)
class CostSnapshot:
    """Cumulative cost after a round.

    ``total_modeled_usd`` is the only scalar used for the primary cost-saving
    ratio.  Measured counters remain dimensioned and are never added together.
    ``pilot_modeled_usd`` and ``pilot_measured`` are cumulative pilot-only
    overheads, reported separately from the full pipeline total.
    """

    total_modeled_usd: Optional[float]
    measured: Mapping[str, float] = field(default_factory=dict)
    pilot_modeled_usd: Optional[float] = None
    pilot_measured: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("total_modeled_usd", self.total_modeled_usd),
            ("pilot_modeled_usd", self.pilot_modeled_usd),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        for namespace, values in (
            ("measured", self.measured),
            ("pilot_measured", self.pilot_measured),
        ):
            for key, value in values.items():
                if not isfinite(value) or value < 0:
                    raise ValueError(f"{namespace}.{key} must be finite and non-negative")


@dataclass(frozen=True)
class DecisionEvidence:
    action: str
    reason: str
    confidence_radius: Optional[float] = None
    certified_gain_cost_stop: bool = False

    def __post_init__(self) -> None:
        if self.action not in {"continue", "stop"}:
            raise ValueError("decision action must be 'continue' or 'stop'")
        if self.confidence_radius is not None:
            if not isfinite(self.confidence_radius) or self.confidence_radius < 0:
                raise ValueError("confidence radius must be finite and non-negative")
        if self.certified_gain_cost_stop and self.action != "stop":
            raise ValueError("only a stop action can be a certified gain-cost stop")


@dataclass(frozen=True)
class RoundRecord:
    round_index: int
    budget: int
    offline_accuracy: Optional[float]
    validation_accuracy: Optional[float]
    cost: CostSnapshot
    decisions: Mapping[str, DecisionEvidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ValueError("round_index must be positive")
        if self.budget < 0:
            raise ValueError("budget must be non-negative")
        for name, value in (
            ("offline_accuracy", self.offline_accuracy),
            ("validation_accuracy", self.validation_accuracy),
        ):
            if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class Trajectory:
    seed: int
    rounds: Tuple[RoundRecord, ...]
    t_max: int = 8
    budget_max: int = 256

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.rounds, key=lambda item: item.round_index))
        if ordered != self.rounds:
            raise ValueError("trajectory rounds must be sorted")
        indices = [item.round_index for item in ordered]
        if len(indices) != len(set(indices)):
            raise ValueError("trajectory has duplicate rounds")
        if any(b.round_index <= a.round_index or b.budget < a.budget for a, b in zip(ordered, ordered[1:])):
            raise ValueError("round and budget must be monotone")
        if ordered and ordered[-1].round_index > self.t_max:
            raise ValueError("trajectory exceeds t_max")
        if ordered and ordered[-1].budget > self.budget_max:
            raise ValueError("trajectory exceeds budget_max")
        known_costs = [
            (item.round_index, item.cost.total_modeled_usd)
            for item in ordered
            if item.cost.total_modeled_usd is not None
        ]
        if any(float(later[1]) < float(earlier[1]) for earlier, later in zip(known_costs, known_costs[1:])):
            raise ValueError("cumulative total modeled cost must be monotone")
        measured_keys = set().union(*(item.cost.measured.keys() for item in ordered))
        for key in measured_keys:
            values = [
                (item.round_index, item.cost.measured[key])
                for item in ordered
                if key in item.cost.measured
            ]
            if key.endswith("_cumulative") and any(
                later[1] < earlier[1] for earlier, later in zip(values, values[1:])
            ):
                raise ValueError(f"cumulative measured counter is not monotone: {key}")

    def by_round(self) -> Dict[int, RoundRecord]:
        return {item.round_index: item for item in self.rounds}

    def require_complete_rounds(self) -> Tuple[str, ...]:
        expected = set(range(1, self.t_max + 1))
        missing = sorted(expected - set(self.by_round()))
        return tuple(f"missing_round:{value}" for value in missing)


@dataclass(frozen=True)
class FrozenPolicyStop:
    policy: str
    decision_round: int
    deployed_round: int
    reason: str
    confidence_radius: Optional[float]
    certified_online_stop: bool


@dataclass(frozen=True)
class OracleResult:
    a_star: float
    oracle_round: int
    oracle_cost: float


@dataclass(frozen=True)
class BenchmarkResult:
    seed: int
    policy: str
    status: str
    classification: Optional[str]
    decision_round: Optional[int]
    deployed_round: Optional[int]
    budget: Optional[int]
    decision_reason: Optional[str]
    confidence_radius: Optional[float]
    certified_online_stop: bool
    a_star: Optional[float]
    a_stop: Optional[float]
    quality_regret: Optional[float]
    epsilon_optimal: Optional[bool]
    oracle_round: Optional[int]
    full_modeled_usd: Optional[float]
    stopped_modeled_usd: Optional[float]
    cost_saving_ratio: Optional[float]
    avoidable_modeled_usd: Optional[float]
    pilot_overhead_modeled_usd: Optional[float]
    measured_full: Mapping[str, float]
    measured_stopped: Mapping[str, float]
    measured_saving_ratio: Mapping[str, float]
    pilot_measured: Mapping[str, float]
    missing_evidence: Tuple[str, ...] = ()


def freeze_recorded_policy(trajectory: Trajectory, decision_key: str) -> FrozenPolicyStop:
    """Freeze a stop using recorded online decisions only.

    Budget exhaustion and operational termination are deliberately not treated
    as certified gain-cost STOP events.
    """

    if not trajectory.rounds:
        raise ValueError("empty trajectory")
    recorded = {
        item.round_index: item.decisions[decision_key]
        for item in trajectory.rounds
        if decision_key in item.decisions
    }
    first_certified = next(
        (
            round_index
            for round_index in sorted(recorded)
            if recorded[round_index].certified_gain_cost_stop
        ),
        None,
    )
    horizon = first_certified if first_certified is not None else trajectory.t_max
    missing = [index for index in range(2, horizon + 1) if index not in recorded]
    if missing:
        last = trajectory.rounds[-1]
        return FrozenPolicyStop(
            policy=decision_key,
            decision_round=last.round_index,
            deployed_round=last.round_index,
            reason="missing_recorded_policy_decisions:" + ",".join(map(str, missing)),
            confidence_radius=None,
            certified_online_stop=False,
        )
    for item in trajectory.rounds:
        decision = item.decisions.get(decision_key)
        if decision is not None and decision.certified_gain_cost_stop:
            return FrozenPolicyStop(
                policy=decision_key,
                decision_round=item.round_index,
                deployed_round=item.round_index,
                reason=decision.reason,
                confidence_radius=decision.confidence_radius,
                certified_online_stop=True,
            )
    last = trajectory.rounds[-1]
    return FrozenPolicyStop(
        policy=decision_key,
        decision_round=last.round_index,
        deployed_round=last.round_index,
        reason="budget_exhausted_without_certified_gain_cost_stop",
        confidence_radius=(
            last.decisions[decision_key].confidence_radius
            if decision_key in last.decisions
            else None
        ),
        certified_online_stop=False,
    )


def freeze_no_stopping(trajectory: Trajectory) -> FrozenPolicyStop:
    if not trajectory.rounds:
        raise ValueError("empty trajectory")
    last = trajectory.rounds[-1]
    return FrozenPolicyStop(
        policy="v7_no_stopping",
        decision_round=last.round_index,
        deployed_round=last.round_index,
        reason="fixed_full_trajectory",
        confidence_radius=None,
        certified_online_stop=False,
    )


def freeze_validation_patience(
    trajectory: Trajectory,
    *,
    patience_rounds: int = 1,
    min_delta: float = 0.001,
) -> FrozenPolicyStop:
    """Round-level baseline using current/past validation values only.

    When patience is exhausted after round ``t``, cost is incurred through
    ``t`` but the best validation checkpoint observed up to ``t`` is deployed.
    No offline/future quality enters this policy.
    """

    if patience_rounds < 1:
        raise ValueError("patience_rounds must be at least one")
    if min_delta < 0:
        raise ValueError("min_delta must be non-negative")
    best_value: Optional[float] = None
    best_round: Optional[int] = None
    bad_rounds = 0
    for item in trajectory.rounds:
        value = item.validation_accuracy
        if value is None:
            raise ValueError(f"round {item.round_index} lacks validation_accuracy")
        if best_value is None or value > best_value + min_delta:
            best_value = value
            best_round = item.round_index
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= patience_rounds:
                assert best_round is not None
                return FrozenPolicyStop(
                    policy="validation_patience",
                    decision_round=item.round_index,
                    deployed_round=best_round,
                    reason=(
                        f"validation_patience_exhausted:p={patience_rounds}:"
                        f"min_delta={min_delta}"
                    ),
                    confidence_radius=None,
                    certified_online_stop=True,
                )
    if not trajectory.rounds:
        raise ValueError("empty trajectory")
    last = trajectory.rounds[-1]
    return FrozenPolicyStop(
        policy="validation_patience",
        decision_round=last.round_index,
        deployed_round=(best_round if best_round is not None else last.round_index),
        reason="validation_patience_not_exhausted_before_budget",
        confidence_radius=None,
        certified_online_stop=False,
    )


def retrospective_oracle(trajectory: Trajectory, epsilon: float) -> OracleResult:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    missing = []
    for item in trajectory.rounds:
        if item.offline_accuracy is None:
            missing.append(f"round_{item.round_index}.offline_accuracy")
        if item.cost.total_modeled_usd is None:
            missing.append(f"round_{item.round_index}.cost.total_modeled_usd")
    if missing:
        raise ValueError("oracle evidence missing: " + ", ".join(missing))
    accuracies = [float(item.offline_accuracy) for item in trajectory.rounds]
    a_star = max(accuracies)
    candidates = [
        item
        for item in trajectory.rounds
        if a_star - float(item.offline_accuracy) <= epsilon
    ]
    selected = min(
        candidates,
        key=lambda item: (float(item.cost.total_modeled_usd), item.round_index),
    )
    return OracleResult(
        a_star=a_star,
        oracle_round=selected.round_index,
        oracle_cost=float(selected.cost.total_modeled_usd),
    )


def freeze_retrospective_oracle(
    trajectory: Trajectory, *, epsilon: float
) -> FrozenPolicyStop:
    """Create an evaluation-only oracle row.

    This function intentionally consumes future/offline quality.  Its result
    must never be passed back to the online controller or training pipeline.
    """

    oracle = retrospective_oracle(trajectory, epsilon)
    return FrozenPolicyStop(
        policy="retrospective_oracle",
        decision_round=oracle.oracle_round,
        deployed_round=oracle.oracle_round,
        reason="offline_evaluation_oracle_only",
        confidence_radius=None,
        certified_online_stop=False,
    )


def _resource_savings(
    full: Mapping[str, float], stopped: Mapping[str, float]
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for key in sorted(set(full) & set(stopped)):
        denominator = full[key]
        if denominator > 0:
            result[key] = (denominator - stopped[key]) / denominator
    return result


def _classify(
    *,
    stop: FrozenPolicyStop,
    t_max: int,
    epsilon_optimal: bool,
    oracle_round: int,
) -> str:
    if stop.decision_round < t_max:
        return "good_stop" if epsilon_optimal else "premature_stop"
    if oracle_round < t_max:
        return "unnecessary_continue"
    return "budget_exhausted"


def evaluate_policy(
    trajectory: Trajectory,
    stop: FrozenPolicyStop,
    *,
    epsilon: float = 0.01,
) -> BenchmarkResult:
    by_round = trajectory.by_round()
    missing = list(trajectory.require_complete_rounds())
    if stop.reason.startswith("missing_recorded_policy_decisions"):
        missing.append(stop.reason)
    if stop.policy == "validation_patience" and stop.reason == "missing_validation_evidence":
        missing.extend(
            f"round_{item.round_index}.validation_accuracy"
            for item in trajectory.rounds
            if item.validation_accuracy is None
        )
    if stop.decision_round not in by_round:
        missing.append(f"missing_decision_round:{stop.decision_round}")
    if stop.deployed_round not in by_round:
        missing.append(f"missing_deployed_round:{stop.deployed_round}")
    for item in trajectory.rounds:
        if item.offline_accuracy is None:
            missing.append(f"round_{item.round_index}.offline_accuracy")
        if item.cost.total_modeled_usd is None:
            missing.append(f"round_{item.round_index}.cost.total_modeled_usd")
    if missing:
        return BenchmarkResult(
            seed=trajectory.seed,
            policy=stop.policy,
            status="incomplete_evidence",
            classification=None,
            decision_round=stop.decision_round,
            deployed_round=stop.deployed_round,
            budget=(by_round[stop.decision_round].budget if stop.decision_round in by_round else None),
            decision_reason=stop.reason,
            confidence_radius=stop.confidence_radius,
            certified_online_stop=stop.certified_online_stop,
            a_star=None,
            a_stop=None,
            quality_regret=None,
            epsilon_optimal=None,
            oracle_round=None,
            full_modeled_usd=None,
            stopped_modeled_usd=None,
            cost_saving_ratio=None,
            avoidable_modeled_usd=None,
            pilot_overhead_modeled_usd=None,
            measured_full={},
            measured_stopped={},
            measured_saving_ratio={},
            pilot_measured={},
            missing_evidence=tuple(sorted(set(missing))),
        )

    oracle = retrospective_oracle(trajectory, epsilon)
    full = by_round[trajectory.t_max]
    decision_round = by_round[stop.decision_round]
    deployed = by_round[stop.deployed_round]
    a_stop = float(deployed.offline_accuracy)
    regret = oracle.a_star - a_stop
    epsilon_optimal = regret <= epsilon
    full_cost = float(full.cost.total_modeled_usd)
    stopped_cost = float(decision_round.cost.total_modeled_usd)
    saving_ratio = (full_cost - stopped_cost) / full_cost if full_cost > 0 else None
    return BenchmarkResult(
        seed=trajectory.seed,
        policy=stop.policy,
        status="complete",
        classification=(
            "oracle_reference"
            if stop.policy == "retrospective_oracle"
            else _classify(
                stop=stop,
                t_max=trajectory.t_max,
                epsilon_optimal=epsilon_optimal,
                oracle_round=oracle.oracle_round,
            )
        ),
        decision_round=stop.decision_round,
        deployed_round=stop.deployed_round,
        budget=decision_round.budget,
        decision_reason=stop.reason,
        confidence_radius=stop.confidence_radius,
        certified_online_stop=stop.certified_online_stop,
        a_star=oracle.a_star,
        a_stop=a_stop,
        quality_regret=regret,
        epsilon_optimal=epsilon_optimal,
        oracle_round=oracle.oracle_round,
        full_modeled_usd=full_cost,
        stopped_modeled_usd=stopped_cost,
        cost_saving_ratio=saving_ratio,
        avoidable_modeled_usd=full_cost - stopped_cost,
        pilot_overhead_modeled_usd=decision_round.cost.pilot_modeled_usd,
        measured_full=dict(full.cost.measured),
        measured_stopped=dict(decision_round.cost.measured),
        measured_saving_ratio=_resource_savings(
            full.cost.measured, decision_round.cost.measured
        ),
        pilot_measured=dict(decision_round.cost.pilot_measured),
        missing_evidence=(),
    )


def benchmark_trajectory(
    trajectory: Trajectory,
    *,
    epsilon: float = 0.01,
    patience_rounds: int = 1,
    validation_min_delta: float = 0.001,
) -> Tuple[BenchmarkResult, ...]:
    stops = [
        freeze_recorded_policy(trajectory, "v2_hoeffding"),
        freeze_recorded_policy(trajectory, "v2_1_range_aware"),
        freeze_no_stopping(trajectory),
    ]
    try:
        stops.append(
            freeze_validation_patience(
                trajectory,
                patience_rounds=patience_rounds,
                min_delta=validation_min_delta,
            )
        )
    except ValueError:
        # Preserve an explicit incomplete row instead of substituting another
        # metric for missing validation evidence.
        last = trajectory.rounds[-1]
        stops.append(
            FrozenPolicyStop(
                policy="validation_patience",
                decision_round=last.round_index,
                deployed_round=last.round_index,
                reason="missing_validation_evidence",
                confidence_radius=None,
                certified_online_stop=False,
            )
        )
    try:
        stops.append(freeze_retrospective_oracle(trajectory, epsilon=epsilon))
    except ValueError:
        last = trajectory.rounds[-1]
        stops.append(
            FrozenPolicyStop(
                policy="retrospective_oracle",
                decision_round=last.round_index,
                deployed_round=last.round_index,
                reason="missing_offline_or_cost_evidence_for_oracle",
                confidence_radius=None,
                certified_online_stop=False,
            )
        )
    return tuple(evaluate_policy(trajectory, stop, epsilon=epsilon) for stop in stops)
