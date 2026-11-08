"""Pure accounting and replay primitives for incremental Cost-Aware V7.

The online decision uses only the projected marginal cost that can still be
avoided at decision time.  Exploration already incurred is a sunk cost and is
reported separately.  This module performs no I/O and does not alter either
the V7 controller or historical artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Iterable, Mapping, Optional


class CostAccountingError(ValueError):
    """Raised when cost units/scopes cannot be audited safely."""


@dataclass(frozen=True)
class CostComponent:
    component_id: str
    category: str
    scope: str
    amount: float
    unit: str
    timing: str
    known_before_decision: bool
    source: str
    measured: bool = False
    projected: bool = False
    modeled: bool = False

    def __post_init__(self) -> None:
        if self.category not in {
            "teacher_inference",
            "verification",
            "synthetic_generation_qc",
            "pilot_training",
            "pilot_evaluation",
            "main_training",
            "epoch_validation",
            "evaluation",
            "other",
        }:
            raise CostAccountingError(f"unknown cost category: {self.category}")
        if self.scope not in {
            "exploration",
            "decision",
            "pipeline_only",
        }:
            raise CostAccountingError(f"unknown cost scope: {self.scope}")
        if not isfinite(self.amount) or self.amount < 0:
            raise CostAccountingError("cost component amount must be finite and non-negative")
        if not self.unit:
            raise CostAccountingError("cost component unit is required")
        if sum((self.measured, self.projected, self.modeled)) > 1:
            raise CostAccountingError("component cannot be measured, projected and modeled simultaneously")
        if self.scope == "decision" and not self.known_before_decision:
            raise CostAccountingError("decision cost must be known or projected before decision")


def deduplicate_components(
    components: Iterable[CostComponent],
) -> tuple[CostComponent, ...]:
    """Reject duplicate accounting identities instead of silently summing them."""

    unique: dict[tuple[str, str, str, str], CostComponent] = {}
    for component in components:
        identity = (
            component.component_id,
            component.scope,
            component.unit,
            component.timing,
        )
        if identity in unique:
            previous = unique[identity]
            raise CostAccountingError(
                "double-counted component identity: "
                f"{identity}; sources={previous.source},{component.source}"
            )
        unique[identity] = component
    return tuple(unique.values())


@dataclass(frozen=True)
class CostPartition:
    exploration_cost: Optional[float]
    decision_cost: Optional[float]
    actual_intervention_cost: Optional[float]
    total_pipeline_cost: Optional[float]
    cost_unit: str
    modeled_usd_cost: Optional[float] = None
    shared_initial_pool_cost_available: bool = False
    round_one_cost_available: bool = False
    missing_components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("exploration_cost", self.exploration_cost),
            ("decision_cost", self.decision_cost),
            ("actual_intervention_cost", self.actual_intervention_cost),
            ("total_pipeline_cost", self.total_pipeline_cost),
            ("modeled_usd_cost", self.modeled_usd_cost),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise CostAccountingError(f"{name} must be finite and non-negative")
        if not self.cost_unit:
            raise CostAccountingError("cost_unit is required")

    @property
    def partial_total_cost(self) -> bool:
        return (
            self.total_pipeline_cost is None
            or not self.shared_initial_pool_cost_available
            or not self.round_one_cost_available
            or bool(self.missing_components)
        )

    @property
    def internally_consistent(self) -> Optional[bool]:
        if (
            self.exploration_cost is None
            or self.actual_intervention_cost is None
            or self.total_pipeline_cost is None
        ):
            return None
        # Total may contain extra reporting/evaluation overhead, but it cannot
        # be lower than already incurred exploration + actual intervention.
        return self.total_pipeline_cost + 1e-9 >= (
            self.exploration_cost + self.actual_intervention_cost
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partial_total_cost"] = self.partial_total_cost
        payload["internally_consistent"] = self.internally_consistent
        return payload


@dataclass(frozen=True)
class ReplayInputs:
    seed: int
    round_index: int
    policy: str
    delta_hat: Optional[float]
    confidence_radius: Optional[float]
    support_upper: float
    lambda_cost: float
    costs: CostPartition
    recorded_action: str
    recorded_reason: str
    evidence_gate: Optional[str] = None
    observed_one_step_gain: Optional[float] = None

    def __post_init__(self) -> None:
        if self.policy not in {"v2_hoeffding", "v2_1_range_aware"}:
            raise ValueError(f"unsupported replay policy: {self.policy}")
        if self.recorded_action not in {"continue", "stop"}:
            raise ValueError("recorded_action must be continue or stop")
        if self.round_index < 2:
            raise ValueError("decision replay starts at round 2")
        if self.support_upper <= 0 or self.lambda_cost < 0:
            raise ValueError("invalid support/lambda")
        if self.confidence_radius is not None and self.confidence_radius < 0:
            raise ValueError("confidence radius must be non-negative")


@dataclass(frozen=True)
class ReplayDecision:
    seed: int
    round_index: int
    policy: str
    lambda_cost: float
    delta_hat: Optional[float]
    confidence_radius: Optional[float]
    gain_ucb: Optional[float]
    decision_cost: Optional[float]
    estimated_net_gain: Optional[float]
    net_ucb: Optional[float]
    replay_action: str
    replay_reason: str
    recorded_action: str
    recorded_reason: str
    replay_matches_recorded: bool
    evidence_gate: Optional[str]
    observed_one_step_gain: Optional[float]
    observed_one_step_net_gain: Optional[float]
    actual_intervention_cost: Optional[float]
    exploration_cost: Optional[float]
    total_pipeline_cost: Optional[float]
    partial_total_cost: bool
    counterfactual_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def replay_incremental_decision(inputs: ReplayInputs) -> ReplayDecision:
    """Replay J and U without using incurred or future actual cost online."""

    decision_cost = inputs.costs.decision_cost
    gain_ucb = None
    estimated_net_gain = None
    net_ucb = None
    if inputs.delta_hat is not None and inputs.confidence_radius is not None:
        gain_ucb = min(
            inputs.support_upper,
            inputs.delta_hat + inputs.confidence_radius,
        )
    if inputs.delta_hat is not None and decision_cost is not None:
        estimated_net_gain = inputs.delta_hat - inputs.lambda_cost * decision_cost
    if gain_ucb is not None and decision_cost is not None:
        net_ucb = gain_ucb - inputs.lambda_cost * decision_cost

    if inputs.evidence_gate:
        action = inputs.recorded_action
        reason = inputs.recorded_reason
    elif net_ucb is None:
        action = inputs.recorded_action
        reason = "insufficient_cost_or_gain_evidence_preserve_recorded_action"
    else:
        action = "continue" if net_ucb > 0 else "stop"
        reason = "positive_incremental_net_ucb" if action == "continue" else "nonpositive_incremental_net_ucb"

    observed_net = None
    if (
        inputs.observed_one_step_gain is not None
        and inputs.costs.actual_intervention_cost is not None
    ):
        observed_net = (
            inputs.observed_one_step_gain
            - inputs.lambda_cost * inputs.costs.actual_intervention_cost
        )

    return ReplayDecision(
        seed=inputs.seed,
        round_index=inputs.round_index,
        policy=inputs.policy,
        lambda_cost=inputs.lambda_cost,
        delta_hat=inputs.delta_hat,
        confidence_radius=inputs.confidence_radius,
        gain_ucb=gain_ucb,
        decision_cost=decision_cost,
        estimated_net_gain=estimated_net_gain,
        net_ucb=net_ucb,
        replay_action=action,
        replay_reason=reason,
        recorded_action=inputs.recorded_action,
        recorded_reason=inputs.recorded_reason,
        replay_matches_recorded=action == inputs.recorded_action,
        evidence_gate=inputs.evidence_gate,
        observed_one_step_gain=inputs.observed_one_step_gain,
        observed_one_step_net_gain=observed_net,
        actual_intervention_cost=inputs.costs.actual_intervention_cost,
        exploration_cost=inputs.costs.exploration_cost,
        total_pipeline_cost=inputs.costs.total_pipeline_cost,
        partial_total_cost=inputs.costs.partial_total_cost,
    )


def audit_evaluation_rate_modes(
    rates: Mapping[str, Any],
    *,
    explicit_additive_policy: bool,
) -> tuple[str, ...]:
    """Flag ambiguous simultaneous evaluation charging without changing rates."""

    active: list[str] = []
    for key in (
        "evaluation_per_item",
        "evaluation_per_run",
        "evaluation_gpu_second",
        "evaluation_gpu_hour",
    ):
        value = rates.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != 0:
            active.append(key)
    if len(active) > 1 and not explicit_additive_policy:
        return ("ambiguous_evaluation_charging_modes:" + ",".join(sorted(active)),)
    return ()

