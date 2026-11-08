"""Pure one-step retrospective evaluation for V7 Cost-Aware decisions.

This module does not load models or inspect future checkpoints.  It combines
one recorded online decision with one optional, separately measured main-update
outcome.  A single realized update is diagnostic evidence and is never treated
as an estimate of the expected intervention gain or as proof of a confidence
theorem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class DecisionInputs:
    seed: int
    round_index: int
    policy: str
    action: str
    reason: str
    delta_hat: Optional[float]
    radius: Optional[float]
    decision_cost: Optional[float]
    actual_main_update_cost: Optional[float]
    lambda_cost: float
    support_upper: float
    certified_hypotheses: Optional[int]
    unresolved_hypotheses: Optional[int]
    evidence_gate: Optional[str] = None

    def __post_init__(self) -> None:
        if self.round_index < 2:
            raise ValueError("Cost-Aware one-step decisions start at round 2")
        if self.policy not in {"v2_hoeffding", "v2_1_range_aware"}:
            raise ValueError(f"Unsupported policy: {self.policy}")
        if self.action not in {"continue", "stop"}:
            raise ValueError("action must be continue or stop")
        for name, value in (
            ("delta_hat", self.delta_hat),
            ("radius", self.radius),
            ("decision_cost", self.decision_cost),
            ("actual_main_update_cost", self.actual_main_update_cost),
            ("lambda_cost", self.lambda_cost),
            ("support_upper", self.support_upper),
        ):
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.radius is not None and self.radius < 0:
            raise ValueError("radius must be non-negative")
        if self.decision_cost is not None and self.decision_cost < 0:
            raise ValueError("decision_cost must be non-negative")
        if self.actual_main_update_cost is not None and self.actual_main_update_cost < 0:
            raise ValueError("actual_main_update_cost must be non-negative")
        if self.lambda_cost < 0:
            raise ValueError("lambda_cost must be non-negative")
        if self.support_upper <= 0:
            raise ValueError("support_upper must be positive")
        if self.certified_hypotheses is not None and self.certified_hypotheses < 0:
            raise ValueError("hypothesis counts must be non-negative when available")
        if self.unresolved_hypotheses is not None and self.unresolved_hypotheses < 0:
            raise ValueError("hypothesis counts must be non-negative")


@dataclass(frozen=True)
class ObservedUpdate:
    accuracy_before: Optional[float]
    accuracy_after: Optional[float]
    evaluation_manifest_hash: Optional[str]
    checkpoint_before_hash: Optional[str]
    checkpoint_after_hash: Optional[str]
    num_examples: Optional[int]

    def __post_init__(self) -> None:
        for name, value in (
            ("accuracy_before", self.accuracy_before),
            ("accuracy_after", self.accuracy_after),
        ):
            if value is not None and (
                not isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.num_examples is not None and self.num_examples <= 0:
            raise ValueError("num_examples must be positive")

    @property
    def gain(self) -> Optional[float]:
        if self.accuracy_before is None or self.accuracy_after is None:
            return None
        return self.accuracy_after - self.accuracy_before


@dataclass(frozen=True)
class OneStepAssessment:
    seed: int
    round_index: int
    policy: str
    action: str
    reason: str
    evidence_gate: Optional[str]
    certified_hypotheses: Optional[int]
    unresolved_hypotheses: Optional[int]
    delta_hat: Optional[float]
    radius: Optional[float]
    gain_ucb: Optional[float]
    decision_cost: Optional[float]
    actual_main_update_cost: Optional[float]
    estimated_net_gain: Optional[float]
    net_ucb: Optional[float]
    observed_gain: Optional[float]
    observed_net_gain: Optional[float]
    confidence_gap: Optional[float]
    observed_below_gain_ucb: Optional[bool]
    one_step_action_alignment: Optional[bool]
    evaluable: bool
    missing_evidence: tuple[str, ...]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_one_step(
    decision: DecisionInputs,
    observed: ObservedUpdate,
) -> OneStepAssessment:
    """Compare one recorded decision to one realized main update.

    ``decision_cost`` is the projected marginal cost used by the online
    controller. ``actual_main_update_cost`` is used only for the realized net
    value. Incurred pilot/exploration cost is intentionally absent from both
    expressions and must be reported separately by the artifact adapter.
    """

    missing: list[str] = []
    if decision.delta_hat is None:
        missing.append("delta_hat")
    if decision.radius is None:
        missing.append("confidence_radius")
    if decision.decision_cost is None:
        missing.append("projected_decision_cost")
    if decision.actual_main_update_cost is None:
        missing.append("actual_main_update_cost")
    if observed.gain is None:
        missing.append("observed_one_step_gain")
    if not observed.evaluation_manifest_hash:
        missing.append("evaluation_manifest_hash")
    if not observed.checkpoint_before_hash:
        missing.append("checkpoint_before_hash")
    if not observed.checkpoint_after_hash:
        missing.append("checkpoint_after_hash")

    gain_ucb = None
    estimated_net_gain = None
    net_ucb = None
    if decision.delta_hat is not None and decision.radius is not None:
        gain_ucb = min(
            decision.support_upper,
            decision.delta_hat + decision.radius,
        )
    if decision.delta_hat is not None and decision.decision_cost is not None:
        estimated_net_gain = (
            decision.delta_hat - decision.lambda_cost * decision.decision_cost
        )
    if gain_ucb is not None and decision.decision_cost is not None:
        net_ucb = gain_ucb - decision.lambda_cost * decision.decision_cost

    observed_gain = observed.gain
    observed_net_gain = None
    if observed_gain is not None and decision.actual_main_update_cost is not None:
        observed_net_gain = (
            observed_gain
            - decision.lambda_cost * decision.actual_main_update_cost
        )
    confidence_gap = (
        gain_ucb - observed_gain
        if gain_ucb is not None and observed_gain is not None
        else None
    )
    covered = confidence_gap >= 0.0 if confidence_gap is not None else None
    alignment = None
    if observed_net_gain is not None:
        realized_continue = observed_net_gain > 0.0
        alignment = (decision.action == "continue") == realized_continue

    return OneStepAssessment(
        seed=decision.seed,
        round_index=decision.round_index,
        policy=decision.policy,
        action=decision.action,
        reason=decision.reason,
        evidence_gate=decision.evidence_gate,
        certified_hypotheses=decision.certified_hypotheses,
        unresolved_hypotheses=decision.unresolved_hypotheses,
        delta_hat=decision.delta_hat,
        radius=decision.radius,
        gain_ucb=gain_ucb,
        decision_cost=decision.decision_cost,
        actual_main_update_cost=decision.actual_main_update_cost,
        estimated_net_gain=estimated_net_gain,
        net_ucb=net_ucb,
        observed_gain=observed_gain,
        observed_net_gain=observed_net_gain,
        confidence_gap=confidence_gap,
        observed_below_gain_ucb=covered,
        one_step_action_alignment=alignment,
        evaluable=not missing,
        missing_evidence=tuple(sorted(set(missing))),
        interpretation=(
            "one_realized_main_update_not_expected_gain_or_theorem_validation"
        ),
    )


def cost_accounting_row(
    decision_cost: Mapping[str, Any] | None,
) -> dict[str, Optional[float]]:
    """Extract disjoint cost scopes without adding them together again."""

    source = decision_cost if isinstance(decision_cost, Mapping) else {}

    def number(key: str) -> Optional[float]:
        value = source.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if isfinite(value) and value >= 0.0 else None

    return {
        "projected_decision_cost": number("decision_cost"),
        "actual_main_update_cost": number("actual_main_update_cost"),
        "incurred_exploration_cost": number("incurred_exploration_cost"),
        "total_pipeline_cost": number("total_pipeline_cost"),
    }
