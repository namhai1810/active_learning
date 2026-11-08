"""Minimal next-round Cost-Aware stopping contract for V7 Active Learning.

This module deliberately contains no estimator, pilot allocator, risk-ceiling
rule, training code, or artifact reader.  It only combines evidence that was
constructed from information available before the next round:

    J = E[next_round_gain | F_t] - lambda * E[next_round_cost | F_t]
    U = valid_gain_UCB - lambda * valid_full_round_cost_lower_bound

STOP is certified only when U <= 0.  Missing or invalid evidence produces an
uncertified CONTINUE.  Budget exhaustion is a separate terminal condition.

The frozen V7 v2/v2.1 implementation remains in ``v7_minimal_cost_aware.py``
for historical reproducibility; this contract is not wired into the runtime.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


Action = Literal["stop", "continue"]
Certification = Literal["certified_stop", "uncertified_continue", "budget_stop"]


@dataclass(frozen=True)
class PreDecisionBound:
    """One confidence bound plus the provenance needed for online use."""

    value: float | None
    valid: bool
    available_pre_decision: bool
    uses_future_round_data: bool = False
    uses_sealed_test: bool = False
    source: str = "unspecified"

    def invalid_reasons(self, *, nonnegative: bool) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.valid:
            reasons.append("bound_not_statistically_valid")
        if not self.available_pre_decision:
            reasons.append("bound_not_available_pre_decision")
        if self.uses_future_round_data:
            reasons.append("future_round_data_used")
        if self.uses_sealed_test:
            reasons.append("sealed_test_used")
        if self.value is None:
            reasons.append("bound_value_missing")
        elif not math.isfinite(self.value):
            reasons.append("bound_value_not_finite")
        elif nonnegative and self.value < 0:
            reasons.append("cost_lower_bound_negative")
        return tuple(reasons)


@dataclass(frozen=True)
class NextRoundEvidence:
    """Evidence for one V7 round action, conditional on F_t."""

    gain_ucb: PreDecisionBound
    full_round_cost_lower_bound: PreDecisionBound
    lambda_cost: float
    expected_gain: float | None = None
    expected_full_round_cost: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.lambda_cost) or self.lambda_cost < 0:
            raise ValueError("lambda_cost must be finite and non-negative")
        if self.expected_gain is not None and not math.isfinite(self.expected_gain):
            raise ValueError("expected_gain must be finite when provided")
        if self.expected_full_round_cost is not None and (
            not math.isfinite(self.expected_full_round_cost)
            or self.expected_full_round_cost < 0
        ):
            raise ValueError(
                "expected_full_round_cost must be finite and non-negative when provided"
            )


@dataclass(frozen=True)
class NextRoundDecision:
    action: Action
    certification: Certification
    reason: str
    expected_net_gain: float | None
    gain_ucb: float | None
    full_round_cost_lower_bound: float | None
    optimistic_net_gain: float | None
    lambda_cost: float
    evidence_valid: bool
    evidence_issues: tuple[str, ...]
    budget_exhausted: bool

    @property
    def certified_stop(self) -> bool:
        return self.certification == "certified_stop"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_next_round(
    evidence: NextRoundEvidence,
    *,
    budget_exhausted: bool = False,
) -> NextRoundDecision:
    """Apply the single next-round expected-net-gain stopping objective.

    This function never treats missing evidence as a STOP certificate.  It
    also never upgrades a positive UCB to a statistically certified CONTINUE;
    CONTINUE simply means that the STOP condition has not been proven.
    """

    expected_net = None
    if (
        evidence.expected_gain is not None
        and evidence.expected_full_round_cost is not None
    ):
        expected_net = (
            evidence.expected_gain
            - evidence.lambda_cost * evidence.expected_full_round_cost
        )

    if budget_exhausted:
        return NextRoundDecision(
            action="stop",
            certification="budget_stop",
            reason="maximum_budget_reached",
            expected_net_gain=expected_net,
            gain_ucb=None,
            full_round_cost_lower_bound=None,
            optimistic_net_gain=None,
            lambda_cost=evidence.lambda_cost,
            evidence_valid=False,
            evidence_issues=("budget_exhaustion_is_not_a_gain_cost_certificate",),
            budget_exhausted=True,
        )

    gain_issues = tuple(
        f"gain:{reason}"
        for reason in evidence.gain_ucb.invalid_reasons(nonnegative=False)
    )
    cost_issues = tuple(
        f"cost:{reason}"
        for reason in evidence.full_round_cost_lower_bound.invalid_reasons(
            nonnegative=True
        )
    )
    issues = gain_issues + cost_issues
    if issues:
        return NextRoundDecision(
            action="continue",
            certification="uncertified_continue",
            reason="insufficient_valid_next_round_evidence",
            expected_net_gain=expected_net,
            gain_ucb=None,
            full_round_cost_lower_bound=None,
            optimistic_net_gain=None,
            lambda_cost=evidence.lambda_cost,
            evidence_valid=False,
            evidence_issues=issues,
            budget_exhausted=False,
        )

    assert evidence.gain_ucb.value is not None
    assert evidence.full_round_cost_lower_bound.value is not None
    optimistic_net = (
        evidence.gain_ucb.value
        - evidence.lambda_cost * evidence.full_round_cost_lower_bound.value
    )
    if optimistic_net <= 0:
        action: Action = "stop"
        certification: Certification = "certified_stop"
        reason = "nonpositive_valid_next_round_net_ucb"
    else:
        action = "continue"
        certification = "uncertified_continue"
        reason = "positive_next_round_net_ucb"
    return NextRoundDecision(
        action=action,
        certification=certification,
        reason=reason,
        expected_net_gain=expected_net,
        gain_ucb=evidence.gain_ucb.value,
        full_round_cost_lower_bound=evidence.full_round_cost_lower_bound.value,
        optimistic_net_gain=optimistic_net,
        lambda_cost=evidence.lambda_cost,
        evidence_valid=True,
        evidence_issues=(),
        budget_exhausted=False,
    )
