"""Time-adaptive, sliding-window primitives for offline V7 stopping audits.

The estimator concentrates around an observation-weighted average of historical
conditional pilot means.  It is not, by itself, an estimator or confidence
bound for the next-round V7 policy gain.  A next-round certificate additionally
requires justified sampling, temporal-drift, policy-mismatch, and cost-bound
assumptions.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence


EstimatorKind = Literal["fixed_window", "cumulative"]


@dataclass(frozen=True)
class HistoricalWindowEstimate:
    decision_round: int
    estimator: EstimatorKind
    requested_window_length: int | None
    calendar_window_start: int
    calendar_window_end: int
    calendar_window_length: int
    calendar_rounds: tuple[int, ...]
    observation_counts_by_round: dict[int, int]
    observed_rounds: tuple[int, ...]
    zero_pilot_rounds: tuple[int, ...]
    n_eff: int
    mean_gain: float | None
    sample_variance: float | None
    weighted_mean_of_round_means: float | None
    maximum_source_round: int | None
    strict_predecision_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TimeAdaptiveDecision:
    action: str
    certification: str
    reason: str
    historical_mean: float | None
    statistical_radius: float | None
    rho: float | None
    eta: float | None
    cost_lcb: float
    lambda_cost: float
    gain_ucb: float | None
    net_ucb: float | None
    optimistic_net_ucb: float | None
    maximum_combined_rho_eta_for_stop: float | None
    evidence_issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def planned_test_family_size(
    *, decision_rounds: int, fixed_windows: int, include_cumulative: bool
) -> int:
    if decision_rounds <= 0 or fixed_windows <= 0:
        raise ValueError("decision_rounds and fixed_windows must be positive")
    estimators = fixed_windows + int(include_cumulative)
    return decision_rounds * estimators


def _calendar_rounds(
    *, decision_round: int, window_length: int | None
) -> tuple[EstimatorKind, tuple[int, ...]]:
    if decision_round < 2:
        raise ValueError("decision_round must be at least 2")
    last_completed = decision_round - 1
    if window_length is None:
        estimator: EstimatorKind = "cumulative"
        start = 1
    else:
        if window_length <= 0:
            raise ValueError("window_length must be positive")
        estimator = "fixed_window"
        start = max(1, last_completed - window_length + 1)
    return estimator, tuple(range(start, last_completed + 1))


def historical_window_estimate(
    observations_by_round: Mapping[int, Sequence[float]],
    *,
    decision_round: int,
    window_length: int | None,
) -> HistoricalWindowEstimate:
    """Pool eligible historical observations with observation-level weights.

    Values from ``decision_round`` or later are excluded even if supplied by
    the caller.  Rounds with no pilots remain in the calendar window and are
    reported explicitly, but contribute zero statistical weight.
    """
    estimator, rounds = _calendar_rounds(
        decision_round=decision_round, window_length=window_length
    )
    counts: dict[int, int] = {}
    values: list[float] = []
    round_means: list[tuple[int, float, int]] = []
    for round_index in rounds:
        current = [float(value) for value in observations_by_round.get(round_index, ())]
        if any(not math.isfinite(value) for value in current):
            raise ValueError("pilot gains must be finite")
        counts[round_index] = len(current)
        values.extend(current)
        if current:
            round_means.append((round_index, statistics.fmean(current), len(current)))

    n_eff = len(values)
    mean = statistics.fmean(values) if values else None
    variance = statistics.variance(values) if len(values) >= 2 else (
        0.0 if len(values) == 1 else None
    )
    weighted_round_mean = (
        math.fsum(mean_value * count for _, mean_value, count in round_means)
        / n_eff
        if n_eff
        else None
    )
    observed_rounds = tuple(index for index in rounds if counts[index] > 0)
    zero_rounds = tuple(index for index in rounds if counts[index] == 0)
    maximum_source = max(observed_rounds) if observed_rounds else None
    return HistoricalWindowEstimate(
        decision_round=decision_round,
        estimator=estimator,
        requested_window_length=window_length,
        calendar_window_start=rounds[0],
        calendar_window_end=rounds[-1],
        calendar_window_length=len(rounds),
        calendar_rounds=rounds,
        observation_counts_by_round=counts,
        observed_rounds=observed_rounds,
        zero_pilot_rounds=zero_rounds,
        n_eff=n_eff,
        mean_gain=mean,
        sample_variance=variance,
        weighted_mean_of_round_means=weighted_round_mean,
        maximum_source_round=maximum_source,
        strict_predecision_available=(
            maximum_source is None or maximum_source < decision_round
        ),
    )


def simultaneous_hoeffding_radius(
    n_eff: int,
    *,
    delta: float,
    n_tests: int,
    width: float = 1.0,
) -> float:
    """Bonferroni-corrected radius over the predeclared test family."""
    if n_eff <= 0 or n_tests <= 0:
        raise ValueError("n_eff and n_tests must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("width must be finite and positive")
    return width * math.sqrt(math.log(2.0 * n_tests / delta) / (2.0 * n_eff))


def weighted_temporal_drift(
    observation_counts_by_round: Mapping[int, int],
    *,
    decision_round: int,
    per_round_drift: float,
) -> float:
    """Illustrative drift allowance with observation-age weighting.

    This value is valid only under the separately stated assumption that the
    relevant conditional expected gain changes by at most ``per_round_drift``
    per round.  The function does not estimate or validate that assumption.
    """
    if decision_round < 2:
        raise ValueError("decision_round must be at least 2")
    if not math.isfinite(per_round_drift) or per_round_drift < 0:
        raise ValueError("per_round_drift must be finite and non-negative")
    denominator = 0
    numerator = 0.0
    for source_round, count in observation_counts_by_round.items():
        if source_round >= decision_round:
            raise ValueError("drift calculation cannot use current/future rounds")
        if count < 0:
            raise ValueError("observation counts must be non-negative")
        denominator += count
        numerator += count * per_round_drift * (decision_round - source_round)
    if denominator == 0:
        raise ValueError("drift calculation requires at least one observation")
    return numerator / denominator


def assess_time_adaptive_decision(
    *,
    estimate: HistoricalWindowEstimate,
    statistical_radius: float | None,
    rho: float | None,
    eta: float | None,
    cost_lcb: float,
    lambda_cost: float,
    sampling_assumptions_valid: bool,
    drift_assumption_valid: bool,
    policy_mismatch_assumption_valid: bool,
    cost_lcb_applicable_predecision: bool,
    retrospective_metrics: Mapping[str, Any] | None = None,
    budget_exhausted: bool = False,
) -> TimeAdaptiveDecision:
    """Apply the certificate gate without consuming retrospective metrics."""
    # Retrospective fields are accepted only to make accidental coupling
    # testable; policy calculations deliberately never read them.
    del retrospective_metrics
    if not math.isfinite(cost_lcb) or cost_lcb < 0:
        raise ValueError("cost_lcb must be finite and non-negative")
    if not math.isfinite(lambda_cost) or lambda_cost < 0:
        raise ValueError("lambda_cost must be finite and non-negative")
    if budget_exhausted:
        return TimeAdaptiveDecision(
            action="stop",
            certification="budget_stop",
            reason="maximum_budget_reached_not_gain_cost_certificate",
            historical_mean=estimate.mean_gain,
            statistical_radius=statistical_radius,
            rho=rho,
            eta=eta,
            cost_lcb=cost_lcb,
            lambda_cost=lambda_cost,
            gain_ucb=None,
            net_ucb=None,
            optimistic_net_ucb=None,
            maximum_combined_rho_eta_for_stop=None,
            evidence_issues=("budget_exhaustion_is_not_certified_stop",),
        )

    issues: list[str] = []
    if estimate.n_eff == 0 or estimate.mean_gain is None:
        issues.append("no_eligible_historical_pilots")
    if not estimate.strict_predecision_available:
        issues.append("future_round_leakage")
    if statistical_radius is None:
        issues.append("statistical_radius_missing")
    if not sampling_assumptions_valid:
        issues.append("conditional_independence_or_martingale_assumption_unproved")
    if rho is None or not drift_assumption_valid:
        issues.append("temporal_drift_bound_unknown")
    if eta is None or not policy_mismatch_assumption_valid:
        issues.append("policy_mismatch_bound_unknown")
    if not cost_lcb_applicable_predecision:
        issues.append("conditional_cost_lcb_applicability_unproved_predecision")

    optimistic = None
    allowance = None
    if estimate.mean_gain is not None and statistical_radius is not None:
        optimistic = (
            estimate.mean_gain
            + statistical_radius
            - lambda_cost * cost_lcb
        )
        allowance = -optimistic
    gain_ucb = None
    net = None
    if (
        estimate.mean_gain is not None
        and statistical_radius is not None
        and rho is not None
        and eta is not None
    ):
        gain_ucb = estimate.mean_gain + statistical_radius + rho + eta
        net = gain_ucb - lambda_cost * cost_lcb

    if issues:
        action = "continue"
        certification = "uncertified_continue"
        reason = "insufficient_valid_pre_round_evidence"
    elif net is not None and net <= 0:
        action = "stop"
        certification = "certified_stop"
        reason = "nonpositive_valid_time_adaptive_net_ucb"
    else:
        action = "continue"
        certification = "uncertified_continue"
        reason = "positive_time_adaptive_net_ucb"
    return TimeAdaptiveDecision(
        action=action,
        certification=certification,
        reason=reason,
        historical_mean=estimate.mean_gain,
        statistical_radius=statistical_radius,
        rho=rho,
        eta=eta,
        cost_lcb=cost_lcb,
        lambda_cost=lambda_cost,
        gain_ucb=gain_ucb,
        net_ucb=net,
        optimistic_net_ucb=optimistic,
        maximum_combined_rho_eta_for_stop=allowance,
        evidence_issues=tuple(issues),
    )
