"""Bernoulli-KL versus range-aware Hoeffding bounds for cached V7 pilots.

The KL calculation is a same-round, fixed-shard conditional diagnostic.  Its
arithmetic does not prove that deterministic pilot seeds are conditionally
independent, nor does it make current-round pilots available before Teacher,
verification, generation, or selection.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.decision.final_mathematical_feasibility import (
    range_aware_hoeffding_radius,
)
from src.gain.range_aware import validate_fixed_shard_pilot_observations


@dataclass(frozen=True)
class KLHoeffdingComparison:
    m: int
    baseline_accuracy: float
    pilot_mean_gain: float
    mean_normalized_accuracy: float
    confidence_budget: float
    hoeffding_radius: float
    kl_radius: float
    hoeffding_gain_ucb: float
    kl_gain_ucb: float
    radius_reduction: float
    radius_reduction_percent: float
    conditional_independence_proved: bool
    guarantee_scope: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probability(value: float, name: str) -> float:
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1]")
    return checked


def bernoulli_kl(p: float, q: float) -> float:
    """Return binary relative entropy KL(Ber(p) || Ber(q))."""
    p_value = _probability(p, "p")
    q_value = _probability(q, "q")
    if p_value == q_value:
        return 0.0
    if q_value == 0.0:
        return math.inf if p_value > 0.0 else 0.0
    if q_value == 1.0:
        return math.inf if p_value < 1.0 else 0.0
    first = (
        0.0
        if p_value == 0.0
        else p_value * (math.log(p_value) - math.log(q_value))
    )
    second = (
        0.0
        if p_value == 1.0
        else (1.0 - p_value)
        * (math.log1p(-p_value) - math.log1p(-q_value))
    )
    result = first + second
    # Round-off can produce a tiny negative number near p == q.
    if result < 0.0 and result >= -1e-15:
        return 0.0
    if result < 0.0 or not math.isfinite(result):
        raise ValueError("binary KL produced an invalid finite result")
    return result


def kl_confidence_budget(*, m: int, delta: float, K: int, T: int) -> float:
    if isinstance(m, bool) or m <= 0:
        raise ValueError("m must be a positive integer")
    if isinstance(K, bool) or K <= 0 or isinstance(T, bool) or T <= 0:
        raise ValueError("K and T must be positive integers")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return math.log(2.0 * K * T / delta) / m


def bernoulli_kl_upper_bound(
    mean: float,
    *,
    confidence_budget: float,
    tolerance: float = 1e-13,
    max_iterations: int = 256,
) -> float:
    """Largest u in [mean, 1] satisfying KL(mean || u) <= budget."""
    p_value = _probability(mean, "mean")
    budget = float(confidence_budget)
    if not math.isfinite(budget) or budget < 0.0:
        raise ValueError("confidence_budget must be finite and non-negative")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if budget == 0.0 or p_value == 1.0:
        return p_value

    low = p_value  # feasible
    high = 1.0  # infeasible for p < 1
    for _ in range(max_iterations):
        if high - low <= tolerance:
            break
        midpoint = (low + high) / 2.0
        divergence = bernoulli_kl(p_value, midpoint)
        if divergence <= budget:
            low = midpoint
        else:
            high = midpoint
    return low


def normalized_pilot_accuracies(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_pilots: int,
    tolerance: float = 1e-12,
) -> tuple[list[float], dict[str, Any]]:
    """Validate X and return Y = X + baseline_accuracy in [0, 1]."""
    audit = validate_fixed_shard_pilot_observations(
        observations, required_pilots=required_pilots, tolerance=tolerance
    )
    if audit.get("status") != "mechanically_valid":
        raise ValueError("A complete mechanically valid pilot set is required")
    baseline = float(audit["baseline_accuracy"])
    normalized: list[float] = []
    for index, observation in enumerate(observations, start=1):
        value = float(observation["gain"]) + baseline
        if not math.isfinite(value) or value < -tolerance or value > 1.0 + tolerance:
            raise ValueError(f"Normalized pilot Y_{index} lies outside [0, 1]")
        # Only remove floating-point dust already admitted by the tolerance.
        normalized.append(min(1.0, max(0.0, value)))
    return normalized, audit


def compare_kl_and_range_aware_hoeffding(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_pilots: int,
    delta: float,
    K: int,
    T: int,
) -> tuple[KLHoeffdingComparison, dict[str, Any]]:
    normalized, audit = normalized_pilot_accuracies(
        observations, required_pilots=required_pilots
    )
    m = len(normalized)
    baseline = float(audit["baseline_accuracy"])
    mean_y = statistics.fmean(normalized)
    mean_gain = mean_y - baseline
    budget = kl_confidence_budget(m=m, delta=delta, K=K, T=T)
    upper_y = bernoulli_kl_upper_bound(
        mean_y, confidence_budget=budget
    )
    kl_radius = upper_y - mean_y
    hoeffding_radius = range_aware_hoeffding_radius(
        m, delta=delta, K=K, T=T, width=1.0
    )
    if kl_radius > hoeffding_radius + 1e-11:
        raise RuntimeError(
            "KL radius exceeds its range-aware Hoeffding/Pinsker envelope"
        )
    hoeffding_gain_ucb = min(1.0 - baseline, mean_gain + hoeffding_radius)
    kl_gain_ucb = upper_y - baseline
    reduction = hoeffding_radius - kl_radius
    comparison = KLHoeffdingComparison(
        m=m,
        baseline_accuracy=baseline,
        pilot_mean_gain=mean_gain,
        mean_normalized_accuracy=mean_y,
        confidence_budget=budget,
        hoeffding_radius=hoeffding_radius,
        kl_radius=kl_radius,
        hoeffding_gain_ucb=hoeffding_gain_ucb,
        kl_gain_ucb=kl_gain_ucb,
        radius_reduction=reduction,
        radius_reduction_percent=100.0 * reduction / hoeffding_radius,
        conditional_independence_proved=False,
        guarantee_scope=(
            "assumption-conditional same-round fixed-shard mean over training "
            "randomization; not a pre-round or population guarantee"
        ),
    )
    audit = {
        **audit,
        "normalization": "Y_j = X_j + baseline_accuracy = accuracy_after_j",
        "normalized_min": min(normalized),
        "normalized_max": max(normalized),
        "confidence_budget": budget,
        "kl_radius_le_hoeffding_radius": True,
    }
    return comparison, audit


def pre_round_bound_availability(
    *,
    pilot_source_round: int,
    decision_round: int,
    cross_round_transfer_valid: bool,
) -> dict[str, Any]:
    """Audit whether a pilot bound can support a pre-round decision."""
    if pilot_source_round <= 0 or decision_round <= 0:
        raise ValueError("round indices must be positive")
    if pilot_source_round >= decision_round:
        return {
            "available_predecision": False,
            "valid_for_next_round_gain": False,
            "reason": "same_or_future_round_pilots_not_available_predecision",
        }
    if not cross_round_transfer_valid:
        return {
            "available_predecision": True,
            "valid_for_next_round_gain": False,
            "reason": "cross_round_drift_and_policy_transfer_unproved",
        }
    return {
        "available_predecision": True,
        "valid_for_next_round_gain": True,
        "reason": "predecision_and_transfer_assumptions_declared_valid",
    }
