"""Range-aware pilot-gain estimation for a fixed HANS evaluation shard.

The guarantee in this module is deliberately conditional on the frozen shard,
starting checkpoint, selected V7 action, and preceding run history.  It does
not turn repeated training randomizations into independent evaluation samples
and it does not claim a population-level HANS guarantee.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.gain.estimation import estimate_gain, gain_confidence_radius
from src.gain.types import GainEstimate


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fixed_baseline_gain_support(baseline_accuracy: float) -> tuple[float, float]:
    """Return the exact support of ``accuracy_after - baseline_accuracy``."""
    accuracy = float(baseline_accuracy)
    if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
        raise ValueError("baseline_accuracy must be a finite probability in [0, 1]")
    return -accuracy, 1.0 - accuracy


def validate_fixed_shard_pilot_observations(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_pilots: int,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Validate the mechanical assumptions needed by the range-aware bound.

    This validates shared metadata, the gain identity, bounded support, and
    distinct training seeds.  It cannot prove stochastic independence or
    unbiasedness of the deterministic seed schedule.
    """
    if required_pilots <= 0:
        raise ValueError("required_pilots must be positive")
    if not observations:
        return {
            "status": "insufficient_pilots",
            "observed_pilots": 0,
            "required_pilots": required_pilots,
            "conditional_independence_proved": False,
            "population_unbiasedness_claimed": False,
        }
    if len(observations) != required_pilots:
        raise ValueError(
            f"Expected {required_pilots} pilot observations, got {len(observations)}"
        )

    first = observations[0]
    eval_signature = _canonical(first.get("eval_ids"))
    train_signature = _canonical(first.get("train_ids"))
    checkpoint_signature = _canonical(first.get("starting_checkpoint"))
    risk_before = float(first["risk_before"])
    if not math.isfinite(risk_before) or not 0.0 <= risk_before <= 1.0:
        raise ValueError("risk_before must lie in [0, 1]")
    baseline_accuracy = 1.0 - risk_before
    support_lower, support_upper = fixed_baseline_gain_support(baseline_accuracy)

    seeds: list[int] = []
    for index, observation in enumerate(observations, start=1):
        if _canonical(observation.get("eval_ids")) != eval_signature:
            raise ValueError("All pilots in a round must use the same eval_ids")
        if _canonical(observation.get("train_ids")) != train_signature:
            raise ValueError("All pilots in a round must clone the same selected action")
        if _canonical(observation.get("starting_checkpoint")) != checkpoint_signature:
            raise ValueError("All pilots in a round must share the starting checkpoint")
        current_before = float(observation["risk_before"])
        if abs(current_before - risk_before) > tolerance:
            raise ValueError("Pilot baseline risk is not fixed within the round")
        risk_after = float(observation["risk_after"])
        if not math.isfinite(risk_after) or not 0.0 <= risk_after <= 1.0:
            raise ValueError("risk_after must lie in [0, 1]")
        gain = float(observation["gain"])
        expected_gain = (1.0 - risk_after) - baseline_accuracy
        if abs(gain - expected_gain) > tolerance:
            raise ValueError(
                f"Pilot {index} gain is not accuracy_after - accuracy_before"
            )
        if gain < support_lower - tolerance or gain > support_upper + tolerance:
            raise ValueError(
                f"Pilot {index} gain {gain} violates fixed-baseline support "
                f"[{support_lower}, {support_upper}]"
            )
        seeds.append(int(observation["pilot_seed"]))
    if len(seeds) != len(set(seeds)):
        raise ValueError("Pilot training seeds must be distinct within a round")

    return {
        "status": "mechanically_valid",
        "observed_pilots": len(observations),
        "required_pilots": required_pilots,
        "baseline_accuracy": baseline_accuracy,
        "support_lower": support_lower,
        "support_upper": support_upper,
        "support_width": support_upper - support_lower,
        "fixed_eval_ids": list(first.get("eval_ids") or []),
        "fixed_train_ids": list(first.get("train_ids") or []),
        "distinct_training_seeds": True,
        "gain_identity": "accuracy_after - accuracy_before",
        "target": (
            "expected_fixed_shard_empirical_accuracy_gain_over_configured_"
            "training_randomization"
        ),
        "conditional_independence_proved": False,
        "population_unbiasedness_claimed": False,
        "shared_eval_examples_counted_as_independent": False,
    }


def estimate_fixed_shard_range_aware_gain(
    observations: Sequence[Mapping[str, Any]],
    *,
    hypothesis_id: str,
    required_pilots: int,
    M: float,
    delta: float,
    K: int,
    T: int,
) -> tuple[GainEstimate, dict[str, Any]]:
    """Estimate gain using the width-one fixed-baseline Hoeffding radius."""
    audit = validate_fixed_shard_pilot_observations(
        observations,
        required_pilots=required_pilots,
    )
    estimate = estimate_gain(
        hypothesis_id,
        [float(item["gain"]) for item in observations],
        M=M,
        delta=delta,
        K=K,
        T=T,
        support_width=(1.0 if observations else None),
    )
    audit["confidence_method"] = "fixed_baseline_range_aware_hoeffding"
    audit["confidence_radius_formula"] = "sqrt(log(2*K*T/delta)/(2*m))"
    audit["confidence_radius"] = estimate.radius
    return estimate, audit


def compare_legacy_and_range_aware_bounds(
    observations: Sequence[Mapping[str, Any]],
    *,
    hypothesis_id: str,
    required_pilots: int,
    M: float,
    delta: float,
    K: int,
    T: int,
    decision_cost: float,
    lambda_cost: float,
) -> dict[str, Any]:
    """Return a pure CPU comparison without mutating a historical decision."""
    estimate, audit = estimate_fixed_shard_range_aware_gain(
        observations,
        hypothesis_id=hypothesis_id,
        required_pilots=required_pilots,
        M=M,
        delta=delta,
        K=K,
        T=T,
    )
    if estimate.delta_hat is None or estimate.radius is None:
        raise ValueError("A bound comparison requires a complete pilot set")
    old_radius = gain_confidence_radius(len(observations), M, K, T, delta)
    cost = float(decision_cost)
    penalty = float(lambda_cost) * cost

    def summarize(radius: float) -> dict[str, Any]:
        raw_ucb = float(estimate.delta_hat) + radius
        bounded_ucb = min(float(M), raw_ucb)
        net_ucb = bounded_ucb - penalty
        maximum_radius_for_stop = penalty - float(estimate.delta_hat)
        return {
            "radius": radius,
            "raw_gain_ucb": raw_ucb,
            "bounded_gain_ucb": bounded_ucb,
            "cost_penalty": penalty,
            "net_ucb": net_ucb,
            "action": "continue" if net_ucb > 0.0 else "stop",
            "reason": (
                "positive_mixed_policy_optimistic_net_gain"
                if net_ucb > 0.0
                else "nonpositive_mixed_policy_optimistic_net_gain"
            ),
            "net_ucb_gap_to_stop": max(0.0, net_ucb),
            "maximum_radius_for_stop": maximum_radius_for_stop,
            "radius_reduction_needed_to_stop": (
                max(0.0, radius - maximum_radius_for_stop)
                if maximum_radius_for_stop >= 0.0
                else None
            ),
            "additional_negative_gain_needed_at_zero_radius": max(
                0.0, float(estimate.delta_hat) - penalty
            ),
        }

    return {
        "delta_hat": estimate.delta_hat,
        "m": estimate.m,
        "decision_cost": cost,
        "lambda_cost": float(lambda_cost),
        "legacy_symmetric_support": summarize(old_radius),
        "range_aware_fixed_baseline": summarize(float(estimate.radius)),
        "range_aware_audit": audit,
    }
