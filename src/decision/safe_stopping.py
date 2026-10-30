"""Audits required by the verified-failure safe-stopping appendix."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def verification_assumption_audit(
    payload: Mapping[str, Any],
    *,
    gamma: float,
    expected_K: int,
    expected_T: int,
    expected_delta: float,
) -> dict[str, Any]:
    """Check every mechanically testable part of Appendix Assumption 1.

    Conditional independence, the true verification margin, and pilot
    unbiasedness are distributional assumptions.  They are reported rather
    than incorrectly inferred from finite artifacts.
    """
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("verification margin gamma must be finite and positive")
    tau = float(payload["tau_v"])
    if gamma > min(tau, 1.0 - tau):
        raise ValueError(
            "verification margin gamma must keep tau_v +/- gamma inside [0, 1]"
        )
    observed_K = int(payload["K"])
    observed_T = int(payload["T"])
    observed_delta = float(payload["delta"])
    results = payload.get("results")
    if not isinstance(results, Sequence):
        raise ValueError("verification results must be a sequence")
    per_hypothesis: list[dict[str, Any]] = []
    radii_satisfy_margin = True
    for raw in results:
        if not isinstance(raw, Mapping):
            raise ValueError("verification result rows must be mappings")
        radius_value = raw.get("confidence_radius")
        radius = None if radius_value is None else float(radius_value)
        bound_ok = radius is not None and radius <= gamma / 2.0
        radii_satisfy_margin = radii_satisfy_margin and bound_ok
        per_hypothesis.append(
            {
                "hypothesis_id": str(raw.get("hypothesis_id", "")),
                "state": (
                    "certified"
                    if raw.get("certified") is True
                    else "unresolved_or_refuted"
                ),
                "n": raw.get("n"),
                "confidence_radius": radius,
                "required_max_radius": gamma / 2.0,
                "radius_condition_met": bound_ok,
            }
        )
    accounting_matches = (
        observed_K == expected_K
        and observed_T == expected_T
        and math.isclose(observed_delta, expected_delta, rel_tol=0.0, abs_tol=1e-15)
    )
    mechanical_checks_pass = accounting_matches and radii_satisfy_margin
    return {
        "appendix_assumption": "Assumption 1",
        "gamma": gamma,
        "tau_v": tau,
        "expected": {
            "K": expected_K,
            "T": expected_T,
            "delta": expected_delta,
        },
        "observed": {
            "K": observed_K,
            "T": observed_T,
            "delta": observed_delta,
        },
        "family_wise_accounting_matches": accounting_matches,
        "all_observed_radii_at_most_gamma_over_two": radii_satisfy_margin,
        "mechanical_checks_pass": mechanical_checks_pass,
        "distributional_assumptions": {
            "verification_probes_conditionally_independent": "not_empirically_proven",
            "true_margin_separation": "not_observable_from_finite_samples",
            "pilot_gains_conditionally_independent_and_unbiased": "not_empirically_proven",
            "pilot_gains_bounded": "enforced_by_gain_estimation",
            "cost_known_before_decision": "enforced_by_frozen_cost_artifact",
        },
        "guarantee_status": (
            "conditional_on_unverifiable_distributional_assumptions"
            if mechanical_checks_pass
            else "appendix_preconditions_not_met"
        ),
        "hypotheses": per_hypothesis,
    }


def cumulative_cost_ledger(
    prior_total: float,
    *,
    incurred_exploration: float,
    realized_intervention: float,
) -> dict[str, float]:
    """Add each realized phase once; projected costs never enter the ledger."""
    values = (prior_total, incurred_exploration, realized_intervention)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Cost ledger values must be finite and non-negative")
    return {
        "cumulative_before": prior_total,
        "incurred_exploration": incurred_exploration,
        "realized_intervention": realized_intervention,
        "cumulative_after": (
            prior_total + incurred_exploration + realized_intervention
        ),
    }
