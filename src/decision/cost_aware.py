"""Pure cost-aware selection and safe stopping equations."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.decision.types import DecisionCandidate, InterventionDecision


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_cost_inputs(cost: float, lambda_cost: float) -> tuple[float, float]:
    resolved_cost = _finite_number(cost, "cost")
    resolved_lambda = _finite_number(lambda_cost, "lambda_cost")
    if resolved_cost < 0:
        raise ValueError("cost must be non-negative")
    if resolved_lambda < 0:
        raise ValueError("lambda_cost must be non-negative")
    return resolved_cost, resolved_lambda


def compute_net_value(delta_hat: float, cost: float, lambda_cost: float) -> float:
    """Compute J_hat = Delta_hat - lambda_cost*C."""
    resolved_delta = _finite_number(delta_hat, "delta_hat")
    resolved_cost, resolved_lambda = _validate_cost_inputs(cost, lambda_cost)
    return resolved_delta - resolved_lambda * resolved_cost


def compute_optimistic_net_value(
    delta_hat: float,
    confidence_radius: float,
    cost: float,
    lambda_cost: float,
) -> float:
    """Compute U = Delta_hat + b - lambda_cost*C."""
    resolved_delta = _finite_number(delta_hat, "delta_hat")
    radius = _finite_number(confidence_radius, "confidence_radius")
    if radius < 0:
        raise ValueError("confidence_radius must be non-negative")
    resolved_cost, resolved_lambda = _validate_cost_inputs(cost, lambda_cost)
    return resolved_delta + radius - resolved_lambda * resolved_cost


def build_decision_candidates(
    verification_results: Sequence[Mapping[str, Any]],
    gain_estimates: Sequence[Mapping[str, Any]],
    costs: Mapping[str, float],
    lambda_cost: float,
    cost_breakdowns: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[DecisionCandidate]:
    """Join certified hypotheses to valid gain estimates and rank by U."""
    _validate_cost_inputs(0.0, lambda_cost)
    certified_ids: set[str] = set()
    seen_verification: set[str] = set()
    for result in verification_results:
        hypothesis_id = str(result.get("hypothesis_id", ""))
        if not hypothesis_id:
            raise ValueError("Verification result is missing hypothesis_id")
        if hypothesis_id in seen_verification:
            raise ValueError(f"Duplicate verification result: {hypothesis_id}")
        seen_verification.add(hypothesis_id)
        if result.get("certified") is True:
            certified_ids.add(hypothesis_id)

    seen_estimates: set[str] = set()
    candidates: list[DecisionCandidate] = []
    for estimate in gain_estimates:
        hypothesis_id = str(estimate.get("hypothesis_id", ""))
        if not hypothesis_id:
            raise ValueError("Gain estimate is missing hypothesis_id")
        if hypothesis_id in seen_estimates:
            raise ValueError(f"Duplicate gain estimate: {hypothesis_id}")
        seen_estimates.add(hypothesis_id)
        if hypothesis_id not in certified_ids:
            continue
        if estimate.get("status") != "estimated":
            continue
        m = estimate.get("m")
        if isinstance(m, bool) or not isinstance(m, int) or m <= 0:
            continue
        if estimate.get("delta_hat") is None or estimate.get("radius") is None:
            continue
        if hypothesis_id not in costs:
            raise ValueError(f"Missing intervention cost for {hypothesis_id}")

        delta_hat = _finite_number(estimate["delta_hat"], "delta_hat")
        radius = _finite_number(estimate["radius"], "confidence_radius")
        cost, resolved_lambda = _validate_cost_inputs(
            costs[hypothesis_id], lambda_cost
        )
        candidate = DecisionCandidate(
            hypothesis_id=hypothesis_id,
            delta_hat=delta_hat,
            confidence_radius=radius,
            cost=cost,
            lambda_cost=resolved_lambda,
            estimated_net_value=compute_net_value(
                delta_hat, cost, resolved_lambda
            ),
            optimistic_net_value=compute_optimistic_net_value(
                delta_hat, radius, cost, resolved_lambda
            ),
            cost_breakdown=(
                None
                if cost_breakdowns is None
                else dict(cost_breakdowns.get(hypothesis_id, {}))
            ),
        )
        candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda candidate: (
            -candidate.optimistic_net_value,
            candidate.hypothesis_id,
        ),
    )


def select_intervention(
    candidates: Sequence[DecisionCandidate],
    lambda_cost: float,
    *,
    max_cost: float | None = None,
) -> InterventionDecision:
    """Select argmax U among budget-eligible candidates, otherwise STOP."""
    _, resolved_lambda = _validate_cost_inputs(0.0, lambda_cost)
    ranked_all = sorted(
        candidates,
        key=lambda candidate: (
            -candidate.optimistic_net_value,
            candidate.hypothesis_id,
        ),
    )
    for candidate in ranked_all:
        if candidate.lambda_cost != resolved_lambda:
            raise ValueError("Candidate lambda_cost does not match decision lambda_cost")
    if max_cost is not None:
        available = _finite_number(max_cost, "max_cost")
        if available < 0:
            raise ValueError("max_cost must be non-negative")
        ranked = [candidate for candidate in ranked_all if candidate.cost <= available]
    else:
        ranked = ranked_all
    if not ranked_all:
        return InterventionDecision(
            action="stop",
            selected_hypothesis_id=None,
            max_optimistic_value=None,
            delta_hat=None,
            confidence_radius=None,
            cost=None,
            lambda_cost=resolved_lambda,
            estimated_net_value=None,
            optimistic_net_value=None,
            reason="no_eligible_certified_hypotheses",
        )
    if not ranked:
        return InterventionDecision(
            action="stop",
            selected_hypothesis_id=None,
            max_optimistic_value=max(item.optimistic_net_value for item in ranked_all),
            delta_hat=None,
            confidence_radius=None,
            cost=None,
            lambda_cost=resolved_lambda,
            estimated_net_value=None,
            optimistic_net_value=None,
            reason="budget_exhausted",
        )

    best = ranked[0]
    if best.optimistic_net_value <= 0.0:
        return InterventionDecision(
            action="stop",
            selected_hypothesis_id=None,
            max_optimistic_value=best.optimistic_net_value,
            delta_hat=None,
            confidence_radius=None,
            cost=None,
            lambda_cost=resolved_lambda,
            estimated_net_value=None,
            optimistic_net_value=None,
            reason="no_positive_optimistic_net_value",
        )
    return InterventionDecision(
        action="continue",
        selected_hypothesis_id=best.hypothesis_id,
        max_optimistic_value=best.optimistic_net_value,
        delta_hat=best.delta_hat,
        confidence_radius=best.confidence_radius,
        cost=best.cost,
        lambda_cost=resolved_lambda,
        estimated_net_value=best.estimated_net_value,
        optimistic_net_value=best.optimistic_net_value,
        reason="positive_optimistic_net_value",
        cost_breakdown=best.cost_breakdown,
    )
