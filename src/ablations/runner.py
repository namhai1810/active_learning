"""Pure strategies for the seven controlled one-round ablations."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.ablations.types import AblationOutcome
from src.decision.cost_aware import (
    build_decision_candidates,
    compute_net_value,
    compute_optimistic_net_value,
    select_intervention,
)
from src.decision.types import DecisionCandidate, InterventionDecision


ABLATION_VARIANTS = (
    "full",
    "no_verification",
    "point_verification",
    "no_gain_ucb",
    "no_cost",
    "random_hypothesis",
    "fixed_budget",
)


def _accepted_ids(
    variant: str,
    hypothesis_ids: Sequence[str],
    verification_results: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    known = set(hypothesis_ids)
    if len(known) != len(hypothesis_ids):
        raise ValueError("Candidate hypothesis IDs must be unique")
    if variant == "no_verification":
        return tuple(sorted(known))

    accepted = []
    seen: set[str] = set()
    for result in verification_results:
        hypothesis_id = str(result.get("hypothesis_id", ""))
        if not hypothesis_id:
            raise ValueError("Verification result is missing hypothesis_id")
        if hypothesis_id in seen:
            raise ValueError(f"Duplicate verification result: {hypothesis_id}")
        seen.add(hypothesis_id)
        if hypothesis_id not in known:
            raise ValueError(f"Verification result references unknown {hypothesis_id}")
        if variant == "point_verification":
            nu_hat = result.get("nu_hat")
            threshold = result.get("threshold")
            accept = (
                isinstance(nu_hat, (int, float))
                and not isinstance(nu_hat, bool)
                and isinstance(threshold, (int, float))
                and not isinstance(threshold, bool)
                and float(nu_hat) >= float(threshold)
            )
        else:
            accept = result.get("certified") is True
        if accept:
            accepted.append(hypothesis_id)
    return tuple(sorted(accepted))


def _variant_candidates(
    variant: str,
    candidates: Sequence[DecisionCandidate],
) -> list[DecisionCandidate]:
    changed = []
    for candidate in candidates:
        if variant == "no_gain_ucb":
            value = compute_net_value(
                candidate.delta_hat,
                candidate.cost,
                candidate.lambda_cost,
            )
            changed.append(
                replace(
                    candidate,
                    estimated_net_value=value,
                    optimistic_net_value=value,
                )
            )
        elif variant == "no_cost":
            changed.append(
                replace(
                    candidate,
                    estimated_net_value=candidate.delta_hat,
                    optimistic_net_value=compute_optimistic_net_value(
                        candidate.delta_hat,
                        candidate.confidence_radius,
                        0.0,
                        candidate.lambda_cost,
                    ),
                )
            )
        else:
            changed.append(candidate)
    if variant == "random_hypothesis":
        return sorted(changed, key=lambda item: item.hypothesis_id)
    return sorted(
        changed,
        key=lambda item: (-item.optimistic_net_value, item.hypothesis_id),
    )


def _continue(
    candidate: DecisionCandidate,
    *,
    max_optimistic_value: float,
    reason: str,
) -> InterventionDecision:
    return InterventionDecision(
        action="continue",
        selected_hypothesis_id=candidate.hypothesis_id,
        max_optimistic_value=max_optimistic_value,
        delta_hat=candidate.delta_hat,
        confidence_radius=candidate.confidence_radius,
        cost=candidate.cost,
        lambda_cost=candidate.lambda_cost,
        estimated_net_value=candidate.estimated_net_value,
        optimistic_net_value=candidate.optimistic_net_value,
        reason=reason,
    )


def evaluate_ablation(
    *,
    variant: str,
    hypothesis_ids: Sequence[str],
    verification_results: Sequence[Mapping[str, Any]],
    gain_estimates: Sequence[Mapping[str, Any]],
    costs: Mapping[str, float],
    lambda_cost: float,
    seed: int,
) -> AblationOutcome:
    """Apply only the strategy change named by ``variant``."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant!r}")
    accepted = _accepted_ids(variant, hypothesis_ids, verification_results)
    accepted_set = set(accepted)
    synthetic_verification = [
        {"hypothesis_id": hypothesis_id, "certified": hypothesis_id in accepted_set}
        for hypothesis_id in hypothesis_ids
    ]
    full_candidates = build_decision_candidates(
        synthetic_verification,
        gain_estimates,
        costs,
        lambda_cost,
    )
    candidates = _variant_candidates(variant, full_candidates)

    if variant == "random_hypothesis" and candidates:
        max_value = max(item.optimistic_net_value for item in candidates)
        if max_value <= 0.0:
            decision = select_intervention(candidates, lambda_cost)
        else:
            selected = random.Random(seed).choice(candidates)
            decision = _continue(
                selected,
                max_optimistic_value=max_value,
                reason="positive_optimistic_net_value_random_selection",
            )
    elif variant == "fixed_budget" and candidates:
        selected = candidates[0]
        decision = _continue(
            selected,
            max_optimistic_value=selected.optimistic_net_value,
            reason="fixed_budget_never_stop",
        )
    else:
        decision = select_intervention(candidates, lambda_cost)

    return AblationOutcome(
        variant=variant,
        accepted_hypothesis_ids=accepted,
        candidates=tuple(candidates),
        decision=decision,
    )
