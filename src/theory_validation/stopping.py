"""Offline-only one-step audit of a previously saved online STOP decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.decision.cost_aware import compute_net_value


def audit_safe_stopping(
    *,
    online_action: str,
    reference_statuses: Mapping[str, str],
    delta_references: Mapping[str, float],
    costs: Mapping[str, float],
    lambda_cost: float,
) -> dict[str, Any]:
    if online_action not in {"stop", "continue"}:
        raise ValueError("online_action must be stop or continue")
    if online_action == "continue":
        return {
            "online_action": "continue",
            "audit_performed": False,
            "reason": "online_method_did_not_stop",
            "candidate_results": [],
            "best_remaining_reference_net_value": None,
            "empirical_safe_stop_pass": None,
        }

    robust_ids = sorted(
        hypothesis_id
        for hypothesis_id, status in reference_statuses.items()
        if status == "robust_valid"
    )
    missing = [
        hypothesis_id
        for hypothesis_id in robust_ids
        if hypothesis_id not in delta_references or hypothesis_id not in costs
    ]
    if missing:
        return {
            "online_action": "stop",
            "audit_performed": False,
            "reason": "missing_reference_gain_or_cost",
            "missing_hypothesis_ids": missing,
            "candidate_results": [],
            "best_remaining_reference_net_value": None,
            "empirical_safe_stop_pass": None,
        }

    candidate_results = [
        {
            "hypothesis_id": hypothesis_id,
            "delta_reference": delta_references[hypothesis_id],
            "cost": costs[hypothesis_id],
            "lambda_cost": lambda_cost,
            "reference_net_value": compute_net_value(
                delta_references[hypothesis_id],
                costs[hypothesis_id],
                lambda_cost,
            ),
        }
        for hypothesis_id in robust_ids
    ]
    best = (
        max(row["reference_net_value"] for row in candidate_results)
        if candidate_results
        else None
    )
    return {
        "online_action": "stop",
        "audit_performed": True,
        "reason": "offline_reference_audit_completed",
        "candidate_results": candidate_results,
        "best_remaining_reference_net_value": best,
        "empirical_safe_stop_pass": best is None or best <= 0.0,
        "scope_statement": (
            "No robustly valid candidate failure in the evaluated candidate set "
            "had positive offline estimated one-step net value."
            if best is None or best <= 0.0
            else "At least one robustly valid candidate failure in the evaluated "
            "candidate set had positive offline estimated one-step net value."
        ),
        "not_ruled_out": [
            "multi-step repair sequences",
            "interactions between hypotheses",
            "unseen hypotheses",
            "future hypotheses outside the current candidate set",
        ],
    }
