"""Pure helpers for checkpoint-conditioned hypothesis lifecycle management."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def _tokens(hypothesis: Mapping[str, Any]) -> frozenset[str]:
    text = " ".join(
        str(hypothesis.get(field, ""))
        for field in (
            "name",
            "description",
            "predicted_failure_behavior",
            "intervention_target",
        )
    ).lower()
    return frozenset(re.findall(r"[a-z0-9]+", text))


def hypothesis_similarity(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    """Token-Jaccard similarity over mechanism-defining hypothesis fields."""
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def merge_hypothesis_candidates(
    archive: Sequence[Mapping[str, Any]],
    proposed: Sequence[Mapping[str, Any]],
    *,
    active_ids: Sequence[str],
    maximum: int,
    near_duplicate_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retain active mechanisms and append genuinely novel proposals.

    The operation is deterministic and never silently evicts an active
    hypothesis.  New proposals are compared against the complete historical
    archive, including already resolved mechanisms, so renamed rediscoveries do
    not masquerade as novelty.
    """
    if maximum <= 0:
        raise ValueError("maximum hypotheses must be positive")
    if not 0.0 <= near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must lie in [0, 1]")
    archive_rows = [dict(item) for item in archive]
    archive_by_id = {str(item.get("id", "")): item for item in archive_rows}
    if "" in archive_by_id or len(archive_by_id) != len(archive_rows):
        raise ValueError("Archived hypotheses require unique non-empty IDs")
    active = []
    for hypothesis_id in active_ids:
        if hypothesis_id not in archive_by_id:
            raise ValueError(f"Unknown active hypothesis {hypothesis_id!r}")
        if hypothesis_id not in {str(item["id"]) for item in active}:
            active.append(archive_by_id[hypothesis_id])
    if len(active) > maximum:
        raise ValueError("maximum cannot be smaller than the active archive")

    novel: list[dict[str, Any]] = []
    reactivated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    comparison = list(archive_rows)
    for raw in proposed:
        item = dict(raw)
        hypothesis_id = str(item.get("id", ""))
        if not hypothesis_id:
            raise ValueError("Proposed hypotheses require non-empty IDs")
        duplicate_of: str | None = None
        similarity = 0.0
        for prior in comparison:
            current = (
                1.0
                if hypothesis_id == str(prior["id"])
                else hypothesis_similarity(item, prior)
            )
            if current > similarity:
                similarity = current
                duplicate_of = str(prior["id"])
        if similarity >= near_duplicate_threshold:
            chosen_ids = {
                str(value["id"]) for value in [*active, *reactivated, *novel]
            }
            if (
                duplicate_of is not None
                and duplicate_of not in chosen_ids
                and len(active) + len(reactivated) + len(novel) < maximum
            ):
                # A previously resolved mechanism may recur after later
                # training. Re-verify the canonical archived hypothesis rather
                # than admitting a renamed duplicate or suppressing recurrence.
                reactivated.append(archive_by_id[duplicate_of])
                rejected.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "reason": "historical_mechanism_reactivated",
                        "duplicate_of": duplicate_of,
                        "similarity": similarity,
                    }
                )
                continue
            rejected.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "reason": "historical_near_duplicate",
                    "duplicate_of": duplicate_of,
                    "similarity": similarity,
                }
            )
            continue
        if len(active) + len(reactivated) + len(novel) >= maximum:
            rejected.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "reason": "maximum_candidates_reached",
                }
            )
            continue
        novel.append(item)
        comparison.append(item)
    return active + reactivated + novel, {
        "active_retained": [str(item["id"]) for item in active],
        "historical_reactivated": [str(item["id"]) for item in reactivated],
        "novel_accepted": [str(item["id"]) for item in novel],
        "proposals_rejected": rejected,
    }


def verification_states(results: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert fresh Hoeffding results into certified/unresolved/refuted states."""
    tau = float(results["tau_v"])
    states: dict[str, dict[str, Any]] = {}
    for result in results.get("results", []):
        hypothesis_id = str(result["hypothesis_id"])
        estimate = result.get("nu_hat")
        radius = result.get("confidence_radius")
        lower = result.get("lower_bound")
        upper = None
        if estimate is not None and radius is not None:
            upper = min(1.0, float(estimate) + float(radius))
        reliable = bool(result.get("probe_design", {}).get("reliable_for_lcb", False))
        if reliable and lower is not None and float(lower) >= tau:
            state = "certified"
        elif reliable and upper is not None and upper < tau:
            state = "refuted"
        else:
            state = "unresolved"
        states[hypothesis_id] = {
            "state": state,
            "certified": state == "certified",
            "nu_hat": estimate,
            "lower_bound": lower,
            "upper_bound": upper,
            "n": result.get("n"),
            "metric": result.get("certification_metric"),
            "threshold": tau,
        }
    return states

