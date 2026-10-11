"""Deterministic fixed-budget acquisition helpers for Stage 2."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample


def select_complete_pairs(
    examples: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    excluded_ids: Sequence[str] = (),
) -> tuple[list[SyntheticNLIExample], dict[str, int | bool]]:
    """Select up to ``budget`` examples without splitting any synthetic pair."""
    if budget <= 0 or budget % 2:
        raise ValueError("Synthetic example budget must be positive and even")
    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)
    complete: list[list[SyntheticNLIExample]] = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        pair_type = members[0].pair_type
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        if {item.pair_member for item in members} != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        if not excluded & {item.id for item in members}:
            complete.append(members)
    ranked = sorted(
        complete,
        key=lambda group: (
            hashlib.sha256(f"{seed}:stage2-pair:{group[0].pair_id}".encode()).digest(),
            group[0].pair_id,
        ),
    )
    selected = [item for group in ranked[: budget // 2] for item in group]
    return selected, {
        "requested_examples": budget,
        "actual_examples": len(selected),
        "requested_pairs": budget // 2,
        "actual_pairs": len(selected) // 2,
        "shortage": len(selected) < budget,
    }
