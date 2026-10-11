"""Validation for dedicated offline reference roles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


REFERENCE_ROLES = ("verification_reference", "gain_reference")


def validate_reference_disjointness(
    reference_records: Sequence[Mapping[str, Any]],
    online_ids: Sequence[str],
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Reject overlap with every online role and between reference roles."""
    online = {str(value) for value in online_ids}
    seen: set[str] = set()
    grouped: dict[str, list[dict[str, Any]]] = {
        role: [] for role in REFERENCE_ROLES
    }
    for record in reference_records:
        example_id = str(record.get("id", ""))
        if not example_id:
            raise ValueError("Reference record is missing a stable id")
        if example_id in online:
            raise ValueError(f"Reference ID overlaps online data: {example_id}")
        if example_id in seen:
            raise ValueError(f"Duplicate reference ID: {example_id}")
        seen.add(example_id)
        role = str(record.get("split_role", ""))
        if role not in grouped:
            raise ValueError(
                f"Reference role must be one of {REFERENCE_ROLES}; got {role!r}"
            )
        grouped[role].append(dict(record))
    if not grouped["verification_reference"]:
        raise ValueError("verification_reference must not be empty")
    if not grouped["gain_reference"]:
        raise ValueError("gain_reference must not be empty")
    return {role: tuple(records) for role, records in grouped.items()}
