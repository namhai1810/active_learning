"""Shared five-way split and manifest guards for the primary pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.data.hans_split import (
    HANS_FIVE_WAY_SPLIT_VERSION,
    HansFiveWaySplit,
    split_hans_five_way,
)


ROLE_ID_FIELDS = {
    "hypothesis_discovery": "hypothesis_discovery_ids",
    "verification": "verification_ids",
    "intervention_pool": "intervention_pool_ids",
    "gain_eval": "gain_eval_ids",
    "final_test": "final_test_ids",
}


def reconstruct_five_way_split(
    records: Sequence[Mapping[str, Any]], hans_config: Mapping[str, Any]
) -> HansFiveWaySplit:
    """Reconstruct roles from IDs while refusing pre-split source artifacts."""
    if {record.get("split_role") for record in records} != {None}:
        raise ValueError("Source predictions must be an unsplit evaluation artifact")
    id_views = [{"id": record.get("id")} for record in records]
    return split_hans_five_way(
        id_views,
        float(hans_config["discovery_fraction"]),
        float(hans_config["verification_fraction"]),
        float(hans_config["intervention_fraction"]),
        float(hans_config["gain_eval_fraction"]),
        int(hans_config["split_seed"]),
    )


def manifest_payload(
    split: HansFiveWaySplit,
    hans_config: Mapping[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "oracle_debug": False,
        "split_version": HANS_FIVE_WAY_SPLIT_VERSION,
        "split_seed": int(hans_config["split_seed"]),
        "discovery_fraction": float(hans_config["discovery_fraction"]),
        "verification_fraction": float(hans_config["verification_fraction"]),
        "intervention_fraction": float(hans_config["intervention_fraction"]),
        "gain_eval_fraction": float(hans_config["gain_eval_fraction"]),
        "num_total": sum(
            len(getattr(split, role)) for role in ROLE_ID_FIELDS
        ),
        "num_hypothesis_discovery": len(split.hypothesis_discovery),
        "num_verification": len(split.verification),
        "num_intervention_pool": len(split.intervention_pool),
        "num_gain_eval": len(split.gain_eval),
        "num_final_test": len(split.final_test),
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }


def validate_manifest(
    manifest: Mapping[str, Any], split: HansFiveWaySplit
) -> None:
    if manifest.get("split_version") != HANS_FIVE_WAY_SPLIT_VERSION:
        raise ValueError("Unsupported or missing five-way split version")
    expected = {
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }
    for field, ids in expected.items():
        if manifest.get(field) != ids:
            raise ValueError(f"Split manifest field {field!r} does not match source IDs")
    id_sets = [set(ids) for ids in expected.values()]
    if any(
        id_sets[left] & id_sets[right]
        for left in range(len(id_sets))
        for right in range(left + 1, len(id_sets))
    ):
        raise ValueError("Five-way split roles overlap")


def records_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        example_id = str(record.get("id", ""))
        if not example_id or example_id in indexed:
            raise ValueError(f"Missing or duplicate source ID: {example_id!r}")
        indexed[example_id] = dict(record)
    return indexed


def materialize_role(
    indexed: Mapping[str, Mapping[str, Any]],
    ids: Sequence[str],
    role: str,
) -> tuple[dict[str, Any], ...]:
    if role not in ROLE_ID_FIELDS:
        raise ValueError(f"Unknown split role: {role!r}")
    records = []
    for example_id in ids:
        if example_id not in indexed:
            raise ValueError(f"Split ID is absent from predictions: {example_id}")
        records.append({**indexed[example_id], "split_role": role})
    return tuple(records)
