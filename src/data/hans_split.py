"""Deterministic diagnostic/final partitioning for HANS oracle debugging."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


HANS_FIVE_WAY_SPLIT_VERSION = "hans_five_way_v1"


@dataclass(frozen=True)
class HansRoleSplit:
    diagnostic: tuple[dict[str, Any], ...]
    final: tuple[dict[str, Any], ...]

    @property
    def diagnostic_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.diagnostic)

    @property
    def final_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.final)


@dataclass(frozen=True)
class HansThreeWaySplit:
    hypothesis_discovery: tuple[dict[str, Any], ...]
    verification: tuple[dict[str, Any], ...]
    final_test: tuple[dict[str, Any], ...]

    @property
    def discovery_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.hypothesis_discovery)

    @property
    def verification_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.verification)

    @property
    def final_test_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.final_test)


@dataclass(frozen=True)
class HansFiveWaySplit:
    """Five disjoint HANS roles used by Milestones 4--6.

    The role boundaries are prefixes of the same stable hash ordering used by
    the earlier three-way split. With four fractions that sum to 0.5, the
    final-test half is therefore byte-for-byte identical to the old split.
    """

    hypothesis_discovery: tuple[dict[str, Any], ...]
    verification: tuple[dict[str, Any], ...]
    intervention_pool: tuple[dict[str, Any], ...]
    gain_eval: tuple[dict[str, Any], ...]
    final_test: tuple[dict[str, Any], ...]
    version: str = HANS_FIVE_WAY_SPLIT_VERSION

    @property
    def discovery_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.hypothesis_discovery)

    @property
    def verification_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.verification)

    @property
    def intervention_pool_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.intervention_pool)

    @property
    def gain_eval_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.gain_eval)

    @property
    def final_test_ids(self) -> tuple[str, ...]:
        return tuple(str(record["id"]) for record in self.final_test)


def split_hans_records(
    records: Sequence[Mapping[str, Any]],
    diagnostic_fraction: float,
    split_seed: int,
) -> HansRoleSplit:
    """Split by a deterministic hash ranking over stable example IDs."""
    if not 0.0 < diagnostic_fraction < 1.0:
        raise ValueError("diagnostic_fraction must be strictly between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two HANS records are required for two roles")

    ids = [str(record.get("id", "")) for record in records]
    if any(not example_id for example_id in ids):
        raise ValueError("Every HANS record must have a non-empty stable id")
    if len(ids) != len(set(ids)):
        raise ValueError("HANS split requires unique stable example IDs")

    diagnostic_size = round(len(records) * diagnostic_fraction)
    diagnostic_size = min(max(diagnostic_size, 1), len(records) - 1)

    ranked_ids = sorted(
        ids,
        key=lambda example_id: (
            hashlib.sha256(
                f"{split_seed}:{example_id}".encode("utf-8")
            ).digest(),
            example_id,
        ),
    )
    diagnostic_ids = set(ranked_ids[:diagnostic_size])
    diagnostic: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        if str(copied["id"]) in diagnostic_ids:
            copied["split_role"] = "diagnostic"
            diagnostic.append(copied)
        else:
            copied["split_role"] = "final"
            final.append(copied)

    result = HansRoleSplit(tuple(diagnostic), tuple(final))
    overlap = set(result.diagnostic_ids) & set(result.final_ids)
    if overlap:
        raise RuntimeError(f"Diagnostic/final split overlap detected: {overlap}")
    return result


def split_hans_three_way(
    records: Sequence[Mapping[str, Any]],
    discovery_fraction: float,
    verification_fraction: float,
    split_seed: int,
) -> HansThreeWaySplit:
    """Create deterministic, pairwise-disjoint HANS experiment roles."""
    if not 0.0 < discovery_fraction < 1.0:
        raise ValueError("discovery_fraction must be strictly between 0 and 1")
    if not 0.0 < verification_fraction < 1.0:
        raise ValueError("verification_fraction must be strictly between 0 and 1")
    if discovery_fraction + verification_fraction >= 1.0:
        raise ValueError(
            "discovery_fraction + verification_fraction must be below 1"
        )
    if len(records) < 3:
        raise ValueError("At least three HANS records are required for three roles")

    ids = [str(record.get("id", "")) for record in records]
    if any(not example_id for example_id in ids):
        raise ValueError("Every HANS record must have a non-empty stable id")
    if len(ids) != len(set(ids)):
        raise ValueError("HANS split requires unique stable example IDs")

    discovery_size = max(1, round(len(records) * discovery_fraction))
    verification_size = max(1, round(len(records) * verification_fraction))
    if discovery_size + verification_size >= len(records):
        overflow = discovery_size + verification_size - (len(records) - 1)
        if verification_size > discovery_size:
            verification_size -= overflow
        else:
            discovery_size -= overflow
    if discovery_size < 1 or verification_size < 1:
        raise ValueError("Fractions leave too few records for all three roles")

    ranked_ids = sorted(
        ids,
        key=lambda example_id: (
            hashlib.sha256(
                f"{split_seed}:{example_id}".encode("utf-8")
            ).digest(),
            example_id,
        ),
    )
    discovery_ids = set(ranked_ids[:discovery_size])
    verification_ids = set(
        ranked_ids[discovery_size : discovery_size + verification_size]
    )

    discovery: list[dict[str, Any]] = []
    verification: list[dict[str, Any]] = []
    final_test: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        example_id = str(copied["id"])
        if example_id in discovery_ids:
            copied["split_role"] = "hypothesis_discovery"
            discovery.append(copied)
        elif example_id in verification_ids:
            copied["split_role"] = "verification"
            verification.append(copied)
        else:
            copied["split_role"] = "final_test"
            final_test.append(copied)

    result = HansThreeWaySplit(
        tuple(discovery), tuple(verification), tuple(final_test)
    )
    id_sets = (
        set(result.discovery_ids),
        set(result.verification_ids),
        set(result.final_test_ids),
    )
    if any(id_sets[left] & id_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("HANS three-way split overlap detected")
    if set().union(*id_sets) != set(ids):
        raise RuntimeError("HANS three-way split does not cover every input ID")
    return result


def split_hans_five_way(
    records: Sequence[Mapping[str, Any]],
    discovery_fraction: float,
    verification_fraction: float,
    intervention_fraction: float,
    gain_eval_fraction: float,
    split_seed: int,
) -> HansFiveWaySplit:
    """Create a deterministic and exhaustive five-role HANS partition."""
    fractions = {
        "discovery_fraction": discovery_fraction,
        "verification_fraction": verification_fraction,
        "intervention_fraction": intervention_fraction,
        "gain_eval_fraction": gain_eval_fraction,
    }
    invalid = [name for name, value in fractions.items() if not 0.0 < value < 1.0]
    if invalid:
        raise ValueError(f"Split fractions must be strictly between 0 and 1: {invalid}")
    if sum(fractions.values()) >= 1.0:
        raise ValueError("The four non-final split fractions must sum to below 1")
    if len(records) < 5:
        raise ValueError("At least five HANS records are required for five roles")

    ids = [str(record.get("id", "")) for record in records]
    if any(not example_id for example_id in ids):
        raise ValueError("Every HANS record must have a non-empty stable id")
    if len(ids) != len(set(ids)):
        raise ValueError("HANS split requires unique stable example IDs")

    # Allocate a single non-final prefix first. This makes the final-test suffix
    # stable even when per-role rounding would otherwise move its boundary.
    fraction_values = list(fractions.values())
    non_final_fraction = sum(fraction_values)
    non_final_size = round(len(records) * non_final_fraction)
    non_final_size = min(max(non_final_size, 4), len(records) - 1)
    raw_sizes = [
        non_final_size * value / non_final_fraction for value in fraction_values
    ]
    sizes = [max(1, int(value)) for value in raw_sizes]
    while sum(sizes) < non_final_size:
        candidates = sorted(
            range(len(sizes)),
            key=lambda index: (raw_sizes[index] - int(raw_sizes[index]), -index),
            reverse=True,
        )
        sizes[candidates[0]] += 1
        raw_sizes[candidates[0]] = float(sizes[candidates[0]])
    while sum(sizes) > non_final_size:
        candidates = [index for index, size in enumerate(sizes) if size > 1]
        if not candidates:
            raise ValueError("Fractions leave too few records for all five roles")
        index = max(candidates, key=lambda item: sizes[item] - raw_sizes[item])
        sizes[index] -= 1

    ranked_ids = sorted(
        ids,
        key=lambda example_id: (
            hashlib.sha256(f"{split_seed}:{example_id}".encode("utf-8")).digest(),
            example_id,
        ),
    )
    boundaries: list[set[str]] = []
    offset = 0
    for size in sizes:
        boundaries.append(set(ranked_ids[offset : offset + size]))
        offset += size

    role_names = (
        "hypothesis_discovery",
        "verification",
        "intervention_pool",
        "gain_eval",
    )
    role_records: dict[str, list[dict[str, Any]]] = {
        role: [] for role in (*role_names, "final_test")
    }
    for record in records:
        copied = dict(record)
        example_id = str(copied["id"])
        role = "final_test"
        for role_name, role_ids in zip(role_names, boundaries, strict=True):
            if example_id in role_ids:
                role = role_name
                break
        copied["split_role"] = role
        role_records[role].append(copied)

    result = HansFiveWaySplit(
        hypothesis_discovery=tuple(role_records["hypothesis_discovery"]),
        verification=tuple(role_records["verification"]),
        intervention_pool=tuple(role_records["intervention_pool"]),
        gain_eval=tuple(role_records["gain_eval"]),
        final_test=tuple(role_records["final_test"]),
    )
    id_sets = (
        set(result.discovery_ids),
        set(result.verification_ids),
        set(result.intervention_pool_ids),
        set(result.gain_eval_ids),
        set(result.final_test_ids),
    )
    if any(
        id_sets[left] & id_sets[right]
        for left in range(5)
        for right in range(left + 1, 5)
    ):
        raise RuntimeError("HANS five-way split overlap detected")
    if set().union(*id_sets) != set(ids):
        raise RuntimeError("HANS five-way split does not cover every input ID")
    return result
