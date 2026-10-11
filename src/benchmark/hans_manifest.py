"""One seed-independent frozen HANS manifest shared by every benchmark method."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from src.data.hans_split import HANS_FIVE_WAY_SPLIT_VERSION, split_hans_five_way


STAGE2_MANIFEST_VERSION = "stage2_hans_frozen_v1"
ROLE_FIELDS = (
    "hypothesis_discovery_ids",
    "verification_ids",
    "intervention_pool_ids",
    "gain_eval_ids",
    "final_test_ids",
)


def _hash_json(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenHansManifest:
    version: str
    split_algorithm: str
    split_seed: int
    dataset_id_hash: str
    hypothesis_discovery_ids: tuple[str, ...]
    verification_ids: tuple[str, ...]
    intervention_pool_ids: tuple[str, ...]
    gain_eval_ids: tuple[str, ...]
    final_test_ids: tuple[str, ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ROLE_FIELDS:
            value[field] = list(value[field])
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FrozenHansManifest":
        required = {
            "version",
            "split_algorithm",
            "split_seed",
            "dataset_id_hash",
            *ROLE_FIELDS,
            "manifest_hash",
        }
        if set(value) != required:
            raise ValueError(
                f"Frozen HANS manifest fields differ; missing={sorted(required-set(value))}, "
                f"extra={sorted(set(value)-required)}"
            )
        manifest = cls(
            version=str(value["version"]),
            split_algorithm=str(value["split_algorithm"]),
            split_seed=int(value["split_seed"]),
            dataset_id_hash=str(value["dataset_id_hash"]),
            **{field: tuple(str(item) for item in value[field]) for field in ROLE_FIELDS},
            manifest_hash=str(value["manifest_hash"]),
        )
        expected_hash = _hash_json(
            {key: item for key, item in manifest.to_dict().items() if key != "manifest_hash"}
        )
        if manifest.manifest_hash != expected_hash:
            raise ValueError("Frozen HANS manifest hash is invalid")
        return manifest


def build_frozen_hans_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    split_seed: int,
    discovery_fraction: float,
    verification_fraction: float,
    intervention_fraction: float,
    gain_eval_fraction: float,
) -> FrozenHansManifest:
    ids = [str(record.get("id", "")) for record in records]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError("Frozen HANS manifest requires unique non-empty IDs")
    split = split_hans_five_way(
        [{"id": item} for item in ids],
        discovery_fraction,
        verification_fraction,
        intervention_fraction,
        gain_eval_fraction,
        split_seed,
    )
    payload: dict[str, Any] = {
        "version": STAGE2_MANIFEST_VERSION,
        "split_algorithm": HANS_FIVE_WAY_SPLIT_VERSION,
        "split_seed": split_seed,
        "dataset_id_hash": _hash_json(sorted(ids)),
        "hypothesis_discovery_ids": list(split.discovery_ids),
        "verification_ids": list(split.verification_ids),
        "intervention_pool_ids": list(split.intervention_pool_ids),
        "gain_eval_ids": list(split.gain_eval_ids),
        "final_test_ids": list(split.final_test_ids),
    }
    payload["manifest_hash"] = _hash_json(payload)
    return FrozenHansManifest.from_mapping(payload)


def validate_frozen_hans_manifest(
    manifest: FrozenHansManifest,
    records: Sequence[Mapping[str, Any]],
) -> None:
    ids = [str(record.get("id", "")) for record in records]
    if manifest.version != STAGE2_MANIFEST_VERSION:
        raise ValueError("Unsupported Stage 2 HANS manifest version")
    if manifest.dataset_id_hash != _hash_json(sorted(ids)):
        raise ValueError("HANS dataset IDs do not match the frozen manifest")
    role_sets = [set(getattr(manifest, field)) for field in ROLE_FIELDS]
    if any(
        role_sets[left] & role_sets[right]
        for left in range(len(role_sets))
        for right in range(left + 1, len(role_sets))
    ):
        raise ValueError("Frozen HANS manifest roles overlap")
    if set().union(*role_sets) != set(ids):
        raise ValueError("Frozen HANS manifest is not exhaustive")


def _tokens(premise: str, hypothesis: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", f"{premise} {hypothesis}".lower()))


def audit_final_test_overlap(
    training_records: Sequence[Mapping[str, Any]],
    final_test_records: Sequence[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float,
) -> dict[str, Any]:
    """Post-selection leakage audit; results must never feed acquisition/training."""
    if not 0 < near_duplicate_threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0, 1]")
    train_ids = {str(row.get("id", "")) for row in training_records}
    final_ids = {str(row.get("id", "")) for row in final_test_records}
    id_overlap = sorted(train_ids & final_ids)
    final_tokens = [
        (str(row["id"]), _tokens(str(row["premise"]), str(row["hypothesis"])))
        for row in final_test_records
    ]
    near_duplicates: list[dict[str, Any]] = []
    for train in training_records:
        left = _tokens(str(train["premise"]), str(train["hypothesis"]))
        for final_id, right in final_tokens:
            union = left | right
            similarity = 1.0 if not union else len(left & right) / len(union)
            if similarity >= near_duplicate_threshold:
                near_duplicates.append(
                    {
                        "training_id": str(train.get("id", "")),
                        "final_test_id": final_id,
                        "token_jaccard": similarity,
                    }
                )
    return {
        "audit_only_after_selection_and_training": True,
        "used_to_change_training_or_selection": False,
        "id_overlap": id_overlap,
        "near_duplicates": near_duplicates,
        "passed": not id_overlap and not near_duplicates,
    }
