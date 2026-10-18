"""Prediction-blind frozen stopping/report partitions for Stage 2 gain_eval."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


STOPPING_SUBSET_VERSION = "stage2_gain_eval_stopping_subset_v1"


def _hash_json(value: Any) -> str:
    serialized = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_stopping_subset_manifest(
    rows: Sequence[Mapping[str, Any]],
    gain_eval_ids: Sequence[str],
    *,
    frozen_manifest_hash: str,
    examples_per_subcase: int = 30,
    namespace: str = "stopping-eval-v1",
) -> dict[str, Any]:
    """Select the same number of IDs per subcase without reading predictions."""
    if not frozen_manifest_hash or not namespace:
        raise ValueError("Frozen manifest hash and namespace must be non-empty")
    if examples_per_subcase <= 0:
        raise ValueError("examples_per_subcase must be positive")
    ids = [str(item) for item in gain_eval_ids]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("gain_eval_ids must be non-empty and unique")
    index = {str(row.get("id", "")): row for row in rows}
    if "" in index or len(index) != len(rows):
        raise ValueError("Rows must have unique non-empty IDs")
    if set(index) != set(ids):
        raise ValueError("Rows must exactly match the frozen gain_eval IDs")

    by_subcase: dict[str, list[str]] = {}
    subcase_metadata: dict[str, tuple[str, str]] = {}
    for item_id in ids:
        row = index[item_id]
        subcase = str(row.get("subcase", ""))
        heuristic = str(row.get("heuristic", ""))
        gold = str(row.get("gold", ""))
        if not subcase or not heuristic or not gold:
            raise ValueError("gain_eval rows require subcase, heuristic, and gold")
        metadata = (heuristic, gold)
        if subcase in subcase_metadata and subcase_metadata[subcase] != metadata:
            raise ValueError(f"Subcase {subcase!r} has inconsistent metadata")
        subcase_metadata[subcase] = metadata
        by_subcase.setdefault(subcase, []).append(item_id)
    if any(len(values) < examples_per_subcase for values in by_subcase.values()):
        too_small = {
            key: len(values)
            for key, values in by_subcase.items()
            if len(values) < examples_per_subcase
        }
        raise ValueError(f"Subcases cannot satisfy the requested size: {too_small}")

    selected: set[str] = set()
    counts: dict[str, int] = {}
    for subcase, subcase_ids in sorted(by_subcase.items()):
        ranked = sorted(
            subcase_ids,
            key=lambda item: (
                hashlib.sha256(
                    f"{frozen_manifest_hash}:{namespace}:{item}".encode("utf-8")
                ).digest(),
                item,
            ),
        )
        chosen = ranked[:examples_per_subcase]
        selected.update(chosen)
        counts[subcase] = len(chosen)

    # Preserve the frozen gain_eval order in both roles. Ranking affects only
    # membership, never downstream evaluation order.
    stopping_ids = [item for item in ids if item in selected]
    report_ids = [item for item in ids if item not in selected]
    payload: dict[str, Any] = {
        "version": STOPPING_SUBSET_VERSION,
        "frozen_manifest_hash": frozen_manifest_hash,
        "source_role": "gain_eval",
        "selection_namespace": namespace,
        "selection_algorithm": "equal_per_subcase_sha256_rank_v1",
        "selection_uses_predictions": False,
        "selection_metadata_fields": ["id", "subcase", "heuristic", "gold"],
        "examples_per_subcase": examples_per_subcase,
        "num_subcases": len(by_subcase),
        "stopping_eval_ids": stopping_ids,
        "gain_report_ids": report_ids,
        "stopping_counts_by_subcase": counts,
        "stopping_counts_by_heuristic": dict(
            sorted(
                Counter(index[item]["heuristic"] for item in stopping_ids).items()
            )
        ),
        "stopping_counts_by_gold": dict(
            sorted(Counter(index[item]["gold"] for item in stopping_ids).items())
        ),
    }
    payload["partition_hash"] = _hash_json(payload)
    return payload


def validate_stopping_subset_manifest(
    value: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    gain_eval_ids: Sequence[str],
    *,
    frozen_manifest_hash: str,
) -> tuple[list[str], list[str]]:
    """Validate the frozen partition and return stopping/report IDs."""
    required = {
        "version",
        "frozen_manifest_hash",
        "source_role",
        "selection_namespace",
        "selection_algorithm",
        "selection_uses_predictions",
        "selection_metadata_fields",
        "examples_per_subcase",
        "num_subcases",
        "stopping_eval_ids",
        "gain_report_ids",
        "stopping_counts_by_subcase",
        "stopping_counts_by_heuristic",
        "stopping_counts_by_gold",
        "partition_hash",
    }
    if set(value) != required:
        raise ValueError("Stopping subset manifest fields differ from the v1 schema")
    if value.get("version") != STOPPING_SUBSET_VERSION:
        raise ValueError("Unsupported stopping subset manifest version")
    if value.get("frozen_manifest_hash") != frozen_manifest_hash:
        raise ValueError("Stopping subset belongs to a different frozen manifest")
    if value.get("source_role") != "gain_eval":
        raise ValueError("Stopping subset source role must be gain_eval")
    if value.get("selection_uses_predictions") is not False:
        raise ValueError("Stopping subset selection must be prediction-blind")
    expected_hash = _hash_json(
        {key: item for key, item in value.items() if key != "partition_hash"}
    )
    if value.get("partition_hash") != expected_hash:
        raise ValueError("Stopping subset partition hash is invalid")

    stopping_ids = [str(item) for item in value["stopping_eval_ids"]]
    report_ids = [str(item) for item in value["gain_report_ids"]]
    frozen_ids = [str(item) for item in gain_eval_ids]
    if (
        len(stopping_ids) != len(set(stopping_ids))
        or len(report_ids) != len(set(report_ids))
        or set(stopping_ids) & set(report_ids)
        or set(stopping_ids) | set(report_ids) != set(frozen_ids)
    ):
        raise ValueError("Stopping/report IDs must be unique, disjoint, and exhaustive")
    index = {str(row.get("id", "")): row for row in rows}
    if set(index) != set(frozen_ids):
        raise ValueError("Rows must exactly match frozen gain_eval IDs")
    observed_subcases = Counter(str(index[item]["subcase"]) for item in stopping_ids)
    declared_subcases = {
        str(key): int(count)
        for key, count in value["stopping_counts_by_subcase"].items()
    }
    if dict(sorted(observed_subcases.items())) != dict(sorted(declared_subcases.items())):
        raise ValueError("Stopping subset subcase counts do not match the manifest")
    return stopping_ids, report_ids
