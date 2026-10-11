"""Outcome-independent targeting for the debug-only HANS oracle."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.mnli import MNLIExample
from src.hypotheses.types import FailureHypothesis


def _eligible(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
    expected_role: str,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    eligible: list[dict[str, Any]] = []
    for record in records:
        role = str(record.get("split_role", ""))
        if role != expected_role:
            raise ValueError(
                f"Targeting accepts only split_role={expected_role!r}; "
                f"received {role or '<missing>'!r}"
            )
        example_id = str(record.get("id", ""))
        if not example_id:
            raise ValueError("Targeting record is missing a stable id")
        if example_id in seen:
            raise ValueError(f"Duplicate targeting example ID: {example_id}")
        seen.add(example_id)
        if str(record.get("heuristic", "")).strip().lower() == hypothesis.heuristic:
            eligible.append(dict(record))
    return eligible


def _rank(
    records: Sequence[Mapping[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    return sorted(
        (dict(record) for record in records),
        key=lambda record: (
            hashlib.sha256(
                f"{namespace}:{seed}:{record['id']}".encode("utf-8")
            ).digest(),
            str(record["id"]),
        ),
    )


def pilot_seed(base_seed: int, hypothesis_id: str, pilot_index: int) -> int:
    if pilot_index < 0:
        raise ValueError("pilot_index must be non-negative")
    digest = hashlib.sha256(
        f"pilot:{base_seed}:{hypothesis_id}:{pilot_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def select_intervention_examples(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
    pilot_train_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select by hypothesis, split role, ID and seed--never by outcomes."""
    if pilot_train_size <= 0:
        raise ValueError("pilot_train_size must be positive")
    eligible = _eligible(hypothesis, records, "intervention_pool")
    ranked = _rank(eligible, seed, f"train:{hypothesis.id}")
    return ranked[:pilot_train_size]


def select_gain_eval_batches(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
    *,
    m: int,
    gain_eval_size: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Prefer disjoint fixed eval batches and overlap only when unavoidable."""
    if m <= 0 or gain_eval_size <= 0:
        raise ValueError("m and gain_eval_size must be positive")
    eligible = _eligible(hypothesis, records, "gain_eval")
    ranked = _rank(eligible, seed, f"eval:{hypothesis.id}")
    if not ranked:
        return [[] for _ in range(m)]
    if len(ranked) >= m * gain_eval_size:
        return [
            ranked[index * gain_eval_size : (index + 1) * gain_eval_size]
            for index in range(m)
        ]

    actual_size = min(gain_eval_size, len(ranked))
    batches = []
    for index in range(m):
        start = (index * actual_size) % len(ranked)
        rotated = ranked[start:] + ranked[:start]
        batches.append(rotated[:actual_size])
    return batches


def select_matching_gain_eval_examples(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return all matching gain-eval examples in stable ID order."""
    eligible = _eligible(hypothesis, records, "gain_eval")
    return sorted(eligible, key=lambda record: str(record["id"]))


def to_mnli_training_examples(
    records: Sequence[Mapping[str, Any]],
    non_entailment_label: str = "contradiction",
) -> list[MNLIExample]:
    """Map binary HANS labels into the student's three-label training space."""
    if non_entailment_label not in {"neutral", "contradiction"}:
        raise ValueError("non_entailment_label must be neutral or contradiction")
    examples = []
    for record in records:
        gold = str(record.get("gold", ""))
        if gold not in {"entailment", "non-entailment"}:
            raise ValueError(f"Unsupported HANS gold label: {gold!r}")
        examples.append(
            MNLIExample(
                id=str(record["id"]),
                premise=str(record["premise"]),
                hypothesis=str(record["hypothesis"]),
                label="entailment" if gold == "entailment" else non_entailment_label,
            )
        )
    return examples
