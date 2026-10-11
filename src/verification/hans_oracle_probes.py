"""Outcome-independent HANS probe selection for oracle debugging."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.labels import normalize_hans_label
from src.hypotheses.types import FailureHypothesis


def select_hans_oracle_probes(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
    *,
    n_probes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select probes without consulting predictions, correctness, or outcomes."""
    if isinstance(n_probes, bool) or n_probes <= 0:
        raise ValueError("n_probes must be a positive integer")

    eligible: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        role = str(record.get("split_role", "")).strip().lower()
        if role != "verification":
            raise ValueError(
                "HANS verification probe builder accepts only records marked "
                f"split_role='verification'; received {role or '<missing>'!r}"
            )
        example_id = str(record.get("id", ""))
        if not example_id:
            raise ValueError("Verification record is missing a stable id")
        if example_id in seen_ids:
            raise ValueError(f"Duplicate verification example ID: {example_id}")
        seen_ids.add(example_id)
        heuristic = str(record.get("heuristic", "")).strip().lower()
        gold = normalize_hans_label(record.get("gold"))
        if heuristic == hypothesis.heuristic and gold == "non-entailment":
            eligible.append(dict(record))

    eligible.sort(
        key=lambda record: (
            hashlib.sha256(
                f"{seed}:{hypothesis.id}:{record['id']}".encode("utf-8")
            ).digest(),
            str(record["id"]),
        )
    )
    return eligible[:n_probes]


def build_probe_outcomes(
    hypothesis: FailureHypothesis,
    selected_probes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    outcomes = []
    for record in selected_probes:
        role = str(record.get("split_role", "")).strip().lower()
        if role != "verification":
            raise ValueError("Probe outcome requires split_role='verification'")
        heuristic = str(record.get("heuristic", "")).strip().lower()
        if heuristic != hypothesis.heuristic:
            raise ValueError("Probe heuristic does not match the hypothesis")
        gold = normalize_hans_label(record.get("gold"))
        if gold != "non-entailment":
            raise ValueError("HANS oracle probes require gold non-entailment")
        prediction = normalize_hans_label(record.get("prediction"))
        outcomes.append(
            {
                "hypothesis_id": hypothesis.id,
                "example_id": str(record["id"]),
                "heuristic": heuristic,
                "gold": gold,
                "prediction": prediction,
                "z": 1 if prediction == "entailment" else 0,
                "split_role": "verification",
            }
        )
    return outcomes
