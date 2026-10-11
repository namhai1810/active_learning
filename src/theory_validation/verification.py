"""Offline verification-confidence sweeps over dedicated reference data."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.labels import normalize_hans_label
from src.hypotheses.types import FailureHypothesis
from src.verification.bounds import verify_binary_outcomes


REFERENCE_STATUSES = (
    "robust_valid",
    "ambiguous",
    "robust_invalid",
    "insufficient_reference",
)


def classify_reference(
    nu_reference: float,
    *,
    tau_v: float,
    gamma: float,
) -> str:
    if not 0.0 <= nu_reference <= 1.0:
        raise ValueError("nu_reference must be in [0, 1]")
    if not 0.0 <= tau_v <= 1.0:
        raise ValueError("tau_v must be in [0, 1]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if nu_reference >= tau_v + gamma:
        return "robust_valid"
    if nu_reference <= tau_v - gamma:
        return "robust_invalid"
    return "ambiguous"


def _select_outcomes(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
    *,
    expected_role: str,
    count: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    if count <= 0:
        raise ValueError("Probe count must be positive")
    eligible = []
    seen: set[str] = set()
    for record in records:
        role = str(record.get("split_role", ""))
        if role != expected_role:
            raise ValueError(
                f"Expected split_role={expected_role!r}; got {role or '<missing>'!r}"
            )
        example_id = str(record.get("id", ""))
        if not example_id or example_id in seen:
            raise ValueError(f"Missing or duplicate probe ID: {example_id!r}")
        seen.add(example_id)
        if (
            str(record.get("heuristic", "")).strip().lower()
            == hypothesis.heuristic
            and normalize_hans_label(record.get("gold")) == "non-entailment"
        ):
            eligible.append(dict(record))
    eligible.sort(
        key=lambda record: (
            hashlib.sha256(
                f"{seed}:{expected_role}:{hypothesis.id}:{record['id']}".encode(
                    "utf-8"
                )
            ).digest(),
            str(record["id"]),
        )
    )
    selected = eligible[:count]
    ids = [str(record["id"]) for record in selected]
    outcomes = [
        1
        if normalize_hans_label(record.get("prediction")) == "entailment"
        else 0
        for record in selected
    ]
    return ids, outcomes


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_verification_confidence(
    *,
    hypotheses: Sequence[FailureHypothesis],
    online_verification_records: Sequence[Mapping[str, Any]],
    reference_records: Sequence[Mapping[str, Any]],
    n_values: Sequence[int],
    reference_probe_count: int,
    delta: float,
    tau_v: float,
    gamma: float,
    K: int,
    T: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not n_values or any(value <= 0 for value in n_values):
        raise ValueError("n_values must contain positive integers")
    reference: dict[str, dict[str, Any]] = {}
    for hypothesis in hypotheses:
        ids, outcomes = _select_outcomes(
            hypothesis,
            reference_records,
            expected_role="verification_reference",
            count=reference_probe_count,
            seed=seed,
        )
        if outcomes:
            nu_reference = sum(outcomes) / len(outcomes)
            status = classify_reference(
                nu_reference, tau_v=tau_v, gamma=gamma
            )
        else:
            nu_reference = None
            status = "insufficient_reference"
        reference[hypothesis.id] = {
            "nu_reference": nu_reference,
            "reference_status": status,
            "reference_n": len(outcomes),
            "reference_probe_ids": ids,
        }

    rows = []
    summaries = []
    for requested_n in n_values:
        n_rows = []
        for hypothesis in hypotheses:
            probe_ids, outcomes = _select_outcomes(
                hypothesis,
                online_verification_records,
                expected_role="verification",
                count=requested_n,
                seed=seed,
            )
            result = verify_binary_outcomes(
                hypothesis.id,
                outcomes,
                delta=delta,
                tau_v=tau_v,
                K=K,
                T=T,
            )
            row = {
                "hypothesis_id": hypothesis.id,
                "requested_n": requested_n,
                "actual_n": result.n,
                "online_probe_ids": probe_ids,
                "nu_hat": result.nu_hat,
                "confidence_radius": result.confidence_radius,
                "lower_bound": result.lower_bound,
                "certified": result.certified,
                **reference[hypothesis.id],
            }
            rows.append(row)
            n_rows.append(row)

        classified = [
            row
            for row in n_rows
            if row["reference_status"] in {"robust_valid", "robust_invalid"}
        ]
        certified_classified = [row for row in classified if row["certified"]]
        robust_valid = [
            row for row in classified if row["reference_status"] == "robust_valid"
        ]
        robust_invalid = [
            row for row in classified if row["reference_status"] == "robust_invalid"
        ]
        true_positive = sum(
            row["certified"] and row["reference_status"] == "robust_valid"
            for row in classified
        )
        false_positive = sum(row["certified"] for row in robust_invalid)
        summaries.append(
            {
                "requested_n": requested_n,
                "num_hypotheses": len(n_rows),
                "num_reference_classified": len(classified),
                "certification_rate": _ratio(
                    sum(row["certified"] for row in n_rows), len(n_rows)
                ),
                "certification_precision": _ratio(
                    true_positive, len(certified_classified)
                ),
                "robust_failure_recall": _ratio(
                    true_positive, len(robust_valid)
                ),
                "false_certification_rate": _ratio(
                    false_positive, len(robust_invalid)
                ),
            }
        )
    return rows, summaries
