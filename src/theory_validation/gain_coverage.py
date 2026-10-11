"""Empirical coverage analysis for bounded pilot-gain UCBs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.gain.estimation import empirical_mean_gain, gain_confidence_radius, validate_gain


def is_covered(
    delta_reference: float,
    delta_hat: float,
    confidence_radius: float,
) -> bool:
    return delta_reference <= delta_hat + confidence_radius


def evaluate_gain_coverage(
    *,
    certified_hypothesis_ids: Sequence[str],
    online_gains: Mapping[str, Sequence[float]],
    reference_gains: Mapping[str, Sequence[float]],
    m_values: Sequence[int],
    reference_repetitions: int,
    M: float,
    delta: float,
    K: int,
    T: int,
    tolerance: float = 1e-12,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not m_values or any(value <= 0 for value in m_values):
        raise ValueError("m_values must contain positive integers")
    if reference_repetitions <= 0:
        raise ValueError("reference_repetitions must be positive")
    rows = []
    for hypothesis_id in certified_hypothesis_ids:
        online = [validate_gain(value, M, tolerance) for value in online_gains.get(hypothesis_id, ())]
        reference = [
            validate_gain(value, M, tolerance)
            for value in reference_gains.get(hypothesis_id, ())[:reference_repetitions]
        ]
        delta_reference = empirical_mean_gain(reference) if reference else None
        for requested_m in m_values:
            selected = online[:requested_m]
            if len(selected) < requested_m or delta_reference is None:
                rows.append(
                    {
                        "hypothesis_id": hypothesis_id,
                        "requested_m": requested_m,
                        "actual_m": len(selected),
                        "reference_repetitions": len(reference),
                        "delta_hat": None,
                        "delta_reference": delta_reference,
                        "confidence_radius": None,
                        "optimistic_gain_bound": None,
                        "absolute_error": None,
                        "covered": None,
                        "status": "insufficient_repetitions",
                    }
                )
                continue
            delta_hat = empirical_mean_gain(selected)
            radius = gain_confidence_radius(requested_m, M, K, T, delta)
            rows.append(
                {
                    "hypothesis_id": hypothesis_id,
                    "requested_m": requested_m,
                    "actual_m": requested_m,
                    "reference_repetitions": len(reference),
                    "delta_hat": delta_hat,
                    "delta_reference": delta_reference,
                    "confidence_radius": radius,
                    "optimistic_gain_bound": delta_hat + radius,
                    "absolute_error": abs(delta_hat - delta_reference),
                    "covered": is_covered(delta_reference, delta_hat, radius),
                    "status": "evaluated",
                }
            )

    summaries = {}
    for requested_m in m_values:
        evaluated = [
            row
            for row in rows
            if row["requested_m"] == requested_m and row["status"] == "evaluated"
        ]
        summaries[str(requested_m)] = {
            "m": requested_m,
            "num_cases": len(evaluated),
            "coverage_rate": (
                sum(row["covered"] for row in evaluated) / len(evaluated)
                if evaluated
                else None
            ),
            "mean_confidence_radius": (
                sum(row["confidence_radius"] for row in evaluated) / len(evaluated)
                if evaluated
                else None
            ),
            "mean_absolute_error": (
                sum(row["absolute_error"] for row in evaluated) / len(evaluated)
                if evaluated
                else None
            ),
        }
    return rows, summaries
