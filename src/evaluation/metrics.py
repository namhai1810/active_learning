"""Metrics for HANS."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from src.data.hans import HANS_HEURISTICS
from src.data.labels import normalize_hans_label


def _accuracy(correctness: Sequence[bool]) -> float | None:
    if not correctness:
        return None
    return sum(correctness) / len(correctness)


def compute_hans_metrics(
    gold_labels: Sequence[str],
    predictions: Sequence[str],
    heuristics: Sequence[str],
    subcases: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute overall, per-label, and per-heuristic HANS accuracy."""
    if not (len(gold_labels) == len(predictions) == len(heuristics)):
        raise ValueError("gold_labels, predictions, and heuristics must align")
    if subcases is not None and len(subcases) != len(gold_labels):
        raise ValueError("subcases must align with gold_labels")
    if not gold_labels:
        raise ValueError("Cannot compute HANS metrics for an empty dataset")

    normalized_gold = [normalize_hans_label(label) for label in gold_labels]
    normalized_predictions = [
        normalize_hans_label(label) for label in predictions
    ]
    normalized_heuristics = [
        str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for value in heuristics
    ]
    unknown = sorted(set(normalized_heuristics) - set(HANS_HEURISTICS))
    if unknown:
        raise ValueError(f"Unknown HANS heuristics: {unknown}")

    correct = [
        gold == prediction
        for gold, prediction in zip(
            normalized_gold, normalized_predictions, strict=True
        )
    ]

    label_accuracy: dict[str, float | None] = {}
    for label in ("entailment", "non-entailment"):
        subset = [
            is_correct
            for gold, is_correct in zip(normalized_gold, correct, strict=True)
            if gold == label
        ]
        label_accuracy[label] = _accuracy(subset)

    heuristic_accuracy: dict[str, float | None] = {}
    for heuristic in HANS_HEURISTICS:
        subset = [
            is_correct
            for observed, is_correct in zip(
                normalized_heuristics, correct, strict=True
            )
            if observed == heuristic
        ]
        heuristic_accuracy[heuristic] = _accuracy(subset)

    subcase_accuracy: dict[str, float | None] = {}
    normalized_subcases: list[str] = []
    if subcases is not None:
        normalized_subcases = [str(value).strip() for value in subcases]
        for subcase in sorted(set(normalized_subcases)):
            subset = [
                is_correct
                for observed, is_correct in zip(
                    normalized_subcases, correct, strict=True
                )
                if observed == subcase
            ]
            subcase_accuracy[subcase] = _accuracy(subset)

    return {
        "num_examples": len(correct),
        "num_correct": sum(correct),
        "overall_accuracy": _accuracy(correct),
        "entailment_accuracy": label_accuracy["entailment"],
        "non_entailment_accuracy": label_accuracy["non-entailment"],
        "accuracy_by_heuristic": heuristic_accuracy,
        "accuracy_by_subcase": subcase_accuracy,
        "label_counts": dict(Counter(normalized_gold)),
        "heuristic_counts": dict(Counter(normalized_heuristics)),
        "subcase_counts": dict(Counter(normalized_subcases)),
    }
