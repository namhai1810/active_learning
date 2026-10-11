"""Entropy-based acquisition baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.baselines.base import Baseline, ProbabilityPredictor, validate_acquisition_request
from src.data.mnli import MNLIExample


def entropy_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must have shape [num_examples, num_classes]")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Each probability row must have positive mass")
    normalized = values / row_sums
    safe = np.clip(normalized, np.finfo(np.float64).tiny, 1.0)
    return -(normalized * np.log(safe)).sum(axis=1)


def entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("logits must have shape [num_examples, num_classes]")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return entropy_from_probabilities(probabilities)


def select_top_uncertain_indices(logits: np.ndarray, budget: int) -> list[int]:
    if budget <= 0:
        raise ValueError("budget must be positive")
    entropies = entropy_from_logits(logits)
    if budget > len(entropies):
        raise ValueError("budget cannot exceed the number of examples")
    return np.argsort(-entropies, kind="stable")[:budget].tolist()


class UncertaintyBaseline(Baseline):
    method = "uncertainty"

    def acquire(
        self,
        pool: Sequence[MNLIExample],
        budget: int,
        seed: int,
        student: ProbabilityPredictor | None,
        batch_size: int,
    ) -> list[MNLIExample]:
        validate_acquisition_request(pool, budget)
        if student is None:
            raise ValueError("Uncertainty acquisition requires a student")
        probabilities = student.predict_proba(
            [example.premise for example in pool],
            [example.hypothesis for example in pool],
            batch_size=batch_size,
        )
        if len(probabilities) != len(pool):
            raise RuntimeError("Student probability count does not match pool size")
        entropies = entropy_from_probabilities(np.asarray(probabilities))
        indices = np.argsort(-entropies, kind="stable")[:budget]
        return [pool[int(index)] for index in indices]
