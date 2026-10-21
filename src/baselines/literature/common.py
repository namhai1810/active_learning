"""Numerical utilities shared by literature active-learning baselines."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np


def probability_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must have shape [examples, classes]")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    totals = matrix.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("each probability row must have positive mass")
    return matrix / totals


def embedding_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must have shape [examples, dimensions]")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("embedding rows must have non-zero norm")
    return matrix / norms


def validate_budget(size: int, budget: int) -> None:
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget > size:
        raise ValueError("budget cannot exceed the candidate count")


def deterministic_top_indices(
    scores: Sequence[float],
    *,
    budget: int,
    seed: int,
    namespace: str,
) -> list[int]:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite one-dimensional vector")
    validate_budget(len(values), budget)

    def key(index: int) -> tuple[float, bytes, int]:
        tie = hashlib.sha256(
            f"{seed}:{namespace}:{index}".encode("utf-8")
        ).digest()
        return (-float(values[index]), tie, index)

    return sorted(range(len(values)), key=key)[:budget]


def mean_unit_vectors(
    groups: Sequence[Sequence[Sequence[float]]],
) -> list[list[float]]:
    """Average member embeddings and renormalize each acquisition unit."""
    result: list[list[float]] = []
    for group in groups:
        matrix = embedding_matrix(group)
        mean = matrix.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("mean acquisition-unit embedding has zero norm")
        result.append((mean / norm).astype(np.float32).tolist())
    return result
