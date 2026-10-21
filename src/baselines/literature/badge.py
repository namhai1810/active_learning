"""BADGE acquisition (Ash et al., ICLR 2020)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.baselines.literature.common import (
    embedding_matrix,
    probability_matrix,
    validate_budget,
)


def gradient_embeddings(
    probabilities: Sequence[Sequence[float]],
    embeddings: Sequence[Sequence[float]],
) -> np.ndarray:
    """Construct BADGE pseudo-label gradient embeddings.

    For representation ``h`` and pseudo-label ``y_hat``, the flattened last-layer
    cross-entropy gradient is ``h outer (p - one_hot(y_hat))``.
    """
    posterior = probability_matrix(probabilities).astype(np.float32)
    features = embedding_matrix(embeddings)
    if len(posterior) != len(features):
        raise ValueError("probability and embedding counts differ")
    pseudo = np.argmax(posterior, axis=1)
    residual = posterior.copy()
    residual[np.arange(len(residual)), pseudo] -= 1.0
    return np.einsum("nd,nc->ncd", features, residual, optimize=True).reshape(
        len(features), -1
    )


def select_badge_indices(
    probabilities: Sequence[Sequence[float]],
    embeddings: Sequence[Sequence[float]],
    *,
    budget: int,
    seed: int,
) -> list[int]:
    """Select a diverse uncertain batch by D² seeding in gradient space."""
    gradients = gradient_embeddings(probabilities, embeddings)
    validate_budget(len(gradients), budget)
    rng = np.random.default_rng(seed)
    norms = np.einsum("nd,nd->n", gradients, gradients)
    first = int(np.argmax(norms))
    selected = [first]
    available = np.ones(len(gradients), dtype=bool)
    available[first] = False
    min_distances = np.maximum(
        0.0,
        norms + norms[first] - 2.0 * (gradients @ gradients[first]),
    ).astype(np.float64)
    min_distances[first] = 0.0

    while len(selected) < budget:
        weights = np.where(available, min_distances, 0.0)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            remaining = np.flatnonzero(available)
            chosen = int(remaining[int(rng.integers(len(remaining)))])
        else:
            chosen = int(rng.choice(len(weights), p=weights / total))
            if not available[chosen]:  # numerical guard
                chosen = int(np.flatnonzero(available)[0])
        selected.append(chosen)
        available[chosen] = False
        distances = np.maximum(
            0.0,
            norms + norms[chosen] - 2.0 * (gradients @ gradients[chosen]),
        )
        min_distances = np.minimum(min_distances, distances)
        min_distances[~available] = 0.0
    return selected
