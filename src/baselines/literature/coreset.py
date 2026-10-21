"""Greedy k-center Core-set acquisition (Sener and Savarese, ICLR 2018)."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from src.baselines.literature.common import embedding_matrix, validate_budget


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.maximum(
        0.0,
        np.sum(left * left, axis=1)
        + np.sum(right * right, axis=1)[0]
        - 2.0 * (left @ right[0]),
    )


def select_coreset_indices(
    embeddings: Sequence[Sequence[float]],
    *,
    budget: int,
    seed: int,
    reference_embeddings: Sequence[Sequence[float]] = (),
) -> list[int]:
    """Run deterministic farthest-first traversal in student feature space.

    Existing labeled/reference examples initialize the centers. If no labeled
    example exists, a deterministic seed-dependent point initializes traversal;
    the benchmark normally avoids this branch by using a shared random warm start.
    """
    candidates = embedding_matrix(embeddings)
    validate_budget(len(candidates), budget)
    selected: list[int] = []
    available = np.ones(len(candidates), dtype=bool)

    if len(reference_embeddings) > 0:
        references = embedding_matrix(reference_embeddings)
        if references.shape[1] != candidates.shape[1]:
            raise ValueError("candidate and reference embeddings differ in width")
        min_distances = np.full(len(candidates), np.inf, dtype=np.float64)
        for reference in references:
            min_distances = np.minimum(
                min_distances,
                _squared_distances(candidates, reference[None, :]),
            )
    else:
        first = min(
            range(len(candidates)),
            key=lambda index: (
                hashlib.sha256(
                    f"{seed}:coreset-initial:{index}".encode("utf-8")
                ).digest(),
                index,
            ),
        )
        selected.append(first)
        available[first] = False
        min_distances = _squared_distances(candidates, candidates[first : first + 1])
        min_distances[first] = -np.inf

    while len(selected) < budget:
        maximum = float(np.max(min_distances[available]))
        tied = np.flatnonzero(available & np.isclose(min_distances, maximum))
        chosen = min(
            tied.tolist(),
            key=lambda index: (
                hashlib.sha256(
                    f"{seed}:coreset-tie:{len(selected)}:{index}".encode("utf-8")
                ).digest(),
                index,
            ),
        )
        selected.append(chosen)
        available[chosen] = False
        min_distances = np.minimum(
            min_distances,
            _squared_distances(candidates, candidates[chosen : chosen + 1]),
        )
        min_distances[~available] = -np.inf
    return selected
