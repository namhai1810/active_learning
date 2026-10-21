"""Contrastive Active Learning (Margatina et al., EMNLP 2021)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.baselines.literature.common import (
    deterministic_top_indices,
    embedding_matrix,
    probability_matrix,
    validate_budget,
)


def select_cal_indices(
    candidate_probabilities: Sequence[Sequence[float]],
    candidate_embeddings: Sequence[Sequence[float]],
    reference_probabilities: Sequence[Sequence[float]],
    reference_embeddings: Sequence[Sequence[float]],
    *,
    budget: int,
    seed: int,
    neighbors: int = 10,
) -> list[int]:
    """Rank candidates by mean KL from nearest labeled examples.

    This follows the paper's standard direction: for each candidate ``q``, find
    its nearest labeled examples and average ``KL(p_neighbor || p_q)``.
    """
    if neighbors <= 0:
        raise ValueError("neighbors must be positive")
    candidate_p = probability_matrix(candidate_probabilities)
    reference_p = probability_matrix(reference_probabilities)
    candidate_h = embedding_matrix(candidate_embeddings)
    reference_h = embedding_matrix(reference_embeddings)
    validate_budget(len(candidate_h), budget)
    if len(candidate_p) != len(candidate_h):
        raise ValueError("candidate probability and embedding counts differ")
    if len(reference_p) != len(reference_h):
        raise ValueError("reference probability and embedding counts differ")
    if candidate_h.shape[1] != reference_h.shape[1]:
        raise ValueError("candidate and reference embeddings differ in width")

    k = min(neighbors, len(reference_h))
    distances = np.maximum(0.0, 2.0 - 2.0 * (candidate_h @ reference_h.T))
    nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    safe_candidate = np.clip(candidate_p, np.finfo(np.float64).tiny, 1.0)
    safe_reference = np.clip(reference_p, np.finfo(np.float64).tiny, 1.0)
    scores = np.empty(len(candidate_h), dtype=np.float64)
    for index, neighbor_indices in enumerate(nearest):
        neighbor_p = safe_reference[neighbor_indices]
        kl = np.sum(
            neighbor_p * (np.log(neighbor_p) - np.log(safe_candidate[index])),
            axis=1,
        )
        scores[index] = float(np.mean(kl))
    return deterministic_top_indices(
        scores,
        budget=budget,
        seed=seed,
        namespace="contrastive-active-learning",
    )
