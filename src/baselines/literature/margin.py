"""Maximum-margin (breaking-ties) uncertainty sampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.baselines.literature.common import (
    deterministic_top_indices,
    probability_matrix,
)


def select_margin_indices(
    probabilities: Sequence[Sequence[float]],
    *,
    budget: int,
    seed: int,
) -> list[int]:
    """Select examples with the smallest top-two posterior margin."""
    matrix = probability_matrix(probabilities)
    ordered = np.sort(matrix, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    return deterministic_top_indices(
        -margins,
        budget=budget,
        seed=seed,
        namespace="maximum-margin",
    )
