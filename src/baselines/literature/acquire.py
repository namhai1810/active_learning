"""Adapters from the shared Qwen NLI student to published acquisition rules."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any, Protocol

from src.baselines.base import ProbabilityPredictor, validate_acquisition_request
from src.baselines.literature.badge import select_badge_indices
from src.baselines.literature.cal import select_cal_indices
from src.baselines.literature.coreset import select_coreset_indices
from src.baselines.literature.margin import select_margin_indices
from src.data.mnli import MNLIExample


LITERATURE_STRATEGIES = ("margin", "coreset", "badge", "cal")


class FeaturePredictor(ProbabilityPredictor, Protocol):
    progress_label: str

    def encode_pair_embeddings(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int,
    ) -> list[list[float]]: ...


def _probabilities(
    student: FeaturePredictor,
    examples: Sequence[MNLIExample],
    *,
    batch_size: int,
    label: str,
) -> list[list[float]]:
    student.progress_label = label
    return student.predict_proba(
        [item.premise for item in examples],
        [item.hypothesis for item in examples],
        batch_size=batch_size,
    )


def _embeddings(
    student: FeaturePredictor,
    examples: Sequence[MNLIExample],
    *,
    batch_size: int,
    label: str,
) -> list[list[float]]:
    student.progress_label = label
    return student.encode_pair_embeddings(
        [item.premise for item in examples],
        [item.hypothesis for item in examples],
        batch_size=batch_size,
    )


def acquire_literature_mnli(
    strategy: str,
    candidates: Sequence[MNLIExample],
    *,
    budget: int,
    seed: int,
    student: FeaturePredictor,
    probability_batch_size: int,
    embedding_batch_size: int,
    references: Sequence[MNLIExample] = (),
    warm_start: bool = False,
    cal_neighbors: int = 10,
) -> tuple[list[MNLIExample], dict[str, Any]]:
    """Acquire one MNLI batch and return auditable selector diagnostics."""
    if strategy not in LITERATURE_STRATEGIES:
        raise ValueError(f"Unknown literature strategy: {strategy!r}")
    validate_acquisition_request(candidates, budget)
    if warm_start:
        indices = random.Random(seed).sample(range(len(candidates)), budget)
        return [candidates[index] for index in indices], {
            "selection_strategy": "shared_random_warm_start",
            "literature_strategy": strategy,
            "requested_examples": budget,
            "actual_examples": budget,
            "candidate_examples_scored": 0,
            "reference_examples_scored": 0,
            "shortage": False,
        }

    candidate_probabilities: list[list[float]] | None = None
    candidate_embeddings: list[list[float]] | None = None
    reference_probabilities: list[list[float]] | None = None
    reference_embeddings: list[list[float]] | None = None
    if strategy in {"margin", "badge", "cal"}:
        candidate_probabilities = _probabilities(
            student,
            candidates,
            batch_size=probability_batch_size,
            label=f"{strategy} MNLI probability scoring",
        )
    if strategy in {"coreset", "badge", "cal"}:
        candidate_embeddings = _embeddings(
            student,
            candidates,
            batch_size=embedding_batch_size,
            label=f"{strategy} MNLI embedding extraction",
        )
    if strategy in {"coreset", "cal"}:
        if not references:
            raise ValueError(f"{strategy} requires a labeled reference set")
        reference_embeddings = _embeddings(
            student,
            references,
            batch_size=embedding_batch_size,
            label=f"{strategy} MNLI reference embeddings",
        )
    if strategy == "cal":
        reference_probabilities = _probabilities(
            student,
            references,
            batch_size=probability_batch_size,
            label="cal MNLI reference probability scoring",
        )

    if strategy == "margin":
        assert candidate_probabilities is not None
        indices = select_margin_indices(
            candidate_probabilities, budget=budget, seed=seed
        )
    elif strategy == "coreset":
        assert candidate_embeddings is not None and reference_embeddings is not None
        indices = select_coreset_indices(
            candidate_embeddings,
            budget=budget,
            seed=seed,
            reference_embeddings=reference_embeddings,
        )
    elif strategy == "badge":
        assert candidate_probabilities is not None and candidate_embeddings is not None
        indices = select_badge_indices(
            candidate_probabilities,
            candidate_embeddings,
            budget=budget,
            seed=seed,
        )
    else:
        assert candidate_probabilities is not None
        assert candidate_embeddings is not None
        assert reference_probabilities is not None
        assert reference_embeddings is not None
        indices = select_cal_indices(
            candidate_probabilities,
            candidate_embeddings,
            reference_probabilities,
            reference_embeddings,
            budget=budget,
            seed=seed,
            neighbors=cal_neighbors,
        )

    return [candidates[index] for index in indices], {
        "selection_strategy": strategy,
        "requested_examples": budget,
        "actual_examples": len(indices),
        "candidate_examples_scored": len(candidates),
        "reference_examples_scored": len(references) if references else 0,
        "embedding_batch_size": embedding_batch_size,
        "cal_neighbors": cal_neighbors if strategy == "cal" else None,
        "shortage": len(indices) < budget,
    }
