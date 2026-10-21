"""Complete-pair adapters for controlled synthetic-pool comparisons."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from src.baselines.literature.acquire import FeaturePredictor, LITERATURE_STRATEGIES
from src.baselines.literature.badge import select_badge_indices
from src.baselines.literature.cal import select_cal_indices
from src.baselines.literature.common import mean_unit_vectors
from src.baselines.literature.coreset import select_coreset_indices
from src.baselines.literature.margin import select_margin_indices
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample


def _complete_groups(
    examples: Sequence[SyntheticNLIExample],
) -> list[list[SyntheticNLIExample]]:
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)
    result: list[list[SyntheticNLIExample]] = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        pair_type = members[0].pair_type
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} mixes pair types")
        if {item.pair_member for item in members} != set(
            SYNTHETIC_PAIR_MEMBERS[pair_type]
        ):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        result.append(sorted(members, key=lambda item: item.pair_member))
    return sorted(result, key=lambda group: group[0].pair_id)


def _targets(groups: Sequence[Sequence[SyntheticNLIExample]]) -> list[SyntheticNLIExample]:
    return [next(item for item in group if item.pair_member == "base") for group in groups]


def _probabilities(
    student: FeaturePredictor,
    examples: Sequence[SyntheticNLIExample],
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


def _pair_embeddings(
    student: FeaturePredictor,
    groups: Sequence[Sequence[SyntheticNLIExample]],
    *,
    batch_size: int,
    label: str,
) -> list[list[float]]:
    flattened = [item for group in groups for item in group]
    student.progress_label = label
    values = student.encode_pair_embeddings(
        [item.premise for item in flattened],
        [item.hypothesis for item in flattened],
        batch_size=batch_size,
    )
    grouped_values = [values[index : index + 2] for index in range(0, len(values), 2)]
    return mean_unit_vectors(grouped_values)


def acquire_literature_synthetic_pairs(
    strategy: str,
    candidates: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    student: FeaturePredictor,
    probability_batch_size: int,
    embedding_batch_size: int,
    references: Sequence[SyntheticNLIExample] = (),
    warm_start: bool = False,
    cal_neighbors: int = 10,
) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
    """Select complete pairs while adapting published scores at pair level."""
    if strategy not in LITERATURE_STRATEGIES:
        raise ValueError(f"Unknown literature strategy: {strategy!r}")
    if budget <= 0 or budget % 2:
        raise ValueError("synthetic literature budget must be positive and even")
    groups = _complete_groups(candidates)
    pair_budget = budget // 2
    if pair_budget > len(groups):
        raise ValueError("synthetic pair budget exceeds the complete-pair pool")
    reference_groups = _complete_groups(references) if references else []

    if warm_start:
        indices = random.Random(seed).sample(range(len(groups)), pair_budget)
        selected = [item for index in indices for item in groups[index]]
        return selected, {
            "selection_strategy": "shared_random_warm_start",
            "literature_strategy": strategy,
            "selection_unit": "complete_pair",
            "requested_examples": budget,
            "actual_examples": len(selected),
            "actual_pairs": len(indices),
            "candidate_examples_scored": 0,
            "reference_examples_scored": 0,
            "shortage": False,
        }

    targets = _targets(groups)
    reference_targets = _targets(reference_groups) if reference_groups else []
    candidate_probabilities: list[list[float]] | None = None
    candidate_embeddings: list[list[float]] | None = None
    reference_probabilities: list[list[float]] | None = None
    reference_embeddings: list[list[float]] | None = None
    if strategy in {"margin", "badge", "cal"}:
        candidate_probabilities = _probabilities(
            student,
            targets,
            batch_size=probability_batch_size,
            label=f"{strategy} synthetic target probability scoring",
        )
    if strategy in {"coreset", "badge", "cal"}:
        candidate_embeddings = _pair_embeddings(
            student,
            groups,
            batch_size=embedding_batch_size,
            label=f"{strategy} synthetic pair embedding extraction",
        )
    if strategy in {"coreset", "cal"}:
        if not reference_groups:
            raise ValueError(f"{strategy} requires labeled reference pairs")
        reference_embeddings = _pair_embeddings(
            student,
            reference_groups,
            batch_size=embedding_batch_size,
            label=f"{strategy} synthetic reference pair embeddings",
        )
    if strategy == "cal":
        reference_probabilities = _probabilities(
            student,
            reference_targets,
            batch_size=probability_batch_size,
            label="cal synthetic reference target probability scoring",
        )

    if strategy == "margin":
        assert candidate_probabilities is not None
        indices = select_margin_indices(
            candidate_probabilities, budget=pair_budget, seed=seed
        )
    elif strategy == "coreset":
        assert candidate_embeddings is not None and reference_embeddings is not None
        indices = select_coreset_indices(
            candidate_embeddings,
            budget=pair_budget,
            seed=seed,
            reference_embeddings=reference_embeddings,
        )
    elif strategy == "badge":
        assert candidate_probabilities is not None and candidate_embeddings is not None
        indices = select_badge_indices(
            candidate_probabilities,
            candidate_embeddings,
            budget=pair_budget,
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
            budget=pair_budget,
            seed=seed,
            neighbors=cal_neighbors,
        )

    selected = [item for index in indices for item in groups[index]]
    return selected, {
        "selection_strategy": strategy,
        "selection_unit": "complete_pair",
        "probability_member": "base",
        "embedding_aggregation": "normalized_mean_of_two_pair_members",
        "requested_examples": budget,
        "actual_examples": len(selected),
        "actual_pairs": len(indices),
        "candidate_examples_scored": len(groups) * 2,
        "reference_examples_scored": len(reference_groups) * 2,
        "embedding_batch_size": embedding_batch_size,
        "cal_neighbors": cal_neighbors if strategy == "cal" else None,
        "shortage": len(selected) < budget,
    }
