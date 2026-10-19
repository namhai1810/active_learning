"""Deterministic fixed-budget acquisition helpers for Stage 2."""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from src.data.labels import MNLI_LABELS
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample


def select_complete_pairs(
    examples: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    excluded_ids: Sequence[str] = (),
) -> tuple[list[SyntheticNLIExample], dict[str, int | bool]]:
    """Select up to ``budget`` examples without splitting any synthetic pair."""
    if budget <= 0 or budget % 2:
        raise ValueError("Synthetic example budget must be positive and even")
    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)
    complete: list[list[SyntheticNLIExample]] = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        pair_type = members[0].pair_type
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        if {item.pair_member for item in members} != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        if not excluded & {item.id for item in members}:
            complete.append(members)
    ranked = sorted(
        complete,
        key=lambda group: (
            hashlib.sha256(f"{seed}:stage2-pair:{group[0].pair_id}".encode()).digest(),
            group[0].pair_id,
        ),
    )
    selected = [item for group in ranked[: budget // 2] for item in group]
    return selected, {
        "requested_examples": budget,
        "actual_examples": len(selected),
        "requested_pairs": budget // 2,
        "actual_pairs": len(selected) // 2,
        "shortage": len(selected) < budget,
    }


def _balanced_quotas(keys: Sequence[str], total: int, seed: int, group: str) -> dict[str, int]:
    ordered = sorted(
        set(keys),
        key=lambda key: (
            hashlib.sha256(f"{seed}:{group}:{key}".encode()).digest(),
            key,
        ),
    )
    if not ordered:
        return {}
    return {
        key: total // len(ordered) + int(index < total % len(ordered))
        for index, key in enumerate(ordered)
    }


def _selection_tokens(group: Sequence[SyntheticNLIExample]) -> frozenset[str]:
    return frozenset(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                f"{item.premise} {item.hypothesis}" for item in group
            ).lower(),
        )
    )


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def select_stratified_utility_pairs(
    examples: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    utility_scores: Mapping[str, float],
    certification_scores: Mapping[str, float],
    excluded_ids: Sequence[str] = (),
    utility_weight: float = 1.0,
    diversity_weight: float = 0.35,
    label_balance_weight: float = 1.0,
    certification_weight: float = 0.25,
) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
    """Select complete pairs with balanced strata and student-aware utility.

    Hypothesis and pair-type quotas are hard whenever the pool makes them
    feasible. Label balance is optimized greedily at pair level, because an
    indivisible pair can contain either one or two labels.
    """
    if budget <= 0 or budget % 2:
        raise ValueError("Synthetic example budget must be positive and even")
    weights = (
        utility_weight,
        diversity_weight,
        label_balance_weight,
        certification_weight,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Selection weights must be finite and non-negative")
    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)
    complete: list[list[SyntheticNLIExample]] = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        if excluded & {item.id for item in members}:
            continue
        pair_type = members[0].pair_type
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        if {item.pair_member for item in members} != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        if any(item.id not in utility_scores for item in members):
            raise ValueError(f"Missing utility score for synthetic pair {pair_id!r}")
        if any(
            not math.isfinite(float(utility_scores[item.id])) for item in members
        ):
            raise ValueError(f"Non-finite utility score for synthetic pair {pair_id!r}")
        members.sort(key=lambda item: item.pair_member)
        complete.append(members)

    requested_pairs = budget // 2
    if len(complete) < requested_pairs:
        selected = [item for group in complete for item in group]
        return selected, {
            "selection_strategy": "stratified_student_utility_v1",
            "requested_examples": budget,
            "actual_examples": len(selected),
            "requested_pairs": requested_pairs,
            "actual_pairs": len(complete),
            "shortage": True,
        }

    raw_pair_utility = {
        group[0].pair_id: sum(float(utility_scores[item.id]) for item in group) / 2.0
        for group in complete
    }
    utility_min = min(raw_pair_utility.values())
    utility_max = max(raw_pair_utility.values())

    def normalized_utility(pair_id: str) -> float:
        if utility_max == utility_min:
            return 0.5
        return (raw_pair_utility[pair_id] - utility_min) / (
            utility_max - utility_min
        )

    hypothesis_quotas = _balanced_quotas(
        [group[0].hypothesis_id for group in complete],
        requested_pairs,
        seed,
        "hypothesis-quota",
    )
    pair_type_quotas = _balanced_quotas(
        [group[0].pair_type for group in complete],
        requested_pairs,
        seed,
        "pair-type-quota",
    )
    label_targets = _balanced_quotas(MNLI_LABELS, budget, seed, "label-quota")
    hypothesis_counts: Counter[str] = Counter()
    pair_type_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    selected_groups: list[list[SyntheticNLIExample]] = []
    selected_tokens: list[frozenset[str]] = []
    remaining = list(complete)
    relaxed_quota_steps = 0

    while len(selected_groups) < requested_pairs:
        eligible = [
            group
            for group in remaining
            if hypothesis_counts[group[0].hypothesis_id]
            < hypothesis_quotas[group[0].hypothesis_id]
            and pair_type_counts[group[0].pair_type]
            < pair_type_quotas[group[0].pair_type]
        ]
        if not eligible:
            relaxed_quota_steps += 1
            eligible = list(remaining)
        before_imbalance = sum(
            abs(label_counts[label] - label_targets[label]) for label in MNLI_LABELS
        )

        def rank(group: list[SyntheticNLIExample]) -> tuple[float, bytes, str]:
            tokens = _selection_tokens(group)
            diversity = (
                1.0
                if not selected_tokens
                else 1.0 - max(_jaccard(tokens, prior) for prior in selected_tokens)
            )
            projected = label_counts.copy()
            projected.update(item.label for item in group)
            after_imbalance = sum(
                abs(projected[label] - label_targets[label]) for label in MNLI_LABELS
            )
            label_gain = (before_imbalance - after_imbalance) / 4.0
            certification = min(
                1.0,
                max(0.0, float(certification_scores.get(group[0].hypothesis_id, 0.0))),
            )
            score = (
                utility_weight * normalized_utility(group[0].pair_id)
                + diversity_weight * diversity
                + label_balance_weight * label_gain
                + certification_weight * certification
            )
            tie = hashlib.sha256(
                f"{seed}:stratified-utility:{group[0].pair_id}".encode()
            ).digest()
            return (-score, tie, group[0].pair_id)

        chosen = min(eligible, key=rank)
        remaining.remove(chosen)
        selected_groups.append(chosen)
        selected_tokens.append(_selection_tokens(chosen))
        hypothesis_counts[chosen[0].hypothesis_id] += 1
        pair_type_counts[chosen[0].pair_type] += 1
        label_counts.update(item.label for item in chosen)

    selected = [item for group in selected_groups for item in group]
    return selected, {
        "selection_strategy": "stratified_student_utility_v1",
        "requested_examples": budget,
        "actual_examples": len(selected),
        "requested_pairs": requested_pairs,
        "actual_pairs": len(selected_groups),
        "shortage": len(selected) < budget,
        "hypothesis_pair_targets": dict(sorted(hypothesis_quotas.items())),
        "hypothesis_pair_counts": dict(sorted(hypothesis_counts.items())),
        "pair_type_targets": dict(sorted(pair_type_quotas.items())),
        "pair_type_counts": dict(sorted(pair_type_counts.items())),
        "label_targets": {label: label_targets[label] for label in MNLI_LABELS},
        "label_counts": {label: label_counts[label] for label in MNLI_LABELS},
        "mean_selected_student_nll": sum(
            raw_pair_utility[group[0].pair_id] for group in selected_groups
        )
        / len(selected_groups),
        "relaxed_quota_steps": relaxed_quota_steps,
        "weights": {
            "utility": utility_weight,
            "diversity": diversity_weight,
            "label_balance": label_balance_weight,
            "certification": certification_weight,
        },
    }


def _softmax_distribution(
    values: Mapping[str, float], *, temperature: float
) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot normalize an empty distribution")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    maximum = max(float(value) / temperature for value in values.values())
    weights = {
        key: math.exp(float(value) / temperature - maximum)
        for key, value in values.items()
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _unit_vector(values: Sequence[float]) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding vectors must be non-empty and finite")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        raise ValueError("Embedding vectors must have non-zero norm")
    return tuple(value / norm for value in vector)


def _cosine_novelty(
    candidate: Sequence[float], selected: Sequence[Sequence[float]]
) -> float:
    """Return normalized distance to the closest selected pair embedding."""
    if not selected:
        return 1.0
    similarity = max(
        sum(left * right for left, right in zip(candidate, prior, strict=True))
        for prior in selected
    )
    return min(1.0, max(0.0, (1.0 - similarity) / 2.0))


def _percentile_ranks(values: Mapping[str, float]) -> dict[str, float]:
    """Map finite scores to tie-aware percentiles in ``[0, 1]``."""
    if not values:
        return {}
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Percentile ranks require finite values")
    ordered = sorted((float(value), key) for key, value in values.items())
    denominator = max(1, len(ordered) - 1)
    result: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_position = (start + end - 1) / 2.0
        percentile = average_position / denominator if len(ordered) > 1 else 0.5
        for _, key in ordered[start:end]:
            result[key] = percentile
        start = end
    return result


def select_closed_loop_soft_pairs(
    examples: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    nll_scores: Mapping[str, float],
    entropy_scores: Mapping[str, float],
    embeddings: Mapping[str, Sequence[float]],
    verification_evidence: Mapping[str, Mapping[str, Any]],
    excluded_ids: Sequence[str] = (),
    prior_hypothesis_pair_counts: Mapping[str, int] | None = None,
    nll_weight: float = 0.65,
    entropy_weight: float = 0.35,
    novelty_weight: float = 0.50,
    hypothesis_bonus_weight: float = 0.35,
    new_hypothesis_bonus_weight: float = 0.50,
) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
    """Greedily select complete pairs with scale-compatible soft incentives.

    There are no hard per-hypothesis quotas.  Target NLL, target entropy, and
    embedding novelty are converted to percentiles before combination.  A
    concave ``sqrt`` marginal bonus softly favors mechanisms that have received
    fewer cumulative pairs, while still allowing a strong mechanism to win
    additional slots.  Previously selected pairs anchor novelty and are never
    selected again.
    """
    if budget <= 0 or budget % 2:
        raise ValueError("Synthetic example budget must be positive and even")
    weights = (
        nll_weight,
        entropy_weight,
        novelty_weight,
        hypothesis_bonus_weight,
        new_hypothesis_bonus_weight,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Selection weights must be finite and non-negative")
    if nll_weight + entropy_weight <= 0.0:
        raise ValueError("At least one difficulty weight must be positive")
    prior_counts = {
        str(key): int(value)
        for key, value in (prior_hypothesis_pair_counts or {}).items()
    }
    if any(value < 0 for value in prior_counts.values()):
        raise ValueError("Prior hypothesis pair counts must be non-negative")

    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)

    all_groups: list[list[SyntheticNLIExample]] = []
    candidate_groups: list[list[SyntheticNLIExample]] = []
    pair_embeddings: dict[str, tuple[float, ...]] = {}
    raw_nll: dict[str, float] = {}
    raw_entropy: dict[str, float] = {}
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        pair_type = members[0].pair_type
        hypothesis_id = members[0].hypothesis_id
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        if any(item.hypothesis_id != hypothesis_id for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} mixes hypotheses")
        if {item.pair_member for item in members} != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        evidence = verification_evidence.get(hypothesis_id)
        if not isinstance(evidence, Mapping):
            continue
        if str(evidence.get("state", "")) not in {"certified", "unresolved"}:
            continue
        members.sort(key=lambda item: item.pair_member)
        for item in members:
            if item.id not in embeddings:
                raise ValueError(f"Missing embedding for {item.id!r}")
        vectors = [_unit_vector(embeddings[item.id]) for item in members]
        if len({len(vector) for vector in vectors}) != 1:
            raise ValueError(f"Embedding dimensions differ within pair {pair_id!r}")
        pair_embeddings[pair_id] = _unit_vector([
            sum(vector[index] for vector in vectors) / len(vectors)
            for index in range(len(vectors[0]))
        ])
        all_groups.append(members)
        if excluded & {item.id for item in members}:
            continue
        target = next(item for item in members if item.pair_member == "base")
        if target.id not in nll_scores or target.id not in entropy_scores:
            raise ValueError(f"Missing target difficulty for {target.id!r}")
        raw_nll[pair_id] = float(nll_scores[target.id])
        raw_entropy[pair_id] = float(entropy_scores[target.id])
        if not math.isfinite(raw_nll[pair_id]) or not 0.0 <= raw_entropy[pair_id] <= 1.0 + 1e-9:
            raise ValueError(f"Invalid target difficulty for {target.id!r}")
        candidate_groups.append(members)

    requested_pairs = budget // 2
    if not candidate_groups:
        return [], {
            "selection_strategy": "closed_loop_soft_diminishing_return_v1",
            "requested_examples": budget,
            "actual_examples": 0,
            "requested_pairs": requested_pairs,
            "actual_pairs": 0,
            "shortage": True,
        }
    dimensions = {len(value) for value in pair_embeddings.values()}
    if len(dimensions) != 1:
        raise ValueError("All embeddings must have the same dimension")

    nll_percentile = _percentile_ranks(raw_nll)
    entropy_percentile = _percentile_ranks(raw_entropy)
    difficulty_scale = nll_weight + entropy_weight
    difficulty = {
        pair_id: (
            nll_weight * nll_percentile[pair_id]
            + entropy_weight * entropy_percentile[pair_id]
        ) / difficulty_scale
        for pair_id in raw_nll
    }
    selected_prior_embeddings = [
        pair_embeddings[group[0].pair_id]
        for group in all_groups
        if excluded & {item.id for item in group}
    ]
    counts = Counter(prior_counts)
    remaining = list(candidate_groups)
    selected_groups: list[list[SyntheticNLIExample]] = []
    selected_embeddings = list(selected_prior_embeddings)
    selected_components: dict[str, dict[str, float]] = {}

    def tie_key(pair_id: str) -> tuple[bytes, str]:
        return hashlib.sha256(f"{seed}:closed-loop:{pair_id}".encode()).digest(), pair_id

    while remaining and len(selected_groups) < requested_pairs:
        raw_novelty = {
            group[0].pair_id: _cosine_novelty(
                pair_embeddings[group[0].pair_id], selected_embeddings
            )
            for group in remaining
        }
        novelty_percentile = _percentile_ranks(raw_novelty)

        def components(
            group: list[SyntheticNLIExample],
        ) -> tuple[float, float, float, float, float]:
            pair_id = group[0].pair_id
            hypothesis_id = group[0].hypothesis_id
            count = counts[hypothesis_id]
            bonus = math.sqrt(count + 1.0) - math.sqrt(count)
            new_bonus = 1.0 / (1.0 + count)
            score = (
                difficulty[pair_id]
                + novelty_weight * novelty_percentile[pair_id]
                + hypothesis_bonus_weight * bonus
                + new_hypothesis_bonus_weight * new_bonus
            )
            return (
                score,
                novelty_percentile[pair_id],
                raw_novelty[pair_id],
                bonus,
                new_bonus,
            )

        chosen = min(
            remaining,
            key=lambda group: (-components(group)[0], *tie_key(group[0].pair_id)),
        )
        pair_id = chosen[0].pair_id
        score, novelty_rank, novelty_raw, bonus, new_bonus = components(chosen)
        selected_components[pair_id] = {
            "score": score,
            "difficulty": difficulty[pair_id],
            "nll_percentile": nll_percentile[pair_id],
            "entropy_percentile": entropy_percentile[pair_id],
            "novelty_percentile": novelty_rank,
            "raw_cosine_novelty": novelty_raw,
            "hypothesis_marginal_bonus": bonus,
            "new_hypothesis_bonus": new_bonus,
        }
        remaining.remove(chosen)
        selected_groups.append(chosen)
        selected_embeddings.append(pair_embeddings[pair_id])
        counts[chosen[0].hypothesis_id] += 1

    selected = [item for group in selected_groups for item in group]
    round_counts = Counter(group[0].hypothesis_id for group in selected_groups)
    return selected, {
        "selection_strategy": "closed_loop_soft_diminishing_return_v1",
        "requested_examples": budget,
        "actual_examples": len(selected),
        "requested_pairs": requested_pairs,
        "actual_pairs": len(selected_groups),
        "shortage": len(selected) < budget,
        "hard_hypothesis_quota": False,
        "hypothesis_pair_counts_this_round": dict(sorted(round_counts.items())),
        "hypothesis_pair_counts_cumulative": dict(sorted(counts.items())),
        "prior_novelty_anchor_pairs": len(selected_prior_embeddings),
        "selected_pair_components": selected_components,
        "normalization": "tie_aware_percentile_per_round",
        "weights": {
            "nll": nll_weight,
            "entropy": entropy_weight,
            "novelty": novelty_weight,
            "hypothesis_bonus": hypothesis_bonus_weight,
            "new_hypothesis_bonus": new_hypothesis_bonus_weight,
        },
    }


def select_confidence_safe_entropy_novelty_pairs(
    examples: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    nll_scores: Mapping[str, float],
    entropy_scores: Mapping[str, float],
    embeddings: Mapping[str, Sequence[float]],
    verification_evidence: Mapping[str, Mapping[str, Any]],
    excluded_ids: Sequence[str] = (),
    nll_weight: float = 0.65,
    entropy_weight: float = 0.35,
    novelty_weight: float = 0.50,
    nll_quantile_low: float = 0.10,
    nll_quantile_high: float = 0.90,
) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
    """Select hard, non-redundant pairs while retaining every safe hypothesis.

    Difficulty is measured only on the failure-targeting ``base`` member, so an
    easy control cannot hide a confidently wrong target.  The first pass picks
    one pair from every certified or unresolved hypothesis; the remaining
    budget is filled greedily using target difficulty plus contextual-embedding
    novelty.  Refuted hypotheses are rejected before this function is called.
    """
    if budget <= 0 or budget % 2:
        raise ValueError("Synthetic example budget must be positive and even")
    if not 0.0 <= nll_quantile_low < nll_quantile_high <= 1.0:
        raise ValueError("NLL quantiles must satisfy 0 <= low < high <= 1")
    weights = (nll_weight, entropy_weight, novelty_weight)
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("Selection weights must be finite and non-negative")
    if nll_weight + entropy_weight <= 0.0:
        raise ValueError("At least one difficulty weight must be positive")

    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)

    complete: list[list[SyntheticNLIExample]] = []
    dimensions: set[int] = set()
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        if excluded & {item.id for item in members}:
            continue
        pair_type = members[0].pair_type
        hypothesis_id = members[0].hypothesis_id
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        if any(item.hypothesis_id != hypothesis_id for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} mixes hypotheses")
        if {item.pair_member for item in members} != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        if hypothesis_id not in verification_evidence:
            raise ValueError(f"Missing verification evidence for {hypothesis_id!r}")
        state = str(verification_evidence[hypothesis_id].get("state", ""))
        if state not in {"certified", "unresolved"}:
            raise ValueError(
                f"Pool contains non-retained hypothesis {hypothesis_id!r}: {state!r}"
            )
        for item in members:
            if item.id not in nll_scores or item.id not in entropy_scores:
                raise ValueError(f"Missing difficulty score for {item.id!r}")
            if item.id not in embeddings:
                raise ValueError(f"Missing embedding for {item.id!r}")
            nll = float(nll_scores[item.id])
            entropy = float(entropy_scores[item.id])
            if not math.isfinite(nll) or not math.isfinite(entropy):
                raise ValueError(f"Non-finite difficulty score for {item.id!r}")
            if not 0.0 <= entropy <= 1.0 + 1e-9:
                raise ValueError(f"Normalized entropy is outside [0, 1] for {item.id!r}")
            dimensions.add(len(embeddings[item.id]))
        members.sort(key=lambda item: item.pair_member)
        complete.append(members)
    if len(dimensions) > 1:
        raise ValueError("All embeddings must have the same dimension")

    requested_pairs = budget // 2
    if len(complete) < requested_pairs:
        selected = [item for group in complete for item in group]
        return selected, {
            "selection_strategy": "confidence_safe_entropy_novelty_v1",
            "requested_examples": budget,
            "actual_examples": len(selected),
            "requested_pairs": requested_pairs,
            "actual_pairs": len(complete),
            "shortage": True,
        }

    hypothesis_ids = sorted({group[0].hypothesis_id for group in complete})
    if requested_pairs < len(hypothesis_ids):
        raise ValueError(
            "Budget cannot provide the declared one-pair coverage guarantee: "
            f"pairs={requested_pairs}, retained_hypotheses={len(hypothesis_ids)}"
        )

    target_by_pair: dict[str, SyntheticNLIExample] = {}
    raw_target_nll: dict[str, float] = {}
    target_entropy: dict[str, float] = {}
    pair_embeddings: dict[str, tuple[float, ...]] = {}
    for group in complete:
        pair_id = group[0].pair_id
        target = next(item for item in group if item.pair_member == "base")
        target_by_pair[pair_id] = target
        raw_target_nll[pair_id] = float(nll_scores[target.id])
        target_entropy[pair_id] = min(1.0, float(entropy_scores[target.id]))
        member_vectors = [_unit_vector(embeddings[item.id]) for item in group]
        pair_embeddings[pair_id] = _unit_vector(
            [
                sum(vector[index] for vector in member_vectors) / len(member_vectors)
                for index in range(len(member_vectors[0]))
            ]
        )

    nll_low = _linear_quantile(list(raw_target_nll.values()), nll_quantile_low)
    nll_high = _linear_quantile(list(raw_target_nll.values()), nll_quantile_high)
    normalized_nll = {
        pair_id: (
            0.5
            if nll_high == nll_low
            else (min(nll_high, max(nll_low, value)) - nll_low) / (nll_high - nll_low)
        )
        for pair_id, value in raw_target_nll.items()
    }
    difficulty_scale = nll_weight + entropy_weight
    difficulty = {
        pair_id: (
            nll_weight * normalized_nll[pair_id]
            + entropy_weight * target_entropy[pair_id]
        )
        / difficulty_scale
        for pair_id in raw_target_nll
    }

    def tie_key(pair_id: str, phase: str) -> tuple[bytes, str]:
        return (
            hashlib.sha256(f"{seed}:{phase}:{pair_id}".encode()).digest(),
            pair_id,
        )

    remaining = list(complete)
    selected_groups: list[list[SyntheticNLIExample]] = []
    selected_embeddings: list[tuple[float, ...]] = []
    selected_novelty: dict[str, float] = {}

    # Coverage is a safety constraint, not an equal-allocation quota.
    for hypothesis_id in sorted(
        hypothesis_ids,
        key=lambda value: hashlib.sha256(f"{seed}:coverage:{value}".encode()).digest(),
    ):
        candidates = [group for group in remaining if group[0].hypothesis_id == hypothesis_id]
        chosen = min(
            candidates,
            key=lambda group: (
                -difficulty[group[0].pair_id],
                *tie_key(group[0].pair_id, "coverage"),
            ),
        )
        novelty = _cosine_novelty(
            pair_embeddings[chosen[0].pair_id], selected_embeddings
        )
        remaining.remove(chosen)
        selected_groups.append(chosen)
        selected_embeddings.append(pair_embeddings[chosen[0].pair_id])
        selected_novelty[chosen[0].pair_id] = novelty

    while len(selected_groups) < requested_pairs:
        def rank(group: list[SyntheticNLIExample]) -> tuple[float, bytes, str]:
            pair_id = group[0].pair_id
            novelty = _cosine_novelty(pair_embeddings[pair_id], selected_embeddings)
            score = difficulty[pair_id] + novelty_weight * novelty
            tie, identity = tie_key(pair_id, "greedy")
            return (-score, tie, identity)

        chosen = min(remaining, key=rank)
        pair_id = chosen[0].pair_id
        novelty = _cosine_novelty(pair_embeddings[pair_id], selected_embeddings)
        remaining.remove(chosen)
        selected_groups.append(chosen)
        selected_embeddings.append(pair_embeddings[pair_id])
        selected_novelty[pair_id] = novelty

    selected = [item for group in selected_groups for item in group]
    hypothesis_counts = Counter(group[0].hypothesis_id for group in selected_groups)
    label_counts = Counter(item.label for item in selected)
    pair_type_counts = Counter(group[0].pair_type for group in selected_groups)
    selected_pair_ids = [group[0].pair_id for group in selected_groups]
    return selected, {
        "selection_strategy": "confidence_safe_entropy_novelty_v1",
        "requested_examples": budget,
        "actual_examples": len(selected),
        "requested_pairs": requested_pairs,
        "actual_pairs": len(selected_groups),
        "shortage": len(selected) < budget,
        "coverage_constraint": "at_least_one_pair_per_retained_hypothesis",
        "coverage_seed_pairs": len(hypothesis_ids),
        "retained_hypothesis_states": {
            hypothesis_id: str(verification_evidence[hypothesis_id]["state"])
            for hypothesis_id in hypothesis_ids
        },
        "hypothesis_pair_counts": dict(sorted(hypothesis_counts.items())),
        "pair_type_counts": dict(sorted(pair_type_counts.items())),
        "label_counts": {label: label_counts[label] for label in MNLI_LABELS},
        "mean_selected_target_nll": sum(raw_target_nll[pair_id] for pair_id in selected_pair_ids)
        / len(selected_pair_ids),
        "mean_selected_target_entropy": sum(
            target_entropy[pair_id] for pair_id in selected_pair_ids
        )
        / len(selected_pair_ids),
        "mean_selected_embedding_novelty": sum(
            selected_novelty[pair_id] for pair_id in selected_pair_ids
        )
        / len(selected_pair_ids),
        "nll_normalization": {
            "scope": "global_target_base_members",
            "quantile_low": nll_quantile_low,
            "quantile_high": nll_quantile_high,
            "lower_value": nll_low,
            "upper_value": nll_high,
        },
        "weights": {
            "nll": nll_weight,
            "entropy": entropy_weight,
            "novelty": novelty_weight,
        },
    }


def select_verification_conditioned_primal_dual_pairs(
    examples: Sequence[SyntheticNLIExample],
    *,
    budget: int,
    seed: int,
    utility_scores: Mapping[str, float],
    verification_evidence: Mapping[str, Mapping[str, Any]],
    excluded_ids: Sequence[str] = (),
    label_temperature: float = 1.0,
    hypothesis_temperature: float = 0.25,
    utility_quantile_low: float = 0.10,
    utility_quantile_high: float = 0.90,
    utility_weight: float = 1.0,
    diversity_weight: float = 0.35,
    specificity_weight: float = 0.50,
    label_dual_weight: float = 1.25,
    hypothesis_dual_weight: float = 0.75,
    pair_type_dual_weight: float = 0.50,
    dual_step: float = 0.50,
    dual_clip: float = 3.0,
    sampling_temperature: float = 0.20,
) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
    """Sample complete pairs using verified failures and soft control evidence.

    The target label distribution is learned from verified target failures,
    instead of being hard-coded. Counterfactual and invariant controls affect a
    hypothesis-specificity score, but never act as a rejection gate. Online
    dual variables softly steer label, hypothesis and pair-type proportions;
    Gumbel sampling keeps acquisition stochastic while the seed makes it fully
    reproducible.
    """
    if budget <= 0 or budget % 2:
        raise ValueError("Synthetic example budget must be positive and even")
    if not 0.0 <= utility_quantile_low < utility_quantile_high <= 1.0:
        raise ValueError("Utility quantiles must satisfy 0 <= low < high <= 1")
    nonnegative = {
        "utility_weight": utility_weight,
        "diversity_weight": diversity_weight,
        "specificity_weight": specificity_weight,
        "label_dual_weight": label_dual_weight,
        "hypothesis_dual_weight": hypothesis_dual_weight,
        "pair_type_dual_weight": pair_type_dual_weight,
        "dual_step": dual_step,
        "dual_clip": dual_clip,
    }
    if any(not math.isfinite(value) or value < 0.0 for value in nonnegative.values()):
        raise ValueError("Selection weights and dual parameters must be finite and non-negative")
    if not math.isfinite(sampling_temperature) or sampling_temperature <= 0.0:
        raise ValueError("sampling_temperature must be finite and positive")

    excluded = set(excluded_ids)
    if len(excluded) != len(excluded_ids):
        raise ValueError("excluded_ids must be unique")
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen: set[str] = set()
    for item in examples:
        if item.id in seen:
            raise ValueError(f"Duplicate synthetic example ID: {item.id}")
        seen.add(item.id)
        grouped.setdefault(item.pair_id, []).append(item)

    complete: list[list[SyntheticNLIExample]] = []
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} is incomplete")
        if excluded & {item.id for item in members}:
            continue
        pair_type = members[0].pair_type
        if any(item.pair_type != pair_type for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent types")
        if {item.pair_member for item in members} != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has invalid members")
        if any(item.id not in utility_scores for item in members):
            raise ValueError(f"Missing utility score for synthetic pair {pair_id!r}")
        if any(not math.isfinite(float(utility_scores[item.id])) for item in members):
            raise ValueError(f"Non-finite utility score for synthetic pair {pair_id!r}")
        if members[0].hypothesis_id not in verification_evidence:
            raise ValueError(
                f"Missing verification evidence for {members[0].hypothesis_id!r}"
            )
        if any(item.hypothesis_id != members[0].hypothesis_id for item in members):
            raise ValueError(f"Synthetic pair {pair_id!r} mixes hypotheses")
        members.sort(key=lambda item: item.pair_member)
        complete.append(members)

    requested_pairs = budget // 2
    if len(complete) < requested_pairs:
        selected = [item for group in complete for item in group]
        return selected, {
            "selection_strategy": "verification_conditioned_primal_dual_v1",
            "requested_examples": budget,
            "actual_examples": len(selected),
            "requested_pairs": requested_pairs,
            "actual_pairs": len(complete),
            "shortage": True,
        }

    hypothesis_ids = sorted({group[0].hypothesis_id for group in complete})
    evidence_ids = set(verification_evidence)
    if set(hypothesis_ids) - evidence_ids:
        raise ValueError("Every pool hypothesis must have verification evidence")

    label_failure_evidence = {label: 0.0 for label in MNLI_LABELS}
    specificity: dict[str, float] = {}
    evidence_diagnostics: dict[str, Any] = {}
    for hypothesis_id in hypothesis_ids:
        evidence = verification_evidence[hypothesis_id]
        lower_bound = float(evidence.get("lower_bound", 0.0))
        counterfactual_accuracy = float(evidence.get("counterfactual_accuracy", 0.0))
        invariant_accuracy = float(evidence.get("invariant_accuracy", 0.0))
        if any(
            not math.isfinite(value)
            for value in (lower_bound, counterfactual_accuracy, invariant_accuracy)
        ):
            raise ValueError(f"Non-finite verification evidence for {hypothesis_id!r}")
        if not 0.0 <= counterfactual_accuracy <= 1.0 or not 0.0 <= invariant_accuracy <= 1.0:
            raise ValueError(f"Control accuracies must lie in [0, 1] for {hypothesis_id!r}")
        failures_value = evidence.get("target_failures_by_label", {})
        target_n = int(evidence.get("target_n", 0))
        if not isinstance(failures_value, Mapping) or target_n <= 0:
            raise ValueError(f"Invalid target failure evidence for {hypothesis_id!r}")
        failures = {label: int(failures_value.get(label, 0)) for label in MNLI_LABELS}
        if any(value < 0 for value in failures.values()) or sum(failures.values()) > target_n:
            raise ValueError(f"Invalid target failure counts for {hypothesis_id!r}")
        for label in MNLI_LABELS:
            label_failure_evidence[label] += max(0.0, lower_bound) * (
                failures[label] / target_n
            )
        raw_specificity = lower_bound - 0.5 * (
            (1.0 - counterfactual_accuracy) + (1.0 - invariant_accuracy)
        )
        specificity[hypothesis_id] = raw_specificity
        evidence_diagnostics[hypothesis_id] = {
            "lower_bound": lower_bound,
            "counterfactual_accuracy": counterfactual_accuracy,
            "invariant_accuracy": invariant_accuracy,
            "control_specificity": raw_specificity,
            "target_n": target_n,
            "target_failures_by_label": failures,
        }

    target_label_distribution = _softmax_distribution(
        label_failure_evidence, temperature=label_temperature
    )
    target_hypothesis_distribution = _softmax_distribution(
        specificity, temperature=hypothesis_temperature
    )
    pair_types = sorted({group[0].pair_type for group in complete})
    target_pair_type_distribution = {
        pair_type: 1.0 / len(pair_types) for pair_type in pair_types
    }

    # NLL scales can differ strongly across mechanisms and labels. Normalize
    # within (hypothesis, gold-label) strata after robust quantile clipping.
    strata: dict[tuple[str, str], list[float]] = {}
    for group in complete:
        for item in group:
            strata.setdefault((item.hypothesis_id, item.label), []).append(
                float(utility_scores[item.id])
            )
    stratum_bounds = {
        stratum: (
            _linear_quantile(values, utility_quantile_low),
            _linear_quantile(values, utility_quantile_high),
        )
        for stratum, values in strata.items()
    }
    normalized_item_utility: dict[str, float] = {}
    for group in complete:
        for item in group:
            low, high = stratum_bounds[(item.hypothesis_id, item.label)]
            clipped = min(high, max(low, float(utility_scores[item.id])))
            normalized_item_utility[item.id] = (
                0.5 if high == low else (clipped - low) / (high - low)
            )
    pair_utility = {
        group[0].pair_id: sum(normalized_item_utility[item.id] for item in group) / 2.0
        for group in complete
    }
    spec_min = min(specificity.values())
    spec_max = max(specificity.values())
    normalized_specificity = {
        key: 0.5 if spec_max == spec_min else (value - spec_min) / (spec_max - spec_min)
        for key, value in specificity.items()
    }

    label_dual = {label: 0.0 for label in MNLI_LABELS}
    hypothesis_dual = {key: 0.0 for key in hypothesis_ids}
    pair_type_dual = {key: 0.0 for key in pair_types}
    label_counts: Counter[str] = Counter()
    hypothesis_counts: Counter[str] = Counter()
    pair_type_counts: Counter[str] = Counter()
    selected_groups: list[list[SyntheticNLIExample]] = []
    selected_tokens: list[frozenset[str]] = []
    remaining = list(complete)
    rng = random.Random(seed)

    def update_dual(
        dual: dict[str, float],
        target: Mapping[str, float],
        counts: Mapping[str, int],
        denominator: int,
    ) -> None:
        for key in dual:
            observed = counts.get(key, 0) / denominator if denominator else 0.0
            dual[key] = min(
                dual_clip,
                max(-dual_clip, dual[key] + dual_step * (target[key] - observed)),
            )

    while len(selected_groups) < requested_pairs:
        update_dual(label_dual, target_label_distribution, label_counts, 2 * len(selected_groups))
        update_dual(
            hypothesis_dual,
            target_hypothesis_distribution,
            hypothesis_counts,
            len(selected_groups),
        )
        update_dual(
            pair_type_dual,
            target_pair_type_distribution,
            pair_type_counts,
            len(selected_groups),
        )

        scored: list[tuple[float, bytes, str, list[SyntheticNLIExample]]] = []
        for group in remaining:
            hypothesis_id = group[0].hypothesis_id
            pair_type = group[0].pair_type
            tokens = _selection_tokens(group)
            diversity = (
                1.0
                if not selected_tokens
                else 1.0 - max(_jaccard(tokens, prior) for prior in selected_tokens)
            )
            label_term = sum(label_dual[item.label] for item in group) / 2.0
            score = (
                utility_weight * pair_utility[group[0].pair_id]
                + diversity_weight * diversity
                + specificity_weight * normalized_specificity[hypothesis_id]
                + label_dual_weight * label_term
                + hypothesis_dual_weight * hypothesis_dual[hypothesis_id]
                + pair_type_dual_weight * pair_type_dual[pair_type]
            )
            # Gumbel-max is exactly a softmax sample over the current scores.
            uniform = min(1.0 - 1e-12, max(1e-12, rng.random()))
            gumbel = -math.log(-math.log(uniform))
            sampled_score = score / sampling_temperature + gumbel
            tie = hashlib.sha256(
                f"{seed}:verification-primal-dual:{group[0].pair_id}".encode()
            ).digest()
            scored.append((sampled_score, tie, group[0].pair_id, group))
        chosen = max(scored, key=lambda item: (item[0], item[1], item[2]))[3]
        remaining.remove(chosen)
        selected_groups.append(chosen)
        selected_tokens.append(_selection_tokens(chosen))
        hypothesis_counts[chosen[0].hypothesis_id] += 1
        pair_type_counts[chosen[0].pair_type] += 1
        label_counts.update(item.label for item in chosen)

    selected = [item for group in selected_groups for item in group]
    mean_raw_nll = sum(float(utility_scores[item.id]) for item in selected) / len(selected)
    mean_normalized_nll = sum(normalized_item_utility[item.id] for item in selected) / len(selected)
    return selected, {
        "selection_strategy": "verification_conditioned_primal_dual_v1",
        "requested_examples": budget,
        "actual_examples": len(selected),
        "requested_pairs": requested_pairs,
        "actual_pairs": len(selected_groups),
        "shortage": len(selected) < budget,
        "seeded_sampling": True,
        "target_label_distribution": target_label_distribution,
        "label_failure_evidence": label_failure_evidence,
        "label_counts": {label: label_counts[label] for label in MNLI_LABELS},
        "target_hypothesis_distribution": target_hypothesis_distribution,
        "hypothesis_pair_counts": dict(sorted(hypothesis_counts.items())),
        "target_pair_type_distribution": target_pair_type_distribution,
        "pair_type_counts": dict(sorted(pair_type_counts.items())),
        "verification_evidence": evidence_diagnostics,
        "mean_selected_student_nll": mean_raw_nll,
        "mean_selected_normalized_student_nll": mean_normalized_nll,
        "utility_normalization": {
            "stratum": "hypothesis_id_x_gold_label",
            "quantile_low": utility_quantile_low,
            "quantile_high": utility_quantile_high,
        },
        "final_duals": {
            "label": label_dual,
            "hypothesis": hypothesis_dual,
            "pair_type": pair_type_dual,
        },
        "weights": {
            **nonnegative,
            "sampling_temperature": sampling_temperature,
            "label_temperature": label_temperature,
            "hypothesis_temperature": hypothesis_temperature,
        },
    }
