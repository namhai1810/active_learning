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
