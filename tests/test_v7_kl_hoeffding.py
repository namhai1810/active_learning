import math

import pytest

from src.gain.kl_hoeffding import (
    bernoulli_kl,
    bernoulli_kl_upper_bound,
    compare_kl_and_range_aware_hoeffding,
    kl_confidence_budget,
    normalized_pilot_accuracies,
    pre_round_bound_availability,
)


def pilot_observations(gains, *, baseline_accuracy=0.625):
    return [
        {
            "gain": gain,
            "risk_before": 1.0 - baseline_accuracy,
            "risk_after": 1.0 - (baseline_accuracy + gain),
            "pilot_seed": 1000 + index,
            "eval_ids": ["e1", "e2"],
            "train_ids": ["t1", "t2"],
            "starting_checkpoint": {"sha256": "same"},
        }
        for index, gain in enumerate(gains)
    ]


def test_bernoulli_kl_known_value_and_identity():
    expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert bernoulli_kl(0.25, 0.5) == pytest.approx(expected)
    assert bernoulli_kl(0.0, 0.0) == 0.0
    assert bernoulli_kl(1.0, 1.0) == 0.0
    assert math.isinf(bernoulli_kl(0.5, 0.0))
    assert math.isinf(bernoulli_kl(0.5, 1.0))


def test_binary_search_satisfies_kl_boundary():
    budget = 0.12
    upper = bernoulli_kl_upper_bound(0.4, confidence_budget=budget)
    assert 0.4 < upper < 1.0
    assert bernoulli_kl(0.4, upper) == pytest.approx(budget, abs=1e-11)


def test_binary_search_is_stable_at_zero_and_one():
    budget = 0.2
    upper_zero = bernoulli_kl_upper_bound(0.0, confidence_budget=budget)
    assert upper_zero == pytest.approx(1.0 - math.exp(-budget), abs=1e-12)
    assert bernoulli_kl_upper_bound(1.0, confidence_budget=budget) == 1.0
    assert bernoulli_kl_upper_bound(0.3, confidence_budget=0.0) == 0.3


def test_confidence_budget_matches_frozen_protocol():
    budget = kl_confidence_budget(m=32, delta=0.05, K=1, T=7)
    assert budget == pytest.approx(math.log(280.0) / 32.0)


def test_normalization_is_accuracy_after_and_bounded():
    observations = pilot_observations([-0.125, 0.0, 0.375] + [0.0] * 29)
    normalized, audit = normalized_pilot_accuracies(
        observations, required_pilots=32
    )
    assert normalized[:3] == pytest.approx([0.5, 0.625, 1.0])
    assert min(normalized) >= 0.0
    assert max(normalized) <= 1.0
    assert audit["baseline_accuracy"] == pytest.approx(0.625)


def test_normalization_rejects_invalid_gain_identity_or_support():
    observations = pilot_observations([0.0] * 32)
    observations[0]["gain"] = 0.5
    with pytest.raises(ValueError):
        normalized_pilot_accuracies(observations, required_pilots=32)


@pytest.mark.parametrize("mean", [-0.01, 1.01, float("nan"), float("inf")])
def test_kl_upper_bound_rejects_invalid_means(mean):
    with pytest.raises(ValueError):
        bernoulli_kl_upper_bound(mean, confidence_budget=0.1)


def test_kl_radius_never_exceeds_range_aware_hoeffding_radius():
    gains = [-0.0625, -0.03125, 0.0, 0.03125] * 8
    comparison, audit = compare_kl_and_range_aware_hoeffding(
        pilot_observations(gains),
        required_pilots=32,
        delta=0.05,
        K=1,
        T=7,
    )
    assert comparison.kl_radius <= comparison.hoeffding_radius
    assert comparison.kl_gain_ucb <= comparison.hoeffding_gain_ucb
    assert comparison.radius_reduction_percent >= 0.0
    assert audit["kl_radius_le_hoeffding_radius"] is True
    assert comparison.conditional_independence_proved is False


def test_same_round_pilots_cannot_leak_into_pre_round_decision():
    status = pre_round_bound_availability(
        pilot_source_round=4,
        decision_round=4,
        cross_round_transfer_valid=True,
    )
    assert status["available_predecision"] is False
    assert status["valid_for_next_round_gain"] is False


def test_prior_round_still_requires_cross_round_transfer_assumption():
    invalid = pre_round_bound_availability(
        pilot_source_round=3,
        decision_round=4,
        cross_round_transfer_valid=False,
    )
    assert invalid["available_predecision"] is True
    assert invalid["valid_for_next_round_gain"] is False
    valid = pre_round_bound_availability(
        pilot_source_round=3,
        decision_round=4,
        cross_round_transfer_valid=True,
    )
    assert valid["valid_for_next_round_gain"] is True
