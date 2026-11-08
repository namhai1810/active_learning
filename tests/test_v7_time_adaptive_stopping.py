import math

import pytest

from src.decision.time_adaptive_stopping import (
    assess_time_adaptive_decision,
    historical_window_estimate,
    planned_test_family_size,
    simultaneous_hoeffding_radius,
    weighted_temporal_drift,
)


def test_window_is_strictly_predecision_and_excludes_supplied_future_rounds():
    estimate = historical_window_estimate(
        {2: [0.1], 3: [0.2], 4: [99.0], 5: [100.0]},
        decision_round=4,
        window_length=3,
    )
    assert estimate.calendar_rounds == (1, 2, 3)
    assert estimate.maximum_source_round == 3
    assert estimate.n_eff == 2
    assert estimate.mean_gain == pytest.approx(0.15)
    assert estimate.strict_predecision_available is True


def test_observation_weighting_does_not_average_round_means_equally():
    estimate = historical_window_estimate(
        {2: [0.0], 3: [1.0, 1.0, 1.0]},
        decision_round=4,
        window_length=2,
    )
    assert estimate.observation_counts_by_round == {2: 1, 3: 3}
    assert estimate.n_eff == 4
    assert estimate.mean_gain == pytest.approx(0.75)
    assert estimate.weighted_mean_of_round_means == pytest.approx(0.75)
    assert estimate.mean_gain != pytest.approx(0.5)


def test_missing_pilot_round_stays_in_calendar_but_has_zero_weight():
    estimate = historical_window_estimate(
        {2: [0.1, 0.2], 4: [-0.1]},
        decision_round=5,
        window_length=3,
    )
    assert estimate.calendar_rounds == (2, 3, 4)
    assert estimate.calendar_window_length == 3
    assert estimate.zero_pilot_rounds == (3,)
    assert estimate.observed_rounds == (2, 4)
    assert estimate.n_eff == 3


def test_cumulative_history_starts_at_round_one():
    estimate = historical_window_estimate(
        {2: [0.1], 3: [0.2]}, decision_round=4, window_length=None
    )
    assert estimate.estimator == "cumulative"
    assert estimate.calendar_rounds == (1, 2, 3)
    assert estimate.zero_pilot_rounds == (1,)


def test_bonferroni_family_counts_all_windows_and_decisions():
    n_tests = planned_test_family_size(
        decision_rounds=7, fixed_windows=7, include_cumulative=True
    )
    assert n_tests == 56
    radius = simultaneous_hoeffding_radius(
        32, delta=0.05, n_tests=n_tests, width=1.0
    )
    assert radius == pytest.approx(math.sqrt(math.log(2240.0) / 64.0))


def test_radius_uses_support_width_and_effective_observation_count():
    r32 = simultaneous_hoeffding_radius(
        32, delta=0.05, n_tests=56, width=1.0
    )
    r96 = simultaneous_hoeffding_radius(
        96, delta=0.05, n_tests=56, width=1.0
    )
    assert r96 == pytest.approx(r32 / math.sqrt(3.0))
    assert simultaneous_hoeffding_radius(
        32, delta=0.05, n_tests=56, width=0.5
    ) == pytest.approx(r32 / 2.0)


def test_weighted_drift_accounts_for_every_observation_age():
    rho = weighted_temporal_drift(
        {2: 32, 3: 0, 4: 16}, decision_round=5, per_round_drift=0.01
    )
    expected = (32 * 0.01 * 3 + 16 * 0.01 * 1) / 48
    assert rho == pytest.approx(expected)


def test_weighted_drift_rejects_current_or_future_round():
    with pytest.raises(ValueError, match="current/future"):
        weighted_temporal_drift(
            {4: 32}, decision_round=4, per_round_drift=0.01
        )


def test_unknown_assumptions_prohibit_numerical_stop_certificate():
    estimate = historical_window_estimate(
        {2: [-0.5] * 32}, decision_round=3, window_length=1
    )
    decision = assess_time_adaptive_decision(
        estimate=estimate,
        statistical_radius=0.1,
        rho=0.0,
        eta=0.0,
        cost_lcb=0.2264,
        lambda_cost=0.2,
        sampling_assumptions_valid=False,
        drift_assumption_valid=False,
        policy_mismatch_assumption_valid=False,
        cost_lcb_applicable_predecision=False,
    )
    assert decision.optimistic_net_ucb < 0
    assert decision.action == "continue"
    assert decision.certification == "uncertified_continue"
    assert "temporal_drift_bound_unknown" in decision.evidence_issues


def test_conditional_cost_lcb_is_not_silently_promoted():
    estimate = historical_window_estimate(
        {2: [-0.5] * 32}, decision_round=3, window_length=1
    )
    decision = assess_time_adaptive_decision(
        estimate=estimate,
        statistical_radius=0.1,
        rho=0.0,
        eta=0.0,
        cost_lcb=1.0,
        lambda_cost=0.2,
        sampling_assumptions_valid=True,
        drift_assumption_valid=True,
        policy_mismatch_assumption_valid=True,
        cost_lcb_applicable_predecision=False,
    )
    assert decision.certification == "uncertified_continue"
    assert "conditional_cost_lcb_applicability_unproved_predecision" in (
        decision.evidence_issues
    )


def test_all_valid_assumptions_allow_nonpositive_certificate():
    estimate = historical_window_estimate(
        {2: [-0.5] * 32}, decision_round=3, window_length=1
    )
    decision = assess_time_adaptive_decision(
        estimate=estimate,
        statistical_radius=0.1,
        rho=0.01,
        eta=0.0,
        cost_lcb=0.2264,
        lambda_cost=0.2,
        sampling_assumptions_valid=True,
        drift_assumption_valid=True,
        policy_mismatch_assumption_valid=True,
        cost_lcb_applicable_predecision=True,
    )
    assert decision.net_ucb < 0
    assert decision.certification == "certified_stop"


def test_retrospective_metrics_never_change_policy_decision():
    estimate = historical_window_estimate(
        {2: [-0.5] * 32}, decision_round=3, window_length=1
    )
    kwargs = dict(
        estimate=estimate,
        statistical_radius=0.1,
        rho=0.0,
        eta=0.0,
        cost_lcb=0.2264,
        lambda_cost=0.2,
        sampling_assumptions_valid=False,
        drift_assumption_valid=False,
        policy_mismatch_assumption_valid=False,
        cost_lcb_applicable_predecision=False,
    )
    left = assess_time_adaptive_decision(
        **kwargs, retrospective_metrics={"accuracy": 0.0, "future": -999}
    )
    right = assess_time_adaptive_decision(
        **kwargs, retrospective_metrics={"accuracy": 1.0, "future": 999}
    )
    assert left == right


def test_budget_stop_is_separate_from_certified_stop():
    estimate = historical_window_estimate(
        {2: [0.1] * 32}, decision_round=3, window_length=1
    )
    decision = assess_time_adaptive_decision(
        estimate=estimate,
        statistical_radius=0.1,
        rho=0.0,
        eta=0.0,
        cost_lcb=0.2264,
        lambda_cost=0.2,
        sampling_assumptions_valid=True,
        drift_assumption_valid=True,
        policy_mismatch_assumption_valid=True,
        cost_lcb_applicable_predecision=True,
        budget_exhausted=True,
    )
    assert decision.certification == "budget_stop"
    assert decision.reason == "maximum_budget_reached_not_gain_cost_certificate"
