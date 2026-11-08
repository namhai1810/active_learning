import math

import pytest

from src.decision.cost_model import CostRates
from src.decision.final_mathematical_feasibility import (
    compare_gain_bounds,
    consecutive_gain_drifts,
    empirical_bernstein_radius,
    familywise_alpha,
    maximum_rho_for_stop,
    net_ucb,
    protocol_full_round_cost_lower_bound,
    range_aware_hoeffding_radius,
)


def observations(gains):
    baseline_accuracy = 0.625
    risk_before = 1.0 - baseline_accuracy
    return [
        {
            "gain": gain,
            "risk_before": risk_before,
            "risk_after": 1.0 - (baseline_accuracy + gain),
            "pilot_seed": 100 + index,
            "eval_ids": ["e1", "e2"],
            "train_ids": ["t1", "t2"],
            "starting_checkpoint": {"sha256": "abc"},
        }
        for index, gain in enumerate(gains)
    ]


def frozen_rates():
    return CostRates(
        llm_input_per_1k_tokens=0.0002,
        llm_output_per_1k_tokens=0.0006,
        data_generation_per_item=0.001,
        data_verification_per_item=0.001,
        lora_training_per_run=0.02,
        lora_training_per_gpu_hour=0.5,
        evaluation_per_run=0.005,
        evaluation_per_item=0.0001,
        evaluation_per_gpu_hour=0.5,
    )


def test_familywise_accounting_and_range_aware_radius():
    assert familywise_alpha(delta=0.05, K=1, T=7) == pytest.approx(0.05 / 7)
    radius = range_aware_hoeffding_radius(32, delta=0.05, K=1, T=7)
    assert radius == pytest.approx(math.sqrt(math.log(280.0) / 64.0))
    assert radius == pytest.approx(0.29672139718854035)


def test_empirical_bernstein_keeps_finite_sample_correction():
    values = [0.0] * 32
    radius = empirical_bernstein_radius(
        values, delta=0.05, K=1, T=7, width=1.0
    )
    correction = 7.0 * math.log(280.0) / (3.0 * 31.0)
    assert radius == pytest.approx(correction)
    assert radius > range_aware_hoeffding_radius(32, delta=0.05, K=1, T=7)


def test_gain_comparison_uses_exact_support_and_does_not_claim_independence():
    values = [0.375] * 32  # exact support upper endpoint at A0=0.625
    comparison, audit = compare_gain_bounds(
        observations(values),
        required_pilots=32,
        delta=0.05,
        K=1,
        T=7,
    )
    assert comparison.support_upper == pytest.approx(0.375)
    assert comparison.hoeffding_gain_ucb == pytest.approx(0.375)
    assert comparison.empirical_bernstein_gain_ucb == pytest.approx(0.375)
    assert comparison.conditional_independence_proved is False
    assert comparison.population_unbiasedness_claimed is False
    assert audit["shared_eval_examples_counted_as_independent"] is False


def test_full_round_lcb_excludes_unproved_variable_and_gpu_costs():
    bound = protocol_full_round_cost_lower_bound(
        rates=frozen_rates(),
        currency="USD",
        acquired_examples=32,
        stopping_examples=900,
    )
    # selection: 2*.005 + 64*.0001 = .0164
    # LoRA fixed run: .02
    # minimum validation: 2*.005 + 1800*.0001 = .19
    assert bound.total == pytest.approx(0.2264)
    assert bound.usage["lora_training_gpu_seconds"] == 0.0
    assert bound.usage["evaluation_gpu_seconds"] == 0.0
    by_stage = {item.stage: item for item in bound.components}
    for stage in ("discovery", "verification", "generation_and_qc"):
        assert by_stage[stage].included is False
        assert by_stage[stage].lower_bound == 0.0
    assert by_stage["main_lora_training"].included is True


def test_net_ucb_and_rho_threshold_have_consistent_signs():
    value = net_ucb(
        gain_ucb=0.03, rho=0.01, lambda_cost=0.2, cost_lcb=0.2264
    )
    assert value == pytest.approx(-0.00528)
    threshold = maximum_rho_for_stop(
        gain_ucb=0.03, lambda_cost=0.2, cost_lcb=0.2264
    )
    assert threshold == pytest.approx(0.01528)
    assert net_ucb(
        gain_ucb=0.03,
        rho=threshold,
        lambda_cost=0.2,
        cost_lcb=0.2264,
    ) == pytest.approx(0.0)


def test_drift_only_uses_consecutive_complete_rounds():
    rows = consecutive_gain_drifts({2: 0.01, 3: -0.02, 5: 0.08, 6: 0.04})
    assert [(row["from_round"], row["to_round"]) for row in rows] == [
        (2.0, 3.0),
        (5.0, 6.0),
    ]
    assert rows[0]["signed_drift"] == pytest.approx(-0.03)
    assert rows[1]["absolute_drift"] == pytest.approx(0.04)


@pytest.mark.parametrize("rho", [-0.1, float("nan")])
def test_invalid_uncertainty_is_rejected(rho):
    with pytest.raises(ValueError):
        net_ucb(gain_ucb=0.1, rho=rho, lambda_cost=0.2, cost_lcb=0.2)
