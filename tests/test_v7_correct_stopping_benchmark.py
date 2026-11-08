from __future__ import annotations

from dataclasses import replace

import pytest

from scripts.benchmark_v7_correct_stopping import (
    Evidence,
    _historical_result_summary,
    _ingest_payload,
    _ingest_range_replay,
)
from src.decision.correct_stopping_benchmark import (
    CostSnapshot,
    DecisionEvidence,
    RoundRecord,
    Trajectory,
    benchmark_trajectory,
    evaluate_policy,
    freeze_recorded_policy,
    freeze_validation_patience,
    retrospective_oracle,
)


def _trajectory(
    *,
    seed: int = 42,
    qualities=(0.70, 0.72, 0.75, 0.755, 0.756, 0.754, 0.753, 0.752),
    validations=(0.70, 0.72, 0.75, 0.749, 0.748, 0.747, 0.746, 0.745),
    v2_stop_round: int | None = 4,
    range_stop_round: int | None = None,
) -> Trajectory:
    rounds = []
    for index, (quality, validation) in enumerate(zip(qualities, validations), start=1):
        decisions = {}
        decisions["v2_hoeffding"] = DecisionEvidence(
            action="stop" if index == v2_stop_round else "continue",
            reason=(
                "gain_cost_net_ucb_non_positive"
                if index == v2_stop_round
                else ("maximum_budget_reached" if index == 8 else "positive_net_ucb")
            ),
            confidence_radius=0.5934427943770807,
            certified_gain_cost_stop=index == v2_stop_round,
        )
        decisions["v2_1_range_aware"] = DecisionEvidence(
            action="stop" if index == range_stop_round else "continue",
            reason=(
                "gain_cost_net_ucb_non_positive"
                if index == range_stop_round
                else ("maximum_budget_reached" if index == 8 else "positive_net_ucb")
            ),
            confidence_radius=0.29672139718854035,
            certified_gain_cost_stop=index == range_stop_round,
        )
        rounds.append(
            RoundRecord(
                round_index=index,
                budget=index * 32,
                offline_accuracy=quality,
                validation_accuracy=validation,
                cost=CostSnapshot(
                    total_modeled_usd=float(index * 10),
                    measured={
                        "teacher_tokens_cumulative": float(index * 1000),
                        "gpu_seconds_cumulative": float(index * 20),
                    },
                    pilot_modeled_usd=float(index * 2),
                    pilot_measured={"pilot_gpu_seconds_cumulative": float(index * 5)},
                ),
                decisions=decisions,
            )
        )
    return Trajectory(seed=seed, rounds=tuple(rounds), t_max=8, budget_max=256)


def _by_policy(results):
    return {item.policy: item for item in results}


def test_oracle_uses_lowest_cost_epsilon_optimal_round() -> None:
    oracle = retrospective_oracle(_trajectory(), epsilon=0.01)
    assert oracle.a_star == pytest.approx(0.756)
    assert oracle.oracle_round == 3
    assert oracle.oracle_cost == pytest.approx(30.0)


def test_good_stop_reports_regret_cost_saving_and_pilot_overhead() -> None:
    trajectory = _trajectory(v2_stop_round=4)
    result = evaluate_policy(
        trajectory,
        freeze_recorded_policy(trajectory, "v2_hoeffding"),
        epsilon=0.01,
    )
    assert result.classification == "good_stop"
    assert result.quality_regret == pytest.approx(0.001)
    assert result.cost_saving_ratio == pytest.approx(0.5)
    assert result.avoidable_modeled_usd == pytest.approx(40.0)
    assert result.pilot_overhead_modeled_usd == pytest.approx(8.0)
    assert result.measured_saving_ratio["teacher_tokens_cumulative"] == pytest.approx(0.5)


def test_premature_stop_when_regret_exceeds_epsilon() -> None:
    trajectory = _trajectory(v2_stop_round=2)
    result = evaluate_policy(
        trajectory,
        freeze_recorded_policy(trajectory, "v2_hoeffding"),
        epsilon=0.01,
    )
    assert result.classification == "premature_stop"
    assert result.quality_regret > 0.01


def test_no_stop_is_unnecessary_continue_when_oracle_is_earlier() -> None:
    result = _by_policy(benchmark_trajectory(_trajectory(range_stop_round=None)))[
        "v2_1_range_aware"
    ]
    assert result.certified_online_stop is False
    assert result.classification == "unnecessary_continue"
    assert result.decision_reason == "budget_exhausted_without_certified_gain_cost_stop"


def test_budget_exhaustion_is_not_certified_stop() -> None:
    qualities = (0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.74, 0.76)
    trajectory = _trajectory(
        qualities=qualities,
        v2_stop_round=None,
        range_stop_round=None,
    )
    result = _by_policy(benchmark_trajectory(trajectory))["v2_hoeffding"]
    assert result.certified_online_stop is False
    assert result.classification == "budget_exhausted"
    assert result.oracle_round == 8


def test_validation_patience_uses_only_observed_prefix_and_restores_best_round() -> None:
    first = _trajectory(
        validations=(0.70, 0.72, 0.75, 0.749, 0.99, 0.99, 0.99, 0.99)
    )
    second = _trajectory(
        validations=(0.70, 0.72, 0.75, 0.749, 0.10, 0.10, 0.10, 0.10)
    )
    first_stop = freeze_validation_patience(first, patience_rounds=1, min_delta=0.001)
    second_stop = freeze_validation_patience(second, patience_rounds=1, min_delta=0.001)
    assert first_stop == second_stop
    assert first_stop.decision_round == 4
    assert first_stop.deployed_round == 3


def test_validation_patience_cost_is_charged_through_decision_round() -> None:
    trajectory = _trajectory()
    stop = freeze_validation_patience(trajectory, patience_rounds=1, min_delta=0.001)
    result = evaluate_policy(trajectory, stop, epsilon=0.01)
    assert result.deployed_round == 3
    assert result.decision_round == 4
    assert result.a_stop == pytest.approx(0.75)
    assert result.stopped_modeled_usd == pytest.approx(40.0)


def test_missing_per_round_offline_quality_is_not_replaced_by_validation() -> None:
    trajectory = _trajectory()
    rounds = list(trajectory.rounds)
    rounds[2] = replace(rounds[2], offline_accuracy=None, validation_accuracy=0.99)
    incomplete = replace(trajectory, rounds=tuple(rounds))
    result = evaluate_policy(
        incomplete,
        freeze_recorded_policy(incomplete, "v2_hoeffding"),
        epsilon=0.01,
    )
    assert result.status == "incomplete_evidence"
    assert result.classification is None
    assert "round_3.offline_accuracy" in result.missing_evidence


def test_missing_cumulative_total_cost_prevents_dimension_mixing() -> None:
    trajectory = _trajectory()
    rounds = list(trajectory.rounds)
    rounds[4] = replace(
        rounds[4],
        cost=CostSnapshot(
            total_modeled_usd=None,
            measured={"teacher_tokens_cumulative": 5000.0},
        ),
    )
    incomplete = replace(trajectory, rounds=tuple(rounds))
    result = evaluate_policy(
        incomplete,
        freeze_recorded_policy(incomplete, "v2_hoeffding"),
        epsilon=0.01,
    )
    assert result.status == "incomplete_evidence"
    assert "round_5.cost.total_modeled_usd" in result.missing_evidence


def test_range_aware_radius_and_reason_are_reported_without_redecision() -> None:
    trajectory = _trajectory(range_stop_round=5)
    stop = freeze_recorded_policy(trajectory, "v2_1_range_aware")
    assert stop.decision_round == 5
    assert stop.confidence_radius == pytest.approx(0.29672139718854035)
    assert stop.reason == "gain_cost_net_ucb_non_positive"


def test_retrospective_oracle_is_explicitly_not_certified_online_stop() -> None:
    result = _by_policy(benchmark_trajectory(_trajectory()))["retrospective_oracle"]
    assert result.certified_online_stop is False
    assert result.classification == "oracle_reference"
    assert result.decision_reason == "offline_evaluation_oracle_only"


def test_artifact_adapter_infers_round_from_directory_without_guessing_metric_type() -> None:
    evidence = Evidence(seed=42, t_max=8, budget_max=256)
    payload = {
        "hans_accuracy": 0.75,
        "stopping_accuracy": 0.71,
        "cumulative_total_pipeline_cost": 12.5,
        "cost_aware_decision": {
            "action": "continue",
            "reason": "positive_mixed_policy_optimistic_net_gain",
            "confidence_radius": 0.5934427943770807,
        },
    }
    _ingest_payload(
        evidence,
        payload,
        "outputs/example/refresh/round_003/result.json",
    )
    round_three = evidence.rounds[3]
    assert round_three["offline_accuracy"] == pytest.approx(0.75)
    assert round_three["validation_accuracy"] == pytest.approx(0.71)
    assert round_three["total_modeled_usd"] == pytest.approx(12.5)
    assert round_three["decisions"]["v2_hoeffding"]["action"] == "continue"


def test_adapter_never_uses_validation_scoped_hans_as_offline_quality() -> None:
    evidence = Evidence(seed=42, t_max=8, budget_max=256)
    _ingest_payload(
        evidence,
        {"validation": {"round": 3, "hans_accuracy": 0.99}},
        "outputs/example/result.json",
    )
    assert evidence.rounds[3]["offline_accuracy"] is None


def test_range_replay_adapter_keeps_old_and_new_decisions_separate() -> None:
    evidence = Evidence(seed=43, t_max=8, budget_max=256)
    payload = {
        "rounds": [
            {
                "round": 3,
                "old_action": "continue",
                "old_reason": "positive_mixed_policy_optimistic_net_gain",
                "old_radius": 0.5934427943770807,
                "new_action": "stop",
                "new_reason": "gain_cost_net_ucb_non_positive",
                "new_radius": 0.29672139718854035,
            }
        ]
    }
    _ingest_range_replay(evidence, payload, "comparison.json")
    decisions = evidence.rounds[3]["decisions"]
    assert decisions["v2_hoeffding"]["action"] == "continue"
    assert decisions["v2_1_range_aware"]["action"] == "stop"
    assert decisions["v2_1_range_aware"]["certified_gain_cost_stop"] is True


def test_current_nested_range_replay_schema_is_ingested() -> None:
    evidence = Evidence(seed=42, t_max=8, budget_max=256)
    payload = {
        "rounds": [
            {
                "round": 2,
                "old": {
                    "action": "continue",
                    "reason": "positive_mixed_policy_optimistic_net_gain",
                    "radius": 0.5934427943770807,
                },
                "range_aware": {
                    "action": "stop",
                    "reason": "gain_cost_net_ucb_non_positive",
                    "radius": 0.29672139718854035,
                },
            }
        ]
    }
    _ingest_range_replay(evidence, payload, "comparison.json")
    decisions = evidence.rounds[2]["decisions"]
    assert decisions["v2_hoeffding"]["confidence_radius"] == pytest.approx(
        0.5934427943770807
    )
    assert decisions["v2_1_range_aware"]["action"] == "stop"


def test_result_schema_keeps_learning_curve_as_validation_and_only_final_hans_as_offline() -> None:
    evidence = Evidence(seed=43, t_max=8, budget_max=256)
    payload = {
        "learning_curve": [
            {"round": 1, "metrics": {"overall_accuracy": 0.65}},
            {"round": 8, "metrics": {"overall_accuracy": 0.75}},
        ],
        "hans_prediction_rules": {
            "three_way_argmax_to_binary": {"overall_accuracy": 0.7674}
        },
    }
    _ingest_payload(evidence, payload, "outputs/example/result.json")
    assert evidence.rounds[1]["validation_accuracy"] == pytest.approx(0.65)
    assert evidence.rounds[1]["offline_accuracy"] is None
    assert evidence.rounds[8]["validation_accuracy"] == pytest.approx(0.75)
    assert evidence.rounds[8]["offline_accuracy"] == pytest.approx(0.7674)


def test_historical_summary_keeps_partial_cost_explicit() -> None:
    summary = _historical_result_summary(
        {
            "learning_curve": [
                {"round": 1, "metrics": {"overall_accuracy": 0.65}}
            ],
            "hans_prediction_rules": {
                "three_way_argmax_to_binary": {"overall_accuracy": 0.75},
                "direct_binary_probability_aggregation": {
                    "overall_accuracy": 0.76
                },
            },
            "total_pipeline_cost_rounds_2_plus": 12.5,
            "cost_aware_rounds": [
                {
                    "round": 2,
                    "decision_cost": {"total_pipeline_cost": 3.0},
                }
            ],
            "resource_usage": {
                "scope": "incremental_replay_compute_only",
                "cost_aware_pilot_wall_seconds": 3600.0,
                "teacher": {"input_tokens": 0, "output_tokens": 0},
            },
            "reused_source_resource_usage_full_8_round_historical": {
                "teacher": {"input_tokens": 100, "output_tokens": 20}
            },
        }
    )
    assert summary["rounds_2_plus_pipeline_usd"] == pytest.approx(12.5)
    assert summary["teacher_tokens_observed"] == pytest.approx(120.0)
    assert summary["pilot_wall_seconds_observed"] == pytest.approx(3600.0)
    assert summary["per_round_pipeline_usd"] == {"2": 3.0}


def test_missing_recorded_decision_rounds_are_incomplete_not_no_stop() -> None:
    trajectory = _trajectory(range_stop_round=None)
    rounds = list(trajectory.rounds)
    decisions = dict(rounds[4].decisions)
    decisions.pop("v2_1_range_aware")
    rounds[4] = replace(rounds[4], decisions=decisions)
    incomplete = replace(trajectory, rounds=tuple(rounds))
    stop = freeze_recorded_policy(incomplete, "v2_1_range_aware")
    result = evaluate_policy(incomplete, stop, epsilon=0.01)
    assert result.status == "incomplete_evidence"
    assert "missing_recorded_policy_decisions:5" in result.missing_evidence


def test_non_monotone_cumulative_total_cost_is_rejected() -> None:
    trajectory = _trajectory()
    rounds = list(trajectory.rounds)
    rounds[4] = replace(
        rounds[4],
        cost=replace(rounds[4].cost, total_modeled_usd=5.0),
    )
    with pytest.raises(ValueError, match="cumulative total modeled cost"):
        replace(trajectory, rounds=tuple(rounds))
