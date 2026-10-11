from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from src.decision.cost_aware import (
    build_decision_candidates,
    compute_net_value,
    compute_optimistic_net_value,
    select_intervention,
)
from src.decision.cost_model import CostRates, CostUsage, compute_cost
from src.decision.types import DecisionCandidate, InterventionDecision
from src.hypotheses.oracle_hans import HANS_ORACLE_HYPOTHESES
from src.intervention.full import run_full_intervention


def candidate(hypothesis_id, optimistic, estimated=0.1, lambda_cost=0.01):
    return DecisionCandidate(
        hypothesis_id=hypothesis_id,
        delta_hat=0.2,
        confidence_radius=0.3,
        cost=10.0,
        lambda_cost=lambda_cost,
        estimated_net_value=estimated,
        optimistic_net_value=optimistic,
    )


def continue_decision(hypothesis_id="h_lexical_overlap", cost=3.0):
    return InterventionDecision(
        action="continue",
        selected_hypothesis_id=hypothesis_id,
        max_optimistic_value=0.4,
        delta_hat=0.2,
        confidence_radius=0.3,
        cost=cost,
        lambda_cost=0.01,
        estimated_net_value=0.17,
        optimistic_net_value=0.47,
        reason="positive_optimistic_net_value",
    )


def role_records(role, heuristic, count):
    return [
        {
            "id": f"{role}-{heuristic}-{index}",
            "premise": f"premise {heuristic} {index}",
            "hypothesis": f"hypothesis {heuristic} {index}",
            "gold": "non-entailment",
            "heuristic": heuristic,
            "subcase": "toy",
            "prediction": "entailment",
            "correct": False,
            "split_role": role,
        }
        for index in range(count)
    ]


def gain_estimate(hypothesis_id, status="estimated", m=4):
    return {
        "hypothesis_id": hypothesis_id,
        "m": m,
        "delta_hat": 0.2 if m else None,
        "radius": 0.3 if m else None,
        "status": status,
    }


def test_estimated_net_value_equation():
    assert compute_net_value(0.4, 100, 0.002) == pytest.approx(0.2)


def test_optimistic_net_value_equation():
    assert compute_optimistic_net_value(0.4, 0.3, 100, 0.002) == pytest.approx(0.5)


def test_full_cost_model_accounts_for_every_configurable_component():
    usage = CostUsage(
        llm_input_tokens=1000,
        llm_output_tokens=2000,
        generated_items=4,
        verified_items=5,
        lora_training_runs=1,
        lora_training_gpu_seconds=3600,
        evaluation_runs=2,
        evaluation_items=10,
        evaluation_gpu_seconds=1800,
    )
    rates = CostRates(
        llm_input_per_1k_tokens=3,
        llm_output_per_1k_tokens=4,
        data_generation_per_item=5,
        data_verification_per_item=6,
        lora_training_per_run=7,
        lora_training_per_gpu_hour=8,
        evaluation_per_run=9,
        evaluation_per_item=10,
        evaluation_per_gpu_hour=12,
    )
    result = compute_cost(usage, rates, currency="USD")
    assert result.components == pytest.approx(
        {
            "llm_inference_input": 3,
            "llm_inference_output": 8,
            "data_generation": 20,
            "data_verification": 30,
            "lora_training": 15,
            "evaluation": 124,
        }
    )
    assert result.total == pytest.approx(200)


def test_composite_total_is_the_cost_used_in_utility():
    total = compute_cost(
        CostUsage(llm_input_tokens=1000),
        CostRates(llm_input_per_1k_tokens=2.0),
    ).total
    assert compute_optimistic_net_value(0.4, 0.1, total, 0.25) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("factory", "value"),
    [
        (CostRates.from_mapping, {"human_annotation_per_item": 1.0}),
        (CostRates.from_mapping, {"human_review_per_item": 1.0}),
        (CostUsage.from_mapping, {"human_annotation_items": 1.0}),
        (CostUsage.from_mapping, {"human_review_items": 1.0}),
    ],
)
def test_cost_model_rejects_removed_human_cost_fields(factory, value):
    with pytest.raises(ValueError, match="Unknown cost"):
        factory(value)


def test_cost_model_rejects_negative_resource_usage():
    with pytest.raises(ValueError, match="non-negative"):
        compute_cost(CostUsage(llm_input_tokens=-1), CostRates())


def test_highest_optimistic_value_is_selected():
    decision = select_intervention(
        [candidate("h_b", 0.2), candidate("h_a", 0.7)], 0.01
    )
    assert decision.action == "continue"
    assert decision.selected_hypothesis_id == "h_a"
    assert decision.max_optimistic_value == pytest.approx(0.7)


@pytest.mark.parametrize("value", [-0.1, 0.0])
def test_nonpositive_max_optimistic_value_stops(value):
    decision = select_intervention([candidate("h", value)], 0.01)
    assert decision.action == "stop"
    assert decision.selected_hypothesis_id is None
    assert decision.reason == "no_positive_optimistic_net_value"


def test_positive_max_optimistic_value_continues():
    assert select_intervention([candidate("h", 1e-12)], 0.01).action == "continue"


def test_rejected_hypotheses_are_ignored():
    candidates = build_decision_candidates(
        [
            {"hypothesis_id": "accepted", "certified": True},
            {"hypothesis_id": "rejected", "certified": False},
        ],
        [gain_estimate("accepted"), gain_estimate("rejected")],
        {"accepted": 10, "rejected": 10},
        0.01,
    )
    assert [item.hypothesis_id for item in candidates] == ["accepted"]


@pytest.mark.parametrize(
    "estimate",
    [gain_estimate("h", status="insufficient_pilots"), gain_estimate("h", m=0)],
)
def test_hypotheses_without_valid_gain_estimates_are_ignored(estimate):
    candidates = build_decision_candidates(
        [{"hypothesis_id": "h", "certified": True}],
        [estimate],
        {"h": 10},
        0.01,
    )
    assert candidates == []


def test_empty_candidates_stop_with_distinct_reason():
    decision = select_intervention([], 0.01)
    assert decision.action == "stop"
    assert decision.max_optimistic_value is None
    assert decision.reason == "no_eligible_certified_hypotheses"


def test_budget_filters_candidates_before_selection():
    decision = select_intervention(
        [candidate("too-expensive", 0.8), candidate("affordable", 0.3)],
        0.01,
        max_cost=10.0,
    )
    assert decision.selected_hypothesis_id == "too-expensive"
    assert decision.cost == 10.0

    stopped = select_intervention([candidate("h", 0.8)], 0.01, max_cost=9.0)
    assert stopped.action == "stop"
    assert stopped.reason == "budget_exhausted"


def test_changing_lambda_changes_optimistic_value_exactly():
    low = compute_optimistic_net_value(0.2, 0.3, 100, 0.001)
    high = compute_optimistic_net_value(0.2, 0.3, 100, 0.004)
    assert low - high == pytest.approx((0.004 - 0.001) * 100)


def test_candidate_table_is_sorted_by_optimistic_value():
    candidates = build_decision_candidates(
        [
            {"hypothesis_id": "h1", "certified": True},
            {"hypothesis_id": "h2", "certified": True},
        ],
        [
            {**gain_estimate("h1"), "delta_hat": 0.1},
            {**gain_estimate("h2"), "delta_hat": 0.4},
        ],
        {"h1": 10, "h2": 10},
        0.01,
    )
    assert [item.hypothesis_id for item in candidates] == ["h2", "h1"]
    assert select_intervention(candidates, 0.01).selected_hypothesis_id == "h2"


def test_one_full_intervention_uses_original_checkpoint_and_selected_heuristic(tmp_path):
    selected = HANS_ORACLE_HYPOTHESES["lexical_overlap"]
    decision = continue_decision(cost=3.0)
    starting_checkpoint = {
        "name_or_path": "current-student",
        "adapter_path": "current-adapter",
    }
    untouched_checkpoint = deepcopy(starting_checkpoint)
    starts = []
    train_calls = []

    def factory(seed):
        student = {
            "source": starting_checkpoint["name_or_path"],
            "adapter": starting_checkpoint["adapter_path"],
            "quality": 0.2,
        }
        starts.append((student["source"], student["adapter"], seed))
        return student

    def train(student, examples, adapter_dir: Path, seed):
        train_calls.append(
            {
                "ids": [example.id for example in examples],
                "labels": [example.label for example in examples],
                "adapter_dir": str(adapter_dir),
                "seed": seed,
            }
        )
        student["quality"] = 0.7
        return 0.25

    def risk(student, records):
        assert all(record["split_role"] == "gain_eval" for record in records)
        assert all(record["heuristic"] == "lexical_overlap" for record in records)
        return 1.0 - student["quality"]

    intervention = role_records("intervention_pool", "lexical_overlap", 5)
    intervention += role_records("intervention_pool", "subsequence", 5)
    gain_eval = role_records("gain_eval", "lexical_overlap", 4)
    result = run_full_intervention(
        decision=decision,
        hypothesis=selected,
        intervention_records=intervention,
        gain_eval_records=gain_eval,
        full_intervention_size=3,
        seed=42,
        non_entailment_label="contradiction",
        M=1.0,
        tolerance=1e-12,
        starting_checkpoint=starting_checkpoint,
        training_config={"learning_rate": 1e-5},
        output_dir=tmp_path,
        student_factory=factory,
        train_callback=train,
        risk_callback=risk,
    )

    assert len(starts) == 1
    assert starts == [("current-student", "current-adapter", 42)]
    assert len(train_calls) == 1
    assert len(train_calls[0]["ids"]) == 3
    assert all("lexical_overlap" in item for item in train_calls[0]["ids"])
    assert "pilot" not in train_calls[0]["adapter_dir"]
    assert train_calls[0]["adapter_dir"].endswith("h_lexical_overlap/adapter")
    assert starting_checkpoint == untouched_checkpoint
    assert result.risk_before == pytest.approx(0.8)
    assert result.risk_after == pytest.approx(0.3)
    assert result.observed_full_gain == pytest.approx(0.5)


def test_post_intervention_result_cannot_mutate_frozen_decision(tmp_path):
    decision = continue_decision(cost=1.0)
    frozen = decision.to_dict()
    hypothesis = HANS_ORACLE_HYPOTHESES["lexical_overlap"]

    def factory(seed):
        return {"quality": 0.1}

    def train(student, examples, adapter_dir, seed):
        student["quality"] = 0.9
        return 0.1

    def risk(student, records):
        return 1.0 - student["quality"]

    run_full_intervention(
        decision=decision,
        hypothesis=hypothesis,
        intervention_records=role_records("intervention_pool", "lexical_overlap", 1),
        gain_eval_records=role_records("gain_eval", "lexical_overlap", 1),
        full_intervention_size=1,
        seed=1,
        non_entailment_label="contradiction",
        M=1.0,
        tolerance=1e-12,
        starting_checkpoint={"name_or_path": "current"},
        training_config={},
        output_dir=tmp_path,
        student_factory=factory,
        train_callback=train,
        risk_callback=risk,
    )
    assert decision.to_dict() == frozen


def test_stop_decision_cannot_start_training(tmp_path):
    calls = []
    stop = select_intervention([], 0.01)

    def factory(seed):
        calls.append("factory")
        return {}

    with pytest.raises(ValueError, match="continue decision"):
        run_full_intervention(
            decision=stop,
            hypothesis=HANS_ORACLE_HYPOTHESES["lexical_overlap"],
            intervention_records=[],
            gain_eval_records=[],
            full_intervention_size=1,
            seed=1,
            non_entailment_label="contradiction",
            M=1.0,
            tolerance=1e-12,
            starting_checkpoint={"name_or_path": "current"},
            training_config={},
            output_dir=tmp_path,
            student_factory=factory,
            train_callback=lambda *args: calls.append("train"),
            risk_callback=lambda *args: calls.append("risk"),
        )
    assert calls == []


def test_full_intervention_api_has_no_final_test_input():
    assert "final_test" not in inspect.signature(run_full_intervention).parameters
