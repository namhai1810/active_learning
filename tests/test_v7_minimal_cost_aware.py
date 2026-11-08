from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_stage2_closed_loop as closed_loop_runner
import scripts.replay_v7_minimal_cost_aware as replay_runner
from scripts.audit_v7_minimal_cost_aware import audit_protocol
from scripts.report_v7_minimal_cost_aware import build_report
from src.data.hans import HANSExample
from src.decision.v7_minimal_cost_aware import (
    MIXED_POLICY_ID,
    V7MinimalCostAwareConfig,
    decide_mixed_policy_round,
    predeclare_pilot_eval_plan,
)
from src.gain.estimation import estimate_gain, gain_confidence_radius
from src.models.lora_training import EarlyStoppingConfig, LoraTrainingConfig
from src.synthetic.types import SyntheticNLIExample
from src.utils.config import load_yaml_with_extends
from src.utils.io import write_json, write_jsonl


def _estimate(count=32, gain=0.0):
    return estimate_gain(
        MIXED_POLICY_ID,
        [gain] * count,
        M=1.0,
        delta=0.05,
        K=1,
        T=7,
    )


def test_legacy_v7_config_remains_cost_aware_disabled():
    legacy = load_yaml_with_extends(
        Path("configs/stage2_v7_early_stopping_verification_on.yaml")
    )
    policy = V7MinimalCostAwareConfig.from_mapping(legacy.get("cost_aware"))
    assert policy.enabled is False
    assert legacy["closed_loop"]["total_budget"] == 128
    assert legacy["closed_loop"]["max_rounds"] == 4
    assert legacy["closed_loop"]["per_round_examples"] == 32
    assert legacy["closed_loop"]["selection"] == {
        "nll_weight": 0.65,
        "entropy_weight": 0.35,
        "novelty_weight": 0.5,
        "hypothesis_bonus_weight": 0.35,
        "new_hypothesis_bonus_weight": 0.5,
        "embedding_batch_size": 8,
    }


def test_static_protocol_audit_is_ready_and_calibrates_without_shrinking_radius():
    audit = audit_protocol()
    assert audit["status"] == "ready"
    assert all(audit["checks"].values())
    assert audit["round_1_cost_decision"] is False
    assert audit["gain_ucb"]["radius"] == pytest.approx(0.5934427943770807)
    assert audit["gain_ucb"]["raw_radius_unchanged"] is True
    assert audit["gain_ucb"]["maximum_ucb_break_even_cost"] == pytest.approx(5.0)
    assert audit["gain_ucb"]["support_intersection_preserves_coverage"] is True
    assert "not claimed" in audit["theory_claim"]


def test_minimal_config_preserves_v7_training_and_extends_only_budget_horizon():
    legacy = load_yaml_with_extends(
        Path("configs/stage2_v7_early_stopping_verification_on.yaml")
    )
    current = load_yaml_with_extends(
        Path("configs/stage2_v7_minimal_cost_aware.yaml")
    )
    assert current["training"] == legacy["training"]
    assert current["closed_loop"]["selection"] == legacy["closed_loop"]["selection"]
    assert current["closed_loop"]["initial_pool_dir"] == legacy["closed_loop"]["initial_pool_dir"]
    assert current["closed_loop"]["max_rounds"] == 8
    assert current["closed_loop"]["total_budget"] == 256
    assert current["verification"]["adaptive_verification"] == legacy["verification"]["adaptive_verification"]
    assert current["verification"]["max_rounds"] == 9
    policy = V7MinimalCostAwareConfig.from_mapping(current["cost_aware"])
    assert policy.lambda_cost == pytest.approx(0.2)
    assert policy.calibration_reference_gain == pytest.approx(1.0)
    assert policy.calibration_reference_cost == pytest.approx(5.0)
    assert policy.calibration_currency == "USD"
    assert policy.required_pilots == 32
    assert policy.shared_eval_shard_within_round is True


def test_pilot_plan_is_predeclared_disjoint_and_leaves_held_out_report():
    ids = [f"g-{index}" for index in range(2850)]
    plan = predeclare_pilot_eval_plan(
        ids,
        decision_rounds=range(2, 9),
        pilots_per_round=32,
        eval_size=64,
        seed=42,
        shared_eval_shard_within_round=True,
    )
    flattened = [
        item
        for shards in plan["round_shards"].values()
        for shard in shards
        for item in shard
    ]
    assert len(flattened) == 7 * 64
    assert len(flattened) == len(set(flattened))
    assert set(flattened).isdisjoint(plan["held_out_gain_report_ids"])
    assert len(plan["held_out_gain_report_ids"]) == 2402
    assert plan["pilots_per_round"] == 32
    assert all(len(shards) == 1 for shards in plan["round_shards"].values())
    assert plan["shared_eval_examples_are_not_counted_as_independent_observations"]
    assert plan == predeclare_pilot_eval_plan(
        ids,
        decision_rounds=range(2, 9),
        pilots_per_round=32,
        eval_size=64,
        seed=42,
        shared_eval_shard_within_round=True,
    )


def test_unresolved_is_retained_but_does_not_block_complete_policy_pilots():
    decision = decide_mixed_policy_round(
        states={
            "certified": {"state": "certified"},
            "pending": {"state": "unresolved"},
        },
        estimate=_estimate(),
        required_pilots=32,
        cost=1_000_000.0,
        lambda_cost=0.2,
    )
    assert decision.action == "stop"
    assert decision.reason == "nonpositive_mixed_policy_optimistic_net_gain"
    assert decision.unresolved_hypotheses == ("pending",)
    assert decision.gain_cost_stop_evaluated is True


def test_incomplete_pilots_cannot_trigger_gain_cost_stop():
    decision = decide_mixed_policy_round(
        states={"certified": {"state": "certified"}},
        estimate=_estimate(count=31),
        required_pilots=32,
        cost=1_000_000.0,
        lambda_cost=0.2,
    )
    assert decision.action == "continue"
    assert decision.reason == "insufficient_pilot_evidence"
    assert decision.gain_cost_stop_evaluated is False


def test_complete_policy_ucb_stops_only_when_optimistic_net_is_nonpositive():
    states = {"certified": {"state": "certified"}}
    break_even = 0.5934427943770807 / 0.2
    continuing = decide_mixed_policy_round(
        states=states,
        estimate=_estimate(),
        required_pilots=32,
        cost=break_even - 0.01,
        lambda_cost=0.2,
    )
    stopping = decide_mixed_policy_round(
        states=states,
        estimate=_estimate(),
        required_pilots=32,
        cost=break_even,
        lambda_cost=0.2,
    )
    assert continuing.action == "continue"
    assert continuing.optimistic_net_value > 0
    assert stopping.action == "stop"
    assert stopping.optimistic_net_value <= 0
    assert stopping.reason == "nonpositive_mixed_policy_optimistic_net_gain"
    assert stopping.appendix_theorem_claimed is False
    assert continuing.raw_gain_ucb == pytest.approx(0.5934427943770807)
    assert continuing.bounded_gain_ucb == pytest.approx(0.5934427943770807)
    assert continuing.raw_optimistic_net_value == pytest.approx(
        continuing.optimistic_net_value
    )
    assert stopping.cost_penalty == pytest.approx(0.5934427943770807)


def test_predeclared_ucb_is_wide_and_not_silently_retuned():
    radius = gain_confidence_radius(m=32, M=1.0, K=1, T=7, delta=0.05)
    assert radius == pytest.approx(0.5934427943770807)
    assert min(1.0, radius) / 0.2 == pytest.approx(2.9672139718854034)


def test_enabled_config_rejects_lambda_without_matching_ex_ante_calibration():
    with pytest.raises(ValueError, match="lambda_cost must equal"):
        V7MinimalCostAwareConfig.from_mapping(
            {
                "enabled": True,
                "T": 7,
                "M": 1.0,
                "lambda_cost": 0.001,
                "calibration_reference_gain": 1.0,
                "calibration_reference_cost": 5.0,
                "calibration_currency": "USD",
                "cost": {"currency": "USD"},
            }
        )


def test_completed_pilot_observation_resumes_without_loading_model(
    tmp_path, monkeypatch
):
    class FakeStudent:
        trained = False

        def predict_proba(self, premises, hypotheses, batch_size):
            row = [0.8, 0.1, 0.1] if self.trained else [0.4, 0.3, 0.3]
            return [row[:] for _ in premises]

    def fake_train(
        student,
        examples,
        training,
        early_stopping,
        validation_callback,
        adapter_dir,
        seed,
        *,
        optimizer_state_input,
        optimizer_state_output,
    ):
        Path(adapter_dir).mkdir(parents=True)
        Path(optimizer_state_output).write_bytes(b"optimizer")
        student.trained = True
        validation_callback()
        return SimpleNamespace(
            total_loss=0.1,
            best_epoch=1,
            best_validation_loss=0.2,
            epochs_completed=1,
            stopped_early=False,
            optimizer_step_start=0,
            optimizer_step_end=1,
            optimizer_step_last_attempted=1,
            epoch_history=(),
            validation_seconds=0.0,
        )

    monkeypatch.setattr(closed_loop_runner, "_student", lambda *args: FakeStudent())
    monkeypatch.setattr(
        closed_loop_runner, "train_lora_stateful_early_stopping", fake_train
    )
    monkeypatch.setattr(closed_loop_runner, "_clear_cuda_cache", lambda: None)
    training = LoraTrainingConfig.from_mapping(
        {
            "num_epochs": 1,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "max_length": 32,
            "lora_r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
            "gradient_checkpointing": False,
        }
    )
    early = EarlyStoppingConfig(max_epochs=2, patience=1, min_delta=0.0)
    acquired = [
        SyntheticNLIExample(
            id="train-1",
            premise="A person runs.",
            hypothesis="Someone runs.",
            label="entailment",
            hypothesis_id="h1",
            pair_id="pair-1",
            pair_type="counterfactual",
            pair_member="base",
            pair_rationale="rationale",
            seed_example_ids=("seed-1",),
            label_rationale="rationale",
            targeting_rationale="rationale",
            generator_prompt_version="test",
        )
    ]
    eval_shards = [[
        HANSExample(
            "eval-1", "p", "h", "entailment", "lexical_overlap", "toy"
        )
    ]]
    stopping = [
        HANSExample(
            "stop-1", "p", "h", "entailment", "lexical_overlap", "toy"
        )
    ]
    kwargs = dict(
        output=tmp_path,
        round_index=2,
        acquired=acquired,
        eval_shards=eval_shards,
        model={"name_or_path": "fake", "batch_size": 1},
        current_adapter=None,
        current_optimizer_state=None,
        committed_optimizer_step=0,
        training=training,
        early_stopping=early,
        stopping_examples=stopping,
        pilot_seed=42,
        gain_bound_M=1.0,
        preserve_adapters=False,
    )
    first = closed_loop_runner._run_v7_mixed_policy_pilots(
        **kwargs, resume=False
    )
    monkeypatch.setattr(
        closed_loop_runner,
        "_student",
        lambda *args: pytest.fail("resume must not load a model"),
    )
    second = closed_loop_runner._run_v7_mixed_policy_pilots(
        **kwargs, resume=True
    )
    assert first == second
    loads = {"count": 0}

    def load_only_missing(*args):
        loads["count"] += 1
        return FakeStudent()

    monkeypatch.setattr(closed_loop_runner, "_student", load_only_missing)
    partial_resume = closed_loop_runner._run_v7_mixed_policy_pilots(
        **{**kwargs, "eval_shards": eval_shards * 2}, resume=True
    )
    assert len(partial_resume) == 2
    assert partial_resume[0] == first[0]
    assert loads["count"] == 1
    assert first[0]["gain"] == pytest.approx(1.0)
    assert first[0]["pilot_training_gpu_seconds"] >= 0.0
    assert first[0]["pilot_evaluation_gpu_seconds"] >= 0.0
    altered = SyntheticNLIExample.from_mapping(
        {**acquired[0].to_dict(), "id": "train-2"}
    )
    with pytest.raises(ValueError, match="Cached pilot metadata differs"):
        closed_loop_runner._run_v7_mixed_policy_pilots(
            **{**kwargs, "acquired": [altered]}, resume=True
        )


def test_replay_source_validation_hashes_ids_and_optimizer_continuity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".complete").write_text("complete\n")
    config = {section: {"stable": section} for section in replay_runner.CORE_CONFIG_SECTIONS}
    initial_pool = tmp_path / "initial_pool"
    for relative in replay_runner.INITIAL_REPLAY_FILES:
        path = initial_pool / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"initial:{relative}\n")
    config.update(
        {
            "closed_loop": {
                "initial_pool_dir": str(initial_pool),
                "selection": {"stable": True},
                "per_round_examples": 32,
            },
            "verification": {"stable": True},
        }
    )
    selected = [{"id": f"selected-{index}"} for index in range(256)]
    rounds = []
    previous_adapter = None
    previous_step = 0
    for round_index in range(1, 9):
        adapter = source / "training" / f"round_{round_index:03d}" / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(
            f"adapter-{round_index}".encode()
        )
        optimizer = adapter.parent / "optimizer_state.pt"
        optimizer.write_bytes(f"optimizer-{round_index}".encode())
        rounds.append(
            {
                "round": round_index,
                "starting_adapter": previous_adapter,
                "ending_adapter": str(adapter),
                "optimizer_continuity": {
                    "starting_step": previous_step,
                    "ending_step": round_index * 4,
                    "ending_state": str(optimizer),
                },
                "cost_aware": {
                    "selected_action_example_ids": [
                        item["id"]
                        for item in selected[
                            (round_index - 1) * 32 : round_index * 32
                        ]
                    ]
                },
            }
        )
        previous_adapter = str(adapter)
        previous_step = round_index * 4
        if round_index >= 2:
            refresh = source / "refresh" / f"round_{round_index:03d}"
            for relative in replay_runner.REFRESH_REPLAY_FILES:
                path = refresh / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"round={round_index};artifact={relative}\n")
    write_json(source / "result.json", {"seed": 42, "method": "ours_v7_minimal_cost_aware"})
    write_jsonl(source / "rounds.jsonl", rounds)
    write_jsonl(source / "selected_training_examples.jsonl", selected)
    validated_rounds, validated_selected, payload = replay_runner._validate_source(
        source=source,
        config=config,
        source_config=config,
    )
    assert len(validated_rounds) == 8
    assert len(validated_selected) == 256
    assert len(payload["manifest"]["source_round_checkpoints"]) == 8
    assert len(payload["manifest"]["refresh_artifact_hashes"]) == 7
    assert len(payload["manifest"]["initial_pool_artifact_hashes"]) == 3
    assert payload["manifest"]["teacher_compute_reused"] is True

    bad_rounds = list(rounds)
    bad_rounds[1] = {
        **bad_rounds[1],
        "cost_aware": {"selected_action_example_ids": ["wrong"] * 32},
    }
    write_jsonl(source / "rounds.jsonl", bad_rounds)
    with pytest.raises(ValueError, match="Selected batch ID mismatch"):
        replay_runner._validate_source(
            source=source,
            config=config,
            source_config=config,
        )


def test_cost_accounting_separates_pilot_training_and_evaluation(tmp_path):
    early = EarlyStoppingConfig(max_epochs=8, patience=2, min_delta=0.0)
    payload = closed_loop_runner._mixed_policy_round_cost(
        refresh_dir=tmp_path,
        teacher_usage={
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "teacher_wall_clock_seconds": 0.0,
            "calls": 0.0,
        },
        observations=[
            {
                "eval_ids": ["a", "b", "c", "d"],
                "epochs_completed": 2,
                "pilot_training_gpu_seconds": 70.0,
                "pilot_evaluation_gpu_seconds": 30.0,
                "gpu_seconds": 100.0,
            }
        ],
        eligible_examples=10,
        acquired_examples=32,
        stopping_examples=900,
        early_stopping=early,
        cost_config={
            "type": "full_pipeline",
            "currency": "USD",
            "rates": {},
            "projections": {
                "lora_training_gpu_seconds_fixed": 0.0,
                "lora_training_gpu_seconds_per_item": 0.2,
                "evaluation_gpu_seconds_fixed": 0.0,
                "evaluation_gpu_seconds_per_item": 0.02,
            },
        },
    )
    incurred = payload["phases"]["incurred_exploration"]["usage"]
    assert incurred["lora_training_gpu_seconds"] == pytest.approx(70.0)
    assert incurred["evaluation_gpu_seconds"] == pytest.approx(30.4)
    assert incurred["evaluation_runs"] == 6
    assert incurred["evaluation_items"] == 1828
    assert payload["incurred_exploration_cost"] == pytest.approx(
        payload["phases"]["incurred_exploration"]["total"]
    )
    assert payload["decision_cost"] == pytest.approx(
        payload["phases"]["projected_main_update"]["total"]
    )
    assert payload["total_pipeline_cost"] is None
    assert payload["incurred_cost_is_excluded_from_gain_ucb_decision"] is True
    assert payload["pilot_training_and_evaluation_gpu_time_are_disjoint"] is True


def test_incurred_pilot_cost_does_not_force_stop_after_pilots(tmp_path):
    current = load_yaml_with_extends(
        Path("configs/stage2_v7_minimal_cost_aware.yaml")
    )
    policy = V7MinimalCostAwareConfig.from_mapping(current["cost_aware"])
    write_jsonl(
        tmp_path / "synthetic" / "generated.jsonl",
        [{"id": f"generated-{index}"} for index in range(160)],
    )
    write_jsonl(
        tmp_path / "verification" / "generated_probes.jsonl",
        [{"id": f"probe-{index}"} for index in range(120)],
    )
    observations = [
        {
            "eval_ids": [f"eval-{pilot}-{index}" for index in range(64)],
            "epochs_completed": 8,
            "pilot_training_gpu_seconds": 400.0,
            "pilot_evaluation_gpu_seconds": 200.0,
            "gpu_seconds": 600.0,
        }
        for pilot in range(32)
    ]
    cost = closed_loop_runner._mixed_policy_round_cost(
        refresh_dir=tmp_path,
        teacher_usage={"input_tokens": 100_000.0, "output_tokens": 50_000.0},
        observations=observations,
        eligible_examples=800,
        acquired_examples=32,
        stopping_examples=900,
        early_stopping=EarlyStoppingConfig(
            max_epochs=8, patience=2, min_delta=0.0
        ),
        cost_config=current["cost_aware"]["cost"],
    )
    assert cost["incurred_exploration_cost"] > policy.calibration_reference_cost
    assert cost["decision_cost"] < policy.calibration_reference_cost
    decision = decide_mixed_policy_round(
        states={"h1": {"state": "certified"}},
        estimate=_estimate(),
        required_pilots=32,
        cost=float(cost["decision_cost"]),
        lambda_cost=policy.lambda_cost,
    )
    finalized = closed_loop_runner._finalize_mixed_policy_round_cost(
        cost,
        action=decision.action,
        actual_main_update_usage=closed_loop_runner.CostUsage(
            lora_training_runs=1,
            lora_training_gpu_seconds=5.0,
            evaluation_runs=4,
            evaluation_items=3600,
            evaluation_gpu_seconds=50.0,
        ),
    )
    assert decision.action == "continue"
    assert decision.gain_cost_stop_evaluated is True
    assert decision.optimistic_net_value > 0.0
    assert finalized["total_pipeline_cost"] == pytest.approx(
        finalized["incurred_exploration_cost"]
        + finalized["actual_main_update_cost"]
    )


def test_stop_reports_incurred_actual_cost_but_not_uncommitted_update(tmp_path):
    cost = closed_loop_runner._mixed_policy_round_cost(
        refresh_dir=tmp_path,
        teacher_usage={"input_tokens": 10_000.0, "output_tokens": 5_000.0},
        observations=[],
        eligible_examples=4,
        acquired_examples=32,
        stopping_examples=900,
        early_stopping=EarlyStoppingConfig(max_epochs=8, patience=2, min_delta=0.0),
        cost_config=load_yaml_with_extends(
            Path("configs/stage2_v7_minimal_cost_aware.yaml")
        )["cost_aware"]["cost"],
    )
    finalized = closed_loop_runner._finalize_mixed_policy_round_cost(
        cost, action="stop"
    )
    assert finalized["projected_main_update_committed"] is False
    assert finalized["total_pipeline_cost"] == pytest.approx(
        finalized["incurred_exploration_cost"]
    )
    assert finalized["projected_total_pipeline_cost_if_continue"] == pytest.approx(
        finalized["incurred_exploration_cost"] + finalized["decision_cost"]
    )


def test_hans_final_reports_three_way_and_direct_binary_from_one_pass():
    class FakeStudent:
        def predict_proba(self, premises, hypotheses, batch_size):
            assert len(premises) == len(hypotheses) == 2
            return [
                [0.40, 0.35, 0.25],
                [0.60, 0.20, 0.20],
            ]

    examples = [
        HANSExample(
            "x1", "p1", "h1", "non-entailment", "lexical_overlap", "s1"
        ),
        HANSExample(
            "x2", "p2", "h2", "entailment", "lexical_overlap", "s2"
        ),
    ]
    records, metrics = closed_loop_runner._evaluate_hans_prediction_rules(
        FakeStudent(), examples, batch_size=2
    )
    assert metrics["three_way_argmax_to_binary"]["overall_accuracy"] == pytest.approx(0.5)
    assert metrics["direct_binary_probability_aggregation"]["overall_accuracy"] == pytest.approx(1.0)
    assert metrics["disagreement"] == {"count": 1, "rate": 0.5}
    assert records[0]["prediction"] == "entailment"
    assert records[0]["direct_binary_prediction"] == "non-entailment"


def test_paper_report_contains_both_hans_rules_cost_budget_and_stop_reason():
    row = {
        "seed": 42,
        "rounds": 2,
        "actual_training_examples": 64,
        "stop_reason_category": "gain_cost",
        "stop_reason": "nonpositive_mixed_policy_optimistic_net_gain",
        "hans_final": {"overall_accuracy": 0.70},
        "hans_prediction_rules": {
            "direct_binary_probability_aggregation": {
                "overall_accuracy": 0.72
            },
            "disagreement": {"count": 3, "rate": 0.0002},
        },
        "mnli": {
            "matched": {"accuracy": 0.60},
            "mismatched": {"accuracy": 0.59},
        },
        "total_modeled_decision_cost_rounds_2_plus": 5.1,
        "total_incurred_exploration_cost_rounds_2_plus": 8.2,
        "total_pipeline_cost_rounds_2_plus": 13.3,
        "resource_usage": {
            "teacher": {"input_tokens": 100.0, "output_tokens": 50.0}
        },
        "cost_aware_rounds": [],
        "cost_aware_stop_record": None,
        "learning_curve": [],
        "optimizer_steps": 8,
    }
    report = build_report([row])
    assert "HANS Three-way" in report
    assert "HANS Direct Binary" in report
    assert "70.0000%" in report and "72.0000%" in report
    assert "| 42 | 2 | 64 |" in report
    assert "nonpositive_mixed_policy_optimistic_net_gain" in report
    assert "5.100000" in report

