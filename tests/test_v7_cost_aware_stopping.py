from pathlib import Path

import pytest

from scripts.run_stage2_closed_loop import _apply_seed_override, _cost_aware_decision
from scripts.run_main_pilot_gain import _binary_validation
from src.data.hans import HANSExample
from src.decision.safe_stopping import (
    cumulative_cost_ledger,
    verification_assumption_audit,
)
from src.evaluation.hans import evaluate_hans_prediction_rules
from src.utils.io import write_json, write_jsonl


def _verification(radius=0.18):
    return {
        "K": 5,
        "T": 8,
        "delta": 0.05,
        "tau_v": 0.5,
        "results": [
            {
                "hypothesis_id": "h1",
                "n": 120,
                "confidence_radius": radius,
                "certified": True,
            },
            {
                "hypothesis_id": "h2",
                "n": 120,
                "confidence_radius": radius,
                "certified": False,
            },
        ],
    }


def test_appendix_verification_audit_keeps_unresolved_but_checks_every_radius():
    audit = verification_assumption_audit(
        _verification(), gamma=0.4, expected_K=5, expected_T=8, expected_delta=0.05
    )
    assert audit["mechanical_checks_pass"] is True
    assert [row["state"] for row in audit["hypotheses"]] == [
        "certified",
        "unresolved_or_refuted",
    ]
    assert audit["guarantee_status"] == (
        "conditional_on_unverifiable_distributional_assumptions"
    )


def test_appendix_verification_audit_rejects_radius_larger_than_gamma_over_two():
    audit = verification_assumption_audit(
        _verification(radius=0.21),
        gamma=0.4,
        expected_K=5,
        expected_T=8,
        expected_delta=0.05,
    )
    assert audit["mechanical_checks_pass"] is False
    assert audit["guarantee_status"] == "appendix_preconditions_not_met"


def test_cost_ledger_never_adds_projected_candidate_cost():
    first = cumulative_cost_ledger(
        0.0, incurred_exploration=2.0, realized_intervention=3.0
    )
    second = cumulative_cost_ledger(
        first["cumulative_after"],
        incurred_exploration=1.0,
        realized_intervention=0.0,
    )
    assert first["cumulative_after"] == pytest.approx(5.0)
    assert second["cumulative_after"] == pytest.approx(6.0)


def test_cost_aware_seed_override_updates_pilot_and_intervention_only_not_split():
    config = {
        "seed": 42,
        "closed_loop": {"seed": 42},
        "hans": {"split_seed": 42},
        "pilot": {"seed": 42, "applicability_seed": 42},
        "intervention": {"seed": 42},
    }
    _apply_seed_override(config, 44)
    assert config["pilot"] == {"seed": 44, "applicability_seed": 44}
    assert config["intervention"]["seed"] == 44
    assert config["hans"]["split_seed"] == 42


def _materialize_resumable_decision(
    root: Path, checkpoint: dict, optimizer: dict | None = None
):
    write_json(
        root / "gain" / "gain_estimates.json",
        {
            "starting_checkpoint": checkpoint,
            "starting_optimizer_state": optimizer
            or {"path": "optimizer-r2.pt", "local_sha256": "abc"},
            "main_training_policy_matched": True,
            "gain_bound": {"K": 5, "T": 8, "delta": 0.05},
            "estimates": [],
        },
    )
    write_jsonl(
        root / "gain" / "pilot_observations.jsonl",
        [
            {
                "hypothesis_id": "h1",
                "train_ids": ["pilot-1", "pilot-2"],
            }
        ],
    )
    write_json(root / "gain" / "gain_eval_applicability.json", {})
    write_json(root / "verification" / "results.json", _verification())
    write_json(
        root / "decision" / "decision.json",
        {"action": "continue", "selected_hypothesis_id": "h1"},
    )
    write_json(root / "decision" / "candidates.json", {"candidates": []})
    write_json(
        root / "decision" / "round_cost.json",
        {"incurred_exploration_cost": 1.0},
    )


def _config():
    return {
        "gain": {"K": 5, "T": 8, "delta": 0.05},
        "verification": {"delta": 0.05},
        "safe_stopping": {
            "verification_margin_gamma": 0.4,
            "require_mechanical_assumptions": True,
        },
    }


def test_resume_reuses_frozen_pilot_and_decision_artifacts(tmp_path, monkeypatch):
    checkpoint = {"name_or_path": "student", "adapter_path": "adapter-r2"}
    _materialize_resumable_decision(tmp_path, checkpoint)
    monkeypatch.setattr(
        "scripts.run_stage2_closed_loop._run_stage",
        lambda *args, **kwargs: pytest.fail("resume must not rerun GPU/teacher stages"),
    )
    decision, excluded, _, cost = _cost_aware_decision(
        round_dir=tmp_path,
        config_path=tmp_path / "config.yaml",
        config=_config(),
        resume=True,
        expected_checkpoint=checkpoint,
        expected_optimizer_state={
            "path": "optimizer-r2.pt",
            "local_sha256": "abc",
        },
    )
    assert decision["action"] == "continue"
    assert excluded == ["pilot-1", "pilot-2"]
    assert cost["theorem_assumption_audit"]["mechanical_checks_pass"] is True


def test_resume_rejects_checkpoint_discontinuity(tmp_path):
    checkpoint = {"name_or_path": "student", "adapter_path": "adapter-r2"}
    _materialize_resumable_decision(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="checkpoint differs"):
        _cost_aware_decision(
            round_dir=tmp_path,
            config_path=tmp_path / "config.yaml",
            config=_config(),
            resume=True,
            expected_checkpoint={
                "name_or_path": "student",
                "adapter_path": "adapter-r1",
            },
            expected_optimizer_state={
                "path": "optimizer-r2.pt",
                "local_sha256": "abc",
            },
        )


def test_resume_rejects_optimizer_discontinuity(tmp_path):
    checkpoint = {"name_or_path": "student", "adapter_path": "adapter-r2"}
    _materialize_resumable_decision(tmp_path, checkpoint)
    with pytest.raises(ValueError, match="optimizer state differs"):
        _cost_aware_decision(
            round_dir=tmp_path,
            config_path=tmp_path / "config.yaml",
            config=_config(),
            resume=True,
            expected_checkpoint=checkpoint,
            expected_optimizer_state={
                "path": "optimizer-r1.pt",
                "local_sha256": "old",
            },
        )


class _ProbabilityStudent:
    def predict_proba(self, premises, hypotheses, batch_size):
        return [[0.40, 0.35, 0.25], [0.60, 0.20, 0.20]]


def test_pilot_gain_risk_uses_bounded_direct_binary_accuracy():
    examples = [
        HANSExample("a", "p", "h", "non-entailment", "lexical_overlap", "x"),
        HANSExample("b", "p", "h", "entailment", "subsequence", "y"),
    ]
    metrics = _binary_validation(
        [[0.40, 0.35, 0.25], [0.60, 0.20, 0.20]], examples
    )
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["sealed_final_test_used"] is False


def test_final_hans_rules_are_reported_from_one_probability_pass():
    examples = [
        HANSExample("a", "p", "h", "non-entailment", "lexical_overlap", "x"),
        HANSExample("b", "p", "h", "entailment", "subsequence", "y"),
    ]
    rows, result = evaluate_hans_prediction_rules(
        _ProbabilityStudent(), examples, batch_size=2
    )
    assert result["three_way_to_binary"]["overall_accuracy"] == pytest.approx(0.5)
    assert result["direct_binary"]["overall_accuracy"] == pytest.approx(1.0)
    assert result["disagreement"]["count"] == 1
    assert rows[0]["probabilities"] == [0.40, 0.35, 0.25]
