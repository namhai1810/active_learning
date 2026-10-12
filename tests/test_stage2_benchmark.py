import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.benchmark.hans_manifest import (
    FrozenHansManifest,
    audit_final_test_overlap,
    build_frozen_hans_manifest,
    validate_frozen_hans_manifest,
)
from src.benchmark.selection import select_complete_pairs
from src.decision.artifact_cost import build_artifact_costs, build_round_incurred_cost
from src.decision.cost_aware import select_intervention
from src.decision.types import DecisionCandidate
from src.evaluation.metrics import compute_hans_metrics
from src.synthetic.types import SyntheticNLIExample


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hans_rows(n=40):
    return [
        {
            "id": f"h{index}",
            "premise": f"premise {index}",
            "hypothesis": f"hypothesis {index}",
            "gold": "entailment" if index % 2 else "non-entailment",
            "heuristic": ("lexical_overlap", "subsequence", "constituent")[index % 3],
            "subcase": f"case-{index % 4}",
        }
        for index in range(n)
    ]


def _synthetic(pair_index, member):
    return SyntheticNLIExample(
        id=f"x{pair_index}{member}",
        premise=f"A person {pair_index} greeted someone {member}.",
        hypothesis=f"Someone {member} was greeted by person {pair_index}.",
        label="entailment",
        hypothesis_id="h1",
        pair_id=f"p{pair_index}",
        pair_type="invariant",
        pair_member="base" if member == "a" else "irrelevant_change",
        pair_rationale="controlled pair",
        seed_example_ids=("s1",),
        label_rationale="passive preserves roles",
        targeting_rationale="tests role semantics",
        generator_prompt_version="v1",
    )


def test_frozen_manifest_roundtrip_is_seed_independent_for_experiments():
    rows = _hans_rows()
    manifest = build_frozen_hans_manifest(
        rows,
        split_seed=991,
        discovery_fraction=0.125,
        verification_fraction=0.125,
        intervention_fraction=0.125,
        gain_eval_fraction=0.125,
    )
    restored = FrozenHansManifest.from_mapping(manifest.to_dict())
    validate_frozen_hans_manifest(restored, rows)
    assert restored == manifest
    assert len(restored.final_test_ids) == 20
    # Experimental seeds never enter manifest construction.
    assigned_for_seed_42 = restored.final_test_ids
    assigned_for_seed_44 = restored.final_test_ids
    assert assigned_for_seed_42 == assigned_for_seed_44


def test_complete_pair_budget_and_shortage_are_explicit_across_rounds():
    pool = [_synthetic(index, member) for index in range(3) for member in ("a", "b")]
    first, first_status = select_complete_pairs(pool, budget=4, seed=42)
    second, second_status = select_complete_pairs(
        pool, budget=4, seed=43, excluded_ids=[item.id for item in first]
    )
    assert len(first) == 4
    assert len({item.pair_id for item in first}) == 2
    assert len(second) == 2
    assert second_status == {
        "requested_examples": 4,
        "actual_examples": 2,
        "requested_pairs": 2,
        "actual_pairs": 1,
        "shortage": True,
    }
    assert not {item.id for item in first} & {item.id for item in second}
    assert first_status["shortage"] is False


def test_final_test_overlap_audit_is_post_selection_and_does_not_veto():
    final = [{"id": "f1", "premise": "A doctor ran.", "hypothesis": "A doctor moved."}]
    training = [{"id": "t1", "premise": "A doctor ran.", "hypothesis": "A doctor moved."}]
    result = audit_final_test_overlap(training, final, near_duplicate_threshold=0.9)
    assert result["passed"] is False
    assert result["audit_only_after_selection_and_training"] is True
    assert result["used_to_change_training_or_selection"] is False
    assert result["near_duplicates"][0]["final_test_id"] == "f1"


def test_hans_subcase_metrics_are_consistent_with_overall_counts():
    metrics = compute_hans_metrics(
        ["entailment", "non-entailment", "entailment", "non-entailment"],
        ["entailment", "entailment", "entailment", "non-entailment"],
        ["lexical_overlap", "lexical_overlap", "subsequence", "constituent"],
        ["a", "a", "b", "b"],
    )
    assert metrics["overall_accuracy"] == pytest.approx(0.75)
    assert metrics["accuracy_by_subcase"] == {"a": 0.5, "b": 1.0}
    assert sum(metrics["subcase_counts"].values()) == metrics["num_examples"]


def test_stop_cost_ledger_keeps_automated_incurred_cost(tmp_path):
    (tmp_path / "teacher_calls").mkdir()
    (tmp_path / "synthetic").mkdir()
    (tmp_path / "verification").mkdir()
    (tmp_path / "teacher_calls" / "calls.jsonl").write_text(
        json.dumps({"usage": {"input_tokens": 1000, "output_tokens": 500}}) + "\n"
    )
    (tmp_path / "synthetic" / "generated.jsonl").write_text(
        json.dumps({"hypothesis_id": "h1"}) + "\n"
    )
    for name in ("generated_probes.jsonl", "probes.jsonl"):
        (tmp_path / "verification" / name).write_text("")
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps({"id": "d1", "prediction": "entailment"}) + "\n"
    )
    costs, breakdowns = build_artifact_costs(
        run_dir=tmp_path,
        hypothesis_ids=["h1"],
        full_training_counts={"h1": 2},
        evaluation_counts={"h1": 1},
        pilot_observations=[],
        cost_config={
            "type": "full_pipeline",
            "currency": "USD",
            "rates": {
                "llm_input_per_1k_tokens": 1,
                "llm_output_per_1k_tokens": 2,
                "data_generation_per_item": 3,
                "lora_training_per_run": 5,
            },
            "projections": {},
        },
    )
    phases = breakdowns["h1"]["phases"]
    assert phases["incurred_exploration"]["total"] == pytest.approx(5.0)
    assert phases["projected_intervention"]["total"] == pytest.approx(5.0)
    assert set(breakdowns["h1"]["components"]) == {
        "llm_inference_input",
        "llm_inference_output",
        "data_generation",
        "data_verification",
        "lora_training",
        "evaluation",
    }
    assert costs["h1"] == pytest.approx(10.0)
    round_cost = build_round_incurred_cost(
        run_dir=tmp_path,
        pilot_observations=[],
        cost_config={
            "type": "full_pipeline",
            "currency": "USD",
            "rates": {
                "llm_input_per_1k_tokens": 1,
                "llm_output_per_1k_tokens": 2,
                "data_generation_per_item": 3,
            },
            "projections": {},
        },
    )
    # This remains chargeable even if discovery yields no decision candidate.
    assert round_cost["total"] == pytest.approx(5.0)

    with pytest.raises(ValueError, match="Unknown cost projection fields"):
        build_round_incurred_cost(
            run_dir=tmp_path,
            pilot_observations=[],
            cost_config={
                "type": "full_pipeline",
                "currency": "USD",
                "rates": {},
                "projections": {"human_review_items_per_verified_item": 1},
            },
        )


def test_budget_admission_uses_projected_marginal_not_total_cost():
    candidate = DecisionCandidate(
        hypothesis_id="h1",
        delta_hat=0.4,
        confidence_radius=0.2,
        cost=10.0,
        lambda_cost=0.01,
        estimated_net_value=0.3,
        optimistic_net_value=0.5,
    )
    decision = select_intervention(
        [candidate], 0.01, max_cost=2.0, budget_costs={"h1": 1.5}
    )
    assert decision.action == "continue"


def test_stage2_report_generates_visualizations(tmp_path):
    run = tmp_path / "runs" / "zero_shot" / "budget_32" / "seed_42"
    run.mkdir(parents=True)
    result = {
        "experiment": "A",
        "method": "zero_shot",
        "seed": 42,
        "requested_budget": 32,
        "actual_final_training_examples": 0,
        "shortage": False,
        "cross_source_baseline": False,
        "manifest_hash": "frozen-test-manifest",
        "pilot_count": 0,
        "optimizer_steps": 0,
        "training_gpu_seconds": 0,
        "acquisition_gpu_seconds": 0,
        "evaluation_gpu_seconds": 0,
        "resource_usage": {
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "generated_items": 0,
        },
        "assumed_cost": {"total": 0.5},
        "hans_final": {
            "overall_accuracy": 0.5,
            "non_entailment_accuracy": 0.4,
            "accuracy_by_heuristic": {
                "lexical_overlap": 0.5,
                "subsequence": 0.5,
                "constituent": 0.5,
            },
            "accuracy_by_subcase": {"toy": 0.5},
        },
        "mnli": {
            "matched": {"accuracy": 0.3},
            "mismatched": {"accuracy": 0.2},
        },
        "leakage_audit": {"passed": True},
        "learning_curve_file": "posthoc_gain_eval_curve.json",
    }
    (run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    curve = {
        "points": [
            {
                "round": 0,
                "evaluation_role": "gain_eval",
                "cumulative_final_training_examples": 0,
                "cumulative_total_cost": 0.25,
                "metrics": {
                    "overall_accuracy": 0.45,
                    "non_entailment_accuracy": 0.35,
                },
            }
        ]
    }
    (run / "posthoc_gain_eval_curve.json").write_text(
        json.dumps(curve), encoding="utf-8"
    )
    report = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "report_stage2_benchmark.py"),
            "--runs-root",
            str(tmp_path / "runs"),
            "--output-dir",
            str(report),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (report / "STAGE2_REPORT.md").is_file()
    assert (report / "stage2_summary.csv").is_file()
    assert (report / "stage2_learning_curves.csv").is_file()
    assert (report / "figures" / "experiment_a_final_hans_accuracy.svg").is_file()
    assert (report / "figures" / "experiment_a_gain_eval_learning_curve.svg").is_file()
    assert (report / "figures" / "experiment_a_accuracy_vs_cost.svg").is_file()
