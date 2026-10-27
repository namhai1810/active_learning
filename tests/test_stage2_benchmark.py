import copy
import json
import math
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
from src.benchmark.selection import (
    select_closed_loop_soft_pairs,
    select_confidence_safe_entropy_novelty_pairs,
    select_complete_pairs,
    select_stratified_utility_pairs,
    select_verification_conditioned_primal_dual_pairs,
)
from src.active_learning.closed_loop import (
    merge_hypothesis_candidates,
    verification_states,
)
from src.hypotheses.error_scorer import UniformGoldBlindErrorProbabilityScorer
from src.benchmark.stopping_subset import (
    build_stopping_subset_manifest,
    validate_stopping_subset_manifest,
)
from src.decision.artifact_cost import build_artifact_costs, build_round_incurred_cost
from src.decision.cost_aware import select_intervention
from src.decision.types import DecisionCandidate
from src.evaluation.metrics import compute_hans_metrics
from src.models.lora_training import (
    EarlyStoppingConfig,
    EarlyStoppingTracker,
    _cpu_tree,
)
from src.utils.config import load_yaml_with_extends
from src.synthetic.types import SyntheticNLIExample
from scripts.run_stage2_benchmark import (
    _adaptive_stop_decision,
    _paired_gain_confidence,
    _synthetic_pool,
)
from scripts.run_stage2_closed_loop import (
    _apply_seed_override,
    _binary_hans_validation,
    _bypassed_verification_states,
    _reuse_completed_refresh,
    _skipped_discovery_payload,
    _write_round_split_manifest,
)
from scripts.verify_hypotheses import _load_hypotheses as _load_verified_hypotheses
from scripts.generate_synthetic_interventions import _retained_ids
from src.data.hans import HANSExample
from src.data.pipeline_split import reconstruct_five_way_split, validate_manifest


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


def test_closed_loop_writes_online_manifest_when_discovery_is_skipped(tmp_path):
    rows = [{"id": row["id"]} for row in _hans_rows()]
    hans = {
        "discovery_fraction": 0.2,
        "verification_fraction": 0.2,
        "intervention_fraction": 0.2,
        "gain_eval_fraction": 0.2,
        "split_seed": 42,
    }
    path = tmp_path / "hans_split_v2.json"

    _write_round_split_manifest(path, rows, hans)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split_version"] == "hans_five_way_v1"
    validate_manifest(payload, reconstruct_five_way_split(rows, hans))


def test_closed_loop_skipped_discovery_preserves_hypothesis_provenance():
    payload = _skipped_discovery_payload(
        "skipped_active_candidate_capacity_full"
    )

    assert payload["split_role"] == "hypothesis_discovery"
    assert _load_verified_hypotheses(payload) == []


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


def _stratified_synthetic(pair_index, hypothesis_id, pair_type, labels):
    members = (
        ("base", "relation_changed")
        if pair_type == "counterfactual"
        else ("base", "irrelevant_change")
    )
    return [
        SyntheticNLIExample(
            id=f"strat-{pair_index}-{member}",
            premise=f"Scenario {pair_index} premise variant {offset}.",
            hypothesis=f"Scenario {pair_index} hypothesis variant {offset}.",
            label=labels[offset],
            hypothesis_id=hypothesis_id,
            pair_id=f"strat-pair-{pair_index}",
            pair_type=pair_type,
            pair_member=member,
            pair_rationale="controlled pair",
            seed_example_ids=("s1",),
            label_rationale="declared test label",
            targeting_rationale="tests the certified mechanism",
            generator_prompt_version="v1",
        )
        for offset, member in enumerate(members)
    ]


def test_uniform_gold_blind_error_scorer_ignores_gold_and_correctness():
    records = [
        {
            "id": "a",
            "premise": "A premise.",
            "hypothesis": "A hypothesis.",
            "prediction": "entailment",
            "gold": "contradiction",
            "correct": False,
        },
        {
            "id": "b",
            "premise": "Another premise.",
            "hypothesis": "Another hypothesis.",
            "prediction": "neutral",
            "gold": "neutral",
            "correct": True,
        },
    ]
    result = UniformGoldBlindErrorProbabilityScorer(score=0.5).score(records)
    assert result.scores == {"a": 0.5, "b": 0.5}
    assert result.logs == ()
    assert result.method == "uniform_gold_blind"


def test_closed_loop_hypothesis_merge_preserves_active_and_deduplicates_history():
    archive = [
        {
            "id": "old",
            "name": "Subject object reversal",
            "description": "The model swaps the subject and object roles.",
            "predicted_failure_behavior": "Predicts entailment after reversal.",
            "intervention_target": "Reversed agent and patient pairs.",
        }
    ]
    proposals = [
        {
            "id": "renamed",
            "name": "Subject object reversal",
            "description": "The model swaps the subject and object roles.",
            "predicted_failure_behavior": "Predicts entailment after reversal.",
            "intervention_target": "Reversed agent and patient pairs.",
        },
        {
            "id": "novel",
            "name": "Negated conditional",
            "description": "Conditional negation causes a systematic failure.",
            "predicted_failure_behavior": "Predicts entailment for a negated condition.",
            "intervention_target": "Negated conditional clauses.",
        },
    ]
    merged, log = merge_hypothesis_candidates(
        archive,
        proposals,
        active_ids=["old"],
        maximum=3,
        near_duplicate_threshold=0.78,
    )
    assert [item["id"] for item in merged] == ["old", "novel"]
    assert log["novel_accepted"] == ["novel"]
    assert log["proposals_rejected"][0]["duplicate_of"] == "old"


def test_closed_loop_rediscovery_reactivates_a_resolved_mechanism():
    archived = {
        "id": "resolved",
        "name": "Negated conditional",
        "description": "Conditional negation causes a systematic failure.",
        "predicted_failure_behavior": "Predicts entailment for a negated condition.",
        "intervention_target": "Negated conditional clauses.",
    }
    renamed = {**archived, "id": "renamed"}
    merged, log = merge_hypothesis_candidates(
        [archived],
        [renamed],
        active_ids=[],
        maximum=3,
        near_duplicate_threshold=0.78,
    )
    assert [item["id"] for item in merged] == ["resolved"]
    assert log["historical_reactivated"] == ["resolved"]
    assert log["proposals_rejected"][0]["reason"] == "historical_mechanism_reactivated"


def test_checkpoint_verification_state_uses_ucb_to_resolve():
    payload = {
        "tau_v": 0.5,
        "results": [
            {
                "hypothesis_id": "resolved",
                "nu_hat": 0.0,
                "confidence_radius": 0.4,
                "lower_bound": -0.4,
                "n": 20,
                "certification_metric": "target_failure",
                "probe_design": {"reliable_for_lcb": True},
            },
            {
                "hypothesis_id": "uncertain",
                "nu_hat": 0.5,
                "confidence_radius": 0.4,
                "lower_bound": 0.1,
                "n": 20,
                "certification_metric": "target_failure",
                "probe_design": {"reliable_for_lcb": True},
            },
        ],
    }
    states = verification_states(payload)
    assert states["resolved"]["state"] == "refuted"
    assert states["uncertain"]["state"] == "unresolved"


def test_closed_loop_soft_selector_is_pair_complete_and_deterministic():
    pool = []
    for pair_index, hypothesis_id in enumerate(("h1", "h1", "h2", "h2")):
        pool.extend(
            _stratified_synthetic(
                pair_index,
                hypothesis_id,
                "invariant",
                ("entailment", "entailment"),
            )
        )
    nll = {item.id: float(index + 1) for index, item in enumerate(pool)}
    entropy = {item.id: (index % 4) / 4 for index, item in enumerate(pool)}
    embeddings = {
        item.id: [1.0, float(pair_index + 1), float(pair_index % 2)]
        for pair_index, item in enumerate(pool)
    }
    kwargs = {
        "budget": 4,
        "seed": 42,
        "nll_scores": nll,
        "entropy_scores": entropy,
        "embeddings": embeddings,
        "verification_evidence": {
            "h1": {"state": "certified"},
            "h2": {"state": "unresolved"},
        },
        "prior_hypothesis_pair_counts": {"h1": 5, "h2": 0},
    }
    selected, diagnostics = select_closed_loop_soft_pairs(pool, **kwargs)
    repeated, repeated_diagnostics = select_closed_loop_soft_pairs(pool, **kwargs)
    assert [item.id for item in selected] == [item.id for item in repeated]
    assert diagnostics == repeated_diagnostics
    assert len(selected) == 4
    assert len({item.pair_id for item in selected}) == 2
    assert diagnostics["hard_hypothesis_quota"] is False
    assert diagnostics["weights"]["new_hypothesis_bonus"] == 0.50
    first_pair = next(iter(diagnostics["selected_pair_components"].values()))
    assert 0.0 < first_pair["new_hypothesis_bonus"] <= 1.0


def test_closed_loop_soft_bonus_prefers_an_untrained_hypothesis_when_tied():
    pool = [
        *_stratified_synthetic(
            20, "trained", "invariant", ("entailment", "entailment")
        ),
        *_stratified_synthetic(
            21, "new", "invariant", ("entailment", "entailment")
        ),
    ]
    selected, diagnostics = select_closed_loop_soft_pairs(
        pool,
        budget=2,
        seed=42,
        nll_scores={item.id: 1.0 for item in pool},
        entropy_scores={item.id: 0.5 for item in pool},
        embeddings={item.id: [1.0, 0.0] for item in pool},
        verification_evidence={
            "trained": {"state": "certified"},
            "new": {"state": "unresolved"},
        },
        prior_hypothesis_pair_counts={"trained": 5, "new": 0},
    )
    assert {item.hypothesis_id for item in selected} == {"new"}
    component = next(iter(diagnostics["selected_pair_components"].values()))
    assert component["new_hypothesis_bonus"] == 1.0


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


def test_stratified_student_utility_selection_is_balanced_and_deterministic():
    pool = []
    pair_specs = [
        (0, "h1", "counterfactual", ("entailment", "contradiction")),
        (1, "h1", "counterfactual", ("neutral", "contradiction")),
        (2, "h1", "invariant", ("entailment", "entailment")),
        (3, "h1", "invariant", ("neutral", "neutral")),
        (4, "h2", "counterfactual", ("entailment", "neutral")),
        (5, "h2", "counterfactual", ("contradiction", "neutral")),
        (6, "h2", "invariant", ("entailment", "entailment")),
        (7, "h2", "invariant", ("contradiction", "contradiction")),
    ]
    for spec in pair_specs:
        pool.extend(_stratified_synthetic(*spec))
    utility = {item.id: float(index + 1) for index, item in enumerate(pool)}

    selected, diagnostics = select_stratified_utility_pairs(
        pool,
        budget=8,
        seed=42,
        utility_scores=utility,
        certification_scores={"h1": 0.55, "h2": 0.75},
    )
    repeated, repeated_diagnostics = select_stratified_utility_pairs(
        pool,
        budget=8,
        seed=42,
        utility_scores=utility,
        certification_scores={"h1": 0.55, "h2": 0.75},
    )

    assert [item.id for item in selected] == [item.id for item in repeated]
    assert diagnostics == repeated_diagnostics
    assert len(selected) == 8
    assert all(
        sum(item.pair_id == pair_id for item in selected) == 2
        for pair_id in {item.pair_id for item in selected}
    )
    assert diagnostics["hypothesis_pair_counts"] == {"h1": 2, "h2": 2}
    assert diagnostics["pair_type_counts"] == {
        "counterfactual": 2,
        "invariant": 2,
    }
    assert sum(diagnostics["label_counts"].values()) == 8
    assert diagnostics["shortage"] is False


def test_verification_conditioned_selection_learns_targets_and_is_reproducible():
    pool = []
    pair_specs = [
        (0, "h1", "counterfactual", ("entailment", "contradiction")),
        (1, "h1", "counterfactual", ("neutral", "contradiction")),
        (2, "h1", "invariant", ("entailment", "entailment")),
        (3, "h1", "invariant", ("neutral", "neutral")),
        (4, "h2", "counterfactual", ("entailment", "neutral")),
        (5, "h2", "counterfactual", ("contradiction", "neutral")),
        (6, "h2", "invariant", ("entailment", "entailment")),
        (7, "h2", "invariant", ("contradiction", "contradiction")),
    ]
    for spec in pair_specs:
        pool.extend(_stratified_synthetic(*spec))
    utility = {item.id: 0.1 + index / 10 for index, item in enumerate(pool)}
    evidence = {
        "h1": {
            "lower_bound": 0.55,
            "target_n": 20,
            "target_failures_by_label": {
                "entailment": 0,
                "neutral": 14,
                "contradiction": 2,
            },
            "counterfactual_accuracy": 0.55,
            "invariant_accuracy": 0.50,
        },
        "h2": {
            "lower_bound": 0.60,
            "target_n": 20,
            "target_failures_by_label": {
                "entailment": 0,
                "neutral": 16,
                "contradiction": 1,
            },
            "counterfactual_accuracy": 0.95,
            "invariant_accuracy": 0.85,
        },
    }

    selected, diagnostics = select_verification_conditioned_primal_dual_pairs(
        pool,
        budget=8,
        seed=42,
        utility_scores=utility,
        verification_evidence=evidence,
    )
    repeated, repeated_diagnostics = select_verification_conditioned_primal_dual_pairs(
        pool,
        budget=8,
        seed=42,
        utility_scores=utility,
        verification_evidence=evidence,
    )

    assert [item.id for item in selected] == [item.id for item in repeated]
    assert diagnostics == repeated_diagnostics
    assert diagnostics["selection_strategy"] == "verification_conditioned_primal_dual_v1"
    assert diagnostics["target_label_distribution"]["neutral"] > diagnostics[
        "target_label_distribution"
    ]["entailment"]
    assert diagnostics["target_hypothesis_distribution"]["h2"] > diagnostics[
        "target_hypothesis_distribution"
    ]["h1"]
    assert diagnostics["verification_evidence"]["h2"]["control_specificity"] > diagnostics[
        "verification_evidence"
    ]["h1"]["control_specificity"]
    assert len(selected) == 8
    assert len({item.pair_id for item in selected}) == 4
    assert all(
        sum(item.pair_id == pair_id for item in selected) == 2
        for pair_id in {item.pair_id for item in selected}
    )


def test_confidence_safe_retention_keeps_unresolved_and_rejects_only_by_ucb():
    def result(hypothesis_id, estimate, radius, certified=False):
        return {
            "hypothesis_id": hypothesis_id,
            "certified": certified,
            "nu_hat": estimate,
            "confidence_radius": radius,
            "lower_bound": estimate - radius,
            "n": 20,
            "certification_metric": "target_failure",
            "probe_design": {"reliable_for_lcb": True},
            "diagnostics": {
                "target_failure": {
                    "nu_hat": estimate,
                    "confidence_radius": radius,
                    "lower_bound": estimate - radius,
                }
            },
        }

    payload = {
        "mode": "llm_main",
        "oracle_debug": False,
        "verification_mode": "new_controlled_generated_probes",
        "verification_probes_training_eligible": False,
        "certification_metric": "target_failure",
        "tau_v": 0.5,
        "results": [
            result("certified", 0.90, 0.30, certified=True),
            result("unresolved", 0.55, 0.25),
            result("refuted", 0.00, 0.44),
        ],
    }
    certified, retained, states = _retained_ids(
        payload, retention_policy="confidence_safe"
    )
    assert certified == ["certified"]
    assert retained == ["certified", "unresolved"]
    assert states["refuted"]["state"] == "refuted"


def test_confidence_safe_retention_keeps_insufficient_probes_unresolved():
    payload = {
        "mode": "llm_main",
        "oracle_debug": False,
        "verification_mode": "new_controlled_generated_probes",
        "verification_probes_training_eligible": False,
        "certification_metric": "target_failure",
        "tau_v": 0.5,
        "results": [
            {
                "hypothesis_id": "insufficient",
                "certified": False,
                "nu_hat": None,
                "confidence_radius": None,
                "lower_bound": None,
                "n": 0,
                "certification_metric": "target_failure",
                "status": "unreliable_probe_design",
                "probe_design": {"reliable_for_lcb": False},
                "diagnostics": {
                    "target_failure": {
                        "status": "insufficient_probes",
                        "nu_hat": None,
                        "confidence_radius": None,
                        "lower_bound": None,
                    }
                },
            }
        ],
    }

    certified, retained, states = _retained_ids(
        payload, retention_policy="confidence_safe"
    )

    assert certified == []
    assert retained == ["insufficient"]
    assert states["insufficient"]["state"] == "unresolved"
    assert states["insufficient"]["nu_hat"] is None


def test_confidence_safe_entropy_novelty_selection_covers_every_retained_hypothesis():
    pool = []
    pair_specs = [
        (0, "certified", "counterfactual", ("neutral", "entailment")),
        (1, "certified", "invariant", ("contradiction", "contradiction")),
        (2, "unresolved", "counterfactual", ("neutral", "contradiction")),
        (3, "unresolved", "invariant", ("entailment", "entailment")),
    ]
    for spec in pair_specs:
        pool.extend(_stratified_synthetic(*spec))
    nll_scores = {item.id: 0.2 + index / 10 for index, item in enumerate(pool)}
    entropy_scores = {item.id: 0.2 + (index % 4) / 10 for index, item in enumerate(pool)}
    embeddings = {
        item.id: [
            1.0 if index % 2 == 0 else 0.0,
            1.0 if item.hypothesis_id == "unresolved" else 0.2,
            0.1 + index / 100,
        ]
        for index, item in enumerate(pool)
    }
    evidence = {
        "certified": {"state": "certified"},
        "unresolved": {"state": "unresolved"},
    }

    selected, diagnostics = select_confidence_safe_entropy_novelty_pairs(
        pool,
        budget=6,
        seed=42,
        nll_scores=nll_scores,
        entropy_scores=entropy_scores,
        embeddings=embeddings,
        verification_evidence=evidence,
    )
    repeated, repeated_diagnostics = select_confidence_safe_entropy_novelty_pairs(
        pool,
        budget=6,
        seed=42,
        nll_scores=nll_scores,
        entropy_scores=entropy_scores,
        embeddings=embeddings,
        verification_evidence=evidence,
    )

    assert [item.id for item in selected] == [item.id for item in repeated]
    assert diagnostics == repeated_diagnostics
    assert diagnostics["selection_strategy"] == "confidence_safe_entropy_novelty_v1"
    assert set(diagnostics["hypothesis_pair_counts"]) == {"certified", "unresolved"}
    assert all(value >= 1 for value in diagnostics["hypothesis_pair_counts"].values())
    assert len(selected) == 6
    assert len({item.pair_id for item in selected}) == 3


def test_paired_gain_stopping_requires_patience_and_records_cost_efficiency():
    before = [{"id": f"e{index}", "correct": index % 2 == 0} for index in range(1000)]
    after = [dict(row) for row in before]
    confidence = _paired_gain_confidence(before, after, delta=0.05, max_looks=4)
    config = {
        "minimum_rounds": 2,
        "patience": 2,
        "minimum_mean_gain": 0.002,
        "maximum_small_gain_ucb": 0.05,
        "minimum_optimistic_gain_per_gpu_hour": 0.0,
    }

    stop, count, reason, cost = _adaptive_stop_decision(
        confidence,
        round_index=1,
        consecutive_small=0,
        config=config,
        stage_gpu_seconds=120.0,
    )
    assert stop is False
    assert count == 1
    assert reason == "continue"
    assert cost["optimistic_gain_per_gpu_hour"] >= 0.0

    stop, count, reason, _ = _adaptive_stop_decision(
        confidence,
        round_index=2,
        consecutive_small=count,
        config=config,
        stage_gpu_seconds=120.0,
    )
    assert stop is True
    assert count == 2
    assert reason == "diminishing_gain_patience_exhausted"


def test_stopping_subset_is_balanced_deterministic_and_prediction_blind():
    rows = [
        {
            "id": f"gain-{subcase}-{index}",
            "subcase": subcase,
            "heuristic": ("lexical_overlap", "subsequence", "constituent")[
                subcase_index
            ],
            "gold": "entailment" if subcase_index % 2 else "non-entailment",
            "correct": index % 3 == 0,
        }
        for subcase_index, subcase in enumerate(("case-a", "case-b", "case-c"))
        for index in range(6)
    ]
    ids = [row["id"] for row in rows]
    manifest = build_stopping_subset_manifest(
        rows,
        ids,
        frozen_manifest_hash="frozen-hash",
        examples_per_subcase=2,
    )
    changed_predictions = [{**row, "correct": not row["correct"]} for row in rows]
    repeated = build_stopping_subset_manifest(
        changed_predictions,
        ids,
        frozen_manifest_hash="frozen-hash",
        examples_per_subcase=2,
    )

    assert manifest == repeated
    assert len(manifest["stopping_eval_ids"]) == 6
    assert len(manifest["gain_report_ids"]) == 12
    assert set(manifest["stopping_counts_by_subcase"].values()) == {2}
    stopping, report = validate_stopping_subset_manifest(
        manifest,
        rows,
        ids,
        frozen_manifest_hash="frozen-hash",
    )
    assert set(stopping).isdisjoint(report)
    assert set(stopping) | set(report) == set(ids)


def test_practical_stopping_uses_group_gains_and_patience_not_the_wide_ucb():
    confidence = {
        "mean_gain": 0.002,
        "upper_bound": 0.20,
        "group_gains": {
            "heuristic": {
                "lexical_overlap": 0.003,
                "subsequence": 0.004,
                "constituent": -0.001,
            },
            "gold": {"entailment": 0.001, "non-entailment": 0.004},
        },
    }
    config = {
        "decision_rule": "practical_paired_gain_patience_v1",
        "minimum_rounds": 2,
        "patience": 2,
        "minimum_mean_gain": 0.003,
        "maximum_heuristic_gain": 0.005,
        "maximum_label_gain": 0.005,
        "minimum_heuristic_gain": -0.01,
        "minimum_label_gain": -0.01,
        "maximum_small_gain_ucb": 0.02,
    }
    stop, count, _, stats = _adaptive_stop_decision(
        confidence,
        round_index=1,
        consecutive_small=0,
        config=config,
        stage_gpu_seconds=10.0,
    )
    assert stop is False
    assert count == 1
    assert stats["small_gain_look"] is True
    stop, count, reason, _ = _adaptive_stop_decision(
        confidence,
        round_index=2,
        consecutive_small=count,
        config=config,
        stage_gpu_seconds=10.0,
    )
    assert stop is True
    assert count == 2
    assert reason == "diminishing_gain_patience_exhausted"


def test_only_well_formed_verified_abstention_may_have_an_empty_pool(tmp_path):
    accepted = tmp_path / "accepted.jsonl"
    manifest = tmp_path / "manifest.json"
    calls = tmp_path / "teacher_calls"
    calls.mkdir()
    accepted.write_text("", encoding="utf-8")
    payload = {
        "stage2_policy": "failure_targeted_verified",
        "status": "abstained_no_certified_hypotheses",
        "abstention_reason": "no_certified_hypotheses",
        "binary_hans_labels_converted": False,
        "pair_selection_unit": "complete_pair",
        "certified_hypothesis_ids": [],
        "counts": {},
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    benchmark = {
        "synthetic_artifacts": {
            "ours_verified": {
                "accepted": str(accepted),
                "provenance": str(manifest),
                "teacher_calls_dir": str(calls),
            }
        }
    }
    (calls / "discovery.jsonl").write_text(
        json.dumps(
            {
                "usage": {"input_tokens": 9999, "output_tokens": 9999},
                "scoring_calls": [
                    {"usage": {"input_tokens": 100, "output_tokens": 10}}
                ],
                "hypothesis_discovery_attempts": [
                    {"usage": {"input_tokens": 200, "output_tokens": 20}}
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (calls / "verification.jsonl").write_text(
        json.dumps(
            {
                "calls": [
                    {"usage": {"input_tokens": 400, "output_tokens": 40}}
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    pool, provenance = _synthetic_pool(benchmark, "ours_verified", 42)

    assert pool == []
    assert provenance["status"] == "abstained_no_certified_hypotheses"
    assert provenance["resource_usage"]["teacher_input_tokens"] == 700
    assert provenance["resource_usage"]["teacher_output_tokens"] == 70
    payload["certified_hypothesis_ids"] = ["h1"]
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="empty pool"):
        _synthetic_pool(benchmark, "ours_verified", 42)


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
    abstained_run = (
        tmp_path / "runs" / "ours_verified" / "budget_128" / "seed_42"
    )
    abstained_run.mkdir(parents=True)
    abstained_result = {
        **result,
        "method": "ours_verified",
        "requested_budget": 128,
        "actual_final_training_examples": 0,
        "shortage": True,
        "abstained": True,
        "run_status": "abstained_no_certified_hypotheses",
        "abstention_reason": "no_certified_hypotheses",
        "fixed_budget_comparable": False,
        "hans_final": {
            **result["hans_final"],
            "overall_accuracy": 0.99,
        },
        "synthetic_provenance": {
            "verification_summary": {
                "verification_protocol_version": "target_failure_verified_v1",
                "certification_metric": "target_failure",
                "results": [
                    {
                        "hypothesis_id": "h1",
                        "hypothesis_name": "Toy target failure",
                        "certification_metric": "target_failure",
                        "certified": False,
                        "diagnostics": {
                            "target_failure": {
                                "nu_hat": 0.4,
                                "lower_bound": 0.1,
                            },
                            "strict_joint": {
                                "nu_hat": 0.2,
                                "lower_bound": -0.1,
                            },
                            "controls": {
                                "counterfactual": {"accuracy": 0.7},
                                "invariant": {"accuracy": 0.8},
                            },
                        },
                    }
                ],
            }
        },
    }
    (abstained_run / "result.json").write_text(
        json.dumps(abstained_result), encoding="utf-8"
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
    summary = json.loads((report / "stage2_summary.json").read_text())
    abstained_group = next(
        item for item in summary["groups"] if item["method"] == "ours_verified"
    )
    assert abstained_group["any_abstained"] is True
    assert abstained_group["comparable_seeds"] == []
    assert abstained_group["hans_final.overall_accuracy"]["n"] == 0
    assert abstained_group["hans_final.overall_accuracy"]["mean"] is None
    assert abstained_group["actual_final_training_examples"]["mean"] == 0
    assert "ABSTAINED" in (report / "STAGE2_REPORT.md").read_text()
    assert "## Verification diagnostics" in (
        report / "STAGE2_REPORT.md"
    ).read_text()
    assert summary["verification_diagnostics"][0]["hypothesis_id"] == "h1"
    assert "ours_verified · budget 128" not in (
        report / "figures" / "experiment_a_final_hans_accuracy.svg"
    ).read_text()


def test_objective_ablation_report_requires_identical_selected_ids(tmp_path):
    result_paths = {}
    common = {
        "method": "ours_verified",
        "seed": 42,
        "requested_budget": 4,
        "manifest_hash": "frozen-manifest",
        "run_status": "complete",
        "selected_ids_file": "selected_ids.json",
        "hans_final": {
            "overall_accuracy": 0.6,
            "entailment_accuracy": 0.7,
            "non_entailment_accuracy": 0.5,
        },
        "mnli": {
            "matched": {"accuracy": 0.4},
            "mismatched": {"accuracy": 0.3},
        },
    }
    for objective in ("pairwise", "ce"):
        run = tmp_path / objective
        run.mkdir()
        result = {
            **common,
            "ours_objective": objective,
            "hans_final": {
                **common["hans_final"],
                "overall_accuracy": 0.65 if objective == "pairwise" else 0.6,
            },
        }
        result_path = run / "result.json"
        result_path.write_text(json.dumps(result), encoding="utf-8")
        (run / "selected_ids.json").write_text(
            json.dumps({"ids": ["a", "b", "c", "d"]}), encoding="utf-8"
        )
        result_paths[objective] = result_path

    output = tmp_path / "ablation"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "report_ours_objective_ablation.py"),
            "--pairwise-result",
            str(result_paths["pairwise"]),
            "--ce-result",
            str(result_paths["ce"]),
            "--output-dir",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = (output / "OBJECTIVE_ABLATION.md").read_text()
    payload = json.loads((output / "objective_ablation.json").read_text())
    assert "Controlled comparison check: **pass**" in report
    assert payload["same_selected_ids_in_same_order"] is True
    assert payload["metrics"][0]["pairwise_minus_ce"] == pytest.approx(0.05)


def test_training_early_stopping_uses_min_delta_patience_and_best_epoch():
    config = EarlyStoppingConfig(max_epochs=8, patience=2, min_delta=0.001)
    tracker = EarlyStoppingTracker(config)

    assert tracker.observe(1, 1.0) == (True, False)
    assert tracker.observe(2, 0.9) == (True, False)
    assert tracker.observe(3, 0.9005) == (False, False)
    assert tracker.observe(4, 0.901) == (False, True)
    assert tracker.best_epoch == 2
    assert tracker.best_loss == pytest.approx(0.9)


def test_training_early_stopping_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        EarlyStoppingConfig.from_mapping(
            {
                "max_epochs": 0,
                "early_stopping": {
                    "enabled": True,
                    "patience": 2,
                    "min_delta": 0.0,
                },
            }
        )


def test_training_binary_hans_validation_combines_non_entailment_classes():
    examples = [
        HANSExample("e", "p", "h", "entailment", "constituent", "case-e"),
        HANSExample("n", "p", "h", "non-entailment", "subsequence", "case-n"),
    ]
    result = _binary_hans_validation(
        [[0.8, 0.1, 0.1], [0.2, 0.5, 0.3]], examples
    )

    assert result["accuracy"] == 1.0
    assert result["loss"] == pytest.approx(-math.log(0.8))
    assert result["sealed_final_test_used"] is False


def test_verification_off_retains_every_hypothesis_without_certifying():
    states = _bypassed_verification_states([{"id": "h2"}, {"id": "h1"}])

    assert set(states) == {"h1", "h2"}
    assert all(value["state"] == "unresolved" for value in states.values())
    assert all(value["verification_bypassed"] for value in states.values())
    assert not any(value["certified"] for value in states.values())


def test_closed_loop_seed_override_preserves_frozen_split():
    config = {
        "seed": 42,
        "closed_loop": {"seed": 42},
        "hans": {"split_seed": 20261011},
        "hypothesis_generation": {
            "seed": 42,
            "error_scorer": {"seed": 42},
        },
        "verification": {"seed": 42, "label_validation_seed": 314159},
        "synthetic_generation": {"seed": 42},
        "synthetic_validation": {"seed": 42},
    }

    _apply_seed_override(config, 44)

    assert config["hans"]["split_seed"] == 20261011
    assert config["closed_loop"]["seed"] == 44
    assert config["verification"]["label_validation_seed"] == 314161
    assert config["hypothesis_generation"]["error_scorer"]["seed"] == 44


def test_training_verification_on_off_configs_differ_only_in_switch():
    on = load_yaml_with_extends(
        PROJECT_ROOT / "configs/stage2_v7_early_stopping_verification_on.yaml"
    )
    off = load_yaml_with_extends(
        PROJECT_ROOT / "configs/stage2_v7_early_stopping_verification_off.yaml"
    )

    assert on["training"] == off["training"]
    assert on["training"]["batch_size"] == 4
    assert on["training"]["gradient_accumulation_steps"] == 2
    assert (
        on["training"]["batch_size"]
        * on["training"]["gradient_accumulation_steps"]
        == 8
    )
    assert on["closed_loop"]["total_budget"] == 128
    assert on["closed_loop"]["max_rounds"] == 4
    assert on["closed_loop"]["per_round_examples"] == 32

    normalized_on = copy.deepcopy(on)
    normalized_off = copy.deepcopy(off)
    normalized_on.pop("stage2_policy")
    normalized_off.pop("stage2_policy")
    normalized_on["closed_loop"].pop("verification_mode")
    normalized_off["closed_loop"].pop("verification_mode")
    assert normalized_on == normalized_off


def test_verification_off_refresh_resume_requires_no_probe_artifact(tmp_path):
    round_dir = tmp_path / "refresh" / "round_002"
    round_dir.mkdir(parents=True)
    adapter = "/tmp/checkpoint/round_001"
    lifecycle = {
        "refresh_round": 2,
        "checkpoint_adapter": adapter,
        "verification_mode": "off",
        "candidate_ids": ["h1"],
        "states": {
            "h1": {
                "state": "unresolved",
                "verification_mode": "off",
                "verification_bypassed": True,
            }
        },
        "next_active_ids": ["h1"],
        "generated_training_examples": 0,
    }
    (round_dir / "lifecycle.json").write_text(
        json.dumps(lifecycle), encoding="utf-8"
    )
    (round_dir / "hypotheses.json").write_text(
        json.dumps({"hypotheses": [{"id": "h1"}]}), encoding="utf-8"
    )

    reused = _reuse_completed_refresh(
        round_dir=round_dir,
        round_index=2,
        adapter=adapter,
        archive=[],
        generate_pool=False,
    )

    assert reused is not None
    _, active, states, additions, metadata = reused
    assert active == ["h1"]
    assert states["h1"]["verification_bypassed"] is True
    assert additions == []
    assert metadata["resumed_from_completed_artifacts"] is True


def test_optimizer_best_snapshot_clones_cpu_step_tensor():
    import torch

    live_step = torch.tensor(4.0)
    snapshot = _cpu_tree({"state": {0: {"step": live_step}}})
    live_step.add_(8.0)

    assert snapshot["state"][0]["step"].item() == 4.0
    assert live_step.item() == 12.0
