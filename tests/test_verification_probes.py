import copy
from dataclasses import replace

import pytest

from src.hypotheses.types import FailureHypothesis
from src.verification.generated_probes import (
    ControlledProbeConfig,
    ControlledVerificationProbeGenerator,
    TeacherVerificationProbeValidator,
    _normalized_pair_text,
    _parse_generated_probe_batch,
)
from scripts.verify_hypotheses import (
    _adaptive_action,
    _adaptive_schedule,
    _metric_outcomes,
    _verification_statistic,
)


def _hypothesis() -> FailureHypothesis:
    return FailureHypothesis(
        id="hypothesis-1",
        name="role confusion",
        description="The student confuses semantic roles.",
        predicted_failure_behavior="Predicts entailment after swapping roles.",
        predicted_failure_labels=("entailment",),
        verification_criteria=("Role-swapped targets fail while controls pass.",),
        intervention_target="Subject-object role assignment.",
        support_example_ids=("discovery-1",),
        source="llm_teacher",
        prompt_version="failure_discovery_v1",
    )


def _probe(
    premise: str,
    hypothesis: str,
    expected_label: str,
    expected_failure_label: str | None,
) -> dict[str, object]:
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "expected_label": expected_label,
        "expected_failure_label": expected_failure_label,
        "label_rationale": "The declared MNLI relation follows from the text.",
        "constraint_rationale": "This pair tests the declared mechanism.",
    }


def _response() -> dict[str, object]:
    return {
        "probe_sets": [
            {
                "seed_context_ids": ["verification-1"],
                "mechanism_rationale": "Compare a role swap with two controls.",
                "target": _probe(
                    "The baker thanked the courier.",
                    "The courier thanked the baker.",
                    "neutral",
                    "entailment",
                ),
                "counterfactual_control": _probe(
                    "The courier thanked the baker.",
                    "The courier thanked the baker.",
                    "entailment",
                    None,
                ),
                "invariant_control": _probe(
                    "Yesterday, the baker thanked the courier.",
                    "The baker thanked the courier.",
                    "entailment",
                    None,
                ),
            }
        ]
    }


def _parse(response: dict[str, object], **kwargs):
    return _parse_generated_probe_batch(
        response,
        hypothesis=_hypothesis(),
        permitted_ids={"verification-1"},
        requested=1,
        ordinal_start=0,
        prompt_version="controlled_verification_probes_v4",
        **kwargs,
    )


def test_generated_probe_parser_accepts_three_distinct_pairs():
    probes = _parse(_response())

    assert [probe.probe_type for probe in probes] == [
        "target",
        "counterfactual_control",
        "invariant_control",
    ]


def test_generated_probe_parser_rejects_duplicate_within_set_before_qc():
    response = _response()
    probe_set = response["probe_sets"][0]
    probe_set["invariant_control"]["premise"] = probe_set["target"]["premise"]
    probe_set["invariant_control"]["hypothesis"] = probe_set["target"]["hypothesis"]

    with pytest.raises(ValueError, match="pairwise-distinct"):
        _parse(response)


def test_generated_probe_parser_rejects_invalid_identical_sentence_label():
    response = _response()
    target = response["probe_sets"][0]["target"]
    target["hypothesis"] = target["premise"]

    with pytest.raises(ValueError, match="identical premise and hypothesis"):
        _parse(response)


def test_generated_probe_parser_rejects_pair_seen_in_an_earlier_batch():
    response = _response()
    target = copy.deepcopy(response["probe_sets"][0]["target"])
    pair = _normalized_pair_text(target["premise"], target["hypothesis"])

    with pytest.raises(ValueError, match="duplicates an earlier probe"):
        _parse(response, forbidden_pairs={pair: "vprobe_previous"})


def test_generated_probe_parser_rejects_near_duplicate_before_qc():
    response = _response()
    reverse_role_pair = _normalized_pair_text(
        "The courier thanked the baker.",
        "The baker thanked the courier.",
    )

    with pytest.raises(ValueError, match="near-duplicate"):
        _parse(response, forbidden_pairs={reverse_role_pair: "verification_context:7"})


def test_generator_skips_one_exhausted_set_when_reliability_is_still_possible():
    invalid = _response()
    invalid_set = invalid["probe_sets"][0]
    invalid_set["invariant_control"]["premise"] = invalid_set["target"]["premise"]
    invalid_set["invariant_control"]["hypothesis"] = invalid_set["target"][
        "hypothesis"
    ]

    class QueueTeacher:
        def __init__(self):
            self.responses = [invalid, _response()]
            self.calls = []

        def generate_json(self, call):
            self.calls.append(call)
            return self.responses.pop(0)

    teacher = QueueTeacher()
    generator = ControlledVerificationProbeGenerator(
        teacher,
        ControlledProbeConfig(
            n_probe_sets=2,
            min_valid_probe_sets=1,
            max_context_records=1,
            max_probe_sets_per_call=1,
            max_attempts_per_batch=1,
            prompt_version="controlled_verification_probes_v4",
        ),
    )

    probes, log = generator.generate(
        _hypothesis(),
        [
            {
                "id": "verification-1",
                "premise": "A pilot slept quietly.",
                "hypothesis": "A person slept.",
                "gold": "entailment",
                "split_role": "verification",
            }
        ],
    )

    assert len(probes) == 3
    assert log["attempted_probe_sets"] == 2
    assert log["generated_probe_sets"] == 1
    assert log["failed_generation_probe_sets"] == 1
    assert [
        call.payload["scenario_assignments"][0]["set_ordinal"]
        for call in teacher.calls
    ] == [0, 1]


def _validation_decisions(probes):
    return {
        "decisions": [
            {
                "probe_id": probe["id"],
                "validated_label": probe["expected_label"],
                "label_valid": True,
                "constraint_satisfied": True,
                "unambiguous": True,
                "grammatical": True,
                "original": True,
                "suitable_control": True,
                "reasons": [],
            }
            for probe in probes
        ]
    }


def test_validator_splits_batch_that_repeatedly_omits_a_probe_id():
    first_set = _parse(_response())
    second_set = [
        replace(
            probe,
            id=f"{probe.id}_second",
            probe_set_id=f"{probe.probe_set_id}_second",
            premise=f"In another scenario, {probe.premise}",
        )
        for probe in first_set
    ]

    class IncompleteLargeBatchTeacher:
        def __init__(self):
            self.calls = []

        def generate_json(self, call):
            self.calls.append(call)
            supplied = call.payload["probe_sets"]
            if len(supplied) > 3:
                supplied = supplied[:-1]
            return _validation_decisions(supplied)

    teacher = IncompleteLargeBatchTeacher()
    validator = TeacherVerificationProbeValidator(
        teacher,
        seed=42,
        max_probe_sets_per_call=2,
        max_attempts_per_batch=1,
    )

    decisions, log = validator.validate(_hypothesis(), [*first_set, *second_set])

    assert set(decisions) == {probe.id for probe in [*first_set, *second_set]}
    assert [len(call.payload["probe_sets"]) for call in teacher.calls] == [6, 3, 3]
    assert log["failed_validation_probe_sets"] == 0


def test_validator_abstains_on_one_persistently_incomplete_set():
    probes = _parse(_response())

    class IncompleteSingletonTeacher:
        def generate_json(self, call):
            return _validation_decisions(call.payload["probe_sets"][:-1])

    validator = TeacherVerificationProbeValidator(
        IncompleteSingletonTeacher(),
        seed=42,
        max_probe_sets_per_call=1,
        max_attempts_per_batch=1,
    )

    decisions, log = validator.validate(_hypothesis(), probes)

    assert decisions == {}
    assert log["failed_validation_probe_sets"] == 1
    abstentions = [
        item
        for item in log["calls"]
        if item.get("verification_substage") == "validation_set_abstention"
    ]
    assert len(abstentions) == 1


def test_target_failure_certification_does_not_use_control_correctness():
    outcomes = []
    for index in range(58):
        target_failure = index < 56
        strict_joint = index < 21
        outcomes.append(
            {
                "target_failure_observed": target_failure,
                "counterfactual_control_correct": index < 30,
                "invariant_control_correct": index < 48,
                "z": int(strict_joint),
            }
        )

    target = _verification_statistic(
        "extra-context",
        outcomes,
        "target_failure",
        delta=0.05,
        tau_v=0.5,
        K=3,
        T=1,
    )
    strict = _verification_statistic(
        "extra-context",
        outcomes,
        "strict_joint",
        delta=0.05,
        tau_v=0.5,
        K=3,
        T=1,
    )

    assert _metric_outcomes(outcomes, "target_failure") == [1] * 56 + [0] * 2
    assert target["nu_hat"] == pytest.approx(56 / 58)
    assert target["lower_bound"] == pytest.approx(0.748153750952841)
    assert target["certified"] is True
    assert strict["nu_hat"] == pytest.approx(21 / 58)
    assert strict["lower_bound"] == pytest.approx(0.14470547440111678)
    assert strict["certified"] is False


def test_unknown_certification_metric_is_rejected():
    with pytest.raises(ValueError, match="certification_metric"):
        _metric_outcomes([], "weighted_posthoc")


def test_adaptive_schedule_and_actions_preserve_predeclared_looks():
    assert _adaptive_schedule(20, 10, 60) == [20, 30, 40, 50, 60]
    assert _adaptive_schedule(20, 15, 60) == [20, 35, 50, 60]
    assert _adaptive_action(
        {"nu_hat": 0.9, "confidence_radius": 0.3, "lower_bound": 0.6},
        tau_v=0.5,
    ) == ("certify", 1.0)
    assert _adaptive_action(
        {"nu_hat": 0.1, "confidence_radius": 0.2, "lower_bound": -0.1},
        tau_v=0.5,
    ) == ("reject", pytest.approx(0.3))
    assert _adaptive_action(
        {"nu_hat": 0.6, "confidence_radius": 0.2, "lower_bound": 0.4},
        tau_v=0.5,
    ) == ("continue", pytest.approx(0.8))


def test_multiple_adaptive_looks_make_hoeffding_no_less_conservative():
    outcomes = [
        {
            "target_failure_observed": True,
            "z": 1,
        }
        for _ in range(20)
    ]
    one_look = _verification_statistic(
        "h1", outcomes, "target_failure", delta=0.05, tau_v=0.5, K=5, T=1
    )
    five_looks = _verification_statistic(
        "h1", outcomes, "target_failure", delta=0.05, tau_v=0.5, K=5, T=5
    )

    assert five_looks["confidence_radius"] > one_look["confidence_radius"]
    assert five_looks["lower_bound"] < one_look["lower_bound"]
