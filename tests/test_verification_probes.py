import copy

import pytest

from src.hypotheses.types import FailureHypothesis
from src.verification.generated_probes import (
    _normalized_pair_text,
    _parse_generated_probe_batch,
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
