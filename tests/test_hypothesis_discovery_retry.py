from src.hypotheses.error_scorer import ErrorScoreBatch
from src.hypotheses.llm_generator import (
    LLMDiscoveryConfig,
    LLMFailureHypothesisGenerator,
)


class QueueTeacher:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, call):
        self.calls.append(call)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class StaticScorer:
    def score(self, records):
        return ErrorScoreBatch(
            scores={str(row["id"]): 0.9 for row in records},
            method="test",
            source="test",
            logs=(),
        )


def _rows():
    return [
        {
            "id": "d1",
            "premise": "The doctor visited the lawyer.",
            "hypothesis": "The lawyer visited the doctor.",
            "gold": "non-entailment",
            "prediction": "entailment",
            "correct": False,
            "confidence": 0.9,
            "split_role": "hypothesis_discovery",
        },
        {
            "id": "d2",
            "premise": "A runner moved.",
            "hypothesis": "A person moved.",
            "gold": "entailment",
            "prediction": "entailment",
            "correct": True,
            "confidence": 0.8,
            "split_role": "hypothesis_discovery",
        },
    ]


def _valid_response():
    return {
        "hypotheses": [
            {
                "name": "Role reversal",
                "description": "The student confuses agent and patient roles.",
                "predicted_failure_behavior": "Predicts entailment after role swaps.",
                "predicted_failure_labels": ["entailment"],
                "verification_criteria": [
                    "Premise and hypothesis exchange agent and patient."
                ],
                "intervention_target": "Controlled role-reversal NLI pairs.",
                "support_example_ids": ["d1"],
            }
        ]
    }


def test_hypothesis_discovery_retries_invalid_json_with_new_request():
    teacher = QueueTeacher(
        [ValueError("Teacher response is not a valid standalone JSON value"), _valid_response()]
    )
    generator = LLMFailureHypothesisGenerator(
        teacher,
        LLMDiscoveryConfig(max_hypotheses=1, max_attempts_per_call=2),
        StaticScorer(),
    )

    hypotheses, _, log = generator.generate(_rows())

    assert len(hypotheses) == 1
    assert len(teacher.calls) == 2
    assert teacher.calls[0].request_id != teacher.calls[1].request_id
    assert "retry_correction" not in teacher.calls[0].payload
    assert "retry_correction" in teacher.calls[1].payload
    assert len(log["hypothesis_discovery_attempts"]) == 2
    assert log["hypothesis_discovery_attempts"][0]["validation_error"].startswith(
        "Teacher response is not"
    )


def test_hypothesis_discovery_deduplicates_support_ids_without_retry():
    response = _valid_response()
    response["hypotheses"][0]["support_example_ids"] = ["d1", "d1"]
    teacher = QueueTeacher([response])
    generator = LLMFailureHypothesisGenerator(
        teacher,
        LLMDiscoveryConfig(max_hypotheses=1, max_attempts_per_call=3),
        StaticScorer(),
    )

    hypotheses, evidence, log = generator.generate(_rows())

    assert hypotheses[0].support_example_ids == ("d1",)
    assert len(teacher.calls) == 1
    assert len(log["hypothesis_discovery_attempts"]) == 1
    assert evidence["response_normalizations"] == [
        {
            "candidate_index": 0,
            "field": "support_example_ids",
            "removed_duplicate_count": 1,
        }
    ]


def test_hypothesis_discovery_retries_near_duplicate_candidates():
    duplicate_response = {
        "hypotheses": [
            _valid_response()["hypotheses"][0],
            {
                **_valid_response()["hypotheses"][0],
                "name": "Role reversal errors",
            },
        ]
    }
    diverse_response = {
        "hypotheses": [
            _valid_response()["hypotheses"][0],
            {
                "name": "Negation blindness",
                "description": "The student ignores explicit sentential negation.",
                "predicted_failure_behavior": "Predicts entailment despite negation.",
                "predicted_failure_labels": ["entailment"],
                "verification_criteria": [
                    "One sentence explicitly negates the proposition in the other."
                ],
                "intervention_target": "Pairs contrasting affirmed and negated events.",
                "support_example_ids": ["d1"],
            },
        ]
    }
    teacher = QueueTeacher([duplicate_response, diverse_response])
    generator = LLMFailureHypothesisGenerator(
        teacher,
        LLMDiscoveryConfig(
            max_hypotheses=2,
            min_hypotheses=2,
            max_candidate_token_jaccard=0.7,
            max_attempts_per_call=2,
            prompt_version="failure_discovery_v3",
        ),
        StaticScorer(),
    )

    hypotheses, _, log = generator.generate(_rows())

    assert len(hypotheses) == 2
    assert len(teacher.calls) == 2
    assert "near-duplicate hypotheses" in log["hypothesis_discovery_attempts"][0][
        "validation_error"
    ]
    assert "retry_correction" in teacher.calls[1].payload
