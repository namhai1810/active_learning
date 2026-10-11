"""Gold-blind error-probability scoring for discovery sampling."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.teacher.base import TeacherCall, TeacherLLM


ERROR_SCORING_PROMPT_VERSION = "discovery_error_probability_v1"
_SCORER_INPUT_FIELDS = ("id", "premise", "hypothesis", "prediction", "confidence")


@dataclass(frozen=True)
class ErrorScoringConfig:
    max_records_per_call: int = 40
    seed: int = 42
    prompt_version: str = ERROR_SCORING_PROMPT_VERSION
    source: str = "teacher_llm_critic"

    def __post_init__(self) -> None:
        if self.max_records_per_call <= 0:
            raise ValueError("max_records_per_call must be positive")
        if not self.prompt_version.strip() or not self.source.strip():
            raise ValueError("Error scorer prompt_version and source must be non-empty")


@dataclass(frozen=True)
class ErrorScoreBatch:
    scores: dict[str, float]
    method: str
    source: str
    logs: tuple[dict[str, Any], ...]


class ErrorProbabilityScorer(Protocol):
    """Estimate P(student prediction is wrong | x, predicted label).

    Implementations must not receive gold labels, correctness, benchmark
    taxonomy metadata, or records outside hypothesis_discovery.
    """

    def score(self, records: Sequence[Mapping[str, Any]]) -> ErrorScoreBatch: ...


def _scorer_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("id", "premise", "hypothesis", "prediction")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Error-scoring record is missing fields: {missing}")
    result = {
        field: record[field]
        for field in _SCORER_INPUT_FIELDS
        if field in record and record[field] is not None
    }
    example_id = str(result["id"])
    if not example_id:
        raise ValueError("Error-scoring records require non-empty IDs")
    result["id"] = example_id
    for field in ("premise", "hypothesis", "prediction"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"Error-scoring field {field!r} must be non-empty text")
    if "confidence" in result:
        confidence = result["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("Student confidence must be numeric when supplied")
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Student confidence must be finite and in [0, 1]")
        result["confidence"] = confidence
    return result


class TeacherErrorProbabilityScorer:
    """Use an explicit LLM critic to estimate calibrated error likelihoods."""

    method = "llm_critic_error_probability"

    def __init__(self, teacher: TeacherLLM, config: ErrorScoringConfig) -> None:
        self.teacher = teacher
        self.config = config

    def score(self, records: Sequence[Mapping[str, Any]]) -> ErrorScoreBatch:
        sanitized = [_scorer_record(record) for record in records]
        sanitized.sort(key=lambda row: str(row["id"]))
        ids = [str(record["id"]) for record in sanitized]
        if len(ids) != len(set(ids)):
            raise ValueError("Error scoring requires unique example IDs")
        scores: dict[str, float] = {}
        logs: list[dict[str, Any]] = []
        for batch_index, start in enumerate(
            range(0, len(sanitized), self.config.max_records_per_call)
        ):
            batch = sanitized[start : start + self.config.max_records_per_call]
            system_prompt = (
                "You are an NLI prediction critic. For each item estimate the "
                "probability that the supplied student prediction is wrong, using "
                "only the premise, hypothesis, prediction, and optional student "
                "confidence. Confidence is only one signal: do not equate it with "
                "correctness, entropy, or error probability. You are not given gold "
                "labels or benchmark metadata. Return one standalone JSON object "
                "with no markdown."
            )
            payload = {
                "task": "estimate_student_error_probability",
                "probability_definition": (
                    "P(student prediction is wrong | premise, hypothesis, "
                    "student prediction, permitted confidence)"
                ),
                "output_schema": {
                    "scores": [
                        {
                            "example_id": "one supplied ID",
                            "error_probability": "number in [0, 1]",
                            "rationale": "short explanation independent of any gold label",
                        }
                    ]
                },
                "records": batch,
            }
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.config.seed}:error-scorer:{batch_index}".encode("utf-8")
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task="discovery_error_probability_scoring",
                prompt_version=self.config.prompt_version,
                seed=derived_seed,
                system_prompt=system_prompt,
                payload=payload,
            )
            response = self.teacher.generate_json(call)
            if set(response) != {"scores"} or not isinstance(response["scores"], list):
                raise ValueError("Error scorer response must contain only scores[]")
            expected_ids = {str(record["id"]) for record in batch}
            returned_ids: set[str] = set()
            for raw in response["scores"]:
                if not isinstance(raw, dict) or set(raw) != {
                    "example_id",
                    "error_probability",
                    "rationale",
                }:
                    raise ValueError("Error scorer item violates its declared schema")
                example_id = str(raw["example_id"])
                probability = raw["error_probability"]
                if example_id in returned_ids or example_id not in expected_ids:
                    raise ValueError("Error scorer returned an unknown or duplicate ID")
                if isinstance(probability, bool) or not isinstance(
                    probability, (int, float)
                ):
                    raise ValueError("Error probability must be numeric")
                probability = float(probability)
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("Error probability must be finite and in [0, 1]")
                if not isinstance(raw["rationale"], str):
                    raise ValueError("Error-score rationale must be a string")
                returned_ids.add(example_id)
                scores[example_id] = probability
            if returned_ids != expected_ids:
                raise ValueError("Error scorer must return exactly one score per input ID")
            logs.append(
                {
                    "scoring_method": self.method,
                    "scoring_source": self.config.source,
                    "teacher_call": call.to_dict(),
                    "teacher_response": response,
                }
            )
        if set(scores) != set(ids):
            raise ValueError("Error scorer did not score the complete discovery pool")
        return ErrorScoreBatch(
            scores=scores,
            method=self.method,
            source=self.config.source,
            logs=tuple(logs),
        )
