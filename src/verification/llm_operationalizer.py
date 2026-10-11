"""Outcome-independent LLM operationalization of generated hypotheses."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.hypotheses.types import FailureHypothesis
from src.teacher.base import TeacherCall, TeacherLLM, teacher_exchange_log


OPERATIONALIZATION_PROMPT_VERSION = "hypothesis_operationalization_v1"
_PROBE_FIELDS = ("id", "premise", "hypothesis", "gold")


@dataclass(frozen=True)
class OperationalizationConfig:
    max_records_per_call: int = 50
    seed: int = 42
    prompt_version: str = OPERATIONALIZATION_PROMPT_VERSION

    def __post_init__(self) -> None:
        if self.max_records_per_call <= 0:
            raise ValueError("max_records_per_call must be positive")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must be non-empty")


@dataclass(frozen=True)
class OperationalizationResult:
    hypothesis_id: str
    testable: bool
    reason: str
    applicable_ids: tuple[str, ...]
    logs: tuple[dict[str, Any], ...]


def _sanitize_probe(record: Mapping[str, Any], expected_role: str) -> dict[str, Any]:
    if record.get("split_role") != expected_role:
        raise ValueError(
            f"Operationalization expected split_role={expected_role!r}"
        )
    missing = [field for field in _PROBE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Probe record is missing fields: {missing}")
    # Student predictions, correctness, HANS heuristic, and subcase are
    # deliberately omitted so applicability cannot depend on the outcome or
    # hidden benchmark metadata.
    return {field: record[field] for field in _PROBE_FIELDS}


class LLMHypothesisOperationalizer:
    def __init__(self, teacher: TeacherLLM, config: OperationalizationConfig) -> None:
        self.teacher = teacher
        self.config = config

    def operationalize(
        self,
        hypothesis: FailureHypothesis,
        records: Sequence[Mapping[str, Any]],
        *,
        expected_role: str,
    ) -> OperationalizationResult:
        hypothesis.validate(require_llm_schema=True)
        sanitized = [_sanitize_probe(record, expected_role) for record in records]
        ids = [str(record["id"]) for record in sanitized]
        if len(ids) != len(set(ids)):
            raise ValueError("Operationalization records require unique IDs")
        if not sanitized:
            return OperationalizationResult(
                hypothesis_id=hypothesis.id,
                testable=False,
                reason="no_records_in_requested_role",
                applicable_ids=(),
                logs=(),
            )

        applicable: list[str] = []
        logs: list[dict[str, Any]] = []
        for batch_index, start in enumerate(
            range(0, len(sanitized), self.config.max_records_per_call)
        ):
            batch = sanitized[start : start + self.config.max_records_per_call]
            system_prompt = (
                "You operationalize a pre-registered NLI failure hypothesis. "
                "For each supplied pair, decide only whether the pair satisfies "
                "the stated applicability criteria. Do not predict the student, "
                "infer benchmark categories, or use any information not supplied. "
                "Return one standalone JSON object and no markdown."
            )
            payload = {
                "task": "outcome_independent_hypothesis_applicability",
                "hypothesis": {
                    "id": hypothesis.id,
                    "description": hypothesis.description,
                    "predicted_failure_behavior": hypothesis.predicted_failure_behavior,
                    "verification_criteria": list(hypothesis.verification_criteria),
                },
                "output_schema": {
                    "testable": "boolean",
                    "reason": "short string",
                    "decisions": [
                        {
                            "example_id": "one supplied ID",
                            "applicable": "boolean",
                            "reason": "short string",
                        }
                    ],
                },
                "records": batch,
            }
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.config.seed}:{hypothesis.id}:{expected_role}:{batch_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task=f"operationalize_{expected_role}",
                prompt_version=self.config.prompt_version,
                seed=derived_seed,
                system_prompt=system_prompt,
                payload=payload,
            )
            response = self.teacher.generate_json(call)
            if set(response) != {"testable", "reason", "decisions"}:
                raise ValueError("Operationalization response has unexpected fields")
            if not isinstance(response["testable"], bool):
                raise ValueError("Operationalization testable field must be boolean")
            if not isinstance(response["reason"], str):
                raise ValueError("Operationalization reason must be a string")
            decisions = response["decisions"]
            if not isinstance(decisions, list):
                raise ValueError("Operationalization decisions must be a list")
            expected_ids = [str(record["id"]) for record in batch]
            returned: dict[str, bool] = {}
            for decision in decisions:
                if not isinstance(decision, dict) or set(decision) != {
                    "example_id",
                    "applicable",
                    "reason",
                }:
                    raise ValueError("Operationalization decision violates its schema")
                example_id = str(decision["example_id"])
                if example_id in returned or not isinstance(
                    decision["applicable"], bool
                ):
                    raise ValueError("Invalid or duplicate operationalization decision")
                if not isinstance(decision["reason"], str):
                    raise ValueError("Decision reason must be a string")
                returned[example_id] = decision["applicable"]
            if set(returned) != set(expected_ids):
                raise ValueError(
                    "Operationalization must return exactly one decision per input ID"
                )
            logs.append(
                {
                    "hypothesis_id": hypothesis.id,
                    "split_role": expected_role,
                    **teacher_exchange_log(self.teacher, call, response),
                }
            )
            if not response["testable"]:
                return OperationalizationResult(
                    hypothesis_id=hypothesis.id,
                    testable=False,
                    reason=response["reason"] or "teacher_marked_untestable",
                    applicable_ids=(),
                    logs=tuple(logs),
                )
            applicable.extend(
                example_id for example_id in expected_ids if returned[example_id]
            )
        return OperationalizationResult(
            hypothesis_id=hypothesis.id,
            testable=True,
            reason="operationalized",
            applicable_ids=tuple(applicable),
            logs=tuple(logs),
        )


def select_probe_ids(
    hypothesis_id: str,
    applicable_ids: Sequence[str],
    *,
    n_probes: int,
    seed: int,
) -> list[str]:
    if n_probes <= 0:
        raise ValueError("n_probes must be positive")
    if len(set(applicable_ids)) != len(applicable_ids):
        raise ValueError("Applicable probe IDs must be unique")
    return sorted(
        applicable_ids,
        key=lambda example_id: (
            hashlib.sha256(
                f"{seed}:{hypothesis_id}:{example_id}".encode("utf-8")
            ).digest(),
            example_id,
        ),
    )[:n_probes]


def build_measurable_outcomes(
    hypothesis: FailureHypothesis,
    records: Sequence[Mapping[str, Any]],
    selected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Measure whether the predeclared student failure occurred on each probe."""
    indexed = {str(record["id"]): record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Verification records require unique IDs")
    outcomes = []
    for example_id in selected_ids:
        if example_id not in indexed:
            raise ValueError(f"Selected probe is absent: {example_id}")
        record = indexed[example_id]
        prediction = str(record.get("prediction", ""))
        correct = record.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("Verification outcome requires boolean correctness")
        z = int((not correct) and prediction in hypothesis.predicted_failure_labels)
        outcomes.append(
            {
                "hypothesis_id": hypothesis.id,
                "example_id": example_id,
                "split_role": "verification",
                "prediction": prediction,
                "gold": record.get("gold"),
                "correct": correct,
                "predicted_failure_labels": list(
                    hypothesis.predicted_failure_labels
                ),
                "z": z,
            }
        )
    return outcomes
