"""LLM-based failure-hypothesis discovery for the primary research method."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.data.labels import HANS_LABELS
from src.hypotheses.error_scorer import ErrorProbabilityScorer, ErrorScoreBatch
from src.hypotheses.types import FailureHypothesis
from src.teacher.base import (
    TeacherCall,
    TeacherLLM,
    teacher_exchange_log,
    teacher_failure_log,
)


DISCOVERY_PROMPT_VERSION = "failure_discovery_v1"
STRICT_JSON_DISCOVERY_PROMPT_VERSION = "failure_discovery_v2"
DIVERSE_DISCOVERY_PROMPT_VERSION = "failure_discovery_v3"
_DISCOVERY_FIELDS = (
    "id",
    "premise",
    "hypothesis",
    "gold",
    "prediction",
    "correct",
    "confidence",
)
_CANDIDATE_FIELDS = {
    "name",
    "description",
    "predicted_failure_behavior",
    "predicted_failure_labels",
    "verification_criteria",
    "intervention_target",
    "support_example_ids",
}


@dataclass(frozen=True)
class LLMDiscoveryConfig:
    max_hypotheses: int = 5
    min_hypotheses: int = 1
    max_error_examples: int = 80
    max_contrast_examples: int = 20
    sampling_strategy: str = "exponential"
    sampling_alpha: float = 0.5
    sampling_beta: float = 10.0
    sampling_threshold: float = 0.7
    contrast_sampling_strategy: str = "highest_error_score"
    diversity_sampling: bool = False
    max_candidate_token_jaccard: float = 1.0
    max_attempts_per_call: int = 3
    seed: int = 42
    prompt_version: str = DISCOVERY_PROMPT_VERSION

    def __post_init__(self) -> None:
        if self.max_hypotheses <= 0:
            raise ValueError("max_hypotheses must be positive")
        if not 1 <= self.min_hypotheses <= self.max_hypotheses:
            raise ValueError("min_hypotheses must be in [1, max_hypotheses]")
        if self.max_error_examples <= 0:
            raise ValueError("max_error_examples must be positive")
        if self.max_contrast_examples < 0:
            raise ValueError("max_contrast_examples must be non-negative")
        if self.sampling_strategy not in {
            "normalization",
            "exponential",
            "threshold",
        }:
            raise ValueError(
                "sampling_strategy must be normalization, exponential, or threshold"
            )
        if not math.isfinite(self.sampling_alpha) or not 0.0 <= self.sampling_alpha <= 1.0:
            raise ValueError("sampling_alpha must be finite and in [0, 1]")
        if not math.isfinite(self.sampling_beta) or self.sampling_beta <= 0.0:
            raise ValueError("sampling_beta must be finite and positive")
        if (
            not math.isfinite(self.sampling_threshold)
            or not 0.0 <= self.sampling_threshold <= 1.0
        ):
            raise ValueError("sampling_threshold must be finite and in [0, 1]")
        if self.contrast_sampling_strategy not in {
            "highest_error_score",
            "same_as_errors",
        }:
            raise ValueError(
                "contrast_sampling_strategy must be highest_error_score or same_as_errors"
            )
        if self.max_attempts_per_call <= 0:
            raise ValueError("max_attempts_per_call must be positive")
        if not 0.0 < self.max_candidate_token_jaccard <= 1.0:
            raise ValueError("max_candidate_token_jaccard must be in (0, 1]")
        if not self.prompt_version.strip():
            raise ValueError("prompt_version must be non-empty")


def _stable_rank(record: Mapping[str, Any], seed: int, group: str) -> tuple[bytes, str]:
    example_id = str(record["id"])
    digest = hashlib.sha256(
        f"{seed}:{group}:{example_id}".encode("utf-8")
    ).digest()
    return digest, example_id


def _teacher_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Expose only approved discovery fields, never HANS metadata."""
    result = {
        field: record[field]
        for field in _DISCOVERY_FIELDS
        if field in record and record[field] is not None
    }
    required = ("id", "premise", "hypothesis", "gold", "prediction", "correct")
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"Discovery record is missing fields: {missing}")
    if not isinstance(result["correct"], bool):
        raise ValueError("Discovery record correctness must be boolean")
    return result


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    identity = json.dumps(
        {
            "name": candidate["name"],
            "description": candidate["description"],
            "behavior": candidate["predicted_failure_behavior"],
            "criteria": candidate["verification_criteria"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "llm_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _candidate_token_jaccard(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> float:
    fields = (
        "name",
        "description",
        "predicted_failure_behavior",
        "intervention_target",
    )
    left_tokens = _word_set(" ".join(str(left.get(field, "")) for field in fields))
    right_tokens = _word_set(" ".join(str(right.get(field, "")) for field in fields))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _discovery_diversity_bucket(record: Mapping[str, Any]) -> str:
    premise_tokens = _word_set(str(record.get("premise", "")))
    hypothesis_tokens = _word_set(str(record.get("hypothesis", "")))
    union = premise_tokens | hypothesis_tokens
    overlap = len(premise_tokens & hypothesis_tokens) / len(union) if union else 0.0
    overlap_band = "low" if overlap < 0.34 else "mid" if overlap < 0.67 else "high"
    length_delta = len(premise_tokens) - len(hypothesis_tokens)
    length_band = (
        "premise_longer"
        if length_delta >= 3
        else "hypothesis_longer"
        if length_delta <= -3
        else "similar_length"
    )
    return (
        f"{record.get('gold', 'unknown')}->{record.get('prediction', 'unknown')}:"
        f"{overlap_band}:{length_band}"
    )


class LLMFailureHypothesisGenerator:
    """Ask a teacher to infer testable patterns from discovery-only errors."""

    def __init__(
        self,
        teacher: TeacherLLM,
        config: LLMDiscoveryConfig,
        error_scorer: ErrorProbabilityScorer,
    ) -> None:
        self.teacher = teacher
        self.config = config
        self.error_scorer = error_scorer

    def _weights(
        self, rows: Sequence[Mapping[str, Any]], scores: Mapping[str, float]
    ) -> list[float]:
        values = [scores[str(row["id"])] for row in rows]
        if self.config.sampling_strategy == "normalization":
            total = sum(values)
            if total <= 0.0:
                raise ValueError(
                    "Normalization sampling requires at least one positive error score"
                )
            return [value / total for value in values]
        if self.config.sampling_strategy == "exponential":
            weights = []
            for value in values:
                offset = self.config.sampling_beta * (
                    value - self.config.sampling_alpha
                )
                if offset >= 0:
                    weights.append(1.0 / (1.0 + math.exp(-offset)))
                else:
                    exp_offset = math.exp(offset)
                    weights.append(exp_offset / (1.0 + exp_offset))
            return weights
        raise RuntimeError("Threshold sampling does not use stochastic weights")

    def _sample_group(
        self,
        rows: Sequence[dict[str, Any]],
        scores: Mapping[str, float],
        *,
        budget: int,
        group: str,
    ) -> list[dict[str, Any]]:
        if budget <= 0 or not rows:
            return []
        canonical = sorted(rows, key=lambda row: str(row["id"]))
        if self.config.sampling_strategy == "threshold":
            eligible = [
                row
                for row in canonical
                if scores[str(row["id"])] >= self.config.sampling_threshold
            ]
            return sorted(
                eligible,
                key=lambda row: (
                    -scores[str(row["id"])],
                    _stable_rank(row, self.config.seed, group),
                ),
            )[:budget]

        weights = self._weights(canonical, scores)
        derived_seed = int.from_bytes(
            hashlib.sha256(
                f"{self.config.seed}:{group}:{self.config.sampling_strategy}".encode(
                    "utf-8"
                )
            ).digest()[:8],
            "big",
        )
        rng = random.Random(derived_seed)
        priority = []
        for row, weight in zip(canonical, weights, strict=True):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("Sampling weights must be finite and non-negative")
            if weight == 0.0:
                key = math.inf
            else:
                uniform = max(rng.random(), 1e-300)
                key = -math.log(uniform) / weight
            priority.append((key, _stable_rank(row, self.config.seed, group), row))
        priority.sort(key=lambda item: (item[0], item[1]))
        positive = [item for item in priority if math.isfinite(item[0])]
        if not self.config.diversity_sampling or group != "error":
            return [item[2] for item in positive[: min(budget, len(positive))]]

        # Spread the evidence budget over observable NLI-error shapes without
        # exposing benchmark taxonomy. Ranking inside every bucket still uses
        # the configured teacher-error score and deterministic weighted key.
        buckets: dict[str, list[dict[str, Any]]] = {}
        for _, _, row in positive:
            buckets.setdefault(_discovery_diversity_bucket(row), []).append(row)
        selected: list[dict[str, Any]] = []
        ordered_buckets = sorted(buckets)
        while len(selected) < min(budget, len(positive)):
            progressed = False
            for bucket in ordered_buckets:
                if buckets[bucket]:
                    selected.append(buckets[bucket].pop(0))
                    progressed = True
                    if len(selected) == min(budget, len(positive)):
                        break
            if not progressed:
                break
        return selected

    def _select_records(
        self, predictions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], ErrorScoreBatch]:
        seen: set[str] = set()
        errors: list[dict[str, Any]] = []
        correct: list[dict[str, Any]] = []
        for record in predictions:
            if record.get("split_role") != "hypothesis_discovery":
                raise ValueError(
                    "LLM discovery accepts only split_role='hypothesis_discovery'"
                )
            sanitized = _teacher_record(record)
            example_id = str(sanitized["id"])
            if not example_id or example_id in seen:
                raise ValueError(f"Missing or duplicate discovery ID: {example_id!r}")
            seen.add(example_id)
            (correct if sanitized["correct"] else errors).append(sanitized)
        scorer_inputs = [
            {
                field: row[field]
                for field in ("id", "premise", "hypothesis", "prediction", "confidence")
                if field in row
            }
            for row in (*errors, *correct)
        ]
        scored = self.error_scorer.score(scorer_inputs)
        expected_ids = {str(row["id"]) for row in (*errors, *correct)}
        if set(scored.scores) != expected_ids:
            raise ValueError("Error scorer must return exactly the discovery IDs")
        for example_id, score in scored.scores.items():
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0.0 <= score <= 1.0
            ):
                raise ValueError(
                    f"Invalid estimated error probability for {example_id!r}: {score!r}"
                )
        selected_errors = self._sample_group(
            errors,
            scored.scores,
            budget=self.config.max_error_examples,
            group="error",
        )
        if self.config.contrast_sampling_strategy == "highest_error_score":
            selected_correct = sorted(
                correct,
                key=lambda row: (
                    -scored.scores[str(row["id"])],
                    _stable_rank(row, self.config.seed, "contrast"),
                ),
            )[: self.config.max_contrast_examples]
        else:
            selected_correct = self._sample_group(
                correct,
                scored.scores,
                budget=self.config.max_contrast_examples,
                group="contrast",
            )
        selected_ids = [str(row["id"]) for row in (*selected_errors, *selected_correct)]
        if len(selected_ids) != len(set(selected_ids)):
            raise RuntimeError("Discovery sampling selected duplicate IDs")
        selection_log = {
            "scoring_method": scored.method,
            "scoring_source": scored.source,
            "sampling_strategy": self.config.sampling_strategy,
            "sampling_parameters": {
                "alpha": self.config.sampling_alpha,
                "beta": self.config.sampling_beta,
                "threshold": self.config.sampling_threshold,
                "threshold_fill_policy": "none",
                "contrast_sampling_strategy": self.config.contrast_sampling_strategy,
                "diversity_sampling": self.config.diversity_sampling,
            },
            "total_eligible_examples": len(errors) + len(correct),
            "eligible_error_examples": len(errors),
            "eligible_correct_controls": len(correct),
            "selection_budget": {
                "errors": self.config.max_error_examples,
                "contrasts": self.config.max_contrast_examples,
            },
            "selected_error_ids": [str(row["id"]) for row in selected_errors],
            "selected_error_scores": [
                scored.scores[str(row["id"])] for row in selected_errors
            ],
            "selected_error_diversity_buckets": {
                bucket: sum(
                    _discovery_diversity_bucket(row) == bucket
                    for row in selected_errors
                )
                for bucket in sorted(
                    {_discovery_diversity_bucket(row) for row in selected_errors}
                )
            },
            "selected_contrast_ids": [str(row["id"]) for row in selected_correct],
            "selected_contrast_scores": [
                scored.scores[str(row["id"])] for row in selected_correct
            ],
            "seed": self.config.seed,
            "prompt_version": self.config.prompt_version,
        }
        return selected_errors, selected_correct, selection_log, scored

    def generate(
        self, predictions: Sequence[Mapping[str, Any]]
    ) -> tuple[list[FailureHypothesis], dict[str, Any], dict[str, Any]]:
        errors, contrasts, selection_log, scored = self._select_records(predictions)
        if not errors:
            return [], {
                "status": "no_discovery_errors",
                "num_errors_presented": 0,
                "num_contrasts_presented": len(contrasts),
                "selection": selection_log,
            }, {
                "scoring_calls": list(scored.logs),
                "teacher_call": None,
                "teacher_response": None,
            }

        system_prompt = (
            "You are a research analyst diagnosing an NLI student's failures. "
            "Infer compact, falsifiable failure hypotheses only from the supplied "
            "examples. Do not assume benchmark categories or external annotations. "
            "Each verification criterion must be decidable from a premise and "
            "hypothesis without seeing the student's prediction. Return exactly one "
            "JSON object matching the requested schema, with no markdown."
        )
        payload = {
            "task": "discover_testable_nli_failure_hypotheses",
            "max_hypotheses": self.config.max_hypotheses,
            "allowed_predicted_failure_labels": list(HANS_LABELS),
            "output_schema": {
                "hypotheses": [
                    {
                        "name": "non-empty string",
                        "description": "non-empty string",
                        "predicted_failure_behavior": "measurable behavior",
                        "predicted_failure_labels": ["one or more allowed labels"],
                        "verification_criteria": [
                            "one or more outcome-independent applicability rules"
                        ],
                        "intervention_target": "what new NLI pairs should exercise",
                        "support_example_ids": ["IDs from supplied examples"],
                    }
                ]
            },
            "incorrect_examples": errors,
            "contrasting_correct_examples": contrasts,
        }
        if self.config.prompt_version in {
            STRICT_JSON_DISCOVERY_PROMPT_VERSION,
            DIVERSE_DISCOVERY_PROMPT_VERSION,
        }:
            payload["response_constraints"] = {
                "standalone_json_only": True,
                "max_verification_criteria_per_hypothesis": 3,
                "max_support_ids_per_hypothesis": 5,
                "concise_text": True,
            }
        if self.config.prompt_version == DIVERSE_DISCOVERY_PROMPT_VERSION:
            payload["diversity_requirements"] = {
                "requested_hypotheses": self.config.max_hypotheses,
                "minimum_distinct_hypotheses": self.config.min_hypotheses,
                "one_mechanism_per_hypothesis": True,
                "avoid_paraphrased_or_nested_failure_mechanisms": True,
                "cover_distinct_error_shapes_from_the_supplied_evidence": True,
            }
        presented_ids = {str(row["id"]) for row in (*errors, *contrasts)}
        discovery_attempts: list[dict[str, Any]] = []
        last_error: ValueError | None = None
        call: TeacherCall | None = None
        response: dict[str, Any] | None = None
        hypotheses: list[FailureHypothesis] | None = None
        support_id_normalizations: list[dict[str, Any]] = []
        for attempt in range(self.config.max_attempts_per_call):
            attempt_payload = dict(payload)
            if last_error is not None:
                diversity_retry = (
                    f" Return between {self.config.min_hypotheses} and "
                    f"{self.config.max_hypotheses} distinct hypotheses, with no "
                    "paraphrased or nested versions of the same mechanism."
                    if self.config.prompt_version
                    == DIVERSE_DISCOVERY_PROMPT_VERSION
                    else ""
                )
                attempt_payload["retry_correction"] = {
                    "previous_response_rejected_because": str(last_error),
                    "mandatory_instruction": (
                        "Return exactly one standalone JSON object with only the key "
                        "hypotheses. Close every string, array, and object; emit no "
                        "markdown or comments. Keep each hypothesis concise: at most "
                        "three verification criteria and five support IDs."
                        + diversity_retry
                    ),
                }
            if attempt == 0:
                derived_seed = self.config.seed
            else:
                derived_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{self.config.seed}:hypothesis-discovery:attempt:{attempt}".encode(
                            "utf-8"
                        )
                    ).digest()[:4],
                    "big",
                )
            call = TeacherCall(
                task="hypothesis_discovery",
                prompt_version=self.config.prompt_version,
                seed=derived_seed,
                system_prompt=system_prompt,
                payload=attempt_payload,
            )
            response = None
            try:
                response = self.teacher.generate_json(call)
                if set(response) != {"hypotheses"} or not isinstance(
                    response["hypotheses"], list
                ):
                    raise ValueError(
                        "Teacher discovery response must contain only hypotheses[]"
                    )
                if len(response["hypotheses"]) > self.config.max_hypotheses:
                    raise ValueError("Teacher returned more hypotheses than configured")
                if len(response["hypotheses"]) < self.config.min_hypotheses:
                    raise ValueError(
                        "Teacher returned fewer hypotheses than the configured minimum"
                    )

                parsed: list[FailureHypothesis] = []
                attempt_normalizations: list[dict[str, Any]] = []
                for candidate_index, raw in enumerate(response["hypotheses"]):
                    if not isinstance(raw, dict) or set(raw) != _CANDIDATE_FIELDS:
                        raise ValueError(
                            "Each teacher hypothesis must exactly match the declared schema"
                        )
                    raw_support_value = raw["support_example_ids"]
                    if not isinstance(raw_support_value, (list, tuple)) or any(
                        not isinstance(item, str) for item in raw_support_value
                    ):
                        raise ValueError(
                            "Main-method hypothesis support_example_ids must be a "
                            "string list"
                        )
                    raw_support_ids = tuple(raw_support_value)
                    # Repeating an otherwise valid evidence ID is a harmless
                    # serialization defect, not a different scientific claim.
                    # Normalize it here while keeping FailureHypothesis strict for
                    # artifacts produced by every other caller.
                    support_ids = tuple(dict.fromkeys(raw_support_ids))
                    if not set(support_ids) <= presented_ids:
                        raise ValueError(
                            "Hypothesis cites an example not shown to the teacher"
                        )
                    normalized_raw = {
                        **raw,
                        "support_example_ids": list(support_ids),
                    }
                    if len(support_ids) != len(raw_support_ids):
                        attempt_normalizations.append(
                            {
                                "candidate_index": candidate_index,
                                "field": "support_example_ids",
                                "removed_duplicate_count": len(raw_support_ids)
                                - len(support_ids),
                            }
                        )
                    value = {
                        "id": _candidate_id(normalized_raw),
                        **normalized_raw,
                        "heuristic": None,
                        "source": "llm_teacher",
                        "prompt_version": self.config.prompt_version,
                    }
                    parsed.append(
                        FailureHypothesis.from_mapping(
                            value, require_llm_schema=True
                        )
                    )
                if len({item.id for item in parsed}) != len(parsed):
                    raise ValueError("Teacher returned duplicate hypotheses")
                for left_index, left in enumerate(response["hypotheses"]):
                    for right_index in range(left_index + 1, len(response["hypotheses"])):
                        similarity = _candidate_token_jaccard(
                            left, response["hypotheses"][right_index]
                        )
                        if similarity > self.config.max_candidate_token_jaccard:
                            raise ValueError(
                                "Teacher returned near-duplicate hypotheses at indices "
                                f"{left_index} and {right_index}: token_jaccard="
                                f"{similarity:.3f} exceeds "
                                f"{self.config.max_candidate_token_jaccard:.3f}"
                            )
            except ValueError as exc:
                discovery_attempts.append(
                    teacher_failure_log(self.teacher, call, exc, response)
                )
                last_error = exc
                if attempt + 1 == self.config.max_attempts_per_call:
                    raise ValueError(
                        "Hypothesis discovery response remained invalid after "
                        f"{self.config.max_attempts_per_call} attempts"
                    ) from exc
                continue
            hypotheses = parsed
            support_id_normalizations = attempt_normalizations
            discovery_attempts.append(
                teacher_exchange_log(self.teacher, call, response)
            )
            break
        if call is None or response is None or hypotheses is None:
            raise RuntimeError("Hypothesis discovery retry loop ended without a result")
        evidence = {
            "status": "generated",
            "num_errors_presented": len(errors),
            "num_contrasts_presented": len(contrasts),
            "presented_error_ids": [str(row["id"]) for row in errors],
            "presented_contrast_ids": [str(row["id"]) for row in contrasts],
            "candidate_support_ids": {
                item.id: list(item.support_example_ids) for item in hypotheses
            },
            "response_normalizations": support_id_normalizations,
            "selection": selection_log,
        }
        log = {
            "scoring_calls": list(scored.logs),
            "hypothesis_discovery_attempts": discovery_attempts,
            "selection": selection_log,
            **teacher_exchange_log(self.teacher, call, response),
        }
        return hypotheses, evidence, log
