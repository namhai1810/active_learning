"""HANS-metadata oracle for candidate generation in debug mode only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.data.hans import HANS_HEURISTICS
from src.hypotheses.types import FailureHypothesis

HANS_ORACLE_HYPOTHESES: dict[str, FailureHypothesis] = {
    "lexical_overlap": FailureHypothesis(
        id="h_lexical_overlap",
        name="Lexical overlap failure",
        description=(
            "The student may rely excessively on lexical overlap between "
            "premise and hypothesis when predicting entailment."
        ),
        heuristic="lexical_overlap",
    ),
    "subsequence": FailureHypothesis(
        id="h_subsequence",
        name="Subsequence failure",
        description=(
            "The student may rely excessively on whether the hypothesis is a "
            "contiguous subsequence of the premise when predicting entailment."
        ),
        heuristic="subsequence",
    ),
    "constituent": FailureHypothesis(
        id="h_constituent",
        name="Constituent failure",
        description=(
            "The student may rely excessively on whether the hypothesis is a "
            "syntactic constituent of the premise when predicting entailment."
        ),
        heuristic="constituent",
    ),
}


class HansOracleHypothesisGenerator:
    """Construct candidates from errors on diagnostic HANS records.

    This class performs no statistical verification. Evidence is descriptive
    only and must not be interpreted as certification of a failure.
    """

    def __init__(self, min_errors: int = 1) -> None:
        if (
            isinstance(min_errors, bool)
            or not isinstance(min_errors, int)
            or min_errors < 1
        ):
            raise ValueError("min_errors must be a positive integer")
        self.min_errors = min_errors

    def generate(
        self,
        predictions: Sequence[Mapping[str, Any]],
    ) -> tuple[list[FailureHypothesis], dict[str, dict[str, Any]]]:
        counts = {
            heuristic: {"num_examples": 0, "num_errors": 0}
            for heuristic in HANS_HEURISTICS
        }
        seen_ids: set[str] = set()
        for record in predictions:
            split_role = str(record.get("split_role", "")).strip().lower()
            if split_role != "hypothesis_discovery":
                raise ValueError(
                    "HANS oracle generation accepts only records explicitly "
                    "marked split_role='hypothesis_discovery'; received "
                    f"{split_role or '<missing>'!r}"
                )
            example_id = str(record.get("id", ""))
            if not example_id:
                raise ValueError("Diagnostic prediction is missing a stable id")
            if example_id in seen_ids:
                raise ValueError(f"Duplicate diagnostic example ID: {example_id}")
            seen_ids.add(example_id)

            heuristic = str(record.get("heuristic", "")).strip().lower()
            if heuristic not in counts:
                raise ValueError(f"Unknown HANS heuristic: {heuristic!r}")
            correct = record.get("correct")
            if not isinstance(correct, bool):
                raise ValueError(
                    f"Prediction {example_id!r} must have boolean correctness"
                )
            counts[heuristic]["num_examples"] += 1
            if not correct:
                counts[heuristic]["num_errors"] += 1

        hypotheses: list[FailureHypothesis] = []
        evidence: dict[str, dict[str, Any]] = {}
        for heuristic in HANS_HEURISTICS:
            hypothesis = HANS_ORACLE_HYPOTHESES[heuristic]
            num_examples = counts[heuristic]["num_examples"]
            num_errors = counts[heuristic]["num_errors"]
            error_rate = num_errors / num_examples if num_examples else 0.0
            candidate = num_errors >= self.min_errors
            if candidate:
                hypotheses.append(hypothesis)
            evidence[hypothesis.id] = {
                "hypothesis_id": hypothesis.id,
                "heuristic": heuristic,
                "num_examples": num_examples,
                "num_errors": num_errors,
                "error_rate": error_rate,
                "source": "hans_oracle_debug",
                "candidate": candidate,
            }
        return hypotheses, evidence
