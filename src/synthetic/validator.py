"""Deterministic and teacher-assisted quality control for synthetic NLI data."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.hypotheses.types import FailureHypothesis
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample
from src.teacher.base import TeacherCall, TeacherLLM, teacher_exchange_log


QUALITY_PROMPT_VERSION = "synthetic_pair_quality_control_v2"
_TOKEN = re.compile(r"[\w']+", re.UNICODE)


def _normalized(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def _pair_text(premise: str, hypothesis: str) -> str:
    return _normalized(premise) + " [pair] " + _normalized(hypothesis)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


@dataclass(frozen=True)
class LeakageReference:
    role: str
    example_id: str
    exact_sha256: str
    token_set: frozenset[str]


class LeakageIndex:
    """Comparison signatures; raw held-out text is never sent to a teacher."""

    def __init__(self, references: Sequence[LeakageReference]) -> None:
        self.references = tuple(references)

    @classmethod
    def from_records(
        cls, records: Sequence[Mapping[str, Any]]
    ) -> "LeakageIndex":
        allowed_roles = {
            "hypothesis_discovery",
            "verification",
            "verification_probe",
            "gain_eval",
            "intervention_pool",
        }
        references = []
        seen: set[tuple[str, str]] = set()
        for record in records:
            role = str(record.get("split_role", ""))
            if role not in allowed_roles:
                raise ValueError(f"Unsupported leakage-reference role: {role!r}")
            example_id = str(record.get("id", ""))
            key = (role, example_id)
            if not example_id or key in seen:
                raise ValueError(f"Missing or duplicate leakage reference: {key}")
            seen.add(key)
            pair = _pair_text(str(record["premise"]), str(record["hypothesis"]))
            references.append(
                LeakageReference(
                    role=role,
                    example_id=example_id,
                    exact_sha256=hashlib.sha256(pair.encode("utf-8")).hexdigest(),
                    token_set=_tokens(pair),
                )
            )
        return cls(references)

    def find_overlap(
        self, example: SyntheticNLIExample, near_duplicate_threshold: float
    ) -> tuple[str, str] | None:
        pair = _pair_text(example.premise, example.hypothesis)
        exact = hashlib.sha256(pair.encode("utf-8")).hexdigest()
        tokens = _tokens(pair)
        for reference in self.references:
            if exact == reference.exact_sha256:
                return reference.role, f"exact:{reference.example_id}"
            if _jaccard(tokens, reference.token_set) >= near_duplicate_threshold:
                return reference.role, f"near:{reference.example_id}"
        return None


@dataclass(frozen=True)
class QualityDecision:
    example_id: str
    label_consistent: bool
    unambiguous: bool
    grammatical: bool
    relevant: bool
    pair_structure_valid: bool
    semantically_consistent: bool
    diverse: bool
    reasons: tuple[str, ...]


def _group_complete_pairs(
    examples: Sequence[SyntheticNLIExample],
) -> dict[str, list[SyntheticNLIExample]]:
    grouped: dict[str, list[SyntheticNLIExample]] = {}
    seen_ids: set[str] = set()
    for example in examples:
        if example.id in seen_ids:
            raise ValueError(f"Duplicate synthetic example ID: {example.id}")
        seen_ids.add(example.id)
        grouped.setdefault(example.pair_id, []).append(example)
    for pair_id, members in grouped.items():
        if len(members) != 2:
            raise ValueError(f"Synthetic pair {pair_id!r} must contain exactly two members")
        pair_types = {member.pair_type for member in members}
        hypothesis_ids = {member.hypothesis_id for member in members}
        member_roles = {member.pair_member for member in members}
        if len(pair_types) != 1 or len(hypothesis_ids) != 1:
            raise ValueError(f"Synthetic pair {pair_id!r} has inconsistent metadata")
        pair_type = next(iter(pair_types))
        if member_roles != set(SYNTHETIC_PAIR_MEMBERS[pair_type]):
            raise ValueError(f"Synthetic pair {pair_id!r} has incomplete member roles")
        members.sort(key=lambda item: SYNTHETIC_PAIR_MEMBERS[pair_type].index(item.pair_member))
    return grouped


class TeacherSyntheticQualityJudge:
    def __init__(
        self,
        teacher: TeacherLLM,
        *,
        seed: int,
        prompt_version: str = QUALITY_PROMPT_VERSION,
        max_examples_per_call: int = 24,
    ) -> None:
        if max_examples_per_call < 2:
            raise ValueError("max_examples_per_call must permit at least one pair")
        self.teacher = teacher
        self.seed = seed
        self.prompt_version = prompt_version
        self.max_examples_per_call = max_examples_per_call

    def judge(
        self,
        hypothesis: FailureHypothesis,
        examples: Sequence[SyntheticNLIExample],
    ) -> tuple[dict[str, QualityDecision], dict[str, Any]]:
        system_prompt = (
            "You are an independent NLI data quality reviewer. Judge the supplied "
            "three-class labels, ambiguity, grammar, failure relevance, pair semantic "
            "consistency, and diversity. Counterfactual labels must change for the "
            "intended relational change; invariant labels must remain unchanged under "
            "irrelevant changes. Reject rather than repair questionable items. Return one "
            "standalone JSON object and no markdown."
        )
        decisions: dict[str, QualityDecision] = {}
        logs: list[dict[str, Any]] = []
        required = {
            "example_id",
            "label_consistent",
            "unambiguous",
            "grammatical",
            "relevant",
            "pair_structure_valid",
            "semantically_consistent",
            "diverse",
            "reasons",
        }
        grouped = _group_complete_pairs(examples)
        pair_ids = sorted(grouped)
        pairs_per_call = max(1, self.max_examples_per_call // 2)
        for batch_index, start in enumerate(
            range(0, len(pair_ids), pairs_per_call)
        ):
            batch_pair_ids = pair_ids[start : start + pairs_per_call]
            batch = [item for pair_id in batch_pair_ids for item in grouped[pair_id]]
            payload = {
                "task": "synthetic_nli_quality_control",
                "failure_hypothesis": {
                    "id": hypothesis.id,
                    "description": hypothesis.description,
                    "intervention_target": hypothesis.intervention_target,
                },
                "output_schema": {
                    "decisions": [
                        {
                            "example_id": "one supplied ID",
                            "label_consistent": "boolean",
                            "unambiguous": "boolean",
                            "grammatical": "boolean",
                            "relevant": "boolean",
                            "pair_structure_valid": "boolean",
                            "semantically_consistent": "boolean",
                            "diverse": "boolean",
                            "reasons": ["zero or more short strings"],
                        }
                    ]
                },
                "examples": [
                    {
                        "id": item.id,
                        "premise": item.premise,
                        "hypothesis": item.hypothesis,
                        "label": item.label,
                        "pair_id": item.pair_id,
                        "pair_type": item.pair_type,
                        "pair_member": item.pair_member,
                        "pair_rationale": item.pair_rationale,
                        "label_rationale": item.label_rationale,
                        "targeting_rationale": item.targeting_rationale,
                    }
                    for item in batch
                ],
            }
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}:{hypothesis.id}:quality:{batch_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task="synthetic_quality_control",
                prompt_version=self.prompt_version,
                seed=derived_seed,
                system_prompt=system_prompt,
                payload=payload,
            )
            response = self.teacher.generate_json(call)
            if set(response) != {"decisions"} or not isinstance(
                response["decisions"], list
            ):
                raise ValueError("Quality response must contain only decisions[]")
            for raw in response["decisions"]:
                if not isinstance(raw, dict) or set(raw) != required:
                    raise ValueError("Quality decision violates the declared schema")
                example_id = str(raw["example_id"])
                boolean_fields = (
                    "label_consistent",
                    "unambiguous",
                    "grammatical",
                    "relevant",
                    "pair_structure_valid",
                    "semantically_consistent",
                    "diverse",
                )
                if example_id in decisions or any(
                    not isinstance(raw[field], bool) for field in boolean_fields
                ):
                    raise ValueError("Invalid or duplicate quality decision")
                if not isinstance(raw["reasons"], list) or any(
                    not isinstance(reason, str) for reason in raw["reasons"]
                ):
                    raise ValueError("Quality reasons must be a list of strings")
                decisions[example_id] = QualityDecision(
                    example_id=example_id,
                    label_consistent=raw["label_consistent"],
                    unambiguous=raw["unambiguous"],
                    grammatical=raw["grammatical"],
                    relevant=raw["relevant"],
                    pair_structure_valid=raw["pair_structure_valid"],
                    semantically_consistent=raw["semantically_consistent"],
                    diverse=raw["diverse"],
                    reasons=tuple(raw["reasons"]),
                )
            expected_batch_ids = {item.id for item in batch}
            returned_batch_ids = {
                str(raw["example_id"]) for raw in response["decisions"]
            }
            if returned_batch_ids != expected_batch_ids:
                raise ValueError(
                    "Quality batch must return exactly one decision per input"
                )
            logs.append(
                teacher_exchange_log(self.teacher, call, response)
            )
        expected_ids = {item.id for item in examples}
        if set(decisions) != expected_ids:
            raise ValueError("Quality judge must return exactly one decision per example")
        return decisions, {
            "hypothesis_id": hypothesis.id,
            "calls": logs,
        }


@dataclass(frozen=True)
class SyntheticValidationConfig:
    near_duplicate_threshold: float = 0.92
    min_sentence_tokens: int = 3
    max_sentence_tokens: int = 80

    def __post_init__(self) -> None:
        if not 0.0 < self.near_duplicate_threshold <= 1.0:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        if self.min_sentence_tokens <= 0:
            raise ValueError("min_sentence_tokens must be positive")
        if self.max_sentence_tokens < self.min_sentence_tokens:
            raise ValueError("max_sentence_tokens must be at least the minimum")


def validate_synthetic_examples(
    examples: Sequence[SyntheticNLIExample],
    quality: Mapping[str, QualityDecision],
    leakage_index: LeakageIndex,
    config: SyntheticValidationConfig,
) -> tuple[list[SyntheticNLIExample], list[dict[str, Any]]]:
    """Apply fail-closed QC to complete pairs without changing any NLI label."""
    accepted: list[SyntheticNLIExample] = []
    rejected: list[dict[str, Any]] = []
    observed_pairs: list[tuple[str, frozenset[str], str]] = []
    grouped = _group_complete_pairs(examples)
    for pair_id in sorted(grouped):
        members = grouped[pair_id]
        reasons: list[str] = []
        if len({member.seed_example_ids for member in members}) != 1:
            reasons.append("pair_seed_ids_inconsistent")
        if len({member.pair_rationale for member in members}) != 1:
            reasons.append("pair_rationale_inconsistent")
        labels = {member.label for member in members}
        pair_type = members[0].pair_type
        if pair_type == "counterfactual" and len(labels) != 2:
            reasons.append("counterfactual_labels_do_not_change")
        if pair_type == "invariant" and len(labels) != 1:
            reasons.append("invariant_labels_do_not_match")

        current_signatures: list[tuple[str, frozenset[str], str]] = []
        for example in members:
            decision = quality.get(example.id)
            if decision is None:
                reasons.append(f"{example.id}:missing_quality_decision")
            else:
                failed_quality = False
                for field in (
                    "label_consistent",
                    "unambiguous",
                    "grammatical",
                    "relevant",
                    "pair_structure_valid",
                    "semantically_consistent",
                    "diverse",
                ):
                    if not getattr(decision, field):
                        failed_quality = True
                        reasons.append(f"{example.id}:quality_{field}_failed")
                if failed_quality:
                    reasons.extend(
                        f"{example.id}:quality:{reason}" for reason in decision.reasons
                    )

            premise_tokens = _tokens(example.premise)
            hypothesis_tokens = _tokens(example.hypothesis)
            for name, tokens in (
                ("premise", premise_tokens),
                ("hypothesis", hypothesis_tokens),
            ):
                if not config.min_sentence_tokens <= len(tokens) <= config.max_sentence_tokens:
                    reasons.append(f"{example.id}:{name}_length_out_of_range")
            pair = _pair_text(example.premise, example.hypothesis)
            pair_tokens = _tokens(pair)
            pair_hash = hashlib.sha256(pair.encode("utf-8")).hexdigest()
            for observed_hash, observed_token_set, observed_id in observed_pairs:
                if pair_hash == observed_hash:
                    reasons.append(
                        f"{example.id}:duplicate_generated_example:{observed_id}"
                    )
                elif _jaccard(pair_tokens, observed_token_set) >= config.near_duplicate_threshold:
                    reasons.append(
                        f"{example.id}:near_duplicate_generated_example:{observed_id}"
                    )
            # Controlled pair members are intentionally close. Within a pair,
            # reject exact duplicates but do not treat the designed minimal edit
            # itself as a near-duplicate defect.
            for current_hash, _, current_id in current_signatures:
                if pair_hash == current_hash:
                    reasons.append(
                        f"{example.id}:duplicate_within_controlled_pair:{current_id}"
                    )
            current_signatures.append((pair_hash, pair_tokens, example.id))
            overlap = leakage_index.find_overlap(example, config.near_duplicate_threshold)
            if overlap is not None:
                role, detail = overlap
                reasons.append(
                    f"{example.id}:training_or_heldout_leakage:{role}:{detail}"
                )
        observed_pairs.extend(current_signatures)
        if reasons:
            unique_reasons = sorted(set(reasons))
            rejected.extend(
                {**example.to_dict(), "rejection_reasons": unique_reasons}
                for example in members
            )
        else:
            accepted.extend(members)
    return accepted, rejected
