"""Deterministic and teacher-assisted quality control for synthetic NLI data."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.hypotheses.types import FailureHypothesis
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample
from src.teacher.base import (
    TeacherCall,
    TeacherLLM,
    teacher_exchange_log,
    teacher_failure_log,
)
from tqdm.auto import tqdm


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


_QUALITY_BOOLEAN_FIELDS = (
    "label_consistent",
    "unambiguous",
    "grammatical",
    "relevant",
    "pair_structure_valid",
    "semantically_consistent",
    "diverse",
)
_QUALITY_DECISION_FIELDS = {
    "example_id",
    *_QUALITY_BOOLEAN_FIELDS,
    "reasons",
}


def _parse_quality_batch(
    response: Mapping[str, Any], expected_ids: set[str]
) -> dict[str, QualityDecision]:
    """Validate one QC response atomically before adding any decisions."""
    if set(response) != {"decisions"} or not isinstance(
        response["decisions"], list
    ):
        raise ValueError("Quality response must contain only decisions[]")
    batch_decisions: dict[str, QualityDecision] = {}
    for raw in response["decisions"]:
        if not isinstance(raw, dict) or set(raw) != _QUALITY_DECISION_FIELDS:
            raise ValueError("Quality decision violates the declared schema")
        example_id = str(raw["example_id"])
        if (
            example_id in batch_decisions
            or example_id not in expected_ids
            or any(not isinstance(raw[field], bool) for field in _QUALITY_BOOLEAN_FIELDS)
        ):
            raise ValueError("Invalid, unknown, or duplicate quality decision")
        if not isinstance(raw["reasons"], list) or any(
            not isinstance(reason, str) for reason in raw["reasons"]
        ):
            raise ValueError("Quality reasons must be a list of strings")
        batch_decisions[example_id] = QualityDecision(
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
    if set(batch_decisions) != expected_ids:
        raise ValueError("Quality batch must return exactly one decision per input")
    return batch_decisions


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
        max_attempts_per_batch: int = 3,
    ) -> None:
        if max_examples_per_call < 2:
            raise ValueError("max_examples_per_call must permit at least one pair")
        if max_attempts_per_batch <= 0:
            raise ValueError("Quality max_attempts_per_batch must be positive")
        self.teacher = teacher
        self.seed = seed
        self.prompt_version = prompt_version
        self.max_examples_per_call = max_examples_per_call
        self.max_attempts_per_batch = max_attempts_per_batch

    def judge(
        self,
        hypothesis: FailureHypothesis,
        examples: Sequence[SyntheticNLIExample],
        *,
        resume_state: Mapping[str, Any] | None = None,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
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
        grouped = _group_complete_pairs(examples)
        pair_ids = sorted(grouped)
        example_ids = [
            item.id for pair_id in pair_ids for item in grouped[pair_id]
        ]
        if resume_state is not None:
            expected_metadata = {
                "version": 1,
                "hypothesis_id": hypothesis.id,
                "prompt_version": self.prompt_version,
                "example_ids": example_ids,
            }
            for field, expected in expected_metadata.items():
                if resume_state.get(field) != expected:
                    raise ValueError(
                        f"Synthetic QC checkpoint has incompatible {field}"
                    )
            raw_decisions = resume_state.get("decisions")
            raw_logs = resume_state.get("calls")
            if not isinstance(raw_decisions, list) or not isinstance(raw_logs, list):
                raise ValueError("Synthetic QC checkpoint state is malformed")
            saved_ids = {
                str(item.get("example_id"))
                for item in raw_decisions
                if isinstance(item, Mapping)
            }
            if len(saved_ids) != len(raw_decisions) or not saved_ids <= set(example_ids):
                raise ValueError("Synthetic QC checkpoint decisions are malformed")
            decisions = _parse_quality_batch(
                {"decisions": raw_decisions}, saved_ids
            )
            logs = [dict(item) for item in raw_logs if isinstance(item, Mapping)]
            if len(logs) != len(raw_logs):
                raise ValueError("Synthetic QC checkpoint calls are malformed")
        pairs_per_call = max(1, self.max_examples_per_call // 2)
        pending_pair_ids = []
        for pair_id in pair_ids:
            current_ids = {item.id for item in grouped[pair_id]}
            completed_ids = current_ids & set(decisions)
            if completed_ids and completed_ids != current_ids:
                raise ValueError(
                    "Synthetic QC checkpoint contains a partially completed pair"
                )
            if not completed_ids:
                pending_pair_ids.append(pair_id)
        batch_starts = range(0, len(pending_pair_ids), pairs_per_call)
        for batch_index, start in enumerate(tqdm(
            batch_starts,
            desc=f"synthetic QC [{hypothesis.id[:16]}]",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
        )):
            batch_pair_ids = pending_pair_ids[start : start + pairs_per_call]
            batch = [item for pair_id in batch_pair_ids for item in grouped[pair_id]]
            expected_batch_ids = {item.id for item in batch}
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
            last_error: ValueError | None = None
            for attempt in range(self.max_attempts_per_batch):
                attempt_payload = dict(payload)
                if last_error is not None:
                    attempt_payload["retry_correction"] = {
                        "previous_response_rejected_because": str(last_error),
                        "mandatory_instruction": (
                            "Return exactly one decision for every supplied example ID, "
                            "no unknown or duplicate IDs, and follow the declared schema "
                            "exactly."
                        ),
                    }
                derived_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{self.seed}:{hypothesis.id}:quality:{batch_index}:"
                        f"attempt:{attempt}".encode("utf-8")
                    ).digest()[:4],
                    "big",
                )
                call = TeacherCall(
                    task="synthetic_quality_control",
                    prompt_version=self.prompt_version,
                    seed=derived_seed,
                    system_prompt=system_prompt,
                    payload=attempt_payload,
                )
                response: dict[str, Any] | None = None
                try:
                    response = self.teacher.generate_json(call)
                    batch_decisions = _parse_quality_batch(
                        response, expected_batch_ids
                    )
                except ValueError as exc:
                    logs.append(teacher_failure_log(self.teacher, call, exc, response))
                    last_error = exc
                    if attempt + 1 == self.max_attempts_per_batch:
                        raise ValueError(
                            "Synthetic QC batch remained invalid after "
                            f"{self.max_attempts_per_batch} attempts"
                        ) from exc
                    tqdm.write(
                        f"Retrying synthetic QC batch {batch_index} after invalid "
                        f"teacher response ({attempt + 1}/"
                        f"{self.max_attempts_per_batch}): {exc}"
                    )
                    continue
                decisions.update(batch_decisions)
                logs.append(teacher_exchange_log(self.teacher, call, response))
                break
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "version": 1,
                        "hypothesis_id": hypothesis.id,
                        "prompt_version": self.prompt_version,
                        "example_ids": example_ids,
                        "decisions": [
                            {
                                "example_id": decision.example_id,
                                "label_consistent": decision.label_consistent,
                                "unambiguous": decision.unambiguous,
                                "grammatical": decision.grammatical,
                                "relevant": decision.relevant,
                                "pair_structure_valid": decision.pair_structure_valid,
                                "semantically_consistent": decision.semantically_consistent,
                                "diverse": decision.diverse,
                                "reasons": list(decision.reasons),
                            }
                            for decision in (
                                decisions[example_id]
                                for example_id in example_ids
                                if example_id in decisions
                            )
                        ],
                        "calls": logs,
                    }
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
