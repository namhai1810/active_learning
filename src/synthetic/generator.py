"""Teacher-driven generation of genuinely new targeted NLI pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.data.labels import MNLI_LABELS
from src.hypotheses.types import FailureHypothesis
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample
from src.teacher.base import TeacherCall, TeacherLLM


SYNTHETIC_PROMPT_VERSION = "targeted_synthetic_nli_pairs_v2"
_GENERATED_PAIR_FIELDS = {
    "pair_type",
    "seed_example_ids",
    "pair_rationale",
    "examples",
}
_GENERATED_EXAMPLE_FIELDS = {
    "pair_member",
    "premise",
    "hypothesis",
    "label",
    "label_rationale",
    "targeting_rationale",
}


@dataclass(frozen=True)
class SyntheticGenerationConfig:
    counterfactual_pairs_per_hypothesis: int
    invariant_pairs_per_hypothesis: int
    reasoning_contrast_pairs_per_hypothesis: int = 0
    max_seed_examples: int = 24
    max_pairs_per_call: int = 8
    seed: int = 42
    prompt_version: str = SYNTHETIC_PROMPT_VERSION

    def __post_init__(self) -> None:
        if self.counterfactual_pairs_per_hypothesis <= 0:
            raise ValueError("At least one counterfactual pair is required")
        if self.invariant_pairs_per_hypothesis <= 0:
            raise ValueError("At least one invariant pair is required")
        if self.reasoning_contrast_pairs_per_hypothesis < 0:
            raise ValueError("reasoning_contrast_pairs_per_hypothesis cannot be negative")
        if self.max_seed_examples <= 0 or self.max_pairs_per_call <= 0:
            raise ValueError("Synthetic seed and pair batch counts must be positive")
        if not self.prompt_version.strip():
            raise ValueError("Synthetic prompt_version must be non-empty")

    @property
    def total_pairs(self) -> int:
        return (
            self.counterfactual_pairs_per_hypothesis
            + self.invariant_pairs_per_hypothesis
            + self.reasoning_contrast_pairs_per_hypothesis
        )

    @property
    def total_examples(self) -> int:
        return 2 * self.total_pairs


def _rank_seed(record: Mapping[str, Any], seed: int, hypothesis_id: str) -> tuple[bytes, str]:
    example_id = str(record["id"])
    return (
        hashlib.sha256(
            f"{seed}:{hypothesis_id}:synthetic-seed:{example_id}".encode("utf-8")
        ).digest(),
        example_id,
    )


def _synthetic_pair_id(
    hypothesis_id: str,
    pair_type: str,
    raw_examples: Sequence[Mapping[str, Any]],
    ordinal: int,
) -> str:
    identity = json.dumps(
        [hypothesis_id, pair_type, list(raw_examples), ordinal],
        ensure_ascii=False,
        sort_keys=True,
    )
    return "synpair_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _synthetic_id(pair_id: str, pair_member: str) -> str:
    return "syn_" + hashlib.sha256(
        f"{pair_id}:{pair_member}".encode("utf-8")
    ).hexdigest()[:20]


class TargetedSyntheticGenerator:
    def __init__(
        self, teacher: TeacherLLM, config: SyntheticGenerationConfig
    ) -> None:
        self.teacher = teacher
        self.config = config

    def generate(
        self,
        hypothesis: FailureHypothesis,
        intervention_records: Sequence[Mapping[str, Any]],
    ) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
        hypothesis.validate(require_llm_schema=True)
        seen: set[str] = set()
        seeds = []
        for record in intervention_records:
            if record.get("split_role") != "intervention_pool":
                raise ValueError("Synthetic seeds must come only from intervention_pool")
            required = ("id", "premise", "hypothesis", "gold")
            missing = [field for field in required if field not in record]
            if missing:
                raise ValueError(f"Intervention seed is missing fields: {missing}")
            example_id = str(record["id"])
            if not example_id or example_id in seen:
                raise ValueError(f"Missing or duplicate intervention ID: {example_id!r}")
            seen.add(example_id)
            # Metadata and student outcomes are intentionally excluded.
            seeds.append(
                {
                    "id": example_id,
                    "premise": record["premise"],
                    "hypothesis": record["hypothesis"],
                    "binary_gold": record["gold"],
                }
            )
        seeds.sort(key=lambda row: _rank_seed(row, self.config.seed, hypothesis.id))
        selected_seeds = seeds[: self.config.max_seed_examples]
        if not selected_seeds:
            raise ValueError("No permitted intervention seeds are available")

        system_prompt = (
            "You generate NEW, original, failure-targeted three-class NLI TRAINING "
            "PAIRS. A counterfactual pair changes the failure-relevant relation and "
            "must change its correct label. An invariant pair changes only irrelevant "
            "entities/surface details and must preserve its correct label. A reasoning "
            "contrast pair, when requested, contrasts the required reasoning step. Use "
            "seeds only as structural context; never copy or lightly paraphrase them. "
            "Use exact entailment/neutral/contradiction semantics; binary non-entailment "
            "does not imply contradiction. Return standalone JSON with no markdown."
        )
        permitted_ids = {str(row["id"]) for row in selected_seeds}
        generated: list[SyntheticNLIExample] = []
        call_logs: list[dict[str, Any]] = []
        schedule = (
            ["counterfactual"] * self.config.counterfactual_pairs_per_hypothesis
            + ["invariant"] * self.config.invariant_pairs_per_hypothesis
            + ["reasoning_contrast"]
            * self.config.reasoning_contrast_pairs_per_hypothesis
        )
        batch_index = 0
        offset = 0
        while offset < len(schedule):
            pair_type = schedule[offset]
            run_end = offset
            while run_end < len(schedule) and schedule[run_end] == pair_type:
                run_end += 1
            requested = min(run_end - offset, self.config.max_pairs_per_call)
            member_roles = SYNTHETIC_PAIR_MEMBERS[pair_type]
            payload = {
                "task": "generate_new_targeted_structured_nli_training_pairs",
                "failure_hypothesis": {
                    "id": hypothesis.id,
                    "description": hypothesis.description,
                    "predicted_failure_behavior": hypothesis.predicted_failure_behavior,
                    "intervention_target": hypothesis.intervention_target,
                },
                "batch_index": batch_index,
                "requested_pairs": requested,
                "required_pair_type": pair_type,
                "required_pair_members": list(member_roles),
                "allowed_labels": list(MNLI_LABELS),
                "pair_semantics": {
                    "counterfactual": "two members, different labels, only failure-relevant relation changes",
                    "invariant": "two members, same label, only irrelevant content changes",
                    "reasoning_contrast": "two members requiring contrasting reasoning steps",
                },
                "output_schema": {
                    "pairs": [
                        {
                            "pair_type": pair_type,
                            "seed_example_ids": ["one or more supplied seed IDs"],
                            "pair_rationale": "why the pair has the required structure",
                            "examples": [
                                {
                                    "pair_member": member_roles[0],
                                    "premise": "new sentence",
                                    "hypothesis": "new sentence",
                                    "label": "entailment | neutral | contradiction",
                                    "label_rationale": "why this exact label is valid",
                                    "targeting_rationale": "why this member targets the failure",
                                },
                                {
                                    "pair_member": member_roles[1],
                                    "premise": "new sentence",
                                    "hypothesis": "new sentence",
                                    "label": "entailment | neutral | contradiction",
                                    "label_rationale": "why this exact label is valid",
                                    "targeting_rationale": "why this member targets the failure",
                                },
                            ],
                        }
                    ]
                },
                "permitted_seed_examples": selected_seeds,
            }
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.config.seed}:{hypothesis.id}:generation:{batch_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task="targeted_synthetic_generation",
                prompt_version=self.config.prompt_version,
                seed=derived_seed,
                system_prompt=system_prompt,
                payload=payload,
            )
            response = self.teacher.generate_json(call)
            if set(response) != {"pairs"} or not isinstance(
                response["pairs"], list
            ):
                raise ValueError("Synthetic teacher response must contain only pairs[]")
            if len(response["pairs"]) != requested:
                raise ValueError("Teacher returned a different pair count than requested")
            for raw_pair in response["pairs"]:
                if not isinstance(raw_pair, dict) or set(raw_pair) != _GENERATED_PAIR_FIELDS:
                    raise ValueError("Synthetic pair violates the declared schema")
                if raw_pair["pair_type"] != pair_type:
                    raise ValueError("Teacher returned an unrequested synthetic pair type")
                if not isinstance(raw_pair["pair_rationale"], str):
                    raise ValueError("Synthetic pair_rationale must be a string")
                if not isinstance(raw_pair["seed_example_ids"], list) or any(
                    not isinstance(item, str) for item in raw_pair["seed_example_ids"]
                ):
                    raise ValueError("Synthetic seed_example_ids must be a string list")
                cited = tuple(str(item) for item in raw_pair["seed_example_ids"])
                if not cited or len(set(cited)) != len(cited) or not set(cited) <= permitted_ids:
                    raise ValueError("Synthetic pair cites invalid/non-permitted seeds")
                raw_examples = raw_pair["examples"]
                if not isinstance(raw_examples, list) or len(raw_examples) != 2:
                    raise ValueError("Every synthetic pair must contain exactly two examples")
                pair_id = _synthetic_pair_id(
                    hypothesis.id, pair_type, raw_examples, len(generated) // 2
                )
                returned_members: set[str] = set()
                for raw in raw_examples:
                    if not isinstance(raw, dict) or set(raw) != _GENERATED_EXAMPLE_FIELDS:
                        raise ValueError("Synthetic pair member violates its schema")
                    text_fields = tuple(_GENERATED_EXAMPLE_FIELDS)
                    if any(not isinstance(raw[field], str) for field in text_fields):
                        raise ValueError("Synthetic pair member fields must be strings")
                    pair_member = str(raw["pair_member"])
                    if pair_member in returned_members or pair_member not in member_roles:
                        raise ValueError("Synthetic pair has invalid/duplicate member roles")
                    returned_members.add(pair_member)
                    label = str(raw["label"]).strip().lower()
                    generated.append(
                        SyntheticNLIExample(
                            id=_synthetic_id(pair_id, pair_member),
                            premise=str(raw["premise"]),
                            hypothesis=str(raw["hypothesis"]),
                            label=label,
                            hypothesis_id=hypothesis.id,
                            pair_id=pair_id,
                            pair_type=pair_type,
                            pair_member=pair_member,
                            pair_rationale=str(raw_pair["pair_rationale"]),
                            seed_example_ids=cited,
                            label_rationale=str(raw["label_rationale"]),
                            targeting_rationale=str(raw["targeting_rationale"]),
                            generator_prompt_version=self.config.prompt_version,
                        )
                    )
                if returned_members != set(member_roles):
                    raise ValueError("Synthetic pair is missing a required member role")
            call_logs.append(
                {"teacher_call": call.to_dict(), "teacher_response": response}
            )
            offset += requested
            batch_index += 1
        return generated, {
            "hypothesis_id": hypothesis.id,
            "selected_seed_ids": [str(row["id"]) for row in selected_seeds],
            "generated_pairs": self.config.total_pairs,
            "generated_examples": self.config.total_examples,
            "calls": call_logs,
        }
