"""Teacher-driven generation of genuinely new targeted NLI pairs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.data.labels import MNLI_LABELS
from src.hypotheses.types import FailureHypothesis
from src.synthetic.types import SYNTHETIC_PAIR_MEMBERS, SyntheticNLIExample
from src.teacher.base import (
    TeacherCall,
    TeacherLLM,
    teacher_exchange_log,
    teacher_failure_log,
)
from tqdm.auto import tqdm


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
    max_attempts_per_batch: int = 3
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
        if self.max_attempts_per_batch <= 0:
            raise ValueError("Synthetic max_attempts_per_batch must be positive")
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


def _parse_generated_batch(
    response: Mapping[str, Any],
    *,
    hypothesis: FailureHypothesis,
    pair_type: str,
    requested: int,
    member_roles: Sequence[str],
    permitted_ids: set[str],
    pair_ordinal: int,
    prompt_version: str,
) -> list[SyntheticNLIExample]:
    """Validate one teacher batch without partially mutating run state."""
    if set(response) != {"pairs"} or not isinstance(response["pairs"], list):
        raise ValueError("Synthetic teacher response must contain only pairs[]")
    if len(response["pairs"]) != requested:
        raise ValueError("Teacher returned a different pair count than requested")

    batch: list[SyntheticNLIExample] = []
    for pair_offset, raw_pair in enumerate(response["pairs"]):
        if not isinstance(raw_pair, dict) or set(raw_pair) != _GENERATED_PAIR_FIELDS:
            raise ValueError("Synthetic pair violates the declared schema")
        if raw_pair["pair_type"] != pair_type:
            raise ValueError(
                "Teacher returned an unrequested synthetic pair type: "
                f"expected {pair_type!r}, got {raw_pair['pair_type']!r}"
            )
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
            hypothesis.id,
            pair_type,
            raw_examples,
            pair_ordinal + pair_offset,
        )
        returned_members: set[str] = set()
        pair_examples: list[SyntheticNLIExample] = []
        for raw in raw_examples:
            if not isinstance(raw, dict) or set(raw) != _GENERATED_EXAMPLE_FIELDS:
                raise ValueError("Synthetic pair member violates its schema")
            if any(not isinstance(raw[field], str) for field in _GENERATED_EXAMPLE_FIELDS):
                raise ValueError("Synthetic pair member fields must be strings")
            pair_member = str(raw["pair_member"])
            if pair_member in returned_members or pair_member not in member_roles:
                raise ValueError("Synthetic pair has invalid/duplicate member roles")
            returned_members.add(pair_member)
            pair_examples.append(
                SyntheticNLIExample(
                    id=_synthetic_id(pair_id, pair_member),
                    premise=str(raw["premise"]),
                    hypothesis=str(raw["hypothesis"]),
                    label=str(raw["label"]).strip().lower(),
                    hypothesis_id=hypothesis.id,
                    pair_id=pair_id,
                    pair_type=pair_type,
                    pair_member=pair_member,
                    pair_rationale=str(raw_pair["pair_rationale"]),
                    seed_example_ids=cited,
                    label_rationale=str(raw["label_rationale"]),
                    targeting_rationale=str(raw["targeting_rationale"]),
                    generator_prompt_version=prompt_version,
                )
            )
        if returned_members != set(member_roles):
            raise ValueError("Synthetic pair is missing a required member role")
        batch.extend(pair_examples)
    return batch


class TargetedSyntheticGenerator:
    def __init__(
        self,
        teacher: TeacherLLM,
        config: SyntheticGenerationConfig,
        *,
        failure_targeted: bool = True,
    ) -> None:
        self.teacher = teacher
        self.config = config
        self.failure_targeted = failure_targeted

    def generate(
        self,
        hypothesis: FailureHypothesis,
        intervention_records: Sequence[Mapping[str, Any]],
        *,
        resume_state: Mapping[str, Any] | None = None,
        checkpoint_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[list[SyntheticNLIExample], dict[str, Any]]:
        hypothesis.validate(require_llm_schema=self.failure_targeted)
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

        targeting_text = (
            "failure-targeted"
            if self.failure_targeted
            else "broad-coverage, non-failure-targeted"
        )
        system_prompt = (
            f"You generate NEW, original, {targeting_text} three-class NLI TRAINING "
            "PAIRS. A counterfactual pair changes the central semantic relation and "
            "must change its correct label. An invariant pair changes only irrelevant "
            "entities/surface details and must preserve its correct label. A reasoning "
            "contrast pair, when requested, contrasts the required reasoning step. Use "
            "seeds only as structural context; never copy or lightly paraphrase them. "
            "Use exact entailment/neutral/contradiction semantics; binary non-entailment "
            "does not imply contradiction. Return standalone JSON with no markdown."
        )
        permitted_ids = {str(row["id"]) for row in selected_seeds}
        schedule = (
            ["counterfactual"] * self.config.counterfactual_pairs_per_hypothesis
            + ["invariant"] * self.config.invariant_pairs_per_hypothesis
            + ["reasoning_contrast"]
            * self.config.reasoning_contrast_pairs_per_hypothesis
        )
        generated: list[SyntheticNLIExample] = []
        call_logs: list[dict[str, Any]] = []
        batch_index = 0
        offset = 0
        if resume_state is not None:
            expected_metadata = {
                "version": 1,
                "hypothesis_id": hypothesis.id,
                "failure_targeted": self.failure_targeted,
                "prompt_version": self.config.prompt_version,
                "selected_seed_ids": [str(row["id"]) for row in selected_seeds],
                "schedule": schedule,
            }
            for field, expected in expected_metadata.items():
                if resume_state.get(field) != expected:
                    raise ValueError(
                        f"Synthetic generation checkpoint has incompatible {field}"
                    )
            raw_generated = resume_state.get("generated")
            raw_logs = resume_state.get("calls")
            raw_offset = resume_state.get("offset")
            raw_batch_index = resume_state.get("batch_index")
            if (
                not isinstance(raw_generated, list)
                or not isinstance(raw_logs, list)
                or not isinstance(raw_offset, int)
                or not isinstance(raw_batch_index, int)
            ):
                raise ValueError("Synthetic generation checkpoint state is malformed")
            generated = [SyntheticNLIExample.from_mapping(item) for item in raw_generated]
            call_logs = [dict(item) for item in raw_logs if isinstance(item, Mapping)]
            if len(call_logs) != len(raw_logs):
                raise ValueError("Synthetic generation checkpoint calls are malformed")
            offset = raw_offset
            batch_index = raw_batch_index
            if (
                offset < 0
                or offset > len(schedule)
                or len(generated) != 2 * offset
                or batch_index < 0
            ):
                raise ValueError("Synthetic generation checkpoint counters are invalid")
            resumed_pair_types = [
                generated[index].pair_type for index in range(0, len(generated), 2)
            ]
            if resumed_pair_types != schedule[:offset]:
                raise ValueError("Synthetic generation checkpoint schedule is inconsistent")
            if len({item.id for item in generated}) != len(generated):
                raise ValueError("Synthetic generation checkpoint contains duplicate IDs")
        progress = tqdm(
            total=len(schedule),
            initial=offset,
            desc=f"synthetic generation [{hypothesis.id[:16]}]",
            unit="pair",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        while offset < len(schedule):
            pair_type = schedule[offset]
            run_end = offset
            while run_end < len(schedule) and schedule[run_end] == pair_type:
                run_end += 1
            requested = min(run_end - offset, self.config.max_pairs_per_call)
            member_roles = SYNTHETIC_PAIR_MEMBERS[pair_type]
            payload = {
                "task": (
                    "generate_new_targeted_structured_nli_training_pairs"
                    if self.failure_targeted
                    else "generate_new_random_structured_nli_training_pairs"
                ),
                "generation_objective": {
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
            last_error: ValueError | None = None
            for attempt in range(self.config.max_attempts_per_batch):
                attempt_payload = dict(payload)
                if last_error is not None:
                    attempt_payload["retry_correction"] = {
                        "previous_response_rejected_because": str(last_error),
                        "mandatory_instruction": (
                            f"Return exactly {requested} pair(s). Every pair must set "
                            f"pair_type to {pair_type!r} and use exactly the member roles "
                            f"{list(member_roles)!r}. Follow the declared schema exactly."
                        ),
                    }
                derived_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{self.config.seed}:{hypothesis.id}:generation:"
                        f"{batch_index}:attempt:{attempt}".encode("utf-8")
                    ).digest()[:4],
                    "big",
                )
                call = TeacherCall(
                    task=(
                        "targeted_synthetic_generation"
                        if self.failure_targeted
                        else "random_synthetic_generation"
                    ),
                    prompt_version=self.config.prompt_version,
                    seed=derived_seed,
                    system_prompt=system_prompt,
                    payload=attempt_payload,
                )
                response: dict[str, Any] | None = None
                try:
                    response = self.teacher.generate_json(call)
                    batch_generated = _parse_generated_batch(
                        response,
                        hypothesis=hypothesis,
                        pair_type=pair_type,
                        requested=requested,
                        member_roles=member_roles,
                        permitted_ids=permitted_ids,
                        pair_ordinal=len(generated) // 2,
                        prompt_version=self.config.prompt_version,
                    )
                except ValueError as exc:
                    call_logs.append(
                        teacher_failure_log(self.teacher, call, exc, response)
                    )
                    last_error = exc
                    if attempt + 1 == self.config.max_attempts_per_batch:
                        raise ValueError(
                            "Synthetic teacher batch remained invalid after "
                            f"{self.config.max_attempts_per_batch} attempts"
                        ) from exc
                    progress.write(
                        f"Retrying synthetic batch {batch_index} after invalid "
                        f"teacher response ({attempt + 1}/"
                        f"{self.config.max_attempts_per_batch}): {exc}"
                    )
                    continue
                generated.extend(batch_generated)
                call_logs.append(teacher_exchange_log(self.teacher, call, response))
                break
            offset += requested
            batch_index += 1
            progress.update(requested)
            progress.set_postfix(
                call=batch_index,
                pair_type=pair_type,
                refresh=False,
            )
            if checkpoint_callback is not None:
                checkpoint_callback(
                    {
                        "version": 1,
                        "hypothesis_id": hypothesis.id,
                        "failure_targeted": self.failure_targeted,
                        "prompt_version": self.config.prompt_version,
                        "selected_seed_ids": [
                            str(row["id"]) for row in selected_seeds
                        ],
                        "schedule": schedule,
                        "offset": offset,
                        "batch_index": batch_index,
                        "generated": [item.to_dict() for item in generated],
                        "calls": call_logs,
                    }
                )
        progress.close()
        return generated, {
            "hypothesis_id": hypothesis.id,
            "selected_seed_ids": [str(row["id"]) for row in selected_seeds],
            "generated_pairs": self.config.total_pairs,
            "generated_examples": self.config.total_examples,
            "calls": call_logs,
        }
