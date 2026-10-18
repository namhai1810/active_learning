"""One-shot generation, validation, and scoring of controlled NLI probes."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.data.labels import HANS_LABELS, MNLI_LABELS, mnli_to_hans, normalize_mnli_label
from src.hypotheses.types import FailureHypothesis
from src.teacher.base import (
    TeacherCall,
    TeacherLLM,
    teacher_exchange_log,
    teacher_failure_log,
)
from tqdm.auto import tqdm


VERIFICATION_PROBE_PROMPT_VERSION = "controlled_verification_probes_v3"
PRE_QC_INDEPENDENCE_PROMPT_VERSION = "controlled_verification_probes_v4"
VERIFICATION_LABEL_PROMPT_VERSION = "verification_probe_validation_v1"
PROBE_TYPES = ("target", "counterfactual_control", "invariant_control")
_SCENARIO_DOMAINS = (
    "marine research vessel",
    "pottery workshop",
    "astronomical observatory",
    "artisan bakery",
    "railway maintenance depot",
    "wildlife rehabilitation clinic",
    "architecture studio",
    "orchestra rehearsal hall",
    "botanical conservatory",
    "archaeological excavation",
    "robotics laboratory",
    "mountain rescue station",
    "textile design workshop",
)
_SCENARIO_LEXICAL_ANCHORS = (
    ("diver", "biologist", "catalogued"),
    ("potter", "apprentice", "glazed"),
    ("astronomer", "technician", "calibrated"),
    ("baker", "courier", "delivered"),
    ("engineer", "inspector", "repaired"),
    ("veterinarian", "ranger", "examined"),
    ("architect", "surveyor", "reviewed"),
    ("conductor", "cellist", "rehearsed"),
    ("botanist", "gardener", "transplanted"),
    ("archaeologist", "curator", "documented"),
    ("programmer", "mechanic", "tested"),
    ("paramedic", "climber", "rescued"),
    ("weaver", "designer", "dyed"),
    ("cartographer", "navigator", "charted"),
    ("chemist", "analyst", "measured"),
    ("carpenter", "decorator", "assembled"),
    ("photographer", "editor", "selected"),
)
_TOKEN = re.compile(r"[\w']+", re.UNICODE)
_RAW_PROBE_FIELDS = {
    "premise",
    "hypothesis",
    "expected_label",
    "expected_failure_label",
    "label_rationale",
    "constraint_rationale",
}
_PROBE_VALIDATION_FIELDS = {
    "probe_id",
    "validated_label",
    "label_valid",
    "constraint_satisfied",
    "unambiguous",
    "grammatical",
    "original",
    "suitable_control",
    "reasons",
}


@dataclass(frozen=True)
class ControlledProbeConfig:
    n_probe_sets: int
    min_valid_probe_sets: int
    max_context_records: int = 24
    max_probe_sets_per_call: int = 8
    max_attempts_per_batch: int = 3
    near_duplicate_threshold: float = 0.92
    seed: int = 42
    prompt_version: str = VERIFICATION_PROBE_PROMPT_VERSION

    def __post_init__(self) -> None:
        if self.n_probe_sets <= 0:
            raise ValueError("n_probe_sets must be positive")
        if not 1 <= self.min_valid_probe_sets <= self.n_probe_sets:
            raise ValueError("min_valid_probe_sets must be in [1, n_probe_sets]")
        if self.max_context_records <= 0 or self.max_probe_sets_per_call <= 0:
            raise ValueError("Probe context and call sizes must be positive")
        if self.max_attempts_per_batch <= 0:
            raise ValueError("Probe max_attempts_per_batch must be positive")
        if not 0.0 < self.near_duplicate_threshold <= 1.0:
            raise ValueError("Probe near_duplicate_threshold must be in (0, 1]")
        if not self.prompt_version.strip():
            raise ValueError("Probe prompt_version must be non-empty")


@dataclass(frozen=True)
class VerificationProbe:
    id: str
    probe_set_id: str
    probe_type: str
    hypothesis_id: str
    premise: str
    hypothesis: str
    expected_label: str
    expected_failure_label: str | None
    seed_context_ids: tuple[str, ...]
    mechanism_rationale: str
    label_rationale: str
    constraint_rationale: str
    generator_prompt_version: str

    def __post_init__(self) -> None:
        if not self.id or not self.probe_set_id or not self.hypothesis_id:
            raise ValueError("Verification probe IDs must be non-empty")
        if self.probe_type not in PROBE_TYPES:
            raise ValueError(f"Unsupported verification probe type: {self.probe_type}")
        if self.expected_label not in MNLI_LABELS:
            raise ValueError("Verification probes require an explicit MNLI label")
        if self.probe_type == "target":
            if self.expected_failure_label not in HANS_LABELS:
                raise ValueError("Target probes require an expected HANS failure label")
            if self.expected_failure_label == mnli_to_hans(self.expected_label):
                raise ValueError(
                    "Target expected_failure_label must differ from its gold label"
                )
        elif self.expected_failure_label is not None:
            raise ValueError("Control probes must not declare a failure label")
        if not self.premise.strip() or not self.hypothesis.strip():
            raise ValueError("Verification probe text must be non-empty")
        if not self.seed_context_ids or len(set(self.seed_context_ids)) != len(
            self.seed_context_ids
        ):
            raise ValueError("Verification probes require unique permitted context IDs")
        for text in (
            self.mechanism_rationale,
            self.label_rationale,
            self.constraint_rationale,
            self.generator_prompt_version,
        ):
            if not text.strip():
                raise ValueError("Verification probe rationales/version must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeValidationDecision:
    probe_id: str
    validated_label: str
    label_valid: bool
    constraint_satisfied: bool
    unambiguous: bool
    grammatical: bool
    original: bool
    suitable_control: bool
    reasons: tuple[str, ...]


def _parse_probe_validation_batch(
    response: Mapping[str, Any], expected_ids: set[str]
) -> dict[str, ProbeValidationDecision]:
    if set(response) != {"decisions"} or not isinstance(
        response["decisions"], list
    ):
        raise ValueError("Probe validation response must contain only decisions[]")
    parsed: dict[str, ProbeValidationDecision] = {}
    for raw in response["decisions"]:
        if not isinstance(raw, dict) or set(raw) != _PROBE_VALIDATION_FIELDS:
            raise ValueError("Probe validation decision violates its schema")
        probe_id = str(raw["probe_id"])
        if probe_id in parsed or probe_id not in expected_ids:
            raise ValueError("Probe validator returned an unknown or duplicate ID")
        boolean_fields = (
            "label_valid",
            "constraint_satisfied",
            "unambiguous",
            "grammatical",
            "original",
            "suitable_control",
        )
        if any(not isinstance(raw[field], bool) for field in boolean_fields):
            raise ValueError("Probe validation flags must be booleans")
        if not isinstance(raw["reasons"], list) or any(
            not isinstance(reason, str) for reason in raw["reasons"]
        ):
            raise ValueError("Probe validation reasons must be strings")
        validated_label = str(raw["validated_label"]).strip().lower()
        if validated_label not in MNLI_LABELS:
            raise ValueError("Probe validator returned an invalid MNLI label")
        parsed[probe_id] = ProbeValidationDecision(
            probe_id=probe_id,
            validated_label=validated_label,
            label_valid=raw["label_valid"],
            constraint_satisfied=raw["constraint_satisfied"],
            unambiguous=raw["unambiguous"],
            grammatical=raw["grammatical"],
            original=raw["original"],
            suitable_control=raw["suitable_control"],
            reasons=tuple(raw["reasons"]),
        )
    if set(parsed) != expected_ids:
        raise ValueError("Probe validator must decide every supplied probe")
    return parsed


class ProbePredictor(Protocol):
    def predict(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int,
    ) -> list[str]: ...


def _stable_context_rank(
    record: Mapping[str, Any], seed: int, hypothesis_id: str
) -> tuple[bytes, str]:
    example_id = str(record["id"])
    return (
        hashlib.sha256(
            f"{seed}:{hypothesis_id}:verification-context:{example_id}".encode("utf-8")
        ).digest(),
        example_id,
    )


def _sanitize_context(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("split_role") != "verification":
        raise ValueError("Verification probe context must come only from verification")
    required = ("id", "premise", "hypothesis", "gold")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(f"Verification context is missing fields: {missing}")
    # In particular: no prediction, correctness, heuristic, or subcase.
    return {field: record[field] for field in required}


def _probe_set_id(
    hypothesis_id: str, probes: Sequence[Mapping[str, Any]], ordinal: int
) -> str:
    identity = json.dumps(
        [hypothesis_id, ordinal, list(probes)], ensure_ascii=False, sort_keys=True
    )
    return "vset_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _probe_id(probe_set_id: str, probe_type: str) -> str:
    return "vprobe_" + hashlib.sha256(
        f"{probe_set_id}:{probe_type}".encode("utf-8")
    ).hexdigest()[:20]


def _scenario_assignment(ordinal: int) -> dict[str, Any]:
    """Give greedy decoding a distinct, pre-outcome scenario for every set."""
    if ordinal < 0:
        raise ValueError("Probe-set ordinal must be non-negative")
    return {
        "set_ordinal": ordinal,
        "domain": _SCENARIO_DOMAINS[ordinal % len(_SCENARIO_DOMAINS)],
        "required_lexical_anchors": list(
            _SCENARIO_LEXICAL_ANCHORS[
                ordinal % len(_SCENARIO_LEXICAL_ANCHORS)
            ]
        ),
    }


def _normalized_pair_text(premise: str, hypothesis: str) -> str:
    return " ".join(
        _TOKEN.findall(f"{premise} [pair] {hypothesis}".casefold())
    )


def _normalized_sentence(text: str) -> str:
    return " ".join(_TOKEN.findall(text.casefold()))


def _parse_generated_probe_batch(
    response: Mapping[str, Any],
    *,
    hypothesis: FailureHypothesis,
    permitted_ids: set[str],
    requested: int,
    ordinal_start: int,
    prompt_version: str,
    forbidden_pairs: Mapping[str, str] | None = None,
    near_duplicate_threshold: float = 0.92,
) -> list[VerificationProbe]:
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("Probe near_duplicate_threshold must be in (0, 1]")
    if set(response) != {"probe_sets"} or not isinstance(
        response["probe_sets"], list
    ):
        raise ValueError("Probe generator response must contain only probe_sets[]")
    if len(response["probe_sets"]) != requested:
        raise ValueError("Probe generator returned a different set count")
    parsed: list[VerificationProbe] = []
    observed_pairs = dict(forbidden_pairs or {})
    observed_signatures = [
        (_token_set(pair), source_id)
        for pair, source_id in observed_pairs.items()
    ]
    for set_offset, raw_set in enumerate(response["probe_sets"]):
        expected_set_fields = {
            "seed_context_ids",
            "mechanism_rationale",
            *PROBE_TYPES,
        }
        if not isinstance(raw_set, dict):
            raise ValueError("Generated probe set must be a JSON object")
        actual_set_fields = set(raw_set)
        if actual_set_fields != expected_set_fields:
            raise ValueError(
                "Generated probe set violates its schema: "
                f"missing={sorted(expected_set_fields - actual_set_fields)}, "
                f"extra={sorted(actual_set_fields - expected_set_fields)}"
            )
        if not isinstance(raw_set["mechanism_rationale"], str):
            raise ValueError("Probe mechanism_rationale must be a string")
        raw_seed_ids = raw_set["seed_context_ids"]
        if not isinstance(raw_seed_ids, list) or any(
            not isinstance(item, str) for item in raw_seed_ids
        ):
            raise ValueError("Probe seed_context_ids must be a string list")
        seed_ids = tuple(raw_seed_ids)
        if not seed_ids or not set(seed_ids) <= permitted_ids:
            raise ValueError("Generated probe cites a non-permitted context ID")
        raw_probes = []
        for probe_type in PROBE_TYPES:
            raw = raw_set[probe_type]
            if not isinstance(raw, dict):
                raise ValueError(f"Generated {probe_type} probe must be a JSON object")
            actual_probe_fields = set(raw)
            if actual_probe_fields != _RAW_PROBE_FIELDS:
                raise ValueError(
                    f"Generated {probe_type} probe violates its schema: "
                    f"missing={sorted(_RAW_PROBE_FIELDS - actual_probe_fields)}, "
                    f"extra={sorted(actual_probe_fields - _RAW_PROBE_FIELDS)}"
                )
            raw_probes.append(raw)
        normalized_pairs: list[str] = []
        for raw in raw_probes:
            if not isinstance(raw.get("premise"), str) or not isinstance(
                raw.get("hypothesis"), str
            ):
                raise ValueError("Generated probe text fields must be strings")
            normalized_pairs.append(
                _normalized_pair_text(raw["premise"], raw["hypothesis"])
            )
        if len(set(normalized_pairs)) != len(PROBE_TYPES):
            raise ValueError(
                "Target, counterfactual control, and invariant control must use "
                "three pairwise-distinct premise-hypothesis pairs"
            )
        for pair in normalized_pairs:
            if pair in observed_pairs:
                raise ValueError(
                    "Generated probe duplicates an earlier probe or verification "
                    f"context pair: {observed_pairs[pair]}"
                )
            pair_tokens = _token_set(pair)
            for previous_tokens, previous_id in observed_signatures:
                if _jaccard(pair_tokens, previous_tokens) >= near_duplicate_threshold:
                    raise ValueError(
                        "Generated probe is a near-duplicate of an earlier probe or "
                        f"verification context pair: {previous_id}"
                    )
        set_id = _probe_set_id(
            hypothesis.id, raw_probes, ordinal_start + set_offset
        )
        for probe_type, raw in zip(PROBE_TYPES, raw_probes, strict=True):
            text_fields = (
                "premise",
                "hypothesis",
                "expected_label",
                "label_rationale",
                "constraint_rationale",
            )
            if any(not isinstance(raw[field], str) for field in text_fields):
                raise ValueError("Generated probe text fields must be strings")
            failure_label = raw["expected_failure_label"]
            if failure_label is not None and not isinstance(failure_label, str):
                raise ValueError("expected_failure_label must be text or null")
            expected_label = str(raw["expected_label"]).strip().lower()
            normalized_failure = (
                None
                if failure_label is None
                else str(failure_label).strip().lower()
            )
            if expected_label not in MNLI_LABELS:
                raise ValueError(
                    f"Generated {probe_type} expected_label must be exactly one of "
                    f"{list(MNLI_LABELS)!r}; got {expected_label!r}. "
                    "The binary HANS label 'non-entailment' is forbidden in "
                    "expected_label."
                )
            if (
                prompt_version == PRE_QC_INDEPENDENCE_PROMPT_VERSION
                and _normalized_sentence(str(raw["premise"]))
                == _normalized_sentence(str(raw["hypothesis"]))
                and expected_label != "entailment"
            ):
                raise ValueError(
                    "A probe with identical premise and hypothesis must be labeled "
                    "entailment"
                )
            if (
                probe_type == "target"
                and normalized_failure not in hypothesis.predicted_failure_labels
            ):
                raise ValueError(
                    "Target failure label was not declared by the hypothesis"
                )
            parsed.append(
                VerificationProbe(
                    id=_probe_id(set_id, probe_type),
                    probe_set_id=set_id,
                    probe_type=probe_type,
                    hypothesis_id=hypothesis.id,
                    premise=str(raw["premise"]),
                    hypothesis=str(raw["hypothesis"]),
                    expected_label=expected_label,
                    expected_failure_label=normalized_failure,
                    seed_context_ids=seed_ids,
                    mechanism_rationale=str(raw_set["mechanism_rationale"]),
                    label_rationale=str(raw["label_rationale"]),
                    constraint_rationale=str(raw["constraint_rationale"]),
                    generator_prompt_version=prompt_version,
                )
            )
        for probe, pair in zip(
            parsed[-len(PROBE_TYPES) :], normalized_pairs, strict=True
        ):
            observed_pairs[pair] = probe.id
            observed_signatures.append((_token_set(pair), probe.id))
    return parsed


class ControlledVerificationProbeGenerator:
    """Generate fixed probe sets before any student prediction is observed."""

    def __init__(self, teacher: TeacherLLM, config: ControlledProbeConfig) -> None:
        self.teacher = teacher
        self.config = config

    def _repair_invalid_labels(
        self,
        response: dict[str, Any],
        *,
        hypothesis: FailureHypothesis,
        batch_index: int,
        generation_attempt: int,
        logs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Repair only invalid label fields before probes are frozen and queried."""
        if set(response) != {"probe_sets"} or not isinstance(
            response.get("probe_sets"), list
        ):
            return response
        repaired = copy.deepcopy(response)
        repair_items: list[dict[str, Any]] = []
        singleton_failure_label = (
            hypothesis.predicted_failure_labels[0]
            if len(hypothesis.predicted_failure_labels) == 1
            else None
        )
        expected_set_fields = {
            "seed_context_ids",
            "mechanism_rationale",
            *PROBE_TYPES,
        }
        for set_index, raw_set in enumerate(repaired["probe_sets"]):
            if not isinstance(raw_set, dict) or set(raw_set) != expected_set_fields:
                return response
            for probe_type in PROBE_TYPES:
                raw = raw_set.get(probe_type)
                if not isinstance(raw, dict) or set(raw) != _RAW_PROBE_FIELDS:
                    return response
                if probe_type == "target":
                    if singleton_failure_label is not None:
                        raw["expected_failure_label"] = singleton_failure_label
                    failure_label = raw.get("expected_failure_label")
                    allowed_labels = [
                        label
                        for label in MNLI_LABELS
                        if failure_label in HANS_LABELS
                        and mnli_to_hans(label) != failure_label
                    ]
                else:
                    raw["expected_failure_label"] = None
                    allowed_labels = list(MNLI_LABELS)
                current_label = str(raw.get("expected_label", "")).strip().lower()
                if current_label not in allowed_labels:
                    repair_items.append(
                        {
                            "probe_ref": f"set_{set_index}:{probe_type}",
                            "probe_type": probe_type,
                            "premise": raw.get("premise"),
                            "hypothesis": raw.get("hypothesis"),
                            "current_invalid_label": raw.get("expected_label"),
                            "allowed_expected_labels": allowed_labels,
                        }
                    )
        if not repair_items:
            return repaired
        if any(
            not isinstance(item["premise"], str)
            or not isinstance(item["hypothesis"], str)
            for item in repair_items
        ):
            return response

        base_payload = {
            "task": "repair_controlled_probe_three_class_labels",
            "instructions": (
                "Independently solve each premise-hypothesis pair using three-class "
                "MNLI semantics. Select exactly one label from that item's allowed "
                "list. The binary HANS label non-entailment is forbidden. This repair "
                "occurs before any student prediction is observed."
            ),
            "output_schema": {
                "repairs": [
                    {
                        "probe_ref": "one supplied probe_ref",
                        "expected_label": "one allowed three-class MNLI label",
                        "label_rationale": "short explanation of the repaired label",
                    }
                ]
            },
            "items": repair_items,
        }
        expected = {item["probe_ref"]: item for item in repair_items}
        last_error: ValueError | None = None
        for repair_attempt in range(self.config.max_attempts_per_batch):
            attempt_payload = dict(base_payload)
            if last_error is not None:
                attempt_payload["retry_correction"] = {
                    "previous_response_rejected_because": str(last_error),
                    "mandatory_instruction": (
                        "Return only repairs[]. Include every supplied probe_ref once, "
                        "and choose expected_label only from its allowed_expected_labels."
                    ),
                }
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.config.seed}:{hypothesis.id}:probe-label-repair:"
                    f"{batch_index}:generation-attempt:{generation_attempt}:"
                    f"repair-attempt:{repair_attempt}".encode("utf-8")
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task="controlled_verification_probe_label_repair",
                prompt_version=f"{self.config.prompt_version}_label_repair_v1",
                seed=derived_seed,
                system_prompt=(
                    "You are an independent three-class MNLI annotator. Return one "
                    "standalone JSON object with no markdown. Never use the binary "
                    "label non-entailment as an MNLI label."
                ),
                payload=attempt_payload,
            )
            repair_response: dict[str, Any] | None = None
            try:
                repair_response = self.teacher.generate_json(call)
                if set(repair_response) != {"repairs"} or not isinstance(
                    repair_response["repairs"], list
                ):
                    raise ValueError(
                        "Probe label repair response must contain only repairs[]"
                    )
                parsed_repairs: dict[str, tuple[str, str]] = {}
                for raw_repair in repair_response["repairs"]:
                    if not isinstance(raw_repair, dict) or set(raw_repair) != {
                        "probe_ref",
                        "expected_label",
                        "label_rationale",
                    }:
                        raise ValueError("Probe label repair item violates its schema")
                    probe_ref = str(raw_repair["probe_ref"])
                    label = str(raw_repair["expected_label"]).strip().lower()
                    rationale = raw_repair["label_rationale"]
                    if probe_ref in parsed_repairs or probe_ref not in expected:
                        raise ValueError(
                            "Probe label repair returned an unknown or duplicate ref"
                        )
                    if label not in expected[probe_ref]["allowed_expected_labels"]:
                        raise ValueError(
                            f"Probe label repair returned disallowed label {label!r} "
                            f"for {probe_ref}"
                        )
                    if not isinstance(rationale, str) or not rationale.strip():
                        raise ValueError(
                            "Probe label repair rationale must be non-empty text"
                        )
                    parsed_repairs[probe_ref] = (label, rationale)
                if set(parsed_repairs) != set(expected):
                    raise ValueError("Probe label repair must cover every supplied ref")
            except ValueError as exc:
                logs.append(
                    {
                        "verification_substage": "label_repair",
                        **teacher_failure_log(
                            self.teacher, call, exc, repair_response
                        ),
                    }
                )
                last_error = exc
                if repair_attempt + 1 == self.config.max_attempts_per_batch:
                    raise ValueError(
                        "Probe label repair remained invalid after "
                        f"{self.config.max_attempts_per_batch} attempts"
                    ) from exc
                continue
            for probe_ref, (label, rationale) in parsed_repairs.items():
                set_ref, probe_type = probe_ref.split(":", 1)
                set_index = int(set_ref.removeprefix("set_"))
                raw = repaired["probe_sets"][set_index][probe_type]
                raw["expected_label"] = label
                raw["label_rationale"] = rationale
            logs.append(
                {
                    "verification_substage": "label_repair",
                    "repaired_probe_refs": sorted(parsed_repairs),
                    **teacher_exchange_log(self.teacher, call, repair_response),
                }
            )
            return repaired
        raise RuntimeError("Probe label repair loop ended without a result")

    def generate(
        self,
        hypothesis: FailureHypothesis,
        verification_context: Sequence[Mapping[str, Any]],
        *,
        ordinal_start: int = 0,
        avoid_probes: Sequence[VerificationProbe] = (),
    ) -> tuple[list[VerificationProbe], dict[str, Any]]:
        hypothesis.validate(require_llm_schema=True)
        if isinstance(ordinal_start, bool) or not isinstance(ordinal_start, int) or ordinal_start < 0:
            raise ValueError("ordinal_start must be a non-negative integer")
        generation_start_ordinal = ordinal_start
        prior_probes = list(avoid_probes)
        if any(probe.hypothesis_id != hypothesis.id for probe in prior_probes):
            raise ValueError("avoid_probes must belong to the tested hypothesis")
        if len({probe.id for probe in prior_probes}) != len(prior_probes):
            raise ValueError("avoid_probes IDs must be unique")
        context = [_sanitize_context(record) for record in verification_context]
        ids = [str(record["id"]) for record in context]
        if len(ids) != len(set(ids)):
            raise ValueError("Verification context IDs must be unique")
        context.sort(
            key=lambda row: _stable_context_rank(
                row, self.config.seed, hypothesis.id
            )
        )
        selected_context = context[: self.config.max_context_records]
        if not selected_context:
            raise ValueError("No verification context is available for probe generation")
        context_forbidden_pairs = {
            _normalized_pair_text(
                str(record["premise"]), str(record["hypothesis"])
            ): f"verification_context:{record['id']}"
            for record in context
        }
        strict_independence = (
            self.config.prompt_version == PRE_QC_INDEPENDENCE_PROMPT_VERSION
        )
        if strict_independence:
            system_prompt = (
                "Generate NEW controlled three-class NLI verification probes for one "
                "pre-registered failure hypothesis. Each probe set must contain exactly "
                "one target, one counterfactual/positive control, and one invariant "
                "control. The three premise-hypothesis pairs inside a set must be "
                "distinct. Sets must use independent scenarios and must not copy, "
                "lightly paraphrase, or reuse pairs from context or earlier sets. "
                "State labels before any student is queried. Return one standalone "
                "JSON object and no markdown."
            )
        else:
            system_prompt = (
                "Generate NEW controlled three-class NLI verification probes for one "
                "pre-registered failure hypothesis. Each probe set must contain exactly "
                "one target, one counterfactual/positive control, and one invariant "
                "control. Sets must use independent scenarios and must not copy or "
                "lightly paraphrase context records. State labels before any student "
                "is queried. Return one standalone JSON object and no markdown."
            )
        generated: list[VerificationProbe] = []
        logs: list[dict[str, Any]] = []
        remaining = self.config.n_probe_sets
        batch_index = ordinal_start
        next_set_ordinal = ordinal_start
        failed_probe_sets = 0
        max_failed_probe_sets = (
            self.config.n_probe_sets - self.config.min_valid_probe_sets
        )
        adaptive_max_probe_sets = self.config.max_probe_sets_per_call
        progress = tqdm(
            total=self.config.n_probe_sets,
            desc=f"verification generation [{hypothesis.id[:16]}]",
            unit="set",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        while remaining:
            requested = min(remaining, adaptive_max_probe_sets)
            ordinal_start = next_set_ordinal
            context_window_size = min(2, len(selected_context))
            context_start = (
                batch_index * context_window_size
            ) % len(selected_context)
            batch_context = [
                selected_context[(context_start + offset) % len(selected_context)]
                for offset in range(context_window_size)
            ]
            batch_permitted_ids = {
                str(record["id"]) for record in batch_context
            }
            payload = {
                "task": "generate_controlled_hypothesis_verification_probes",
                "failure_hypothesis": {
                    "id": hypothesis.id,
                    "description": hypothesis.description,
                    "predicted_failure_behavior": hypothesis.predicted_failure_behavior,
                    "predicted_failure_labels": list(
                        hypothesis.predicted_failure_labels
                    ),
                    "verification_criteria": list(hypothesis.verification_criteria),
                },
                "batch_index": batch_index,
                "global_set_ordinal_start": ordinal_start,
                "requested_probe_sets": requested,
                "mandatory_output_contract": (
                    "Return exactly one top-level key named probe_sets. Never echo "
                    "task, batch_index, global_set_ordinal_start, "
                    "failure_hypothesis, requirements, scenario_assignments, "
                    "output_schema, or verification_context_only."
                ),
                "requirements": {
                    "target": (
                        "satisfies the hypothesized mechanism; expected_failure_label "
                        "must be one declared predicted failure label and must be wrong"
                    ),
                    "counterfactual_control": (
                        "minimal mechanism-relevant change for which correct behavior "
                        "distinguishes the proposed mechanism"
                    ),
                    "invariant_control": (
                        "irrelevant entity or surface changes preserve the correct label"
                    ),
                    "all_labels": list(MNLI_LABELS),
                    "label_semantics": (
                        "expected_label is the gold three-class MNLI label and MUST "
                        "be exactly entailment, neutral, or contradiction; never use "
                        "non-entailment there. expected_failure_label is a separate "
                        "binary HANS prediction and may be entailment or "
                        "non-entailment only for target probes. If the target failure "
                        "label is entailment, its gold expected_label must be neutral "
                        "or contradiction. If the target failure label is "
                        "non-entailment, its gold expected_label must be entailment."
                    ),
                    "controls_expected_failure_label": None,
                    **(
                        {
                            "within_set_distinctness": (
                                "The target, counterfactual control, and invariant "
                                "control must be three different premise-hypothesis "
                                "pairs. Never reuse one pair with different labels. If "
                                "premise and hypothesis are identical, their gold label "
                                "must be entailment."
                            ),
                            "control_quality": (
                                "The counterfactual must make a mechanism-relevant "
                                "change with one unambiguous exact MNLI label. The "
                                "invariant must preserve the exact MNLI label after an "
                                "irrelevant change and must not collapse into an "
                                "identical-sentence shortcut."
                            ),
                        }
                        if strict_independence
                        else {}
                    ),
                    "originality": (
                        "Every set must instantiate its corresponding scenario "
                        "assignment, using all assigned lexical anchors (inflection "
                        "is allowed). Do not copy, lightly paraphrase, or reuse the "
                        "premise/hypothesis wording of any supplied context record. "
                        "Do not reuse a scenario across output sets."
                    ),
                },
                "scenario_assignments": [
                    _scenario_assignment(ordinal)
                    for ordinal in range(ordinal_start, ordinal_start + requested)
                ],
                "output_schema": {
                    "probe_sets": [
                        {
                            "seed_context_ids": ["one or more supplied IDs"],
                            "mechanism_rationale": "why the three probes form one test",
                            "target": {
                                "premise": "new sentence",
                                "hypothesis": "new sentence",
                                "expected_label": "entailment | neutral | contradiction",
                                "expected_failure_label": "entailment | non-entailment",
                                "label_rationale": "why the expected label is valid",
                                "constraint_rationale": "why this is a target probe",
                            },
                            "counterfactual_control": {
                                "premise": "new sentence",
                                "hypothesis": "new sentence",
                                "expected_label": "entailment | neutral | contradiction",
                                "expected_failure_label": None,
                                "label_rationale": "why the expected label is valid",
                                "constraint_rationale": "why this is a counterfactual control",
                            },
                            "invariant_control": {
                                "premise": "new sentence",
                                "hypothesis": "new sentence",
                                "expected_label": "entailment | neutral | contradiction",
                                "expected_failure_label": None,
                                "label_rationale": "why the expected label is valid",
                                "constraint_rationale": "why this is an invariant control",
                            },
                        }
                    ]
                },
                "verification_context_only": batch_context,
            }
            initial_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.config.seed}:{hypothesis.id}:probe-generation:{batch_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            )
            last_error: ValueError | None = None
            exhausted_error: ValueError | None = None
            batch_generated: list[VerificationProbe] | None = None
            for attempt in range(self.config.max_attempts_per_batch):
                attempt_payload = dict(payload)
                if last_error is not None:
                    retry_correction: dict[str, Any] = {
                        "previous_response_rejected_because": str(last_error),
                        "mandatory_instruction": (
                            "Return exactly one top-level key named probe_sets, with "
                            f"exactly {requested} set(s). Do not echo batch_index, "
                            "failure_hypothesis, requirements, or any other input key. "
                            "Every set must contain exactly these five keys: "
                            "seed_context_ids, mechanism_rationale, target, "
                            "counterfactual_control, invariant_control. "
                            "seed_context_ids must be a non-empty list containing only "
                            f"IDs from {sorted(batch_permitted_ids)!r}. Every "
                            "target/control "
                            "must contain exactly these six keys: premise, hypothesis, "
                            "expected_label, expected_failure_label, label_rationale, "
                            "constraint_rationale. Never omit premise or "
                            "seed_context_ids. The target, counterfactual control, "
                            "and invariant control premise-hypothesis pairs must be "
                            "pairwise distinct. expected_label MUST be exactly one of "
                            "entailment, neutral, contradiction; never put the binary "
                            "label non-entailment in expected_label. Use "
                            "expected_failure_label only for the target."
                        ),
                    }
                    duplicate_source = str(last_error).rsplit(": ", 1)[-1]
                    duplicate_probe = next(
                        (
                            probe
                            for probe in (*prior_probes, *generated)
                            if probe.id == duplicate_source
                        ),
                        None,
                    )
                    if duplicate_probe is not None:
                        retry_correction["duplicate_avoidance_instruction"] = (
                            "Change both the event wording and syntactic construction; "
                            "do not lightly paraphrase this forbidden pair."
                        )
                        retry_correction["forbidden_near_duplicate_pair"] = {
                            "premise": duplicate_probe.premise,
                            "hypothesis": duplicate_probe.hypothesis,
                        }
                    elif duplicate_source.startswith("verification_context:"):
                        context_id = duplicate_source.split(":", 1)[1]
                        duplicate_context = next(
                            (
                                record
                                for record in context
                                if str(record["id"]) == context_id
                            ),
                            None,
                        )
                        if duplicate_context is not None:
                            retry_correction["duplicate_avoidance_instruction"] = (
                                "Change both the event wording and syntactic "
                                "construction; do not lightly paraphrase this "
                                "forbidden pair."
                            )
                            retry_correction["forbidden_near_duplicate_pair"] = {
                                "premise": str(duplicate_context["premise"]),
                                "hypothesis": str(duplicate_context["hypothesis"]),
                            }
                    attempt_payload["retry_correction"] = retry_correction
                if attempt == 0:
                    # Keep existing request IDs so a cached invalid response is
                    # observed, logged, and then corrected by a distinct retry call.
                    derived_seed = initial_seed
                else:
                    derived_seed = int.from_bytes(
                        hashlib.sha256(
                            f"{self.config.seed}:{hypothesis.id}:probe-generation:"
                            f"{batch_index}:attempt:{attempt}".encode("utf-8")
                        ).digest()[:4],
                        "big",
                    )
                call = TeacherCall(
                    task="controlled_verification_probe_generation",
                    prompt_version=self.config.prompt_version,
                    seed=derived_seed,
                    system_prompt=system_prompt,
                    payload=attempt_payload,
                )
                raw_response: dict[str, Any] | None = None
                response: dict[str, Any] | None = None
                try:
                    raw_response = self.teacher.generate_json(call)
                    response = self._repair_invalid_labels(
                        raw_response,
                        hypothesis=hypothesis,
                        batch_index=batch_index,
                        generation_attempt=attempt,
                        logs=logs,
                    )
                    batch_generated = _parse_generated_probe_batch(
                        response,
                        hypothesis=hypothesis,
                        permitted_ids=batch_permitted_ids,
                        requested=requested,
                        ordinal_start=ordinal_start,
                        prompt_version=self.config.prompt_version,
                        forbidden_pairs=(
                            context_forbidden_pairs
                            | {
                                _normalized_pair_text(
                                    probe.premise, probe.hypothesis
                                ): probe.id
                                for probe in (*prior_probes, *generated)
                            }
                            if self.config.prompt_version
                            == PRE_QC_INDEPENDENCE_PROMPT_VERSION
                            else None
                        ),
                        near_duplicate_threshold=self.config.near_duplicate_threshold,
                    )
                except ValueError as exc:
                    logs.append(
                        teacher_failure_log(
                            self.teacher, call, exc, raw_response
                        )
                    )
                    last_error = exc
                    if attempt + 1 == self.config.max_attempts_per_batch:
                        exhausted_error = exc
                        break
                    tqdm.write(
                        f"Retrying verification generation batch {batch_index} for "
                        f"{hypothesis.id} after invalid teacher response "
                        f"({attempt + 1}/{self.config.max_attempts_per_batch}): {exc}"
                    )
                    continue
                generated.extend(batch_generated)
                generation_log = teacher_exchange_log(
                    self.teacher, call, raw_response
                )
                if response != raw_response:
                    generation_log["accepted_response_after_label_repair"] = response
                logs.append(generation_log)
                break
            if exhausted_error is not None:
                if requested > 1:
                    adaptive_max_probe_sets = max(1, requested // 2)
                    tqdm.write(
                        f"Verification generation batch {batch_index} for "
                        f"{hypothesis.id} remained invalid after "
                        f"{self.config.max_attempts_per_batch} attempts; reducing "
                        f"the call from {requested} to "
                        f"{adaptive_max_probe_sets} probe set(s)."
                    )
                    continue
                failed_probe_sets += 1
                if failed_probe_sets > max_failed_probe_sets:
                    progress.close()
                    raise ValueError(
                        "Verification probe generation failures leave fewer than "
                        f"min_valid_probe_sets={self.config.min_valid_probe_sets} "
                        "possible before independent QC"
                    ) from exhausted_error
                logs.append(
                    {
                        "verification_substage": "generation_set_abstention",
                        "hypothesis_id": hypothesis.id,
                        "set_ordinal": ordinal_start,
                        "attempts": self.config.max_attempts_per_batch,
                        "reason": str(exhausted_error),
                    }
                )
                tqdm.write(
                    f"Skipping invalid verification set ordinal {ordinal_start} for "
                    f"{hypothesis.id}; {failed_probe_sets}/"
                    f"{max_failed_probe_sets} pre-QC failures used."
                )
                remaining -= 1
                next_set_ordinal += 1
                batch_index += 1
                progress.update(1)
                progress.set_postfix(
                    call=batch_index, skipped=failed_probe_sets, refresh=False
                )
                continue
            if batch_generated is None:
                progress.close()
                raise RuntimeError("Probe generation retry loop ended without a result")
            remaining -= requested
            next_set_ordinal += requested
            batch_index += 1
            progress.update(requested)
            progress.set_postfix(call=batch_index, refresh=False)
        progress.close()
        return generated, {
            "hypothesis_id": hypothesis.id,
            "selected_context_ids": [str(row["id"]) for row in selected_context],
            "ordinal_start": generation_start_ordinal,
            "ordinal_end": next_set_ordinal,
            "attempted_probe_sets": next_set_ordinal - generation_start_ordinal,
            "generated_probe_sets": len(generated) // len(PROBE_TYPES),
            "failed_generation_probe_sets": failed_probe_sets,
            "student_outcomes_in_teacher_prompt": False,
            "calls": logs,
        }


class TeacherVerificationProbeValidator:
    """Independently validate fixed labels and probe constraints, without outcomes."""

    def __init__(
        self,
        teacher: TeacherLLM,
        *,
        seed: int,
        prompt_version: str = VERIFICATION_LABEL_PROMPT_VERSION,
        max_probe_sets_per_call: int = 8,
        max_attempts_per_batch: int = 3,
    ) -> None:
        if max_probe_sets_per_call <= 0 or not prompt_version.strip():
            raise ValueError("Probe validator call size/version is invalid")
        if max_attempts_per_batch <= 0:
            raise ValueError("Probe validator max_attempts_per_batch must be positive")
        self.teacher = teacher
        self.seed = seed
        self.prompt_version = prompt_version
        self.max_probe_sets_per_call = max_probe_sets_per_call
        self.max_attempts_per_batch = max_attempts_per_batch

    def validate(
        self,
        hypothesis: FailureHypothesis,
        probes: Sequence[VerificationProbe],
    ) -> tuple[dict[str, ProbeValidationDecision], dict[str, Any]]:
        grouped = _group_probe_sets(probes)
        set_ids = sorted(grouped)
        decisions: dict[str, ProbeValidationDecision] = {}
        logs: list[dict[str, Any]] = []
        failed_validation_probe_sets = 0
        pending_batches: list[tuple[int, tuple[int, ...], list[str]]] = [
            (
                batch_index,
                (),
                set_ids[start : start + self.max_probe_sets_per_call],
            )
            for batch_index, start in enumerate(
                range(0, len(set_ids), self.max_probe_sets_per_call)
            )
        ]
        progress = tqdm(
            total=len(set_ids),
            desc=f"verification QC [{hypothesis.id[:16]}]",
            unit="set",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        while pending_batches:
            batch_index, split_path, batch_ids = pending_batches.pop(0)
            batch_label = str(batch_index) + "".join(
                f".{part}" for part in split_path
            )
            batch = [probe for set_id in batch_ids for probe in grouped[set_id]]
            payload = {
                "task": "independent_verification_probe_label_and_constraint_validation",
                "failure_hypothesis": {
                    "id": hypothesis.id,
                    "description": hypothesis.description,
                    "predicted_failure_behavior": hypothesis.predicted_failure_behavior,
                    "verification_criteria": list(hypothesis.verification_criteria),
                },
                "instructions": (
                    "Independently solve every NLI item, validate the stated MNLI "
                    "label, and validate target/control function. Reject rather than "
                    "repair. No student prediction exists yet."
                ),
                "output_schema": {
                    "decisions": [
                        {
                            "probe_id": "one supplied ID",
                            "validated_label": "entailment | neutral | contradiction",
                            "label_valid": "boolean",
                            "constraint_satisfied": "boolean",
                            "unambiguous": "boolean",
                            "grammatical": "boolean",
                            "original": "boolean",
                            "suitable_control": "boolean; true for a valid target too",
                            "reasons": ["zero or more short strings"],
                        }
                    ]
                },
                "probe_sets": [probe.to_dict() for probe in batch],
            }
            expected_ids = {probe.id for probe in batch}
            last_error: ValueError | None = None
            split_required = False
            for attempt in range(self.max_attempts_per_batch):
                attempt_payload = dict(payload)
                if last_error is not None:
                    attempt_payload["retry_correction"] = {
                        "previous_response_rejected_because": str(last_error),
                        "mandatory_instruction": (
                            "Return exactly one top-level key named decisions and "
                            "exactly one decision for every supplied probe_id. Do not "
                            "echo input metadata. Follow the declared schema exactly."
                        ),
                    }
                if not split_path:
                    seed_material = (
                        f"{self.seed}:{hypothesis.id}:probe-validation:{batch_index}"
                    )
                    if attempt:
                        seed_material += f":attempt:{attempt}"
                else:
                    seed_material = (
                        f"{self.seed}:{hypothesis.id}:probe-validation:"
                        f"{batch_index}:split:{'.'.join(map(str, split_path))}:"
                        f"attempt:{attempt}"
                    )
                derived_seed = int.from_bytes(
                    hashlib.sha256(seed_material.encode("utf-8")).digest()[:4],
                    "big",
                )
                call = TeacherCall(
                    task="verification_probe_validation",
                    prompt_version=self.prompt_version,
                    seed=derived_seed,
                    system_prompt=(
                        "You are an independent NLI label and experimental-control "
                        "reviewer. Do not infer or predict student behavior. Return "
                        "one standalone JSON object with no markdown."
                    ),
                    payload=attempt_payload,
                )
                response: dict[str, Any] | None = None
                try:
                    response = self.teacher.generate_json(call)
                    batch_decisions = _parse_probe_validation_batch(
                        response, expected_ids
                    )
                except ValueError as exc:
                    logs.append(teacher_failure_log(self.teacher, call, exc, response))
                    last_error = exc
                    if "Teacher prompt exceeds teacher.max_input_tokens" in str(exc):
                        if len(batch_ids) == 1:
                            failed_validation_probe_sets += 1
                            logs.append(
                                {
                                    "verification_substage": (
                                        "validation_set_abstention"
                                    ),
                                    "hypothesis_id": hypothesis.id,
                                    "probe_set_id": batch_ids[0],
                                    "reason": str(exc),
                                }
                            )
                            tqdm.write(
                                f"Skipping verification QC set {batch_ids[0]} for "
                                f"{hypothesis.id}: its singleton prompt exceeds the "
                                "teacher input limit."
                            )
                            progress.update(1)
                            break
                        midpoint = len(batch_ids) // 2
                        child_batches = [
                            (batch_index, (*split_path, 0), batch_ids[:midpoint]),
                            (batch_index, (*split_path, 1), batch_ids[midpoint:]),
                        ]
                        pending_batches[0:0] = child_batches
                        tqdm.write(
                            f"Verification QC batch {batch_label} for {hypothesis.id} "
                            "exceeds teacher.max_input_tokens; splitting "
                            f"{len(batch_ids)} probe sets into "
                            f"{len(child_batches[0][2])}+{len(child_batches[1][2])}."
                        )
                        split_required = True
                        break
                    if attempt + 1 == self.max_attempts_per_batch:
                        if len(batch_ids) > 1:
                            midpoint = len(batch_ids) // 2
                            child_batches = [
                                (
                                    batch_index,
                                    (*split_path, 0),
                                    batch_ids[:midpoint],
                                ),
                                (
                                    batch_index,
                                    (*split_path, 1),
                                    batch_ids[midpoint:],
                                ),
                            ]
                            pending_batches[0:0] = child_batches
                            tqdm.write(
                                f"Verification QC batch {batch_label} for "
                                f"{hypothesis.id} remained invalid after "
                                f"{self.max_attempts_per_batch} attempts; splitting "
                                f"{len(batch_ids)} probe sets into "
                                f"{len(child_batches[0][2])}+"
                                f"{len(child_batches[1][2])}."
                            )
                            split_required = True
                            break
                        failed_validation_probe_sets += 1
                        logs.append(
                            {
                                "verification_substage": "validation_set_abstention",
                                "hypothesis_id": hypothesis.id,
                                "probe_set_id": batch_ids[0],
                                "attempts": self.max_attempts_per_batch,
                                "reason": str(exc),
                            }
                        )
                        tqdm.write(
                            f"Skipping verification QC set {batch_ids[0]} for "
                            f"{hypothesis.id} after "
                            f"{self.max_attempts_per_batch} invalid attempts."
                        )
                        progress.update(1)
                        break
                    tqdm.write(
                        f"Retrying verification QC batch {batch_label} for "
                        f"{hypothesis.id} after invalid teacher response "
                        f"({attempt + 1}/{self.max_attempts_per_batch}): {exc}"
                    )
                    continue
                decisions.update(batch_decisions)
                logs.append(teacher_exchange_log(self.teacher, call, response))
                progress.update(len(batch_ids))
                break
            if split_required:
                continue
        progress.close()
        return decisions, {
            "hypothesis_id": hypothesis.id,
            "failed_validation_probe_sets": failed_validation_probe_sets,
            "calls": logs,
        }


def _group_probe_sets(
    probes: Sequence[VerificationProbe],
) -> dict[str, list[VerificationProbe]]:
    grouped: dict[str, list[VerificationProbe]] = {}
    ids: set[str] = set()
    for probe in probes:
        if probe.id in ids:
            raise ValueError("Verification probe IDs must be unique")
        ids.add(probe.id)
        grouped.setdefault(probe.probe_set_id, []).append(probe)
    for set_id, members in grouped.items():
        types = [member.probe_type for member in members]
        if sorted(types) != sorted(PROBE_TYPES):
            raise ValueError(f"Probe set {set_id!r} is incomplete or duplicated")
        if len({member.hypothesis_id for member in members}) != 1:
            raise ValueError(f"Probe set {set_id!r} mixes hypotheses")
        members.sort(key=lambda item: PROBE_TYPES.index(item.probe_type))
    return grouped


def _normalized_pair(probe: VerificationProbe) -> str:
    return _normalized_pair_text(probe.premise, probe.hypothesis)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def finalize_valid_probe_sets(
    probes: Sequence[VerificationProbe],
    decisions: Mapping[str, ProbeValidationDecision],
    verification_context: Sequence[Mapping[str, Any]],
    *,
    near_duplicate_threshold: float = 0.92,
    previously_accepted: Sequence[VerificationProbe] = (),
) -> tuple[list[VerificationProbe], list[dict[str, Any]]]:
    """Fail closed per set; validation never sees student predictions."""
    grouped = _group_probe_sets(probes)
    if any(record.get("split_role") != "verification" for record in verification_context):
        raise ValueError("Probe finalization context must remain in verification")
    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("Probe near_duplicate_threshold must be in (0, 1]")
    context_pairs = {
        str(record["id"]): " ".join(
            _TOKEN.findall(
                f"{record['premise']} [pair] {record['hypothesis']}".casefold()
            )
        )
        for record in verification_context
    }
    accepted: list[VerificationProbe] = []
    rejected: list[dict[str, Any]] = []
    previous = list(previously_accepted)
    if len({probe.id for probe in previous}) != len(previous):
        raise ValueError("previously_accepted probe IDs must be unique")
    observed_pairs: dict[str, str] = {
        _normalized_pair(probe): probe.id for probe in previous
    }
    observed_signatures: list[tuple[frozenset[str], str]] = [
        (_token_set(pair), probe_id) for pair, probe_id in observed_pairs.items()
    ]
    for set_id in sorted(grouped):
        members = grouped[set_id]
        reasons: list[str] = []
        local_pairs: set[str] = set()
        for probe in members:
            decision = decisions.get(probe.id)
            if decision is None:
                reasons.append(f"{probe.id}:missing_validation_decision")
            else:
                decision_failed = False
                if decision.validated_label != probe.expected_label:
                    decision_failed = True
                    reasons.append(f"{probe.id}:independent_label_disagreement")
                for field in (
                    "label_valid",
                    "constraint_satisfied",
                    "unambiguous",
                    "grammatical",
                    "original",
                    "suitable_control",
                ):
                    if not getattr(decision, field):
                        decision_failed = True
                        reasons.append(f"{probe.id}:validation_{field}_failed")
                if decision_failed:
                    reasons.extend(
                        f"{probe.id}:validation:{reason}"
                        for reason in decision.reasons
                    )
            pair = _normalized_pair(probe)
            if pair in context_pairs.values():
                reasons.append(f"{probe.id}:copied_verification_context")
            pair_tokens = _token_set(pair)
            for context_id, context_pair in context_pairs.items():
                if (
                    pair != context_pair
                    and _jaccard(pair_tokens, _token_set(context_pair))
                    >= near_duplicate_threshold
                ):
                    reasons.append(
                        f"{probe.id}:near_copy_of_verification_context:{context_id}"
                    )
            if pair in local_pairs:
                reasons.append(f"{probe.id}:duplicate_within_probe_set")
            if pair in observed_pairs:
                reasons.append(
                    f"{probe.id}:duplicate_across_probe_sets:{observed_pairs[pair]}"
                )
            for previous_tokens, previous_id in observed_signatures:
                if _jaccard(pair_tokens, previous_tokens) >= near_duplicate_threshold:
                    reasons.append(
                        f"{probe.id}:near_duplicate_across_probe_sets:{previous_id}"
                    )
            local_pairs.add(pair)
        if reasons:
            rejected.extend(
                {**probe.to_dict(), "rejection_reasons": sorted(set(reasons))}
                for probe in members
            )
        else:
            accepted.extend(members)
        # Only accepted sets participate in the independence audit. A rejected
        # candidate is not part of the statistical sample and must not poison a
        # later original set merely because it was generated first.
        if not reasons:
            for probe in members:
                normalized = _normalized_pair(probe)
                observed_pairs[normalized] = probe.id
                observed_signatures.append((_token_set(normalized), probe.id))
    return accepted, rejected


def predict_probe_sets(
    hypothesis: FailureHypothesis,
    probes: Sequence[VerificationProbe],
    student: ProbePredictor,
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Predict finalized probes and expose target-only and strict-joint outcomes.

    ``z`` remains the historical strict-joint outcome for artifact compatibility.
    Certification code must explicitly choose the configured metric instead of
    assuming that the compatibility field is the selection statistic.
    """
    if batch_size <= 0:
        raise ValueError("Probe prediction batch_size must be positive")
    grouped = _group_probe_sets(probes)
    if any(probe.hypothesis_id != hypothesis.id for probe in probes):
        raise ValueError("Probe collection does not match the tested hypothesis")
    ordered = [probe for set_id in sorted(grouped) for probe in grouped[set_id]]
    raw_predictions = student.predict(
        [probe.premise for probe in ordered],
        [probe.hypothesis for probe in ordered],
        batch_size=batch_size,
    )
    if len(raw_predictions) != len(ordered):
        raise RuntimeError("Student returned a different number of probe predictions")
    prediction_by_id: dict[str, str] = {}
    prediction_rows: list[dict[str, Any]] = []
    for probe, raw_prediction in zip(ordered, raw_predictions, strict=True):
        prediction = normalize_mnli_label(raw_prediction)
        if prediction is None:
            raise ValueError("Student returned an unlabeled probe prediction")
        prediction_by_id[probe.id] = prediction
        prediction_rows.append(
            {
                **probe.to_dict(),
                "split_role": "generated_verification_probe",
                "student_prediction": prediction,
                "student_prediction_hans": mnli_to_hans(prediction),
                "correct": prediction == probe.expected_label,
                "training_eligible": False,
            }
        )
    outcomes: list[dict[str, Any]] = []
    for set_id in sorted(grouped):
        members = {probe.probe_type: probe for probe in grouped[set_id]}
        target = members["target"]
        target_prediction = prediction_by_id[target.id]
        target_failure = (
            mnli_to_hans(target_prediction) == target.expected_failure_label
            and mnli_to_hans(target_prediction) != mnli_to_hans(target.expected_label)
        )
        counterfactual_correct = (
            prediction_by_id[members["counterfactual_control"].id]
            == members["counterfactual_control"].expected_label
        )
        invariant_correct = (
            prediction_by_id[members["invariant_control"].id]
            == members["invariant_control"].expected_label
        )
        strict_joint = (
            target_failure
            and counterfactual_correct
            and invariant_correct
        )
        outcomes.append(
            {
                "hypothesis_id": hypothesis.id,
                "probe_set_id": set_id,
                "target_probe_id": target.id,
                "counterfactual_control_id": members["counterfactual_control"].id,
                "invariant_control_id": members["invariant_control"].id,
                "target_failure_observed": target_failure,
                "counterfactual_control_correct": counterfactual_correct,
                "invariant_control_correct": invariant_correct,
                "target_failure_z": int(target_failure),
                "strict_joint_z": int(strict_joint),
                "z": int(strict_joint),
                "z_definition": (
                    "target exhibits the predeclared wrong HANS label AND both "
                    "counterfactual/invariant controls receive their exact MNLI labels"
                ),
            }
        )
    return prediction_rows, outcomes
