"""One-shot generation, validation, and scoring of controlled NLI probes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.data.labels import HANS_LABELS, MNLI_LABELS, mnli_to_hans, normalize_mnli_label
from src.hypotheses.types import FailureHypothesis
from src.teacher.base import TeacherCall, TeacherLLM


VERIFICATION_PROBE_PROMPT_VERSION = "controlled_verification_probes_v1"
VERIFICATION_LABEL_PROMPT_VERSION = "verification_probe_validation_v1"
PROBE_TYPES = ("target", "counterfactual_control", "invariant_control")
_TOKEN = re.compile(r"[\w']+", re.UNICODE)
_RAW_PROBE_FIELDS = {
    "premise",
    "hypothesis",
    "expected_label",
    "expected_failure_label",
    "label_rationale",
    "constraint_rationale",
}


@dataclass(frozen=True)
class ControlledProbeConfig:
    n_probe_sets: int
    min_valid_probe_sets: int
    max_context_records: int = 24
    max_probe_sets_per_call: int = 8
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


class ControlledVerificationProbeGenerator:
    """Generate fixed probe sets before any student prediction is observed."""

    def __init__(self, teacher: TeacherLLM, config: ControlledProbeConfig) -> None:
        self.teacher = teacher
        self.config = config

    def generate(
        self,
        hypothesis: FailureHypothesis,
        verification_context: Sequence[Mapping[str, Any]],
    ) -> tuple[list[VerificationProbe], dict[str, Any]]:
        hypothesis.validate(require_llm_schema=True)
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
        permitted_ids = {str(record["id"]) for record in selected_context}
        system_prompt = (
            "Generate NEW controlled three-class NLI verification probes for one "
            "pre-registered failure hypothesis. Each probe set must contain exactly "
            "one target, one counterfactual/positive control, and one invariant "
            "control. Sets must use independent scenarios and must not copy or lightly "
            "paraphrase context records. State labels before any student is queried. "
            "Return one standalone JSON object and no markdown."
        )
        generated: list[VerificationProbe] = []
        logs: list[dict[str, Any]] = []
        remaining = self.config.n_probe_sets
        batch_index = 0
        while remaining:
            requested = min(remaining, self.config.max_probe_sets_per_call)
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
                "requested_probe_sets": requested,
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
                    "controls_expected_failure_label": None,
                },
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
                "verification_context_only": selected_context,
            }
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.config.seed}:{hypothesis.id}:probe-generation:{batch_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task="controlled_verification_probe_generation",
                prompt_version=self.config.prompt_version,
                seed=derived_seed,
                system_prompt=system_prompt,
                payload=payload,
            )
            response = self.teacher.generate_json(call)
            if set(response) != {"probe_sets"} or not isinstance(
                response["probe_sets"], list
            ):
                raise ValueError("Probe generator response must contain only probe_sets[]")
            if len(response["probe_sets"]) != requested:
                raise ValueError("Probe generator returned a different set count")
            for raw_set in response["probe_sets"]:
                if not isinstance(raw_set, dict) or set(raw_set) != {
                    "seed_context_ids",
                    "mechanism_rationale",
                    *PROBE_TYPES,
                }:
                    raise ValueError("Generated probe set violates its schema")
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
                    if not isinstance(raw, dict) or set(raw) != _RAW_PROBE_FIELDS:
                        raise ValueError("Generated probe violates its schema")
                    raw_probes.append(raw)
                set_id = _probe_set_id(
                    hypothesis.id, raw_probes, len(generated) // len(PROBE_TYPES)
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
                        None if failure_label is None else str(failure_label).strip().lower()
                    )
                    if (
                        probe_type == "target"
                        and normalized_failure not in hypothesis.predicted_failure_labels
                    ):
                        raise ValueError(
                            "Target failure label was not declared by the hypothesis"
                        )
                    generated.append(
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
                            generator_prompt_version=self.config.prompt_version,
                        )
                    )
            logs.append({"teacher_call": call.to_dict(), "teacher_response": response})
            remaining -= requested
            batch_index += 1
        return generated, {
            "hypothesis_id": hypothesis.id,
            "selected_context_ids": [str(row["id"]) for row in selected_context],
            "generation_finalized_without_student_predictions": True,
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
    ) -> None:
        if max_probe_sets_per_call <= 0 or not prompt_version.strip():
            raise ValueError("Probe validator call size/version is invalid")
        self.teacher = teacher
        self.seed = seed
        self.prompt_version = prompt_version
        self.max_probe_sets_per_call = max_probe_sets_per_call

    def validate(
        self,
        hypothesis: FailureHypothesis,
        probes: Sequence[VerificationProbe],
    ) -> tuple[dict[str, ProbeValidationDecision], dict[str, Any]]:
        grouped = _group_probe_sets(probes)
        set_ids = sorted(grouped)
        decisions: dict[str, ProbeValidationDecision] = {}
        logs: list[dict[str, Any]] = []
        required_fields = {
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
        for batch_index, start in enumerate(
            range(0, len(set_ids), self.max_probe_sets_per_call)
        ):
            batch_ids = set_ids[start : start + self.max_probe_sets_per_call]
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
            derived_seed = int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}:{hypothesis.id}:probe-validation:{batch_index}".encode(
                        "utf-8"
                    )
                ).digest()[:4],
                "big",
            )
            call = TeacherCall(
                task="verification_probe_validation",
                prompt_version=self.prompt_version,
                seed=derived_seed,
                system_prompt=(
                    "You are an independent NLI label and experimental-control "
                    "reviewer. Do not infer or predict student behavior. Return one "
                    "standalone JSON object with no markdown."
                ),
                payload=payload,
            )
            response = self.teacher.generate_json(call)
            if set(response) != {"decisions"} or not isinstance(
                response["decisions"], list
            ):
                raise ValueError("Probe validation response must contain decisions[]")
            expected_ids = {probe.id for probe in batch}
            returned_ids: set[str] = set()
            for raw in response["decisions"]:
                if not isinstance(raw, dict) or set(raw) != required_fields:
                    raise ValueError("Probe validation decision violates its schema")
                probe_id = str(raw["probe_id"])
                if probe_id in returned_ids or probe_id not in expected_ids:
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
                returned_ids.add(probe_id)
                decisions[probe_id] = ProbeValidationDecision(
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
            if returned_ids != expected_ids:
                raise ValueError("Probe validator must decide every supplied probe")
            logs.append({"teacher_call": call.to_dict(), "teacher_response": response})
        return decisions, {"hypothesis_id": hypothesis.id, "calls": logs}


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
    return " ".join(
        _TOKEN.findall(f"{probe.premise} [pair] {probe.hypothesis}".casefold())
    )


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
    observed_pairs: dict[str, str] = {}
    observed_signatures: list[tuple[frozenset[str], str]] = []
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
    """Predict finalized probes and construct one control-aware Z per probe set."""
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
                "z": int(
                    target_failure
                    and counterfactual_correct
                    and invariant_correct
                ),
                "z_definition": (
                    "target exhibits the predeclared wrong HANS label AND both "
                    "counterfactual/invariant controls receive their exact MNLI labels"
                ),
            }
        )
    return prediction_rows, outcomes
