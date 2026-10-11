"""Serializable synthetic NLI data types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from src.data.labels import MNLI_LABELS
from src.data.mnli import MNLIExample


SYNTHETIC_PAIR_MEMBERS = {
    "counterfactual": ("base", "relation_changed"),
    "invariant": ("base", "irrelevant_change"),
    "reasoning_contrast": ("base", "reasoning_contrast"),
}


@dataclass(frozen=True)
class SyntheticNLIExample:
    id: str
    premise: str
    hypothesis: str
    label: str
    hypothesis_id: str
    pair_id: str
    pair_type: str
    pair_member: str
    pair_rationale: str
    seed_example_ids: tuple[str, ...]
    label_rationale: str
    targeting_rationale: str
    generator_prompt_version: str

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.hypothesis_id.strip() or not self.pair_id.strip():
            raise ValueError("Synthetic IDs must be non-empty")
        if self.pair_type not in SYNTHETIC_PAIR_MEMBERS:
            raise ValueError(f"Invalid synthetic pair_type: {self.pair_type!r}")
        if self.pair_member not in SYNTHETIC_PAIR_MEMBERS[self.pair_type]:
            raise ValueError(
                f"Invalid member {self.pair_member!r} for {self.pair_type!r}"
            )
        if not self.premise.strip() or not self.hypothesis.strip():
            raise ValueError("Synthetic premise and hypothesis must be non-empty")
        if self.label not in MNLI_LABELS:
            raise ValueError(f"Invalid synthetic MNLI label: {self.label!r}")
        if not self.seed_example_ids:
            raise ValueError("Synthetic example must cite at least one permitted seed")
        if len(set(self.seed_example_ids)) != len(self.seed_example_ids):
            raise ValueError("Synthetic seed IDs must be unique")
        if (
            not self.pair_rationale.strip()
            or not self.label_rationale.strip()
            or not self.targeting_rationale.strip()
        ):
            raise ValueError("Synthetic rationales must be non-empty")
        if not self.generator_prompt_version.strip():
            raise ValueError("generator_prompt_version must be non-empty")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SyntheticNLIExample":
        allowed = {
            "id",
            "premise",
            "hypothesis",
            "label",
            "hypothesis_id",
            "pair_id",
            "pair_type",
            "pair_member",
            "pair_rationale",
            "seed_example_ids",
            "label_rationale",
            "targeting_rationale",
            "generator_prompt_version",
        }
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"Unexpected synthetic fields: {sorted(extra)}")
        string_fields = allowed - {"seed_example_ids"}
        if any(not isinstance(value.get(field), str) for field in string_fields):
            raise ValueError("Synthetic text and ID fields must be strings")
        seed_ids = value.get("seed_example_ids")
        if not isinstance(seed_ids, (list, tuple)) or any(
            not isinstance(item, str) for item in seed_ids
        ):
            raise ValueError("seed_example_ids must be a string list")
        try:
            return cls(
                id=str(value["id"]),
                premise=str(value["premise"]),
                hypothesis=str(value["hypothesis"]),
                label=str(value["label"]),
                hypothesis_id=str(value["hypothesis_id"]),
                pair_id=str(value["pair_id"]),
                pair_type=str(value["pair_type"]),
                pair_member=str(value["pair_member"]),
                pair_rationale=str(value["pair_rationale"]),
                seed_example_ids=tuple(
                    str(item) for item in value["seed_example_ids"]
                ),
                label_rationale=str(value["label_rationale"]),
                targeting_rationale=str(value["targeting_rationale"]),
                generator_prompt_version=str(value["generator_prompt_version"]),
            )
        except KeyError as exc:
            raise ValueError(
                f"Synthetic example is missing {exc.args[0]!r}"
            ) from exc

    def to_mnli(self) -> MNLIExample:
        return MNLIExample(
            id=self.id,
            premise=self.premise,
            hypothesis=self.hypothesis,
            label=self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
