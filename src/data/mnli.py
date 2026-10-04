"""MNLI loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.data.labels import normalize_mnli_label


@dataclass(frozen=True)
class MNLIExample:
    id: str
    premise: str
    hypothesis: str
    label: str | None


def parse_mnli_example(
    record: Mapping[str, Any], fallback_id: str | int
) -> MNLIExample:
    """Validate and normalize one Hugging Face MNLI record."""
    try:
        premise = str(record["premise"])
        hypothesis = str(record["hypothesis"])
    except KeyError as exc:
        raise ValueError(f"MNLI record is missing field {exc.args[0]!r}") from exc

    raw_label = record.get("label", -1)
    label = normalize_mnli_label(raw_label)
    example_id = record.get("idx", record.get("id", fallback_id))
    return MNLIExample(
        id=str(example_id),
        premise=premise,
        hypothesis=hypothesis,
        label=label,
    )


def load_mnli(
    split: str = "train",
    dataset_name: str = "nyu-mll/glue",
    dataset_config: str = "mnli",
    cache_dir: str | None = None,
    max_examples: int | None = None,
    drop_unlabeled: bool = True,
) -> list[MNLIExample]:
    """Load an MNLI split from the Hugging Face datasets hub/cache."""
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative or None")
    if max_examples == 0:
        return []

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=cache_dir,
    )
    examples: list[MNLIExample] = []
    for index, record in enumerate(dataset):
        example = parse_mnli_example(record, fallback_id=f"{split}-{index}")
        if drop_unlabeled and example.label is None:
            continue
        examples.append(example)
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples
