"""HANS loader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.data.labels import normalize_hans_label

HANS_HEURISTICS = ("lexical_overlap", "subsequence", "constituent")


@dataclass(frozen=True)
class HANSExample:
    id: str
    premise: str
    hypothesis: str
    gold: str
    heuristic: str
    subcase: str


def _normalize_metadata(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def parse_hans_example(
    record: Mapping[str, Any], fallback_id: str | int
) -> HANSExample:
    """Validate one HANS record in HF-normalized or official TSV form."""
    premise = record.get("premise", record.get("sentence1"))
    hypothesis = record.get("hypothesis", record.get("sentence2"))
    raw_label = record.get("label", record.get("gold_label"))
    resolved = {
        "premise/sentence1": premise,
        "hypothesis/sentence2": hypothesis,
        "label/gold_label": raw_label,
        "heuristic": record.get("heuristic"),
        "subcase": record.get("subcase"),
    }
    missing = [field for field, value in resolved.items() if value is None]
    if missing:
        raise ValueError(f"HANS record is missing fields: {', '.join(missing)}")

    heuristic = _normalize_metadata(record["heuristic"])
    if heuristic not in HANS_HEURISTICS:
        raise ValueError(f"Unknown HANS heuristic: {record['heuristic']!r}")
    example_id = record.get("pairID", record.get("id", fallback_id))
    return HANSExample(
        id=str(example_id),
        premise=str(premise),
        hypothesis=str(hypothesis),
        gold=normalize_hans_label(raw_label),
        heuristic=heuristic,
        subcase=str(record["subcase"]),
    )


def load_hans(
    split: str = "validation",
    dataset_name: str = "hans",
    data_file: str | None = None,
    cache_dir: str | None = None,
    max_examples: int | None = None,
) -> list[HANSExample]:
    """Load a HANS split from the Hugging Face datasets hub/cache."""
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative or None")
    if max_examples == 0:
        return []

    from datasets import load_dataset

    if data_file:
        dataset = load_dataset(
            "csv",
            data_files={split: data_file},
            delimiter="\t",
            split=split,
            cache_dir=cache_dir,
        )
    else:
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    examples: list[HANSExample] = []
    for index, record in enumerate(dataset):
        examples.append(
            parse_hans_example(record, fallback_id=f"{split}-{index}")
        )
        if max_examples is not None and len(examples) >= max_examples:
            break
    return examples
