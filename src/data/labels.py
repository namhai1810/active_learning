"""Canonical MNLI/HANS labels and conversion helpers."""

from __future__ import annotations

from typing import Final

MNLI_LABELS: Final[tuple[str, ...]] = (
    "entailment",
    "neutral",
    "contradiction",
)
HANS_LABELS: Final[tuple[str, ...]] = ("entailment", "non-entailment")

_MNLI_INTEGER_LABELS: Final[dict[int, str]] = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}
_HANS_INTEGER_LABELS: Final[dict[int, str]] = {
    0: "entailment",
    1: "non-entailment",
}


def _canonicalize(label: str) -> str:
    return label.strip().lower().replace("_", "-").replace(" ", "-")


def normalize_mnli_label(label: int | str) -> str | None:
    """Return a canonical MNLI label; ``-1`` denotes an unlabeled example."""
    if isinstance(label, bool):
        raise ValueError(f"Boolean is not a valid MNLI label: {label!r}")
    if isinstance(label, int):
        if label == -1:
            return None
        try:
            return _MNLI_INTEGER_LABELS[label]
        except KeyError as exc:
            raise ValueError(f"Unknown integer MNLI label: {label}") from exc

    canonical = _canonicalize(str(label))
    if canonical in {"-1", "unknown", "unlabeled"}:
        return None
    if canonical not in MNLI_LABELS:
        raise ValueError(f"Unknown MNLI label: {label!r}")
    return canonical


def normalize_hans_label(label: int | str) -> str:
    """Return a canonical binary HANS label."""
    if isinstance(label, bool):
        raise ValueError(f"Boolean is not a valid HANS label: {label!r}")
    if isinstance(label, int):
        try:
            return _HANS_INTEGER_LABELS[label]
        except KeyError as exc:
            raise ValueError(f"Unknown integer HANS label: {label}") from exc

    canonical = _canonicalize(str(label))
    aliases = {
        "nonentailment": "non-entailment",
        "not-entailment": "non-entailment",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in HANS_LABELS:
        raise ValueError(f"Unknown HANS label: {label!r}")
    return canonical


def mnli_to_hans(label: int | str) -> str:
    """Map three-way MNLI predictions to HANS's binary label space."""
    canonical = normalize_mnli_label(label)
    if canonical is None:
        raise ValueError("An unlabeled MNLI value cannot be used as a prediction")
    return "entailment" if canonical == "entailment" else "non-entailment"
