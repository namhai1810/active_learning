import pytest

from src.data.labels import (
    mnli_to_hans,
    normalize_hans_label,
    normalize_mnli_label,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0, "entailment"),
        (1, "neutral"),
        (2, "contradiction"),
        (-1, None),
        ("Entailment", "entailment"),
    ],
)
def test_normalize_mnli_label(raw, expected):
    assert normalize_mnli_label(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0, "entailment"), (1, "non-entailment"), ("non_entailment", "non-entailment")],
)
def test_normalize_hans_label(raw, expected):
    assert normalize_hans_label(raw) == expected


@pytest.mark.parametrize(
    ("mnli_label", "hans_label"),
    [
        ("entailment", "entailment"),
        ("neutral", "non-entailment"),
        ("contradiction", "non-entailment"),
    ],
)
def test_mnli_to_hans(mnli_label, hans_label):
    assert mnli_to_hans(mnli_label) == hans_label


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        normalize_mnli_label("maybe")
