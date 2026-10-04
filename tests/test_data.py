import pytest

from src.data.hans import HANSExample, parse_hans_example
from src.data.mnli import MNLIExample, parse_mnli_example


def test_parse_mnli_example():
    record = {
        "idx": 17,
        "premise": "A musician performs.",
        "hypothesis": "A person performs.",
        "label": 0,
    }
    assert parse_mnli_example(record, "fallback") == MNLIExample(
        id="17",
        premise="A musician performs.",
        hypothesis="A person performs.",
        label="entailment",
    )


def test_parse_hans_example():
    record = {
        "premise": "The doctor saw the lawyer.",
        "hypothesis": "The lawyer saw the doctor.",
        "label": 1,
        "heuristic": "lexical_overlap",
        "subcase": "subject_object_swap",
    }
    assert parse_hans_example(record, "validation-3") == HANSExample(
        id="validation-3",
        premise="The doctor saw the lawyer.",
        hypothesis="The lawyer saw the doctor.",
        gold="non-entailment",
        heuristic="lexical_overlap",
        subcase="subject_object_swap",
    )


def test_parse_official_hans_tsv_shape():
    record = {
        "sentence1": "The doctor saw the lawyer.",
        "sentence2": "The lawyer saw the doctor.",
        "gold_label": "non-entailment",
        "pairID": "ex123",
        "heuristic": "lexical_overlap",
        "subcase": "subject_object_swap",
    }
    parsed = parse_hans_example(record, "fallback")
    assert parsed.id == "ex123"
    assert parsed.premise == record["sentence1"]
    assert parsed.hypothesis == record["sentence2"]
    assert parsed.gold == "non-entailment"


def test_parse_hans_rejects_missing_metadata():
    with pytest.raises(ValueError, match="missing fields"):
        parse_hans_example({"premise": "p", "hypothesis": "h"}, "0")
