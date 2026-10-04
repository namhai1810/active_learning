import pytest

from src.evaluation.metrics import compute_hans_metrics


def test_compute_hans_metrics():
    metrics = compute_hans_metrics(
        gold_labels=[
            "entailment",
            "non-entailment",
            "entailment",
            "non-entailment",
            "entailment",
            "non-entailment",
        ],
        predictions=[
            "entailment",
            "entailment",
            "non-entailment",
            "non-entailment",
            "entailment",
            "non-entailment",
        ],
        heuristics=[
            "lexical_overlap",
            "lexical_overlap",
            "subsequence",
            "subsequence",
            "constituent",
            "constituent",
        ],
    )

    assert metrics["num_examples"] == 6
    assert metrics["num_correct"] == 4
    assert metrics["overall_accuracy"] == pytest.approx(4 / 6)
    assert metrics["entailment_accuracy"] == pytest.approx(2 / 3)
    assert metrics["non_entailment_accuracy"] == pytest.approx(2 / 3)
    assert metrics["accuracy_by_heuristic"] == {
        "lexical_overlap": pytest.approx(0.5),
        "subsequence": pytest.approx(0.5),
        "constituent": pytest.approx(1.0),
    }


def test_compute_hans_metrics_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        compute_hans_metrics(["entailment"], [], ["lexical_overlap"])


def test_compute_hans_metrics_uses_null_for_absent_group():
    metrics = compute_hans_metrics(
        ["entailment"], ["entailment"], ["lexical_overlap"]
    )
    assert metrics["non_entailment_accuracy"] is None
    assert metrics["accuracy_by_heuristic"]["subsequence"] is None
