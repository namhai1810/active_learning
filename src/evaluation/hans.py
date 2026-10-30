"""End-to-end HANS evaluation."""

from __future__ import annotations

from typing import Any, Protocol

from src.data.hans import HANSExample
from src.data.labels import MNLI_LABELS, mnli_to_hans
from src.evaluation.metrics import compute_hans_metrics


class NLIPredictor(Protocol):
    def predict(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int,
    ) -> list[str]: ...


class NLIProbabilityPredictor(Protocol):
    def predict_proba(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int,
    ) -> list[list[float]]: ...


def evaluate_hans(
    student: NLIPredictor,
    examples: list[HANSExample],
    batch_size: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the student and return serializable prediction rows and metrics."""
    if not examples:
        raise ValueError("HANS evaluation requires at least one example")
    mnli_predictions = student.predict(
        [example.premise for example in examples],
        [example.hypothesis for example in examples],
        batch_size=batch_size,
    )
    if len(mnli_predictions) != len(examples):
        raise RuntimeError(
            "Student returned a different number of predictions than examples"
        )
    hans_predictions = [mnli_to_hans(label) for label in mnli_predictions]

    records = []
    for example, mnli_prediction, prediction in zip(
        examples, mnli_predictions, hans_predictions, strict=True
    ):
        records.append(
            {
                "id": example.id,
                "premise": example.premise,
                "hypothesis": example.hypothesis,
                "gold": example.gold,
                "prediction": prediction,
                "correct": prediction == example.gold,
                "heuristic": example.heuristic,
                "subcase": example.subcase,
                "mnli_prediction": mnli_prediction,
            }
        )

    metrics = compute_hans_metrics(
        [record["gold"] for record in records],
        [record["prediction"] for record in records],
        [record["heuristic"] for record in records],
        [record["subcase"] for record in records],
    )
    return records, metrics


def evaluate_hans_prediction_rules(
    student: NLIProbabilityPredictor,
    examples: list[HANSExample],
    batch_size: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate both predeclared HANS mappings from one probability pass.

    The primary rule remains three-way MNLI argmax followed by the HANS label
    mapping.  The direct-binary diagnostic compares P(E) with P(N)+P(C).
    Ties favor entailment, matching first-index argmax semantics.
    """
    if not examples:
        raise ValueError("HANS evaluation requires at least one example")
    probabilities = student.predict_proba(
        [example.premise for example in examples],
        [example.hypothesis for example in examples],
        batch_size=batch_size,
    )
    if len(probabilities) != len(examples):
        raise RuntimeError(
            "Student returned a different number of probabilities than examples"
        )
    entailment_index = MNLI_LABELS.index("entailment")
    neutral_index = MNLI_LABELS.index("neutral")
    contradiction_index = MNLI_LABELS.index("contradiction")
    records: list[dict[str, Any]] = []
    three_way: list[str] = []
    direct_binary: list[str] = []
    for example, raw in zip(examples, probabilities, strict=True):
        values = [float(value) for value in raw]
        if len(values) != len(MNLI_LABELS):
            raise RuntimeError("Student probability rows must follow MNLI_LABELS")
        mnli_index = max(range(len(values)), key=values.__getitem__)
        mnli_prediction = MNLI_LABELS[mnli_index]
        three_prediction = mnli_to_hans(mnli_prediction)
        non_entailment = values[neutral_index] + values[contradiction_index]
        binary_prediction = (
            "entailment"
            if values[entailment_index] >= non_entailment
            else "non-entailment"
        )
        three_way.append(three_prediction)
        direct_binary.append(binary_prediction)
        records.append(
            {
                "id": example.id,
                "premise": example.premise,
                "hypothesis": example.hypothesis,
                "gold": example.gold,
                "prediction": three_prediction,
                "correct": three_prediction == example.gold,
                "heuristic": example.heuristic,
                "subcase": example.subcase,
                "mnli_prediction": mnli_prediction,
                "probabilities": values,
                "three_way_to_binary_prediction": three_prediction,
                "direct_binary_prediction": binary_prediction,
                "direct_binary_correct": binary_prediction == example.gold,
            }
        )

    def metrics(predictions: list[str]) -> dict[str, Any]:
        return compute_hans_metrics(
            [example.gold for example in examples],
            predictions,
            [example.heuristic for example in examples],
            [example.subcase for example in examples],
        )

    different = sum(left != right for left, right in zip(three_way, direct_binary))
    return records, {
        "three_way_to_binary": metrics(three_way),
        "direct_binary": metrics(direct_binary),
        "disagreement": {
            "count": different,
            "rate": different / len(examples),
            "num_examples": len(examples),
        },
    }
