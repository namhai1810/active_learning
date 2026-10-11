"""End-to-end HANS evaluation."""

from __future__ import annotations

from typing import Any, Protocol

from src.data.hans import HANSExample
from src.data.labels import mnli_to_hans
from src.evaluation.metrics import compute_hans_metrics


class NLIPredictor(Protocol):
    def predict(
        self,
        premises: list[str],
        hypotheses: list[str],
        batch_size: int,
    ) -> list[str]: ...


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
