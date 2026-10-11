"""Three-class MNLI evaluation used by the Stage 2 benchmark."""

from __future__ import annotations

from typing import Any

from src.data.mnli import MNLIExample


def evaluate_mnli(
    student: Any, examples: list[MNLIExample], *, batch_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not examples:
        raise ValueError("MNLI evaluation requires at least one example")
    if any(item.label is None for item in examples):
        raise ValueError("MNLI evaluation requires labeled examples")
    predictions = student.predict(
        [item.premise for item in examples],
        [item.hypothesis for item in examples],
        batch_size=batch_size,
    )
    if len(predictions) != len(examples):
        raise RuntimeError("MNLI prediction count does not match evaluation data")
    rows = [
        {
            "id": item.id,
            "premise": item.premise,
            "hypothesis": item.hypothesis,
            "gold": item.label,
            "prediction": prediction,
            "correct": prediction == item.label,
        }
        for item, prediction in zip(examples, predictions, strict=True)
    ]
    return rows, {
        "num_examples": len(rows),
        "num_correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
    }
