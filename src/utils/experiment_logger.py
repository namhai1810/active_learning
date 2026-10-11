"""Local experiment artifacts shared by all acquisition baselines."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from src.utils.io import _atomic_text_write, write_json, write_jsonl

ROUND_FIELDS = (
    "round",
    "method",
    "seed",
    "acquisition_budget",
    "num_acquired",
    "train_loss",
    "train_examples",
    "hans_accuracy",
    "hans_entailment_accuracy",
    "hans_non_entailment_accuracy",
    "lexical_overlap_accuracy",
    "subsequence_accuracy",
    "constituent_accuracy",
    "gpu_seconds",
    "wall_clock_seconds",
)

FINAL_METRIC_FIELDS = (
    "method",
    "seed",
    "acquisition_budget",
    "num_acquired",
    "hans_accuracy",
    "hans_entailment_accuracy",
    "hans_non_entailment_accuracy",
    "lexical_overlap_accuracy",
    "subsequence_accuracy",
    "constituent_accuracy",
)

RUN_ARTIFACTS = (
    "config.yaml",
    "run.json",
    "rounds.jsonl",
    "metrics.json",
    "predictions.jsonl",
    "selected_example_ids.json",
    "baseline_result.json",  # Legacy Milestone 2 artifact.
)


def _validate_schema(
    record: Mapping[str, Any], fields: tuple[str, ...], record_name: str
) -> dict[str, Any]:
    missing = [field for field in fields if field not in record]
    extra = [field for field in record if field not in fields]
    if missing or extra:
        raise ValueError(
            f"Invalid {record_name} schema; missing={missing}, extra={extra}"
        )
    return {field: record[field] for field in fields}


def make_evaluation_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Map the unchanged Milestone 1 metrics to the unified flat schema."""
    by_heuristic = metrics["accuracy_by_heuristic"]
    return {
        "hans_accuracy": metrics["overall_accuracy"],
        "hans_entailment_accuracy": metrics["entailment_accuracy"],
        "hans_non_entailment_accuracy": metrics["non_entailment_accuracy"],
        "lexical_overlap_accuracy": by_heuristic["lexical_overlap"],
        "subsequence_accuracy": by_heuristic["subsequence"],
        "constituent_accuracy": by_heuristic["constituent"],
    }


def make_round_record(
    *,
    round_number: int,
    method: str,
    seed: int,
    acquisition_budget: int,
    num_acquired: int,
    train_loss: float | None,
    train_examples: int | None,
    evaluation_metrics: Mapping[str, Any],
    gpu_seconds: float | None,
    wall_clock_seconds: float,
) -> dict[str, Any]:
    record = {
        "round": round_number,
        "method": method,
        "seed": seed,
        "acquisition_budget": acquisition_budget,
        "num_acquired": num_acquired,
        "train_loss": train_loss,
        "train_examples": train_examples,
        **make_evaluation_fields(evaluation_metrics),
        "gpu_seconds": gpu_seconds,
        "wall_clock_seconds": wall_clock_seconds,
    }
    return _validate_schema(record, ROUND_FIELDS, "round record")


def make_final_metrics(
    *,
    method: str,
    seed: int,
    acquisition_budget: int,
    num_acquired: int,
    evaluation_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = {
        "method": method,
        "seed": seed,
        "acquisition_budget": acquisition_budget,
        "num_acquired": num_acquired,
        **make_evaluation_fields(evaluation_metrics),
    }
    return _validate_schema(metrics, FINAL_METRIC_FIELDS, "final metrics")


class ExperimentLogger:
    """Write one reproducible, local run directory.

    Existing non-empty directories are rejected unless ``overwrite`` is
    explicitly enabled by the caller. Even then, unrelated files are not
    deleted automatically.
    """

    def __init__(
        self,
        output_dir: str | Path,
        config: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        if (
            self.output_dir.exists()
            and any(self.output_dir.iterdir())
            and not overwrite
        ):
            raise FileExistsError(
                f"Refusing to overwrite non-empty run directory: "
                f"{self.output_dir}. Pass --overwrite explicitly to replace "
                "this run's logger artifacts."
            )
        if overwrite and self.output_dir.exists():
            for filename in RUN_ARTIFACTS:
                path = self.output_dir / filename
                if path.is_file() or path.is_symlink():
                    path.unlink()
            adapter_path = self.output_dir / "adapter"
            if adapter_path.is_dir():
                shutil.rmtree(adapter_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        self._overwrite = overwrite

    def save_config(self) -> None:
        content = yaml.safe_dump(
            self.config,
            sort_keys=False,
            allow_unicode=True,
        )
        _atomic_text_write(self.output_dir / "config.yaml", content)

    def save_run_metadata(self, metadata: Mapping[str, Any]) -> None:
        write_json(self.output_dir / "run.json", metadata)

    def initialize_rounds(self) -> None:
        """Create a fresh rounds file for this explicitly authorized run."""
        _atomic_text_write(self.output_dir / "rounds.jsonl", "")

    def log_round(self, record: Mapping[str, Any]) -> None:
        validated = _validate_schema(record, ROUND_FIELDS, "round record")
        path = self.output_dir / "rounds.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(validated, ensure_ascii=False, sort_keys=True) + "\n"
            )

    def save_metrics(self, metrics: Mapping[str, Any]) -> None:
        validated = _validate_schema(
            metrics, FINAL_METRIC_FIELDS, "final metrics"
        )
        write_json(self.output_dir / "metrics.json", validated)

    def save_predictions(self, predictions: Sequence[Mapping[str, Any]]) -> None:
        write_jsonl(self.output_dir / "predictions.jsonl", predictions)

    def save_selected_ids(
        self, method: str, budget: int, ids: Sequence[str]
    ) -> None:
        write_json(
            self.output_dir / "selected_example_ids.json",
            {"method": method, "budget": budget, "ids": list(ids)},
        )
