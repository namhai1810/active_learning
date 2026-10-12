import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_baseline_run(root: Path, run_name: str, method: str, wrong_id: str | None):
    run = root / run_name
    run.mkdir(parents=True)
    definitions = (
        ("0", "entailment", "lexical_overlap"),
        ("1", "non-entailment", "lexical_overlap"),
        ("2", "entailment", "subsequence"),
        ("3", "non-entailment", "subsequence"),
        ("4", "entailment", "constituent"),
        ("5", "non-entailment", "constituent"),
    )
    rows = []
    for example_id, gold, heuristic in definitions:
        correct = example_id != wrong_id
        rows.append(
            {
                "id": example_id,
                "premise": f"premise {example_id}",
                "hypothesis": f"hypothesis {example_id}",
                "gold": gold,
                "prediction": gold if correct else (
                    "non-entailment" if gold == "entailment" else "entailment"
                ),
                "correct": correct,
                "heuristic": heuristic,
                "subcase": f"{heuristic}_case",
                "mnli_prediction": "entailment",
            }
        )
    (run / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    def accuracy(predicate):
        selected = [row for row in rows if predicate(row)]
        return sum(row["correct"] for row in selected) / len(selected)

    acquired = 0 if method in {"base", "zero_shot"} else 32
    common_metrics = {
        "method": method,
        "seed": 42,
        "overall": accuracy(lambda row: True),
        "entailment": accuracy(lambda row: row["gold"] == "entailment"),
        "non_entailment": accuracy(
            lambda row: row["gold"] == "non-entailment"
        ),
        "lexical_overlap": accuracy(
            lambda row: row["heuristic"] == "lexical_overlap"
        ),
        "subsequence": accuracy(
            lambda row: row["heuristic"] == "subsequence"
        ),
        "constituent": accuracy(
            lambda row: row["heuristic"] == "constituent"
        ),
    }
    metrics = (
        {
            "overall_accuracy": common_metrics["overall"],
            "entailment_accuracy": common_metrics["entailment"],
            "non_entailment_accuracy": common_metrics["non_entailment"],
            "accuracy_by_heuristic": {
                name: common_metrics[name]
                for name in ("lexical_overlap", "subsequence", "constituent")
            },
        }
        if method == "zero_shot"
        else {
            "method": method,
            "seed": 42,
            "acquisition_budget": 32,
            "num_acquired": acquired,
            "hans_accuracy": common_metrics["overall"],
            "hans_entailment_accuracy": common_metrics["entailment"],
            "hans_non_entailment_accuracy": common_metrics["non_entailment"],
            "lexical_overlap_accuracy": common_metrics["lexical_overlap"],
            "subsequence_accuracy": common_metrics["subsequence"],
            "constituent_accuracy": common_metrics["constituent"],
        }
    )
    _write_json(
        run / "metrics.json",
        metrics,
    )
    _write_json(
        run / "run.json",
        {
            "seed": 42,
            "student_model": "toy",
            "device": "cpu",
            "gpu_name": None,
            "git_commit": "abc123",
            "wall_clock_seconds": 1.0,
            "gpu_seconds": None,
        },
    )
    (run / "rounds.jsonl").write_text(
        json.dumps({"train_loss": None, "wall_clock_seconds": 1.0}) + "\n",
        encoding="utf-8",
    )


def test_reporter_aggregates_and_visualizes_three_independent_baselines(tmp_path):
    outputs = tmp_path / "outputs"
    _write_baseline_run(outputs, "baseline_base_b32_seed42", "base", None)
    _write_baseline_run(outputs, "baseline_random_b32_seed42", "random", "0")
    _write_baseline_run(
        outputs, "baseline_uncertainty_b32_seed42", "uncertainty", "1"
    )
    report = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "report_hans_baselines.py"),
            "--outputs",
            str(outputs),
            "--report-dir",
            str(report),
            "--expected-num-examples",
            "6",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (report / "HANS_BASELINE_REPORT.md").is_file()
    assert (report / "data" / "run_summary.csv").is_file()
    assert (report / "data" / "group_metrics_long.csv").is_file()
    assert (report / "data" / "results.json").is_file()
    assert (report / "figures" / "overall_accuracy.svg").is_file()
    assert (report / "figures" / "label_heuristic_accuracy.svg").is_file()
    assert (report / "figures" / "subcase_accuracy_heatmap.svg").is_file()
    markdown = (report / "HANS_BASELINE_REPORT.md").read_text(encoding="utf-8")
    assert "Δ vs Base" in markdown
    assert "HANS zero-shot" not in markdown


def test_reporter_can_optionally_include_standalone_zero_shot(tmp_path):
    outputs = tmp_path / "outputs"
    _write_baseline_run(outputs, "hans_qwen3_1_7b_base", "zero_shot", None)
    _write_baseline_run(outputs, "baseline_base_b32_seed42", "base", None)
    _write_baseline_run(outputs, "baseline_random_b32_seed42", "random", "0")
    _write_baseline_run(
        outputs, "baseline_uncertainty_b32_seed42", "uncertainty", "1"
    )
    report = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "report_hans_baselines.py"),
            "--outputs",
            str(outputs),
            "--report-dir",
            str(report),
            "--expected-num-examples",
            "6",
            "--include-standalone-zero-shot",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    markdown = (report / "HANS_BASELINE_REPORT.md").read_text(encoding="utf-8")
    assert "Δ vs HANS zero-shot" in markdown
    assert "reproducibility check" in markdown
