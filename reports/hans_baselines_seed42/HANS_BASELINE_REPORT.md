# HANS zero-shot and acquisition baselines

Generated automatically at `2026-10-11T12:20:27.663417+00:00`. This report contains full HANS runs only; smoke artifacts are excluded.

## Main comparison

| Method | N | Acquired | Overall | Δ vs zero-shot | Entail. | Non-entail. | Lexical | Subsequence | Constituent | Wall time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HANS zero-shot | 30,000 | 0 | 63.64% | +0.00 pp | 93.83% | 33.46% | 82.94% | 57.33% | 50.66% | 7.18 |
| Base | 30,000 | 0 | 63.64% | -0.01 pp | 93.77% | 33.51% | 82.99% | 57.32% | 50.60% | 29.41 |
| Random | 30,000 | 32 | 63.80% | +0.15 pp | 93.67% | 33.92% | 83.33% | 57.42% | 50.64% | 31.67 |
| Uncertainty | 30,000 | 32 | 64.48% | +0.84 pp | 93.16% | 35.81% | 84.30% | 58.22% | 50.93% | 32.51 |

The highest observed single-seed HANS accuracy is **Uncertainty** at **64.48%**. Base differs from the standalone zero-shot evaluation by **-0.0067 percentage points**; this is a reproducibility check because both use the same unadapted checkpoint.

![Overall HANS accuracy](figures/overall_accuracy.svg)

## Breakdown by label and heuristic

![Accuracy by label and heuristic](figures/label_heuristic_accuracy.svg)

## Breakdown by HANS subcase

![HANS subcase heatmap](figures/subcase_accuracy_heatmap.svg)

## Run provenance

| Method | Run directory | Seed | Student | Device | GPU | Git commit |
|---|---|---:|---|---|---|---|
| HANS zero-shot | `hans_qwen3_1_7b_base` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `427014121df5` |
| Base | `baseline_base_b32_seed42` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `427014121df5` |
| Random | `baseline_random_b32_seed42` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `427014121df5` |
| Uncertainty | `baseline_uncertainty_b32_seed42` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `427014121df5` |

## Reusable artifacts

- `data/run_summary.csv`: one row per method, suitable for the paper's main table.
- `data/group_metrics_long.csv`: tidy overall/label/heuristic/subcase metrics with counts.
- `data/results.json`: lossless combined summary and provenance.
- `figures/*.svg`: dependency-free, editable vector figures suitable for the paper.
- Source predictions and original metrics remain under `outputs/<run_name>/`.

## Interpretation guardrails

- These are single-seed results (`seed=42`), so the report intentionally shows no confidence intervals.
- HANS zero-shot and Base are separate executions of the same unadapted model; disagreement indicates run-to-run or configuration drift.
- Random and Uncertainty both use acquisition budget 32 and the same LoRA recipe.
- Do not replace missing values with zero. The reporter fails on incomplete artifacts or disagreement between `metrics.json` and recomputed prediction-level metrics.
