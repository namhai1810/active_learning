# HANS acquisition baselines

Generated automatically at `2026-10-11T18:37:54.868030+00:00`. This report contains full HANS runs only; smoke artifacts are excluded.

## Main comparison

| Method | N | Acquired | Overall | Δ vs Base | Entail. | Non-entail. | Lexical | Subsequence | Constituent | Wall time (min) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 30,000 | 0 | 63.64% | +0.00 pp | 93.77% | 33.51% | 82.99% | 57.32% | 50.60% | 63.61 |
| Random | 30,000 | 32 | 63.80% | +0.16 pp | 93.67% | 33.92% | 83.33% | 57.42% | 50.64% | 67.02 |
| Uncertainty | 30,000 | 32 | 64.48% | +0.85 pp | 93.16% | 35.81% | 84.30% | 58.22% | 50.93% | 69.73 |

The highest observed single-seed HANS accuracy is **Uncertainty** at **64.48%**.

![Overall HANS accuracy](figures/overall_accuracy.svg)

## Breakdown by label and heuristic

![Accuracy by label and heuristic](figures/label_heuristic_accuracy.svg)

## Breakdown by HANS subcase

![HANS subcase heatmap](figures/subcase_accuracy_heatmap.svg)

## Run provenance

| Method | Run directory | Seed | Student | Device | GPU | Git commit |
|---|---|---:|---|---|---|---|
| Base | `baseline_base_b32_seed42` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `ece4f206aa94` |
| Random | `baseline_random_b32_seed42` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `ece4f206aa94` |
| Uncertainty | `baseline_uncertainty_b32_seed42` | 42 | `Qwen/Qwen3-1.7B` | `cuda:0` | NVIDIA RTX 5880 Ada Generation | `ece4f206aa94` |

## Reusable artifacts

- `data/run_summary.csv`: one row per method, suitable for the paper's main table.
- `data/group_metrics_long.csv`: tidy overall/label/heuristic/subcase metrics with counts.
- `data/results.json`: lossless combined summary and provenance.
- `figures/*.svg`: dependency-free, editable vector figures suitable for the paper.
- Source predictions and original metrics remain under `outputs/<run_name>/`.

## Interpretation guardrails

- These are single-seed results (`seed=42`), so the report intentionally shows no confidence intervals.
- Base is the unadapted checkpoint and is the default reference for Random and Uncertainty. When `--include-standalone-zero-shot` is used, its agreement with Base is an additional reproducibility check.
- Random and Uncertainty both use acquisition budget 32 and the same LoRA recipe.
- Do not replace missing values with zero. The reporter fails on incomplete artifacts or disagreement between `metrics.json` and recomputed prediction-level metrics.
