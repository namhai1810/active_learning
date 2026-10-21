# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Methods using the MNLI acquisition pool are cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. The objective used by ours is recorded per run.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.
For an adaptive-budget run, Budget is the configured maximum and Actual train is the stopping budget. Its final metrics are reportable, but it is not claimed to be a fixed-budget comparison when it stops early.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| B | badge_synthetic | 128 | 42 | complete | 0.6470 ± 0.0000 | 0.3580 ± 0.0000 | 0.5861 ± 0.0000 / 0.5877 ± 0.0000 | 128.0000 ± 0.0000 | 7.5723 ± 0.0000 | no | pass |
| B | cal_synthetic | 128 | 42 | complete | 0.6503 ± 0.0000 | 0.3686 ± 0.0000 | 0.5910 ± 0.0000 / 0.5930 ± 0.0000 | 128.0000 ± 0.0000 | 7.5748 ± 0.0000 | no | pass |
| B | coreset_synthetic | 128 | 42 | complete | 0.6505 ± 0.0000 | 0.3692 ± 0.0000 | 0.5901 ± 0.0000 / 0.5926 ± 0.0000 | 128.0000 ± 0.0000 | 7.5619 ± 0.0000 | no | pass |
| B | margin_synthetic | 128 | 42 | complete | 0.6471 ± 0.0000 | 0.3574 ± 0.0000 | 0.5856 ± 0.0000 / 0.5872 ± 0.0000 | 128.0000 ± 0.0000 | 7.5583 ± 0.0000 | no | pass |

## Current interpretation (synthetic runs complete; MNLI runs pending)

At this snapshot, all four controlled synthetic runs are complete and no
published-selector MNLI run has completed. Missing MNLI rows are not estimated
from partial checkpoints.

| Rank | Method | HANS final | HANS non-entailment | Difference from v7 continuation |
|---:|---|---:|---:|---:|
| 1 | `coreset_synthetic` | **65.0533%** | **36.9173%** | +0.0533 pp |
| 2 | `cal_synthetic` | 65.0333% | 36.8639% | +0.0333 pp |
| 3 | `margin_synthetic` | 64.7067% | 35.7420% | -0.2933 pp |
| 4 | `badge_synthetic` | 64.7000% | 35.7954% | -0.3000 pp |

Core-set and CAL are numerically above the valid v7 continuation score of
65.0000%, but by only eight and five correct HANS predictions out of 15,000.
With one seed, this is an effective tie rather than evidence that either method
is superior. Every published selector is above the earlier random-synthetic
score of 64.4600%.

The acquisition cost ranges from 28.9 to 71.5 GPU-seconds per method. Most of
the 3.88-hour wall time comes from repeated HANS/MNLI evaluation. The static v5
pool is reused; its historical teacher-token generation cost is not newly paid
by each selector run.

## Published active-learning baselines

All rows below use the shared Qwen/LoRA Stage-2 implementation. Round 1 is an identical random warm start; rounds 2--4 apply the published acquisition rule. LoRA and AdamW state continue across rounds. Synthetic variants preserve complete pairs.

| Run method | Paper | Venue | Acquisition pool | Local code |
|---|---|---|---|---|
| `badge_synthetic` | Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds | International Conference on Learning Representations (ICLR), 2020 | `ours_verified` | `src/baselines/literature/badge.py` |
| `cal_synthetic` | Active Learning by Acquiring Contrastive Examples | Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021 | `ours_verified` | `src/baselines/literature/cal.py` |
| `coreset_synthetic` | Active Learning for Convolutional Neural Networks: A Core-Set Approach | International Conference on Learning Representations (ICLR), 2018 | `ours_verified` | `src/baselines/literature/coreset.py` |
| `margin_synthetic` | Support Vector Machine Active Learning with Applications to Text Classification | Journal of Machine Learning Research (JMLR), 2001 | `ours_verified` | `src/baselines/literature/margin.py` |

## Verification diagnostics

Certification uses the declared metric. Strict-joint and control accuracies are diagnostic and do not gate target-failure certification.

| Seed | Hypothesis | Metric | Target rate / LCB | Strict-joint rate / LCB | Counterfactual | Invariant | Stop | Certified |
|---:|---|---|---:|---:|---:|---:|---|---|
| 42 | Contextual Invalidation Misclassification | target_failure | 55.00% / 29.83% | 38.33% / 13.17% | 83.33% | 75.00% | maximum_probe_sets_reached | no |
| 42 | General-to-Specific Misclassification | target_failure | 100.00% / 56.41% | 5.00% / -38.59% | 55.00% | 50.00% | hoeffding_lcb_certified | yes |
| 42 | Subject-Object Reversal Misclassification | target_failure | 0.00% / -43.59% | 0.00% / -43.59% | 90.00% | 55.00% | hoeffding_ucb_below_threshold | no |
| 42 | Entity Addition Misclassification | target_failure | 86.67% / 51.07% | 60.00% / 24.41% | 100.00% | 66.67% | hoeffding_lcb_certified | yes |
| 42 | Conditional Statement Misclassification | target_failure | 86.67% / 51.07% | 33.33% / -2.26% | 86.67% | 60.00% | hoeffding_lcb_certified | yes |
| 42 | Contextual Invalidation Misclassification | target_failure | 55.00% / 29.83% | 38.33% / 13.17% | 83.33% | 75.00% | maximum_probe_sets_reached | no |
| 42 | General-to-Specific Misclassification | target_failure | 100.00% / 56.41% | 5.00% / -38.59% | 55.00% | 50.00% | hoeffding_lcb_certified | yes |
| 42 | Subject-Object Reversal Misclassification | target_failure | 0.00% / -43.59% | 0.00% / -43.59% | 90.00% | 55.00% | hoeffding_ucb_below_threshold | no |
| 42 | Entity Addition Misclassification | target_failure | 86.67% / 51.07% | 60.00% / 24.41% | 100.00% | 66.67% | hoeffding_lcb_certified | yes |
| 42 | Conditional Statement Misclassification | target_failure | 86.67% / 51.07% | 33.33% / -2.26% | 86.67% | 60.00% | hoeffding_lcb_certified | yes |
| 42 | Contextual Invalidation Misclassification | target_failure | 55.00% / 29.83% | 38.33% / 13.17% | 83.33% | 75.00% | maximum_probe_sets_reached | no |
| 42 | General-to-Specific Misclassification | target_failure | 100.00% / 56.41% | 5.00% / -38.59% | 55.00% | 50.00% | hoeffding_lcb_certified | yes |
| 42 | Subject-Object Reversal Misclassification | target_failure | 0.00% / -43.59% | 0.00% / -43.59% | 90.00% | 55.00% | hoeffding_ucb_below_threshold | no |
| 42 | Entity Addition Misclassification | target_failure | 86.67% / 51.07% | 60.00% / 24.41% | 100.00% | 66.67% | hoeffding_lcb_certified | yes |
| 42 | Conditional Statement Misclassification | target_failure | 86.67% / 51.07% | 33.33% / -2.26% | 86.67% | 60.00% | hoeffding_lcb_certified | yes |
| 42 | Contextual Invalidation Misclassification | target_failure | 55.00% / 29.83% | 38.33% / 13.17% | 83.33% | 75.00% | maximum_probe_sets_reached | no |
| 42 | General-to-Specific Misclassification | target_failure | 100.00% / 56.41% | 5.00% / -38.59% | 55.00% | 50.00% | hoeffding_lcb_certified | yes |
| 42 | Subject-Object Reversal Misclassification | target_failure | 0.00% / -43.59% | 0.00% / -43.59% | 90.00% | 55.00% | hoeffding_ucb_below_threshold | no |
| 42 | Entity Addition Misclassification | target_failure | 86.67% / 51.07% | 60.00% / 24.41% | 100.00% | 66.67% | hoeffding_lcb_certified | yes |
| 42 | Conditional Statement Misclassification | target_failure | 86.67% / 51.07% | 33.33% / -2.26% | 86.67% | 60.00% | hoeffding_lcb_certified | yes |

## Learning curves (`gain_eval`)

| Exp. | Method | Role | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---|---:|---:|---:|---:|---:|
| B | badge_synthetic | gain_eval | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.8723 ± 0.0000 |
| B | badge_synthetic | gain_eval | 128 | 1 | 32 | 0.6309 ± 0.0000 | 2.2971 ± 0.0000 |
| B | badge_synthetic | gain_eval | 128 | 2 | 64 | 0.6336 ± 0.0000 | 2.7775 ± 0.0000 |
| B | badge_synthetic | gain_eval | 128 | 3 | 96 | 0.6355 ± 0.0000 | 3.2578 ± 0.0000 |
| B | badge_synthetic | gain_eval | 128 | 4 | 128 | 0.6360 ± 0.0000 | 3.7379 ± 0.0000 |
| B | cal_synthetic | gain_eval | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.8725 ± 0.0000 |
| B | cal_synthetic | gain_eval | 128 | 1 | 32 | 0.6309 ± 0.0000 | 2.2972 ± 0.0000 |
| B | cal_synthetic | gain_eval | 128 | 2 | 64 | 0.6341 ± 0.0000 | 2.7778 ± 0.0000 |
| B | cal_synthetic | gain_eval | 128 | 3 | 96 | 0.6371 ± 0.0000 | 3.2586 ± 0.0000 |
| B | cal_synthetic | gain_eval | 128 | 4 | 128 | 0.6405 ± 0.0000 | 3.7392 ± 0.0000 |
| B | coreset_synthetic | gain_eval | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.8722 ± 0.0000 |
| B | coreset_synthetic | gain_eval | 128 | 1 | 32 | 0.6309 ± 0.0000 | 2.2965 ± 0.0000 |
| B | coreset_synthetic | gain_eval | 128 | 2 | 64 | 0.6349 ± 0.0000 | 2.7750 ± 0.0000 |
| B | coreset_synthetic | gain_eval | 128 | 3 | 96 | 0.6384 ± 0.0000 | 3.2537 ± 0.0000 |
| B | coreset_synthetic | gain_eval | 128 | 4 | 128 | 0.6416 ± 0.0000 | 3.7323 ± 0.0000 |
| B | margin_synthetic | gain_eval | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.8724 ± 0.0000 |
| B | margin_synthetic | gain_eval | 128 | 1 | 32 | 0.6309 ± 0.0000 | 2.2968 ± 0.0000 |
| B | margin_synthetic | gain_eval | 128 | 2 | 64 | 0.6336 ± 0.0000 | 2.7761 ± 0.0000 |
| B | margin_synthetic | gain_eval | 128 | 3 | 96 | 0.6352 ± 0.0000 | 3.2552 ± 0.0000 |
| B | margin_synthetic | gain_eval | 128 | 4 | 128 | 0.6357 ± 0.0000 | 3.7339 ± 0.0000 |

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
