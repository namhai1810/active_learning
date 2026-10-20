# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Random/Uncertainty are MNLI cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. The objective used by ours is recorded per run.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.
For an adaptive-budget run, Budget is the configured maximum and Actual train is the stopping budget. Its final metrics are reportable, but it is not claimed to be a fixed-budget comparison when it stops early.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| B | ours_verified | 128 | 42 | complete | 0.6475 ± 0.0000 | 0.3610 ± 0.0000 | 0.5877 ± 0.0000 / 0.5894 ± 0.0000 | 128.0000 ± 0.0000 | 7.1700 ± 0.0000 | no | pass |

## Verification diagnostics

Certification uses the declared metric. Strict-joint and control accuracies are diagnostic and do not gate target-failure certification.

| Seed | Hypothesis | Metric | Target rate / LCB | Strict-joint rate / LCB | Counterfactual | Invariant | Stop | Certified |
|---:|---|---|---:|---:|---:|---:|---|---|
| 42 | Contextual Invalidation Misclassification | target_failure | 55.00% / 29.83% | 38.33% / 13.17% | 83.33% | 75.00% | maximum_probe_sets_reached | no |
| 42 | General-to-Specific Misclassification | target_failure | 100.00% / 56.41% | 5.00% / -38.59% | 55.00% | 50.00% | hoeffding_lcb_certified | yes |
| 42 | Subject-Object Reversal Misclassification | target_failure | 0.00% / -43.59% | 0.00% / -43.59% | 90.00% | 55.00% | hoeffding_ucb_below_threshold | no |
| 42 | Entity Addition Misclassification | target_failure | 86.67% / 51.07% | 60.00% / 24.41% | 100.00% | 66.67% | hoeffding_lcb_certified | yes |
| 42 | Conditional Statement Misclassification | target_failure | 86.67% / 51.07% | 33.33% / -2.26% | 86.67% | 60.00% | hoeffding_lcb_certified | yes |

## Adaptive stopping

Stopping decisions use paired gain on `gain_eval`; the sealed final test is not used. Confidence bounds are adjusted for the predeclared maximum number of looks.

| Seed | Maximum budget | Selected budget | Completed stages | Decision | Last marginal gain / UCB |
|---:|---:|---:|---:|---|---:|
| 42 | 128 | 128 | 4 | maximum_budget_reached | 0.240% / 1.381% |

## Verification-conditioned sampling

Target proportions are learned from certified failure evidence. Control accuracy enters only as a soft specificity signal; there are no hard per-label or per-hypothesis quotas.

| Seed | Learned label target (E/N/C) | Selected labels (E/N/C) | Selected hypothesis pairs |
|---:|---:|---:|---|
| 42 | 0.179/0.591/0.230 | 40/66/22 | llm_549a9499b23f3ace:10, llm_cd643127717b39ea:27, llm_e5ccf1de8f397344:27 |

## Post-hoc learning curves (`gain_eval`)

| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---:|---:|---:|---:|---:|
| B | ours_verified | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.5185 ± 0.0000 |
| B | ours_verified | 128 | 1 | 32 | 0.6280 ± 0.0000 | 1.9880 ± 0.0000 |
| B | ours_verified | 128 | 2 | 64 | 0.6320 ± 0.0000 | 2.4179 ± 0.0000 |
| B | ours_verified | 128 | 3 | 96 | 0.6349 ± 0.0000 | 2.8477 ± 0.0000 |
| B | ours_verified | 128 | 4 | 128 | 0.6373 ± 0.0000 | 3.2763 ± 0.0000 |

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
