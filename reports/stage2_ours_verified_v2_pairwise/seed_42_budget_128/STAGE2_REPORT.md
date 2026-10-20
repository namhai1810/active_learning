# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Random/Uncertainty are MNLI cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. Ours uses token cross-entropy plus counterfactual contrastive and invariant consistency losses.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| A | ours_verified | 128 | 42 | complete | 0.6434 ± 0.0000 | 0.3477 ± 0.0000 | 0.5825 ± 0.0000 / 0.5844 ± 0.0000 | 128.0000 ± 0.0000 | 5.9476 ± 0.0000 | no | pass |

## Verification diagnostics

Certification uses the declared metric. Strict-joint and control accuracies are diagnostic and do not gate target-failure certification.

| Seed | Hypothesis | Metric | Target rate / LCB | Strict-joint rate / LCB | Counterfactual | Invariant | Stop | Certified |
|---:|---|---|---:|---:|---:|---:|---|---|
| 42 | Contextual Invalidation Misclassification | target_failure | 55.00% / 29.83% | 38.33% / 13.17% | 83.33% | 75.00% | maximum_probe_sets_reached | no |
| 42 | General-to-Specific Misclassification | target_failure | 100.00% / 56.41% | 5.00% / -38.59% | 55.00% | 50.00% | hoeffding_lcb_certified | yes |
| 42 | Subject-Object Reversal Misclassification | target_failure | 0.00% / -43.59% | 0.00% / -43.59% | 90.00% | 55.00% | hoeffding_ucb_below_threshold | no |
| 42 | Entity Addition Misclassification | target_failure | 86.67% / 51.07% | 60.00% / 24.41% | 100.00% | 66.67% | hoeffding_lcb_certified | yes |
| 42 | Conditional Statement Misclassification | target_failure | 86.67% / 51.07% | 33.33% / -2.26% | 86.67% | 60.00% | hoeffding_lcb_certified | yes |

## Post-hoc learning curves (`gain_eval`)

| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---:|---:|---:|---:|---:|
| A | ours_verified | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.5205 ± 0.0000 |
| A | ours_verified | 128 | 1 | 128 | 0.6331 ± 0.0000 | 1.9919 ± 0.0000 |

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
