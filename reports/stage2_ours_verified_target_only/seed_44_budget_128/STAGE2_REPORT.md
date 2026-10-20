# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Random/Uncertainty are MNLI cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. Ours uses token cross-entropy plus counterfactual contrastive and invariant consistency losses.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| A | ours_verified | 128 | 44 | complete | 0.6641 ± 0.0000 | 0.4068 ± 0.0000 | 0.6017 ± 0.0000 / 0.6072 ± 0.0000 | 128.0000 ± 0.0000 | 5.8048 ± 0.0000 | no | pass |

## Verification diagnostics

Certification uses the declared metric. Strict-joint and control accuracies are diagnostic and do not gate target-failure certification.

| Seed | Hypothesis | Metric | Target rate / LCB | Strict-joint rate / LCB | Counterfactual | Invariant | Certified |
|---:|---|---|---:|---:|---:|---:|---|
| 44 | Modifier Overlook Hypothesis | target_failure | 98.33% / 76.96% | 55.00% / 33.63% | 86.67% | 70.00% | yes |
| 44 | Conditional Assumption Hypothesis | target_failure | 86.67% / 65.30% | 3.33% / -18.04% | 66.67% | 28.33% | yes |
| 44 | Third-Party Misattribution Hypothesis | target_failure | 68.97% / 47.23% | 41.38% / 19.64% | 100.00% | 68.97% | no |

## Post-hoc learning curves (`gain_eval`)

| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---:|---:|---:|---:|---:|
| A | ours_verified | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.5411 ± 0.0000 |
| A | ours_verified | 128 | 1 | 128 | 0.6515 ± 0.0000 | 1.9671 ± 0.0000 |

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
