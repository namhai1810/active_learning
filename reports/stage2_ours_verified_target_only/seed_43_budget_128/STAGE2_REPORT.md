# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Random/Uncertainty are MNLI cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. Ours uses token cross-entropy plus counterfactual contrastive and invariant consistency losses.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| A | ours_verified | 128 | 43 | complete | 0.6455 ± 0.0000 | 0.3568 ± 0.0000 | 0.5854 ± 0.0000 / 0.5872 ± 0.0000 | 128.0000 ± 0.0000 | 5.8487 ± 0.0000 | no | pass |

## Verification diagnostics

Certification uses the declared metric. Strict-joint and control accuracies are diagnostic and do not gate target-failure certification.

| Seed | Hypothesis | Metric | Target rate / LCB | Strict-joint rate / LCB | Counterfactual | Invariant | Certified |
|---:|---|---|---:|---:|---:|---:|---|
| 43 | Direct-Indirect Relationship Misclassification | target_failure | 91.67% / 70.30% | 26.67% / 5.30% | 46.67% | 66.67% | yes |
| 43 | Subject-Object Mismatch | target_failure | 70.91% / 48.59% | 30.91% / 8.59% | 72.73% | 67.27% | no |
| 43 | Modifier-Induced Entailment Error | target_failure | 96.15% / 73.20% | 46.15% / 23.20% | 50.00% | 65.38% | yes |

## Post-hoc learning curves (`gain_eval`)

| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---:|---:|---:|---:|---:|
| A | ours_verified | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.5381 ± 0.0000 |
| A | ours_verified | 128 | 1 | 128 | 0.6341 ± 0.0000 | 1.9676 ± 0.0000 |

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
