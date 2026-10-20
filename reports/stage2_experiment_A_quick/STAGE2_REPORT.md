# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Random/Uncertainty are MNLI cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. Ours uses token cross-entropy plus counterfactual contrastive and invariant consistency losses.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| A | failure_driven_no_verification | 128 | 42 | complete | 0.6547 ± 0.0000 | 0.3837 ± 0.0000 | 0.5934 ± 0.0000 / 0.5963 ± 0.0000 | 128.0000 ± 0.0000 | 5.3384 ± 0.0000 | no | pass |
| A | ours_verified | 128 | 42 | ABSTAINED | NA | NA | NA / NA | 0.0000 ± 0.0000 | 4.1727 ± 0.0000 | yes | pass |
| A | random_mnli | 128 | 42 | complete | 0.6526 ± 0.0000 | 0.3765 ± 0.0000 | 0.5949 ± 0.0000 / 0.5977 ± 0.0000 | 128.0000 ± 0.0000 | 4.6166 ± 0.0000 | no | pass |
| A | random_synthetic | 128 | 42 | complete | 0.6446 ± 0.0000 | 0.3517 ± 0.0000 | 0.5839 ± 0.0000 / 0.5851 ± 0.0000 | 128.0000 ± 0.0000 | 5.3336 ± 0.0000 | no | pass |
| A | uncertainty_mnli | 128 | 42 | complete | 0.6699 ± 0.0000 | 0.4230 ± 0.0000 | 0.6104 ± 0.0000 / 0.6154 ± 0.0000 | 128.0000 ± 0.0000 | 48.0671 ± 0.0000 | no | pass |
| A | zero_shot | 128 | 42 | complete | 0.6385 ± 0.0000 | 0.3375 ± 0.0000 | 0.5766 ± 0.0000 / 0.5794 ± 0.0000 | 0.0000 ± 0.0000 | 4.1729 ± 0.0000 | no | pass |

## Post-hoc learning curves (`gain_eval`)

| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---:|---:|---:|---:|---:|
| A | failure_driven_no_verification | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.1245 ± 0.0000 |
| A | failure_driven_no_verification | 128 | 1 | 128 | 0.6437 ± 0.0000 | 1.5474 ± 0.0000 |
| A | random_mnli | 128 | 0 | 0 | 0.6251 ± 0.0000 | 0.4005 ± 0.0000 |
| A | random_mnli | 128 | 1 | 128 | 0.6432 ± 0.0000 | 0.8239 ± 0.0000 |
| A | random_synthetic | 128 | 0 | 0 | 0.6251 ± 0.0000 | 1.1202 ± 0.0000 |
| A | random_synthetic | 128 | 1 | 128 | 0.6336 ± 0.0000 | 1.5431 ± 0.0000 |
| A | uncertainty_mnli | 128 | 0 | 0 | 0.6251 ± 0.0000 | 0.4000 ± 0.0000 |
| A | uncertainty_mnli | 128 | 1 | 128 | 0.6565 ± 0.0000 | 44.2774 ± 0.0000 |
| A | zero_shot | 128 | 0 | 0 | 0.6251 ± 0.0000 | 0.4007 ± 0.0000 |

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
