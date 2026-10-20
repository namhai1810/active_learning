# Stage 2 benchmark report

Frozen HANS manifest: `871e205c39611510aa92bfd37405fcdc0b120fb9c149779fd3b3237115d4a0f0`

Random/Uncertainty are MNLI cross-source baselines; equal data access with HANS-guided synthetic methods is not claimed. Random-synthetic and the no-verification ablation use pairs for selection but train with per-example cross-entropy. Ours uses token cross-entropy plus counterfactual contrastive and invariant consistency losses.
Runs that abstain because no failure hypothesis is certified are retained as protocol outcomes and costs, but are excluded from equal-budget accuracy figures and learning-curve comparisons.

![Experiment A final HANS accuracy](figures/experiment_a_final_hans_accuracy.svg)

![Experiment A gain-eval learning curve](figures/experiment_a_gain_eval_learning_curve.svg)

![Experiment A accuracy versus cost](figures/experiment_a_accuracy_vs_cost.svg)

| Exp. | Method | Budget | Seeds | Status | HANS acc. | HANS non-ent. | MNLI m/mm | Actual train | Assumed cost | Shortage | Leakage audit |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---|---|
| A | ours_verified | 128 | 42 | ABSTAINED | NA | NA | NA / NA | 0.0000 ± 0.0000 | 4.1723 ± 0.0000 | yes | pass |

## Post-hoc learning curves (`gain_eval`)

| Exp. | Method | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---:|---:|---:|---:|---:|

The learning curve uses `gain_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
