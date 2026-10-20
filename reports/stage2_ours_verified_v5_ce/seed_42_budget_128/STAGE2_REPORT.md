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
| B | ours_verified | 128 | 42 | complete | 0.6482 ± 0.0000 | 0.3648 ± 0.0000 | 0.5881 ± 0.0000 / 0.5894 ± 0.0000 | 128.0000 ± 0.0000 | 6.1913 ± 0.0000 | no | pass |

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

Stopping decisions use the evaluation role recorded for each run; the sealed final test is not used. Confidence bounds are diagnostic and adjusted for the predeclared maximum number of looks.

| Seed | Maximum budget | Selected budget | Completed stages | Decision | Last marginal gain / UCB |
|---:|---:|---:|---:|---|---:|
| 42 | 128 | 128 | 4 | maximum_budget_reached | -0.111% / 3.911% |

## Held-out gain report

This partition is disjoint from `stopping_eval` and is evaluated once, only after checkpoint selection.

| Seed | Examples | Accuracy | Non-entailment accuracy |
|---:|---:|---:|---:|
| 42 | 2850 | 0.6375 | 0.3551 |

## Confidence-safe entropy-novelty sampling

Only hypotheses whose Hoeffding UCB is below the threshold are excluded. Certified and unresolved hypotheses are retained; one coverage pair per retained hypothesis is selected before greedy target NLL, entropy, and contextual-embedding novelty scoring.

| Seed | States | Selected hypothesis pairs | Target NLL | Target entropy | Embedding novelty |
|---:|---|---|---:|---:|---:|
| 42 | llm_05afa8eb8dd489b3:unresolved, llm_549a9499b23f3ace:certified, llm_cd643127717b39ea:certified, llm_e5ccf1de8f397344:certified | llm_05afa8eb8dd489b3:1, llm_549a9499b23f3ace:12, llm_cd643127717b39ea:1, llm_e5ccf1de8f397344:50 | 7.9730 | 0.1630 | 0.0226 |

## Learning curves (`stopping_eval`)

| Exp. | Method | Role | Budget | Round | Cumulative examples | Accuracy | Cumulative cost |
|---|---|---|---:|---:|---:|---:|---:|
| B | ours_verified | stopping_eval | 128 | 0 | 0 | 0.6322 ± 0.0000 | 1.4694 ± 0.0000 |
| B | ours_verified | stopping_eval | 128 | 1 | 32 | 0.6356 ± 0.0000 | 1.6526 ± 0.0000 |
| B | ours_verified | stopping_eval | 128 | 2 | 64 | 0.6422 ± 0.0000 | 1.7745 ± 0.0000 |
| B | ours_verified | stopping_eval | 128 | 3 | 96 | 0.6456 ± 0.0000 | 1.8965 ± 0.0000 |
| B | ours_verified | stopping_eval | 128 | 4 | 128 | 0.6444 ± 0.0000 | 2.0184 ± 0.0000 |

The learning curve uses `stopping_eval`; the sealed HANS final-test is evaluated once on the final checkpoint only.

Intervals are descriptive across seeds and use a normal approximation; with three seeds they should not be interpreted as strong coverage guarantees.
