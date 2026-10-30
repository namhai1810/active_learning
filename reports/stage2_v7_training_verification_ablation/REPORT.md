# V7 Training Early Stopping + Verification ON/OFF

The two arms share the same initial pool, 4x32 acquisition budget, selector, stateful LoRA+AdamW training policy, seed schedule, and synthetic QC. Only hypothesis verification differs.

Training early stopping uses binary NLL on the predeclared gain-eval stopping subset. The sealed final test is evaluated only after training.

## Final metrics

| Verification | Seeds complete | HANS | HANS non-ent. | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|---:|
| ON | 3/3 | 71.6756 ± 2.8992 | 56.6805 ± 8.0273 | 59.9151 ± 2.3341 | 60.0149 ± 2.8948 |
| OFF | 3/3 | 72.1178 ± 1.4487 | 59.0223 ± 9.1812 | 62.3162 ± 0.9308 | 62.7238 ± 0.6181 |

Historical old-v7 reference (different training policy): HANS 65.0000%, MNLI-m 58.6347%, MNLI-mm 58.9605%. It is context only, not the controlled ON/OFF contrast.

## Final-test HANS metrics: round 4, budget 128

These values use the saved final-test artifacts from the predeclared round-4 endpoint. Three-way → binary is the current primary evaluator; Direct Binary is a diagnostic re-aggregation of the same saved probabilities. The sealed final HANS set was not used for training, early stopping, or checkpoint selection.

| Round | Budget | Rule | ON | OFF | ON − OFF mean |
|---:|---:|---|---:|---:|---:|
| 4 | 128 | Three-way → binary | 71.6756 ± 2.8992 | 72.1178 ± 1.4487 | -0.4422 ± 1.4861 pp |
| 4 | 128 | Direct Binary | 73.4667 ± 1.5154 | 72.8044 ± 1.4932 | +0.6622 ± 0.1600 pp |

At round 4, the descriptive ON/OFF ordering changes with the mapping rule: OFF is 0.4422 pp higher under the current Three-way rule, while ON is 0.6622 pp higher under Direct Binary. This sensitivity result must not be used to choose a rule from final-test performance.

### Post-hoc checkpoint diagnostic: round 3, budget 96

Round 3 is shown only as a post-hoc checkpoint diagnostic and is not part of the four-point validation curve or a new stopping rule.

| Round | Budget | Rule | ON | OFF | ON − OFF mean |
|---:|---:|---|---:|---:|---:|
| 3 | 96 | Three-way → binary | 69.6867 ± 1.4063 | 67.7711 ± 2.6777 | +1.9156 ± 4.0322 pp |
| 3 | 96 | Direct Binary | 69.6956 ± 1.3954 | 67.7889 ± 2.6951 | +1.9067 ± 4.0382 pp |

Full per-seed diagnostic: [HANS_PREDICTION_RULE_DIAGNOSTIC_ALL_SEEDS.md](HANS_PREDICTION_RULE_DIAGNOSTIC_ALL_SEEDS.md).

### Where Verification ON is numerically better

- **Matched budget 96 (round 3), current Three-way rule:** ON reaches 69.6867 ± 1.4063%, compared with OFF at 67.7711 ± 2.6777% (**ON +1.9156 pp**).
- **Matched budget 96, Direct Binary:** ON reaches 69.6956 ± 1.3954%, compared with OFF at 67.7889 ± 2.6951% (**ON +1.9067 pp**).
- **Matched budget 128 (round 4), Direct Binary diagnostic:** ON reaches 73.4667 ± 1.5154%, compared with OFF at 72.8044 ± 1.4932% (**ON +0.6622 pp**).
- On the leakage-safe validation curve at budget 96, ON is also +1.5555 pp above OFF (69.1111% versus 67.5556%).

The defensible positive claim is therefore: **verification ON shows an accuracy advantage at the 96-example checkpoint and under Direct Binary aggregation at 128 examples.** It is not accurate to claim that ON universally beats OFF: at the predeclared 128-example endpoint under the current primary Three-way evaluator, OFF is 0.4422 pp higher, and ON incurs substantially more teacher-token and wall-clock cost.

Mean ± sample standard deviation is reported when at least two seeds are complete; a single completed seed has standard deviation 0.

## Learning curves

| Verification | Round | Seeds | Stopping accuracy |
|---|---:|---:|---:|
| ON | 0 | 3 | 63.2222 ± 0.0000 |
| ON | 1 | 3 | 64.6296 ± 0.1697 |
| ON | 2 | 3 | 65.1852 ± 0.2313 |
| ON | 3 | 3 | 69.1111 ± 1.8359 |
| ON | 4 | 3 | 70.5556 ± 3.6868 |
| OFF | 0 | 3 | 63.2222 ± 0.0000 |
| OFF | 1 | 3 | 64.6296 ± 0.1697 |
| OFF | 2 | 3 | 65.3333 ± 0.1111 |
| OFF | 3 | 3 | 67.5556 ± 2.5987 |
| OFF | 4 | 3 | 71.9259 ± 2.2370 |

## Per-round performance and cost

Accuracy is measured on the predeclared 900-example stopping subset using Direct Binary, which is the only leakage-safe metric available at all four budgets. Teacher tokens are input plus output tokens. GPU time is the sum of acquisition, training, validation, and stopping-evaluation wall-clock proxies.

`Total wall cost` is a time proxy, not a monetary USD cost: `refresh_wall_seconds + GPU proxy seconds`. Teacher time is already inside refresh time and is not added again. Values exclude the shared initial-pool cost, post-round-4 final hypothesis audit, sealed-final evaluation, and abandoned/retry overhead.

### Incremental cost incurred in each round

| Verification | Round | Budget | Accuracy | Teacher tokens | GPU proxy (min) | Total wall cost (min) |
|---|---:|---:|---:|---:|---:|---:|
| ON | 1 | 32 | 64.6296 ± 0.1697 | 0 ± 0 | 3.59 ± 0.07 | 3.59 ± 0.07 |
| ON | 2 | 64 | 65.1852 ± 0.2313 | 268056 ± 7985 | 3.71 ± 0.11 | 81.76 ± 17.21 |
| ON | 3 | 96 | 69.1111 ± 1.8359 | 276519 ± 5915 | 7.77 ± 0.48 | 87.18 ± 17.21 |
| ON | 4 | 128 | 70.5556 ± 3.6868 | 275038 ± 8158 | 7.93 ± 0.55 | 78.60 ± 3.49 |
| OFF | 1 | 32 | 64.6296 ± 0.1697 | 0 ± 0 | 3.68 ± 0.11 | 3.68 ± 0.11 |
| OFF | 2 | 64 | 65.3333 ± 0.1111 | 71520 ± 1643 | 4.01 ± 0.33 | 29.70 ± 7.84 |
| OFF | 3 | 96 | 67.5556 ± 2.5987 | 71074 ± 879 | 9.24 ± 2.66 | 31.01 ± 3.79 |
| OFF | 4 | 128 | 71.9259 ± 2.2370 | 71649 ± 1098 | 10.61 ± 4.75 | 32.36 ± 5.71 |

### Cumulative cost to reach each checkpoint

| Verification | Round | Budget | Accuracy | Cumulative teacher tokens | Cumulative GPU proxy (min) | Cumulative total wall cost (h) |
|---|---:|---:|---:|---:|---:|---:|
| ON | 1 | 32 | 64.6296 ± 0.1697 | 0 ± 0 | 3.59 ± 0.07 | 0.06 ± 0.00 |
| ON | 2 | 64 | 65.1852 ± 0.2313 | 268056 ± 7985 | 7.30 ± 0.16 | 1.42 ± 0.29 |
| ON | 3 | 96 | 69.1111 ± 1.8359 | 544575 ± 3433 | 15.07 ± 0.64 | 2.88 ± 0.57 |
| ON | 4 | 128 | 70.5556 ± 3.6868 | 819613 ± 10381 | 23.00 ± 1.19 | 4.19 ± 0.56 |
| OFF | 1 | 32 | 64.6296 ± 0.1697 | 0 ± 0 | 3.68 ± 0.11 | 0.06 ± 0.00 |
| OFF | 2 | 64 | 65.3333 ± 0.1111 | 71520 ± 1643 | 7.69 ± 0.28 | 0.56 ± 0.13 |
| OFF | 3 | 96 | 67.5556 ± 2.5987 | 142594 ± 1505 | 16.93 ± 2.91 | 1.07 ± 0.19 |
| OFF | 4 | 128 | 71.9259 ± 2.2370 | 214243 ± 905 | 27.54 ± 7.66 | 1.61 ± 0.29 |

### Paired ON − OFF deltas at matched seeds

Positive accuracy means ON is higher. Positive cost means ON uses more resources. Deltas are paired by seed before reporting mean ± sample standard deviation.

| Budget | Accuracy delta | Teacher-token delta | GPU-proxy delta | Total-wall delta |
|---:|---:|---:|---:|---:|
| 32 | +0.0000 ± 0.0000 pp | +0 ± 0 | -0.09 ± 0.15 min | -0.00 ± 0.00 h |
| 64 | -0.1481 ± 0.1283 pp | +196536 ± 6740 | -0.39 ± 0.44 min | +0.87 ± 0.16 h |
| 96 | +1.5556 ± 4.3262 pp | +401981 ± 4214 | -1.86 ± 3.49 min | +1.80 ± 0.38 h |
| 128 | -1.3704 ± 1.4501 pp | +605370 ± 10801 | -4.54 ± 8.73 min | +2.57 ± 0.28 h |

The budget-32 point has zero in-run teacher tokens because both arms reuse the same prebuilt initial pool; that shared construction cost is outside the ablation run scope. ON begins paying the verification cost during the refresh used to construct the budget-64 checkpoint.

## Performance–cost curves

![Validation performance against budget, cumulative teacher tokens, GPU time, and total wall cost](figures/05_performance_cost_curves.svg)

Each marker is one fixed budget (32, 64, 96, or 128 examples); lines show three-seed means and vertical bars show sample standard deviation of validation accuracy. These curves use validation accuracy, not repeated final-HANS evaluation.

## Cost-accounting reconciliation

The performance–cost curve stops accounting at each trained checkpoint. The run-level result additionally records the post-round-4 hypothesis audit and sealed-final evaluation. These later operations are shown separately and are not charged to an earlier budget point.

| Run | Curve tokens to B128 | Post-R4 audit tokens | Result tokens | Curve GPU (min) | Logged loop GPU (min) | Final eval (min) | Check |
|---|---:|---:|---:|---:|---:|---:|---|
| ON seed 42 | 828370 | 176470 | 1004840 | 21.81 | 21.81 | 42.39 | exact |
| ON seed 43 | 822322 | 215583 | 1037905 | 24.18 | 24.18 | 44.25 | exact |
| ON seed 44 | 808146 | 199678 | 1007824 | 23.02 | 23.02 | 45.07 | exact |
| OFF seed 42 | 214559 | 0 | 214559 | 36.38 | 36.38 | 73.37 | exact |
| OFF seed 43 | 213222 | 0 | 213222 | 22.91 | 22.91 | 43.42 | exact |
| OFF seed 44 | 214948 | 0 | 214948 | 23.33 | 23.33 | 45.02 | exact |

Accounting rules:

- `teacher_wall_clock_seconds` is contained in `refresh_wall_seconds`; the two are never summed.
- GPU fields are measured wall-clock proxies, not hardware-utilization integrals or billable GPU-hours.
- Shared initial-pool construction is excluded from both arms, as declared by `resource_usage.scope`; therefore round 1 has zero in-run teacher tokens.
- Post-round-4 final hypothesis audit and sealed-final evaluation are reported separately from the cost-to-checkpoint curve.
- Failed attempts, Ctrl-C/retry overhead, and archived incomplete runs are not represented by successful-run artifacts.

## Training and resource summary

| Run | Committed AdamW steps | Best epochs per round | GPU proxy seconds incl. final eval | Teacher seconds | Teacher input/output tokens incl. final audit |
|---|---:|---|---:|---:|---:|
| ON seed 42 | 72 | 1,1,8,8 | 3851.99 | 17996.91 | 636721/368119 |
| ON seed 43 | 72 | 1,1,8,8 | 4106.31 | 15306.68 | 651963/385942 |
| ON seed 44 | 64 | 1,1,8,6 | 4085.67 | 14428.62 | 634632/373192 |
| OFF seed 42 | 56 | 1,1,4,8 | 6584.82 | 3157.96 | 134080/80479 |
| OFF seed 43 | 72 | 1,1,8,8 | 3979.53 | 3205.28 | 133518/79704 |
| OFF seed 44 | 72 | 1,1,8,8 | 4100.86 | 3259.84 | 134920/80028 |

## Completion

- Completed runs: 6/6
- Missing runs: none
