# V7 Training Early Stopping + Verification ON/OFF

The two arms share the same initial pool, 4x32 acquisition budget, selector, stateful LoRA+AdamW training policy, seed schedule, and synthetic QC. Only hypothesis verification differs.

Training early stopping uses binary NLL on the predeclared gain-eval stopping subset. The sealed final test is evaluated only after training.

## Final metrics

| Verification | Seeds complete | HANS | HANS non-ent. | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|---:|
| ON | 1/3 | 68.6333 ± 0.0000 | 48.6710 ± 0.0000 | 57.8095 ± 0.0000 | 57.4451 ± 0.0000 |
| OFF | 1/3 | 70.5067 ± 0.0000 | 50.6478 ± 0.0000 | 61.5792 ± 0.0000 | 62.2254 ± 0.0000 |

Historical old-v7 reference (different training policy): HANS 65.0000%, MNLI-m 58.6347%, MNLI-mm 58.9605%. It is context only, not the controlled ON/OFF contrast.

Mean ± sample standard deviation is reported when at least two seeds are complete; a single completed seed has standard deviation 0.

## Learning curves

| Verification | Round | Seeds | Stopping accuracy |
|---|---:|---:|---:|
| ON | 0 | 1 | 63.2222 ± 0.0000 |
| ON | 1 | 1 | 64.6667 ± 0.0000 |
| ON | 2 | 1 | 65.4444 ± 0.0000 |
| ON | 3 | 1 | 71.0000 ± 0.0000 |
| ON | 4 | 1 | 66.6667 ± 0.0000 |
| OFF | 0 | 1 | 63.2222 ± 0.0000 |
| OFF | 1 | 1 | 64.6667 ± 0.0000 |
| OFF | 2 | 1 | 65.4444 ± 0.0000 |
| OFF | 3 | 1 | 64.5556 ± 0.0000 |
| OFF | 4 | 1 | 69.5556 ± 0.0000 |

## Post-hoc evaluation of the ON round-3 checkpoint

The verification-ON learning curve reached **71.0000% stopping accuracy after round 3**, then fell to 66.6667% after the required fourth acquisition round. The round-3 adapter was preserved, so it was subsequently evaluated on the complete HANS and MNLI evaluation sets without retraining.

This checkpoint had seen 96 acquisition examples (3 rounds x 32), was committed at AdamW optimizer step 40, and passed the adapter-load audit exactly (`maximum_absolute_difference = 0`).

| Checkpoint | Training examples seen | HANS full | HANS entailment | HANS non-ent. | MNLI-m | MNLI-mm |
|---|---:|---:|---:|---:|---:|---:|
| **ON round 3 (post-hoc)** | 96 | **71.2533** | 86.5433 | **55.9102** | **62.5064** | **63.2425** |
| ON round 4 (predeclared final) | 128 | 68.6333 | 88.5266 | 48.6710 | 57.8095 | 57.4451 |
| OFF round 4 (predeclared final) | 128 | 70.5067 | **90.2968** | 50.6478 | 61.5792 | 62.2254 |

Relative to the predeclared ON round-4 checkpoint, ON round 3 gains **+2.6200 percentage points on full HANS**, **+7.2392 points on HANS non-entailment**, **+4.6969 points on MNLI-m**, and **+5.7974 points on MNLI-mm**. It loses 1.9833 points on HANS entailment, but the much larger non-entailment gain produces a higher overall HANS score.

ON round 3 also exceeds the OFF round-4 checkpoint by 0.7466 points on HANS, approximately 0.9272 points on MNLI-m, and approximately 1.0171 points on MNLI-mm for seed 42. This is evidence that the fourth ON round degraded this seed's generalization; it does not establish that stopping after round 3 will generalize across seeds.

### Direct comparison with verification OFF

The following comparison makes the ON-round-3 versus OFF-final result explicit. Positive deltas favor ON round 3; negative deltas favor OFF round 4.

| Metric | ON round 3 (96 examples) | OFF round 4 (128 examples) | ON r3 - OFF r4 |
|---|---:|---:|---:|
| HANS full | **71.2533%** | 70.5067% | **+0.7466 pp** |
| HANS entailment | 86.5433% | **90.2968%** | -3.7535 pp |
| HANS non-entailment | **55.9102%** | 50.6478% | **+5.2624 pp** |
| MNLI matched | **62.5064%** | 61.5792% | **+0.9272 pp** |
| MNLI mismatched | **63.2425%** | 62.2254% | **+1.0171 pp** |

ON round 3 is better overall because its 5.2624-point advantage on HANS non-entailment is larger than its 3.7535-point disadvantage on entailment. It also retains MNLI slightly better than OFF round 4. Notably, this ON checkpoint used 32 fewer acquisition examples.

However, this is a **post-hoc, unequal-budget checkpoint comparison**, not the controlled verification ablation. The controlled comparison remains ON round 4 versus OFF round 4 because those checkpoints share the same 128-example budget and training stage; under that primary comparison, OFF is better for seed 42. At the equal-budget round-3 stage, the available stopping-subset scores are ON 71.0000% versus OFF 64.5556% (+6.4444 points for ON), but OFF round 3 has not been evaluated on full HANS and MNLI, so no equal-round full-set conclusion should be claimed yet.

### What the three reported accuracies mean

The values below come from different evaluation views and must not be conflated:

| Evaluation view | Number of examples | ON round-3 accuracy | Role |
|---|---:|---:|---|
| Predeclared stopping subset | 900 | 71.0000% | Produces the learning-curve value that motivated this diagnostic |
| Gain-report set | 2,850 | 68.0000% | Broader development analysis with heuristic/subcase breakdowns |
| Full HANS evaluation | 15,000 | 71.2533% | Post-hoc full-set result reported in the checkpoint comparison above |

The 71.0000% value is therefore an **accuracy on the 900-example stopping subset**, not the validation loss used by training early stopping and not the original final HANS result. The independent full-HANS evaluation happens to be close at 71.2533%.

### Why this checkpoint was not automatically selected

Training early stopping operates **within each acquisition round**: it selects and restores the epoch with the lowest binary validation NLL for that round. It does not select the best checkpoint across rounds. The experiment protocol also fixes the acquisition budget at four rounds x 32 examples, so the pipeline committed round 3 and continued through round 4 rather than stopping acquisition when the round-3 accuracy reached 71%.

Moreover, the round-3 checkpoint was chosen for this analysis only after its 71% stopping accuracy had been observed. Its full HANS evaluation is consequently marked **post-hoc diagnostic only** and must not replace the predeclared ON round-4 primary result in the `Final metrics` table. The sealed final set was not used to choose round 3, but it has now been observed for this checkpoint and must not be used for further tuning.

For a future primary comparison, a cross-round selection rule must be declared before running seeds 43 and 44 and must use only an allowed validation metric, such as predeclared validation loss. That future rule should be kept distinct from the current within-round epoch early stopping and from adaptive acquisition stopping, which remains outside the scope of this experiment.

Full post-hoc artifacts: [ON seed-42 round-3 result](../../outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_42/posthoc/round_003_stopping_accuracy_71/result.json).

## Training and resource summary

| Run | Committed AdamW steps | Best epochs per round | GPU proxy seconds | Teacher seconds | Teacher input/output tokens |
|---|---:|---|---:|---:|---:|
| ON seed 42 | 72 | 1,1,8,8 | 3851.99 | 17996.91 | 636721/368119 |
| OFF seed 42 | 56 | 1,1,4,8 | 6584.82 | 3157.96 | 134080/80479 |

## Completion

- Completed runs: 2/6
- Missing runs: on:43, on:44, off:43, off:44
