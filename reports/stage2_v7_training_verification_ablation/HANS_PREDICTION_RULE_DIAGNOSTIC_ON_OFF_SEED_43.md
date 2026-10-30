# HANS prediction-rule diagnostic: V7 Verification ON/OFF seed 43

> **Diagnostic only.** This analysis re-scores the frozen HANS final set with existing round-3 and round-4 checkpoints. It does not modify the model, training, evaluator, checkpoints, or primary results.

## Prediction rules

1. `argmax(P(E), P(N), P(C)) -> HANS binary` (current evaluator rule).
2. `argmax(P(E), P(N)+P(C))` (direct binary aggregation).

`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are resolved in favor of entailment for both rules, consistent with the current first-index argmax and binary-validation implementation.

## Full HANS metrics

| Checkpoint | Budget | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---|---:|---|---:|---:|---:|---:|---:|
| Round 3 | 96 examples | Three-way argmax -> binary | 10,425 | 4,575 | 69.5000% | 89.0856% | 49.8464% |
| Round 3 | 96 examples | Direct binary | 10,433 | 4,567 | 69.5533% | 89.0323% | 50.0067% |
| Round 4 | 128 examples | Three-way argmax -> binary | 10,997 | 4,003 | 73.3133% | 77.7719% | 68.8393% |
| Round 4 | 128 examples | Direct binary | 11,109 | 3,891 | 74.0600% | 74.3378% | 73.7812% |

## Prediction disagreement between rules

| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |
|---|---:|---:|---:|---:|---|
| Round 3 | 16 / 15,000 | 0.1067% | 4 | 12 | entailment -> non-entailment: 16 |
| Round 4 | 628 / 15,000 | 4.1867% | 258 | 370 | entailment -> non-entailment: 628 |

## Accuracy change from round 3 to round 4

Positive values mean round 4 is higher than round 3.

| Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---:|---:|---:|
| Three-way argmax -> binary | +3.8133 pp | -11.3137 pp | +18.9929 pp |
| Direct binary | +4.5067 pp | -14.6945 pp | +23.7745 pp |

## Validation history for rounds 3-4

The logged epoch accuracy and NLL use the direct-binary rule `argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping subset. The sealed final HANS set was not used for these epoch logs.

### Round 3

Best epoch by validation NLL: **8**; best NLL: **2.149493**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 4.524763 | 65.0000% | 95.7778% | 34.2222% | yes |
| 2 | 16 | 4.489443 | 62.4444% | 96.2222% | 28.6667% | yes |
| 3 | 20 | 4.051680 | 61.8889% | 96.0000% | 27.7778% | yes |
| 4 | 24 | 3.593722 | 64.1111% | 94.4444% | 33.7778% | yes |
| 5 | 28 | 3.651713 | 65.7778% | 93.5556% | 38.0000% | no |
| 6 | 32 | 3.432030 | 66.3333% | 92.8889% | 39.7778% | yes |
| 7 | 36 | 2.655547 | 68.1111% | 91.5556% | 44.6667% | yes |
| 8 | 40 | 2.149493 | 69.1111% | 90.6667% | 47.5556% | yes |

### Round 4

Best epoch by validation NLL: **8**; best NLL: **0.565837**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44 | 1.597019 | 69.2222% | 90.6667% | 47.7778% | yes |
| 2 | 48 | 1.070490 | 71.3333% | 89.3333% | 53.3333% | yes |
| 3 | 52 | 0.761461 | 72.5556% | 83.7778% | 61.3333% | yes |
| 4 | 56 | 0.689526 | 73.6667% | 80.4444% | 66.8889% | yes |
| 5 | 60 | 0.663338 | 74.0000% | 80.2222% | 67.7778% | yes |
| 6 | 64 | 0.610111 | 74.1111% | 79.5556% | 68.6667% | yes |
| 7 | 68 | 0.579577 | 74.6667% | 77.7778% | 71.5556% | yes |
| 8 | 72 | 0.565837 | 74.7778% | 75.1111% | 74.4444% | yes |

## Verification ON/OFF at the same round and budget

All entries re-score the same frozen 15,000-example HANS set. The final test is used only for this post-hoc diagnostic, not for checkpoint selection or training.

| Round | Budget | Branch | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 3 | 96 examples | ON | Three-way argmax -> binary | 10,391 | 4,609 | 69.2733% | 88.7528% | 49.7262% |
| 3 | 96 examples | ON | Direct binary | 10,391 | 4,609 | 69.2733% | 88.7528% | 49.7262% |
| 3 | 96 examples | OFF | Three-way argmax -> binary | 10,425 | 4,575 | 69.5000% | 89.0856% | 49.8464% |
| 3 | 96 examples | OFF | Direct binary | 10,433 | 4,567 | 69.5533% | 89.0323% | 50.0067% |
| 4 | 128 examples | ON | Three-way argmax -> binary | 11,161 | 3,839 | 74.4067% | 84.0543% | 64.7255% |
| 4 | 128 examples | ON | Direct binary | 11,229 | 3,771 | 74.8600% | 82.2974% | 67.3968% |
| 4 | 128 examples | OFF | Three-way argmax -> binary | 10,997 | 4,003 | 73.3133% | 77.7719% | 68.8393% |
| 4 | 128 examples | OFF | Direct binary | 11,109 | 3,891 | 74.0600% | 74.3378% | 73.7812% |

### ON minus OFF accuracy at matched budget

Positive values mean ON is numerically higher; negative values mean OFF is numerically higher. These are descriptive final-test differences, not a rule-selection criterion.

| Round | Budget | Rule | Overall ON-OFF | Entailment ON-OFF | Non-entailment ON-OFF |
|---:|---:|---|---:|---:|---:|
| 3 | 96 examples | Three-way argmax -> binary | -0.2267 pp | -0.3328 pp | -0.1202 pp |
| 3 | 96 examples | Direct binary | -0.2800 pp | -0.2795 pp | -0.2805 pp |
| 4 | 128 examples | Three-way argmax -> binary | +1.0933 pp | +6.2824 pp | -4.1138 pp |
| 4 | 128 examples | Direct binary | +0.8000 pp | +7.9595 pp | -6.3844 pp |

### Round 3 to round 4 within each branch

Positive values mean round 4 is numerically higher than round 3.

| Branch | Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---|---:|---:|---:|
| ON | Three-way argmax -> binary | +5.1333 pp | -4.6985 pp | +14.9993 pp |
| ON | Direct binary | +5.5867 pp | -6.4555 pp | +17.6706 pp |
| OFF | Three-way argmax -> binary | +3.8133 pp | -11.3137 pp | +18.9929 pp |
| OFF | Direct binary | +4.5067 pp | -14.6945 pp | +23.7745 pp |

## Provenance and consistency checks

- Frozen HANS examples: `15,000`.
- Round-3 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_43/training/round_003/adapter`.
- Round-4 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_43/training/round_004/adapter`.
- Round-3 current-rule consistency check: `not available (no stored full-HANS predictions for this checkpoint)`.
- Round-4 current-rule consistency check: `0 mismatches`.
- Raw probability and per-example comparison artifacts are stored under `/home/llm/hainn/hainn/active_learning/outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_off/seed_43`.

