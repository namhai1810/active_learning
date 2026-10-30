# HANS prediction-rule diagnostic: V7 Verification ON/OFF seed 42

> **Diagnostic only.** This analysis re-scores the frozen HANS final set with existing round-3 and round-4 checkpoints. It does not modify the model, training, evaluator, checkpoints, or primary results.

## Prediction rules

1. `argmax(P(E), P(N), P(C)) -> HANS binary` (current evaluator rule).
2. `argmax(P(E), P(N)+P(C))` (direct binary aggregation).

`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are resolved in favor of entailment for both rules, consistent with the current first-index argmax and binary-validation implementation.

## Full HANS metrics

| Checkpoint | Budget | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---|---:|---|---:|---:|---:|---:|---:|
| Round 3 | 96 examples | Three-way argmax -> binary | 9,703 | 5,297 | 64.6867% | 93.7309% | 35.5416% |
| Round 3 | 96 examples | Direct binary | 9,703 | 5,297 | 64.6867% | 93.7309% | 35.5416% |
| Round 4 | 128 examples | Three-way argmax -> binary | 10,576 | 4,424 | 70.5067% | 90.2968% | 50.6478% |
| Round 4 | 128 examples | Direct binary | 10,673 | 4,327 | 71.1533% | 90.0040% | 52.2372% |

## Prediction disagreement between rules

| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |
|---|---:|---:|---:|---:|---|
| Round 3 | 0 / 15,000 | 0.0000% | 0 | 0 | none |
| Round 4 | 141 / 15,000 | 0.9400% | 22 | 119 | entailment -> non-entailment: 141 |

## Accuracy change from round 3 to round 4

Positive values mean round 4 is higher than round 3.

| Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---:|---:|---:|
| Three-way argmax -> binary | +5.8200 pp | -3.4340 pp | +15.1062 pp |
| Direct binary | +6.4667 pp | -3.7269 pp | +16.6956 pp |

## Validation history for rounds 3-4

The logged epoch accuracy and NLL use the direct-binary rule `argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping subset. The sealed final HANS set was not used for these epoch logs.

### Round 3

Best epoch by validation NLL: **4**; best NLL: **3.184773**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 4.369598 | 64.5556% | 95.7778% | 33.3333% | yes |
| 2 | 16 | 4.257364 | 61.8889% | 96.2222% | 27.5556% | yes |
| 3 | 20 | 3.708589 | 62.1111% | 96.0000% | 28.2222% | yes |
| 4 | 24 | 3.184773 | 64.5556% | 94.0000% | 35.1111% | yes |
| 5 | 28 | 3.388537 | 65.5556% | 93.1111% | 38.0000% | no |
| 6 | 32 | 3.197255 | 66.7778% | 92.4444% | 41.1111% | no |

### Round 4

Best epoch by validation NLL: **8**; best NLL: **1.121457**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 28 | 3.282184 | 65.6667% | 93.1111% | 38.2222% | yes |
| 2 | 32 | 3.090216 | 66.6667% | 92.4444% | 40.8889% | yes |
| 3 | 36 | 2.836118 | 67.2222% | 92.4444% | 42.0000% | yes |
| 4 | 40 | 2.336295 | 67.4444% | 92.0000% | 42.8889% | yes |
| 5 | 44 | 1.986215 | 67.1111% | 92.2222% | 42.0000% | yes |
| 6 | 48 | 1.598739 | 68.1111% | 92.2222% | 44.0000% | yes |
| 7 | 52 | 1.272718 | 69.8889% | 91.7778% | 48.0000% | yes |
| 8 | 56 | 1.121457 | 70.3333% | 91.3333% | 49.3333% | yes |

## Verification ON/OFF at the same round and budget

All entries re-score the same frozen 15,000-example HANS set. The final test is used only for this post-hoc diagnostic, not for checkpoint selection or training.

| Round | Budget | Branch | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| 3 | 96 examples | ON | Three-way argmax -> binary | 10,688 | 4,312 | 71.2533% | 86.5433% | 55.9102% |
| 3 | 96 examples | ON | Direct binary | 10,688 | 4,312 | 71.2533% | 86.5433% | 55.9102% |
| 3 | 96 examples | OFF | Three-way argmax -> binary | 9,703 | 5,297 | 64.6867% | 93.7309% | 35.5416% |
| 3 | 96 examples | OFF | Direct binary | 9,703 | 5,297 | 64.6867% | 93.7309% | 35.5416% |
| 4 | 128 examples | ON | Three-way argmax -> binary | 10,295 | 4,705 | 68.6333% | 88.5266% | 48.6710% |
| 4 | 128 examples | ON | Direct binary | 10,778 | 4,222 | 71.8533% | 85.3188% | 58.3411% |
| 4 | 128 examples | OFF | Three-way argmax -> binary | 10,576 | 4,424 | 70.5067% | 90.2968% | 50.6478% |
| 4 | 128 examples | OFF | Direct binary | 10,673 | 4,327 | 71.1533% | 90.0040% | 52.2372% |

### ON minus OFF accuracy at matched budget

Positive values mean ON is numerically higher; negative values mean OFF is numerically higher. These are descriptive final-test differences, not a rule-selection criterion.

| Round | Budget | Rule | Overall ON-OFF | Entailment ON-OFF | Non-entailment ON-OFF |
|---:|---:|---|---:|---:|---:|
| 3 | 96 examples | Three-way argmax -> binary | +6.5667 pp | -7.1875 pp | +20.3686 pp |
| 3 | 96 examples | Direct binary | +6.5667 pp | -7.1875 pp | +20.3686 pp |
| 4 | 128 examples | Three-way argmax -> binary | -1.8733 pp | -1.7703 pp | -1.9768 pp |
| 4 | 128 examples | Direct binary | +0.7000 pp | -4.6852 pp | +6.1039 pp |

### Round 3 to round 4 within each branch

Positive values mean round 4 is numerically higher than round 3.

| Branch | Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---|---:|---:|---:|
| ON | Three-way argmax -> binary | -2.6200 pp | +1.9832 pp | -7.2392 pp |
| ON | Direct binary | +0.6000 pp | -1.2245 pp | +2.4309 pp |
| OFF | Three-way argmax -> binary | +5.8200 pp | -3.4340 pp | +15.1062 pp |
| OFF | Direct binary | +6.4667 pp | -3.7269 pp | +16.6956 pp |

## Provenance and consistency checks

- Frozen HANS examples: `15,000`.
- Round-3 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_42/training/round_003/adapter`.
- Round-4 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_42/training/round_004/adapter`.
- Round-3 current-rule consistency check: `not available (no stored full-HANS predictions for this checkpoint)`.
- Round-4 current-rule consistency check: `0 mismatches`.
- Raw probability and per-example comparison artifacts are stored under `/home/llm/hainn/hainn/active_learning/outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_off/seed_42`.

