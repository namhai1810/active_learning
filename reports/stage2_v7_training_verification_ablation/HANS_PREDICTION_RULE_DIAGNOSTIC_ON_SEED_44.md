# HANS prediction-rule diagnostic: V7 Verification ON seed 44

> **Diagnostic only.** This analysis re-scores the frozen HANS final set with existing round-3 and round-4 checkpoints. It does not modify the model, training, evaluator, checkpoints, or primary results.

## Prediction rules

1. `argmax(P(E), P(N), P(C)) -> HANS binary` (current evaluator rule).
2. `argmax(P(E), P(N)+P(C))` (direct binary aggregation).

`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are resolved in favor of entailment for both rules, consistent with the current first-index argmax and binary-validation implementation.

## Full HANS metrics

| Checkpoint | Budget | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---|---:|---|---:|---:|---:|---:|---:|
| Round 3 | 96 examples | Three-way argmax -> binary | 10,280 | 4,720 | 68.5333% | 91.6145% | 45.3720% |
| Round 3 | 96 examples | Direct binary | 10,284 | 4,716 | 68.5600% | 91.6145% | 45.4254% |
| Round 4 | 128 examples | Three-way argmax -> binary | 10,798 | 4,202 | 71.9867% | 87.2754% | 56.6449% |
| Round 4 | 128 examples | Direct binary | 11,053 | 3,947 | 73.6867% | 84.4935% | 62.8423% |

## Prediction disagreement between rules

| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |
|---|---:|---:|---:|---:|---|
| Round 3 | 4 / 15,000 | 0.0267% | 0 | 4 | entailment -> non-entailment: 4 |
| Round 4 | 673 / 15,000 | 4.4867% | 209 | 464 | entailment -> non-entailment: 673 |

## Accuracy change from round 3 to round 4

Positive values mean round 4 is higher than round 3.

| Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---:|---:|---:|
| Three-way argmax -> binary | +3.4533 pp | -4.3391 pp | +11.2729 pp |
| Direct binary | +5.1267 pp | -7.1210 pp | +17.4169 pp |

## Validation history for rounds 3-4

The logged epoch accuracy and NLL use the direct-binary rule `argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping subset. The sealed final HANS set was not used for these epoch logs.

### Round 3

Best epoch by validation NLL: **8**; best NLL: **2.109209**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 4.280063 | 64.8889% | 95.5556% | 34.2222% | yes |
| 2 | 16 | 4.103678 | 63.3333% | 95.7778% | 30.8889% | yes |
| 3 | 20 | 3.539351 | 63.5556% | 95.5556% | 31.5556% | yes |
| 4 | 24 | 2.894009 | 64.5556% | 93.7778% | 35.3333% | yes |
| 5 | 28 | 2.791781 | 66.1111% | 92.6667% | 39.5556% | yes |
| 6 | 32 | 2.763311 | 66.7778% | 92.0000% | 41.5556% | yes |
| 7 | 36 | 2.507712 | 67.5556% | 92.0000% | 43.1111% | yes |
| 8 | 40 | 2.109209 | 67.4444% | 92.6667% | 42.2222% | yes |

### Round 4

Best epoch by validation NLL: **6**; best NLL: **0.610089**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44 | 1.619178 | 68.4444% | 93.3333% | 43.5556% | yes |
| 2 | 48 | 1.321129 | 68.5556% | 93.3333% | 43.7778% | yes |
| 3 | 52 | 1.079638 | 69.7778% | 92.4444% | 47.1111% | yes |
| 4 | 56 | 0.976290 | 69.6667% | 92.6667% | 46.6667% | yes |
| 5 | 60 | 0.725650 | 71.5556% | 90.2222% | 52.8889% | yes |
| 6 | 64 | 0.610089 | 73.5556% | 85.5556% | 61.5556% | yes |
| 7 | 68 | 0.699472 | 68.0000% | 89.5556% | 46.4444% | no |
| 8 | 72 | 0.747394 | 67.3333% | 92.6667% | 42.0000% | no |

## Provenance and consistency checks

- Frozen HANS examples: `15,000`.
- Round-3 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_44/training/round_003/adapter`.
- Round-4 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_44/training/round_004/adapter`.
- Round-3 current-rule consistency check: `not available (no stored full-HANS predictions for this checkpoint)`.
- Round-4 current-rule consistency check: `0 mismatches`.
- Raw probability and per-example comparison artifacts are stored under `/home/llm/hainn/hainn/active_learning/outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_on/seed_44`.

