# HANS prediction-rule diagnostic: V7 Verification ON seed 42

> **Diagnostic only.** This analysis re-scores the frozen HANS final set with existing round-3 and round-4 checkpoints. It does not modify the model, training, evaluator, checkpoints, or primary results.

## Prediction rules

1. `argmax(P(E), P(N), P(C)) -> HANS binary` (current evaluator rule).
2. `argmax(P(E), P(N)+P(C))` (direct binary aggregation).

`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are resolved in favor of entailment for both rules, consistent with the current first-index argmax and binary-validation implementation.

## Full HANS metrics

| Checkpoint | Rule | Overall | Entailment | Non-entailment |
|---|---|---:|---:|---:|
| Round 3 | Three-way argmax -> binary | 71.2533% | 86.5433% | 55.9102% |
| Round 3 | Direct binary | 71.2533% | 86.5433% | 55.9102% |
| Round 4 | Three-way argmax -> binary | 68.6333% | 88.5266% | 48.6710% |
| Round 4 | Direct binary | 71.8533% | 85.3188% | 58.3411% |

## Prediction disagreement between rules

| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |
|---|---:|---:|---:|---:|---|
| Round 3 | 0 / 15,000 | 0.0000% | 0 | 0 | none |
| Round 4 | 965 / 15,000 | 6.4333% | 241 | 724 | entailment -> non-entailment: 965 |

## Accuracy change from round 3 to round 4

Positive values mean round 4 is higher than round 3.

| Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---:|---:|---:|
| Three-way argmax -> binary | -2.6200 pp | +1.9832 pp | -7.2392 pp |
| Direct binary | +0.6000 pp | -1.2245 pp | +2.4309 pp |

## Validation history for rounds 3-4

The logged epoch accuracy and NLL use the direct-binary rule `argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping subset. The sealed final HANS set was not used for these epoch logs.

### Round 3

Best epoch by validation NLL: **8**; best NLL: **1.876448**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 4.332918 | 64.6667% | 95.7778% | 33.5556% | yes |
| 2 | 16 | 3.933436 | 63.1111% | 96.0000% | 30.2222% | yes |
| 3 | 20 | 3.185779 | 63.7778% | 94.4444% | 33.1111% | yes |
| 4 | 24 | 2.483528 | 66.6667% | 92.4444% | 40.8889% | yes |
| 5 | 28 | 2.557312 | 68.6667% | 90.6667% | 46.6667% | no |
| 6 | 32 | 2.470068 | 69.0000% | 89.3333% | 48.6667% | yes |
| 7 | 36 | 2.294125 | 69.5556% | 89.3333% | 49.7778% | yes |
| 8 | 40 | 1.876448 | 71.0000% | 88.0000% | 54.0000% | yes |

### Round 4

Best epoch by validation NLL: **8**; best NLL: **0.588438**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44 | 1.556729 | 70.2222% | 87.7778% | 52.6667% | yes |
| 2 | 48 | 1.295879 | 69.5556% | 89.7778% | 49.3333% | yes |
| 3 | 52 | 1.120043 | 69.2222% | 91.5556% | 46.8889% | yes |
| 4 | 56 | 0.999955 | 69.3333% | 93.1111% | 45.5556% | yes |
| 5 | 60 | 0.943660 | 66.7778% | 94.6667% | 38.8889% | yes |
| 6 | 64 | 0.877141 | 64.4444% | 94.6667% | 34.2222% | yes |
| 7 | 68 | 0.657991 | 69.8889% | 90.0000% | 49.7778% | yes |
| 8 | 72 | 0.588438 | 70.6667% | 85.3333% | 56.0000% | yes |

## Provenance and consistency checks

- Frozen HANS examples: `15,000`.
- Round-3 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_42/training/round_003/adapter`.
- Round-4 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_42/training/round_004/adapter`.
- Round-3 current-rule mismatches against stored predictions: `0`.
- Round-4 current-rule mismatches against stored predictions: `0`.
- Raw probability and per-example comparison artifacts are stored under `/home/llm/hainn/hainn/active_learning/outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_on/seed_42`.

