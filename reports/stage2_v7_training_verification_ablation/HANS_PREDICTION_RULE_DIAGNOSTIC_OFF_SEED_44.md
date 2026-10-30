# HANS prediction-rule diagnostic: V7 Verification OFF seed 44

> **Diagnostic only.** This analysis re-scores the frozen HANS final set with existing round-3 and round-4 checkpoints. It does not modify the model, training, evaluator, checkpoints, or primary results.

## Prediction rules

1. `argmax(P(E), P(N), P(C)) -> HANS binary` (current evaluator rule).
2. `argmax(P(E), P(N)+P(C))` (direct binary aggregation).

`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are resolved in favor of entailment for both rules, consistent with the current first-index argmax and binary-validation implementation.

## Full HANS metrics

| Checkpoint | Budget | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---|---:|---|---:|---:|---:|---:|---:|
| Round 3 | 96 examples | Three-way argmax -> binary | 10,369 | 4,631 | 69.1267% | 88.9392% | 49.2454% |
| Round 3 | 96 examples | Direct binary | 10,369 | 4,631 | 69.1267% | 88.9392% | 49.2454% |
| Round 4 | 128 examples | Three-way argmax -> binary | 10,880 | 4,120 | 72.5333% | 87.4351% | 57.5798% |
| Round 4 | 128 examples | Direct binary | 10,980 | 4,020 | 73.2000% | 86.6099% | 59.7436% |

## Prediction disagreement between rules

| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |
|---|---:|---:|---:|---:|---|
| Round 3 | 0 / 15,000 | 0.0000% | 0 | 0 | none |
| Round 4 | 224 / 15,000 | 1.4933% | 62 | 162 | entailment -> non-entailment: 224 |

## Accuracy change from round 3 to round 4

Positive values mean round 4 is higher than round 3.

| Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---:|---:|---:|
| Three-way argmax -> binary | +3.4067 pp | -1.5041 pp | +8.3344 pp |
| Direct binary | +4.0733 pp | -2.3293 pp | +10.4982 pp |

## Validation history for rounds 3-4

The logged epoch accuracy and NLL use the direct-binary rule `argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping subset. The sealed final HANS set was not used for these epoch logs.

### Round 3

Best epoch by validation NLL: **8**; best NLL: **2.309120**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 4.589293 | 63.2222% | 96.2222% | 30.2222% | yes |
| 2 | 16 | 4.728622 | 58.6667% | 98.2222% | 19.1111% | no |
| 3 | 20 | 4.311720 | 58.4444% | 97.3333% | 19.5556% | yes |
| 4 | 24 | 4.093725 | 60.5556% | 96.2222% | 24.8889% | yes |
| 5 | 28 | 3.968838 | 62.7778% | 95.1111% | 30.4444% | yes |
| 6 | 32 | 3.605447 | 65.3333% | 93.5556% | 37.1111% | yes |
| 7 | 36 | 2.943410 | 67.0000% | 92.0000% | 42.0000% | yes |
| 8 | 40 | 2.309120 | 69.1111% | 90.6667% | 47.5556% | yes |

### Round 4

Best epoch by validation NLL: **8**; best NLL: **0.813875**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44 | 2.080532 | 69.8889% | 89.5556% | 50.2222% | yes |
| 2 | 48 | 1.951177 | 69.7778% | 89.1111% | 50.4444% | yes |
| 3 | 52 | 1.731575 | 69.4444% | 89.3333% | 49.5556% | yes |
| 4 | 56 | 1.396108 | 69.8889% | 88.8889% | 50.8889% | yes |
| 5 | 60 | 1.129882 | 70.3333% | 88.8889% | 51.7778% | yes |
| 6 | 64 | 1.052817 | 70.8889% | 89.3333% | 52.4444% | yes |
| 7 | 68 | 0.927737 | 71.1111% | 90.2222% | 52.0000% | yes |
| 8 | 72 | 0.813875 | 72.5556% | 88.6667% | 56.4444% | yes |

## Provenance and consistency checks

- Frozen HANS examples: `15,000`.
- Round-3 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_44/training/round_003/adapter`.
- Round-4 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_off/seed_44/training/round_004/adapter`.
- Round-3 current-rule consistency check: `not available (no stored full-HANS predictions for this checkpoint)`.
- Round-4 current-rule consistency check: `0 mismatches`.
- Raw probability and per-example comparison artifacts are stored under `/home/llm/hainn/hainn/active_learning/outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_off/seed_44`.

