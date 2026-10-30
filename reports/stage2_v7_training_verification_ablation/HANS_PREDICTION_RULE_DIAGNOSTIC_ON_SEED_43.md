# HANS prediction-rule diagnostic: V7 Verification ON seed 43

> **Diagnostic only.** This analysis re-scores the frozen HANS final set with existing round-3 and round-4 checkpoints. It does not modify the model, training, evaluator, checkpoints, or primary results.

## Prediction rules

1. `argmax(P(E), P(N), P(C)) -> HANS binary` (current evaluator rule).
2. `argmax(P(E), P(N)+P(C))` (direct binary aggregation).

`MNLI_LABELS` order is `entailment, neutral, contradiction`. Ties are resolved in favor of entailment for both rules, consistent with the current first-index argmax and binary-validation implementation.

## Full HANS metrics

| Checkpoint | Budget | Rule | Correct | Wrong | Overall | Entailment | Non-entailment |
|---|---:|---|---:|---:|---:|---:|---:|
| Round 3 | 96 examples | Three-way argmax -> binary | 10,391 | 4,609 | 69.2733% | 88.7528% | 49.7262% |
| Round 3 | 96 examples | Direct binary | 10,391 | 4,609 | 69.2733% | 88.7528% | 49.7262% |
| Round 4 | 128 examples | Three-way argmax -> binary | 11,161 | 3,839 | 74.4067% | 84.0543% | 64.7255% |
| Round 4 | 128 examples | Direct binary | 11,229 | 3,771 | 74.8600% | 82.2974% | 67.3968% |

## Prediction disagreement between rules

| Checkpoint | Different predictions | Rate | Three-way-only correct | Direct-binary-only correct | Directions |
|---|---:|---:|---:|---:|---|
| Round 3 | 0 / 15,000 | 0.0000% | 0 | 0 | none |
| Round 4 | 332 / 15,000 | 2.2133% | 132 | 200 | entailment -> non-entailment: 332 |

## Accuracy change from round 3 to round 4

Positive values mean round 4 is higher than round 3.

| Rule | Overall delta | Entailment delta | Non-entailment delta |
|---|---:|---:|---:|
| Three-way argmax -> binary | +5.1333 pp | -4.6985 pp | +14.9993 pp |
| Direct binary | +5.5867 pp | -6.4555 pp | +17.6706 pp |

## Validation history for rounds 3-4

The logged epoch accuracy and NLL use the direct-binary rule `argmax(P(E), P(N)+P(C))` on the predeclared 900-example stopping subset. The sealed final HANS set was not used for these epoch logs.

### Round 3

Best epoch by validation NLL: **8**; best NLL: **2.509666**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 4.396196 | 64.6667% | 95.7778% | 33.5556% | yes |
| 2 | 16 | 4.469596 | 60.6667% | 97.3333% | 24.0000% | no |
| 3 | 20 | 4.064521 | 60.4444% | 96.4444% | 24.4444% | yes |
| 4 | 24 | 3.751658 | 63.5556% | 94.8889% | 32.2222% | yes |
| 5 | 28 | 3.855337 | 65.3333% | 93.1111% | 37.5556% | no |
| 6 | 32 | 3.337520 | 66.6667% | 92.2222% | 41.1111% | yes |
| 7 | 36 | 2.839739 | 68.1111% | 91.3333% | 44.8889% | yes |
| 8 | 40 | 2.509666 | 69.0000% | 90.2222% | 47.7778% | yes |

### Round 4

Best epoch by validation NLL: **8**; best NLL: **0.627107**.

| Epoch | Optimizer step end | Validation NLL | Accuracy | Entailment | Non-entailment | Improved |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 44 | 2.234386 | 69.3333% | 89.1111% | 49.5556% | yes |
| 2 | 48 | 1.757398 | 69.7778% | 88.0000% | 51.5556% | yes |
| 3 | 52 | 1.303386 | 70.5556% | 85.5556% | 55.5556% | yes |
| 4 | 56 | 1.098376 | 71.2222% | 85.1111% | 57.3333% | yes |
| 5 | 60 | 1.066373 | 71.4444% | 86.8889% | 56.0000% | yes |
| 6 | 64 | 1.010374 | 70.2222% | 89.5556% | 50.8889% | yes |
| 7 | 68 | 0.793227 | 72.2222% | 88.6667% | 55.7778% | yes |
| 8 | 72 | 0.627107 | 74.1111% | 83.3333% | 64.8889% | yes |

## Provenance and consistency checks

- Frozen HANS examples: `15,000`.
- Round-3 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_43/training/round_003/adapter`.
- Round-4 adapter: `/home/llm/hainn/hainn/active_learning/outputs/stage2_runs_v7_training_verification_ablation/verification_on/seed_43/training/round_004/adapter`.
- Round-3 current-rule consistency check: `not available (no stored full-HANS predictions for this checkpoint)`.
- Round-4 current-rule consistency check: `0 mismatches`.
- Raw probability and per-example comparison artifacts are stored under `/home/llm/hainn/hainn/active_learning/outputs/stage2_diagnostics/v7_hans_prediction_rules/verification_on/seed_43`.

