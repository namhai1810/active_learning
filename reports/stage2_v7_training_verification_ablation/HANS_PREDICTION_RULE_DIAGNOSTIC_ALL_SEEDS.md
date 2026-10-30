# HANS prediction-rule diagnostic: V7 Verification ON/OFF seeds 42–44

> **Diagnostic only.** All values below re-score frozen round-3 and round-4 checkpoints on the same 15,000-example HANS final set. The analysis does not modify training, checkpoints, the evaluator, or primary results. Final-test results must not be used to select a checkpoint or prediction rule.

## Prediction rules

1. **Three-way → binary (primary/current):** `argmax(P(E), P(N), P(C))`, then map neutral and contradiction to HANS non-entailment.
2. **Direct Binary (diagnostic):** `argmax(P(E), P(N)+P(C))`.

## Round 4: matched 128-example budget

| Seed | Rule | ON | OFF | ON − OFF |
|---:|---|---:|---:|---:|
| 42 | Three-way → binary | 68.6333% | 70.5067% | −1.8733 pp |
| 42 | Direct Binary | 71.8533% | 71.1533% | +0.7000 pp |
| 43 | Three-way → binary | 74.4067% | 73.3133% | +1.0933 pp |
| 43 | Direct Binary | 74.8600% | 74.0600% | +0.8000 pp |
| 44 | Three-way → binary | 71.9867% | 72.5333% | −0.5467 pp |
| 44 | Direct Binary | 73.6867% | 73.2000% | +0.4867 pp |

## Mean ± sample standard deviation across seeds

| Round | Budget | Rule | ON | OFF | ON − OFF mean |
|---:|---:|---|---:|---:|---:|
| 3 | 96 | Three-way → binary | 69.6867 ± 1.4063% | 67.7711 ± 2.6777% | +1.9156 pp |
| 3 | 96 | Direct Binary | 69.6956 ± 1.3954% | 67.7889 ± 2.6951% | +1.9067 pp |
| 4 | 128 | Three-way → binary | 71.6756 ± 2.8992% | 72.1178 ± 1.4487% | −0.4422 pp |
| 4 | 128 | Direct Binary | 73.4667 ± 1.5154% | 72.8044 ± 1.4932% | +0.6622 pp |

The sign of the descriptive ON/OFF comparison at round 4 changes with the mapping rule: OFF is numerically higher by 0.4422 pp under the current Three-way rule, whereas ON is numerically higher by 0.6622 pp under Direct Binary. This is a diagnostic sensitivity result, not evidence for selecting Direct Binary from the final test.

## Round 4 label balance

| Rule | Branch | Entailment mean | Non-entailment mean |
|---|---|---:|---:|
| Three-way → binary | ON | 86.6187% | 56.6805% |
| Three-way → binary | OFF | 85.1679% | 59.0223% |
| Direct Binary | ON | 84.0366% | 62.8601% |
| Direct Binary | OFF | 83.6506% | 61.9207% |

Direct Binary shifts predictions toward non-entailment. Across the three round-4 ON checkpoints, this lowers mean entailment accuracy and raises mean non-entailment accuracy relative to the current rule.

## Round 3 → round 4

| Branch | Rule | Mean round 3 | Mean round 4 | Change |
|---|---|---:|---:|---:|
| ON | Three-way → binary | 69.6867% | 71.6756% | +1.9889 pp |
| ON | Direct Binary | 69.6956% | 73.4667% | +3.7711 pp |
| OFF | Three-way → binary | 67.7711% | 72.1178% | +4.3467 pp |
| OFF | Direct Binary | 67.7889% | 72.8044% | +5.0156 pp |

The three-seed means improve from round 3 to round 4 for both branches and both rules. The exception at individual-seed level is ON seed 42: Three-way accuracy falls from 71.2533% to 68.6333% (−2.6200 pp), while Direct Binary rises from 71.2533% to 71.8533% (+0.6000 pp). ON seeds 43 and 44, and all evaluated OFF checkpoints, improve under both rules.

## Interpretation boundary

- The primary reported results remain the current Three-way → binary evaluator results.
- The Direct Binary values quantify sensitivity to aggregation of the MNLI neutral and contradiction probabilities.
- Round 3 uses 96 acquired examples; round 4 uses the fixed full budget of 128 examples.
- HANS final-test scores were not used for training or validation-loss early stopping.
- These final-test diagnostics are descriptive and must not be used to tune the rule or select round 3 versus round 4.

## Per-seed reports

- `HANS_PREDICTION_RULE_DIAGNOSTIC_ON_OFF.md` — seed 42.
- `HANS_PREDICTION_RULE_DIAGNOSTIC_ON_OFF_SEED_43.md` — seed 43.
- `HANS_PREDICTION_RULE_DIAGNOSTIC_ON_OFF_SEED_44.md` — seed 44.

