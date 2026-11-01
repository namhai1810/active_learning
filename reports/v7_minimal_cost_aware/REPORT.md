# V7 Minimal Cost-Aware Report

> Artifact-only report; no model, teacher, training, or inference is run.

The controller evaluates the exact mixed-hypothesis V7 soft-pair action. It does not claim the appendix theorem for hypothesis-specific interventions.

## Final results

| Seed | Rounds | Examples | Stop category | Stop reason | HANS Three-way | HANS Direct Binary | MNLI m/mm | Modeled decision cost (r2+) | Teacher tokens |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| 42 | 8 | 256 | `maximum_budget` | `maximum_budget_reached_with_remaining_hypotheses` | 75.0800% | 75.8333% | 59.7453% / 60.0081% | 10.489403 | 2174118 |

## Mean ± sample standard deviation

- HANS Three-way: 75.0800 ± 0.0000%
- HANS Direct Binary: 75.8333 ± 0.0000%
- MNLI matched: 59.7453 ± 0.0000%
- MNLI mismatched: 60.0081 ± 0.0000%
- Acquired examples: 256.0000 ± 0.0000
- Completed rounds: 8.0000 ± 0.0000
- Modeled decision cost (rounds 2+): 10.4894 ± 0.0000


## Seed 42 detailed interpretation

### Same-seed historical comparison

| Run | Budget | HANS Three-way | MNLI-m | MNLI-mm | HANS delta vs Cost-Aware |
|---|---:|---:|---:|---:|---:|
| V7 Verification ON round 3 (post-hoc) | 96 | 71.2533% | 62.5064% | 63.2425% | -3.8267 pp |
| V7 Verification ON final | 128 | 68.6333% | 57.8095% | 57.4451% | -6.4467 pp |
| V7 Verification OFF final | 128 | 70.5067% | 61.5792% | 62.2254% | -4.5733 pp |
| **V7 Minimal Cost-Aware seed 42** | **256** | **75.0800%** | 59.7453% | 60.0081% | — |

The comparison is same-seed but not same-budget: the new run uses 256 examples, while the historical primary ON/OFF runs use 128 and the ON round-3 diagnostic uses 96. Seed controls stochastic choices; it does not make different budgets or training trajectories equivalent.

- The validation curve through round 4 matches the historical Verification-ON run. This is regression evidence that the preserved V7 prefix behaves as intended.
- The new run then adds four refresh/acquisition rounds and reaches 104 committed AdamW steps, versus 72 for historical ON.
- Its Three-way HANS result already exceeds the historical checkpoints, so the improvement is not an artifact of switching to Direct Binary.
- MNLI is higher than historical ON-final but lower than OFF-final and ON-round-3, showing a HANS-robustness/MNLI-retention trade-off rather than uniform domination.

### Validation learning curve

| Round | Cumulative budget | Accuracy | Entailment | Non-entailment |
|---:|---:|---:|---:|---:|
| 0 | 0 | 63.2222% | 94.4444% | 32.0000% |
| 1 | 32 | 64.6667% | 93.7778% | 35.5556% |
| 2 | 64 | 65.4444% | 94.2222% | 36.6667% |
| 3 | 96 | 71.0000% | 88.0000% | 54.0000% |
| 4 | 128 | 66.6667% | 87.5556% | 45.7778% |
| 5 | 160 | 63.6667% | 93.1111% | 34.2222% |
| 6 | 192 | 66.0000% | 92.8889% | 39.1111% |
| 7 | 224 | 74.7778% | 80.0000% | 69.5556% |
| 8 | 256 | 73.8889% | 81.5556% | 66.2222% |

The highest stopping-subset accuracy is 74.7778% at round 7. This is diagnostic only: no cross-round rule predeclared that accuracy as an acquisition stopping criterion, and the sealed final set must not be used to select a round.

### What this run establishes about Cost-Aware stopping

- Decision rounds with unresolved hypotheses: 7/7.
- Pilot observations executed: 0.
- Rounds that evaluated `Gain-UCB - lambda*Cost`: 0.
- Terminal category: `maximum_budget`; reason: `maximum_budget_reached_with_remaining_hypotheses`.

Consequently, seed 42 does **not** empirically validate a gain-cost STOP. Every decision round was gated by unresolved evidence, so the observed performance should be interpreted as an eight-round, maximum-budget V7 trajectory under the safe evidence gate—not as an accuracy gain caused by the stopping controller.

### Final prediction-rule diagnostic

- Three-way HANS: 75.0800%.
- Direct Binary HANS: 75.8333% (+0.7533 pp).
- Rule disagreement: 693/15,000 (4.6200%).

### Seed 42 resource summary

- Teacher calls/tokens: 1,048 / 2,174,118.
- Teacher wall time: 8.60 hours.
- Refresh wall time: 9.01 hours.
- Training / validation GPU-wall proxies: 1.42 / 23.33 minutes.
- Final evaluation GPU-wall proxy: 38.14 minutes.
- Modeled decision cost for rounds 2+: 10.489403 configured USD units. Round 1 and the shared initial pool are excluded.

## Decision traces

| Seed | Round | Budget before | Certified | Unresolved | Pilots | Gain | Radius | Raw gain UCB | Bounded gain UCB | Cost | λCost | UCB net | Action | Reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 1 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `continue` | `round_1_preserved_from_v7` |
| 42 | 2 | 32 | 3 | 2 | 0 | None | None | None | None | 1.4423658666666666 | 0.28847317333333333 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |
| 42 | 3 | 64 | 2 | 2 | 0 | None | None | None | None | 1.4661526222222223 | 0.2932305244444445 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |
| 42 | 4 | 96 | 0 | 5 | 0 | None | None | None | None | 1.4958388 | 0.29916776 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |
| 42 | 5 | 128 | 2 | 3 | 0 | None | None | None | None | 1.4846296 | 0.29692592 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |
| 42 | 6 | 160 | 0 | 5 | 0 | None | None | None | None | 1.5315102888888892 | 0.30630205777777786 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |
| 42 | 7 | 192 | 2 | 3 | 0 | None | None | None | None | 1.5331424444444444 | 0.3066284888888889 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |
| 42 | 8 | 224 | 1 | 3 | 0 | None | None | None | None | 1.5357632222222224 | 0.3071526444444445 | None | `continue` | `unresolved_hypotheses_require_more_evidence` |

## Interpretation limits

- Unresolved hypotheses force CONTINUE; they are never treated as refuted.
- Pilot gain uses predeclared, disjoint gain-eval shards and never sealed HANS/MNLI.
- Conditional independence/unbiasedness of related LoRA pilots is not empirically proven.
- With m=4, K=1, T=7, δ=.05 and M=1, the raw radius is about 1.6785. The gain UCB is intersected with the known support bound M; the radius itself is not reduced.
- Lambda is fixed ex ante at 0.2 gain/USD from the declared indifference point 1.0 gain = USD 5; no final-test outcome is used.
- Round-1 and the frozen initial-pool sunk cost are outside the round-2+ decision-cost aggregate.
