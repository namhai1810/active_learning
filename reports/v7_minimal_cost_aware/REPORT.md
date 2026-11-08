# V7 Minimal Cost-Aware Report

> Artifact-only report; no model, teacher, training, or inference is run.

The controller evaluates the exact mixed-hypothesis V7 soft-pair action. It does not claim the appendix theorem for hypothesis-specific interventions.

## Final results

| Seed | Rounds | Examples | Stop category | Stop reason | HANS Three-way | HANS Direct Binary | MNLI m/mm | Decision cost (r2+) | Incurred exploration | Total pipeline | Teacher tokens |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 8 | 256 | `maximum_budget` | `maximum_budget_reached_with_remaining_hypotheses` | 75.0800% | 75.8333% | 59.7453% / 60.0081% | 6.288722 | 86.608397 | 90.963709 | 0 |
| 43 | 8 | 256 | `maximum_budget` | `maximum_budget_reached_with_remaining_hypotheses` | 76.7400% | 76.4600% | 61.2124% / 62.2050% | 6.288722 | 107.542688 | 112.036556 | 2188675 |

## Mean ± sample standard deviation

- HANS Three-way: 75.9100 ± 1.1738%
- HANS Direct Binary: 76.1467 ± 0.4431%
- MNLI matched: 60.4789 ± 1.0374%
- MNLI mismatched: 61.1066 ± 1.5534%
- Acquired examples: 256.0000 ± 0.0000
- Completed rounds: 8.0000 ± 0.0000
- Modeled decision cost (rounds 2+): 6.2887 ± 0.0000
- Incurred exploration cost (rounds 2+): 97.0755 ± 14.8028
- Total actual pipeline cost (rounds 2+): 101.5001 ± 14.9008

## Protocol v2.1: Range-Aware Hoeffding replay

The completed seed-42/43 trajectories were re-evaluated on CPU from their cached pilot observations. This replay did not run teacher inference, pilot training, model training, or final-test inference, and it did not modify the original checkpoints or metrics above.

For pilot gain `X_j = accuracy_after_j - A_0`, all 32 pilots within a round use the same fixed evaluation shard and baseline accuracy `A_0`. Therefore, conditionally on that shard, `X_j` lies in `[-A_0, 1-A_0]`, whose width is 1. Protocol v2.1 consequently uses

`b = sqrt(log(2KT/delta) / (2m)) = 0.296721`

with the frozen settings `m=32`, `K=1`, `T=7`, `delta=0.05`, `lambda=0.2`, and the unchanged marginal decision cost `C=0.898389`. The legacy symmetric-support calculation used `b=0.593443`. The stopping decision remains `STOP` only when `Delta-hat + b - lambda*C <= 0`, subject to the existing evidence gates.

| Seed | Closest round to STOP | Delta-hat | Legacy Net-UCB | Range-Aware Net-UCB | Range-Aware decision | Terminal outcome |
|---:|---:|---:|---:|---:|---|---|
| 42 | 3 | -0.045410 | 0.368355 | 0.071633 | `continue` | Maximum budget, 256 examples |
| 43 | 3 | -0.068848 | 0.344917 | 0.048196 | `continue` | Maximum budget, 256 examples |

### Main result

- The support correction halves the confidence radius from `0.593443` to `0.296721`.
- It does **not** produce a gain-cost STOP on either seed 42 or seed 43. Every evaluable round still has positive Range-Aware Net-UCB.
- With `lambda*C=0.179678`, a round would require `Delta-hat <= -0.117044` to satisfy the configured gain-cost stopping inequality. The lowest observed values were `-0.045410` for seed 42 and `-0.068848` for seed 43.
- Existing evidence gates are unchanged. Seed-42 rounds 4 and 6 and seed-43 round 6 continue because no certified hypothesis was available while evidence was not exhausted; these are not recorded as gain-cost STOP events.
- Because neither replayed trajectory stops early, the final HANS/MNLI values, budgets, checkpoints, and actual resource costs reported above remain unchanged. No cost saving is claimed from this replay.

This is a fixed-shard conditional analysis. Distinct pilot seeds provide training-randomization observations, but they do not establish population-level independence or unbiasedness. The mixed-policy controller therefore does not claim Appendix Theorem 1. The complete round-by-round comparison is in [RANGE_AWARE_REPLAY.md](RANGE_AWARE_REPLAY.md).


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
- Pilot observations executed: 160.
- Rounds that evaluated `Gain-UCB - lambda*Cost`: 5.
- Terminal category: `maximum_budget`; reason: `maximum_budget_reached_with_remaining_hypotheses`.

The run did not reach a gain-cost STOP. Its terminal result must be interpreted using the recorded operational/max-budget reason; unresolved hypotheses did not suppress complete mixed-policy pilots.

### Final prediction-rule diagnostic

- Three-way HANS: 75.0800%.
- Direct Binary HANS: 75.8333% (+0.7533 pp).
- Rule disagreement: 693/15,000 (4.6200%).

### Seed 42 resource summary

- Teacher calls/tokens: 0 / 0.
- Reused immutable source teacher calls/tokens (historical compute, not rerun): 1,048 / 2,174,118.
- Teacher wall time: 0.00 hours.
- Refresh wall time: 0.00 hours.
- Training / validation GPU-wall proxies: 0.00 / 0.00 minutes.
- Final evaluation GPU-wall proxy: 0.00 minutes.
- Modeled decision cost for rounds 2+: 6.288722 configured USD units. Round 1 and the shared initial pool are excluded.
- Incurred exploration cost for rounds 2+: 86.608397.
- Total actual pipeline cost for rounds 2+: 90.963709.

## Decision traces

| Seed | Round | Budget before | Certified | Unresolved | Pilots | Gain | Radius | Raw gain UCB | Bounded gain UCB | Cost | λCost | UCB net | Action | Reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 1 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `continue` | `round_1_preserved_and_reused_from_v7` |
| 42 | 2 | 32 | 3 | 2 | 32 | 0.0166015625 | 0.5934427943770807 | 0.6100443568770807 | 0.6100443568770807 | 0.8983888888888889 | 0.1796777777777778 | 0.43036657909930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 42 | 3 | 64 | 2 | 2 | 32 | -0.04541015625 | 0.5934427943770807 | 0.5480326381270807 | 0.5480326381270807 | 0.8983888888888889 | 0.1796777777777778 | 0.36835486034930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 42 | 4 | 96 | 0 | 5 | 0 | None | None | None | None | 0.8983888888888889 | 0.1796777777777778 | None | `continue` | `no_certified_hypothesis_but_evidence_not_exhausted` |
| 42 | 5 | 128 | 2 | 3 | 32 | -0.0263671875 | 0.5934427943770807 | 0.5670756068770807 | 0.5670756068770807 | 0.8983888888888889 | 0.1796777777777778 | 0.38739782909930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 42 | 6 | 160 | 0 | 5 | 0 | None | None | None | None | 0.8983888888888889 | 0.1796777777777778 | None | `continue` | `no_certified_hypothesis_but_evidence_not_exhausted` |
| 42 | 7 | 192 | 2 | 3 | 32 | 0.037109375 | 0.5934427943770807 | 0.6305521693770807 | 0.6305521693770807 | 0.8983888888888889 | 0.1796777777777778 | 0.45087439159930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 42 | 8 | 224 | 1 | 3 | 32 | 0.0166015625 | 0.5934427943770807 | 0.6100443568770807 | 0.6100443568770807 | 0.8983888888888889 | 0.1796777777777778 | 0.43036657909930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 43 | 1 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | `continue` | `round_1_preserved_from_v7` |
| 43 | 2 | 32 | 2 | 2 | 32 | 0.015625 | 0.5934427943770807 | 0.6090677943770807 | 0.6090677943770807 | 0.8983888888888889 | 0.1796777777777778 | 0.42939001659930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 43 | 3 | 64 | 2 | 3 | 32 | -0.06884765625 | 0.5934427943770807 | 0.5245951381270807 | 0.5245951381270807 | 0.8983888888888889 | 0.1796777777777778 | 0.34491736034930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 43 | 4 | 96 | 1 | 4 | 32 | 0.005859375 | 0.5934427943770807 | 0.5993021693770807 | 0.5993021693770807 | 0.8983888888888889 | 0.1796777777777778 | 0.41962439159930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 43 | 5 | 128 | 1 | 4 | 32 | -0.017578125 | 0.5934427943770807 | 0.5758646693770807 | 0.5758646693770807 | 0.8983888888888889 | 0.1796777777777778 | 0.39618689159930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 43 | 6 | 160 | 0 | 5 | 0 | None | None | None | None | 0.8983888888888889 | 0.1796777777777778 | None | `continue` | `no_certified_hypothesis_but_evidence_not_exhausted` |
| 43 | 7 | 192 | 3 | 2 | 32 | 0.0703125 | 0.5934427943770807 | 0.6637552943770807 | 0.6637552943770807 | 0.8983888888888889 | 0.1796777777777778 | 0.48407751659930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |
| 43 | 8 | 224 | 1 | 3 | 32 | -0.00537109375 | 0.5934427943770807 | 0.5880717006270807 | 0.5880717006270807 | 0.8983888888888889 | 0.1796777777777778 | 0.40839392284930287 | `continue` | `positive_mixed_policy_optimistic_net_gain` |

## Interpretation limits

- Unresolved hypotheses remain active for V7 verification/acquisition; they neither become refuted nor block a complete mixed-policy pilot estimate when certified support exists.
- Pilot gain uses one predeclared 64-example fixed shard per round, disjoint across rounds, the 900 stopping subset, held-out gain report, and sealed tests.
- The 32 Hoeffding observations are pilot-training randomizations conditional on that fixed empirical shard. Shared eval examples are not counted as independent observations; conditional independence/unbiasedness remains an explicit unverified assumption.
- With m=32, K=1, T=7, δ=.05 and M=1, the raw radius is about 0.5934. The gain UCB is intersected with the known support bound M; the radius itself is not reduced.
- Lambda is fixed ex ante at 0.2 gain/USD from the declared indifference point 1.0 gain = USD 5; no final-test outcome is used.
- Protocol v2 excludes already-incurred teacher/verification/generation/pilot cost from U. Decision cost is only the marginal main-update cost if CONTINUE; incurred and total actual cost remain separately reported.
- Round-1 and the frozen initial-pool sunk cost are outside the round-2+ decision-cost aggregate.
