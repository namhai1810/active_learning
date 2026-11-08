# KL-Hoeffding vs Range-Aware Hoeffding for V7

> CPU-only replay over frozen 32-pilot caches for seeds 42/43. No GPU, Teacher API, inference, training, controller modification, or historical-artifact mutation was performed.

## Executive result

Across **11** complete cached rounds, KL radius is `0.138361–0.250092` versus the constant Range-Aware Hoeffding radius `0.296721`. Radius reduction is **15.71%–53.37%**, mean **30.96%**.

Using conditional full-round Cost-LCB `$0.226400` only as an **optimistic hypothetical diagnostic**, Hoeffding produces `0` numerical STOPs and KL produces `0`. Certified same-round STOPs: `0`; certified pre-round STOPs: `0`.

## Per-round same-round conditional diagnostic

| Seed | Round | Pilot mean gain | Hoeffding radius | KL radius | Reduction % | Hoeffding Net-UCB | KL Net-UCB | Numerical STOP |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 2 | 0.016602 | 0.296721 | 0.242934 | 18.13% | 0.268043 | 0.214255 | no |
| 42 | 3 | -0.045410 | 0.296721 | 0.185790 | 37.39% | 0.157845 | 0.095100 | no |
| 42 | 5 | -0.026367 | 0.296721 | 0.212642 | 28.34% | 0.225074 | 0.140995 | no |
| 42 | 7 | 0.037109 | 0.296721 | 0.212234 | 28.47% | 0.288551 | 0.204064 | no |
| 42 | 8 | 0.016602 | 0.296721 | 0.152173 | 48.72% | 0.157845 | 0.123494 | no |
| 43 | 2 | 0.015625 | 0.296721 | 0.238073 | 19.77% | 0.267066 | 0.208418 | no |
| 43 | 3 | -0.068848 | 0.296721 | 0.250092 | 15.71% | 0.182594 | 0.135964 | no |
| 43 | 4 | 0.005859 | 0.296721 | 0.224710 | 24.27% | 0.257301 | 0.185289 | no |
| 43 | 5 | -0.017578 | 0.296721 | 0.179647 | 39.46% | 0.173470 | 0.116789 | no |
| 43 | 7 | 0.070312 | 0.296721 | 0.138361 | 53.37% | 0.189095 | 0.163394 | no |
| 43 | 8 | -0.005371 | 0.296721 | 0.216851 | 26.92% | 0.246070 | 0.166200 | no |

`Y_j = X_j + baseline_accuracy = accuracy_after_j` was mechanically checked in `[0,1]`. Both methods use `m=32`, `K=1`, `T=7`, `delta=0.05`, and the same `log(2KT/delta)` confidence budget. The KL result is intersected with the natural upper endpoint through `u≤1`; its radius is verified not to exceed Hoeffding/Pinsker radius.

The closest KL case is seed 42 round 3: mean gain `-0.045410`, KL radius `0.185790`, and optimistic hypothetical Net-UCB `0.095100`.

This is not a post-acquisition certificate. Teacher/verification/generation/selection and pilot costs are already incurred when current-round pilots exist; they are not subtracted again. The `$0.2264` term is shown only as a hypothetical lower bound for a full future executable round.

## Pre-round feasibility

Current-round pilots are created only after Teacher, verification, generation, and selected-batch construction. Therefore their KL/Hoeffding bounds are unavailable before the same round and cannot implement a valid pre-round STOP.

A prior-round bound is available in time but targets the previous checkpoint and selected batch. Without a proved temporal-drift and policy-transfer bound, it is not a valid upper bound for next-round gain. The replay never substitutes current-round gain into a pre-round decision.

| Comparison | Numerical status | Certified STOP | Reason |
|---|---|:---:|---|
| Same-round KL/Hoeffding | conditional diagnostic | no | conditional independence is unproved; evidence is post-acquisition |
| Pre-round using same-round pilots | unavailable | no | current-round pilots do not exist before Teacher |
| Pre-round using prior-round pilots | uncertified | no | drift and selected-policy transfer are unproved |

## Bottleneck

KL tightens sampling uncertainty, but the closest hypothetical Net-UCB remains `0.095100`. At `lambda=0.2`, that row would require an eligible Cost-LCB of at least `$0.701898`, which is `$0.475498` above the frozen `$0.226400` diagnostic. More importantly, no cached metadata proves conditional independence of deterministic pilot seeds, and no cross-round transfer theorem makes the same-round bound available before Teacher. Thus a smaller radius alone cannot create a certified pre-round STOP.

## Appendix proposition (assumption-conditional)

**Proposition.** Fix a round after its checkpoint and selected batch are measurable with respect to history `F_t`. Suppose the normalized pilot outcomes `Y_1,…,Y_m∈[0,1]` are conditionally independent given `F_t` and share conditional mean `mu_t`. For each of the predeclared `K×T` tests, let `u_t` be the largest `u∈[mean(Y),1]` satisfying

`kl(mean(Y) || u) ≤ log(2KT/delta)/m`.

Then, by the one-sided conditional KL-Hoeffding/Chernoff inequality and a union bound over `K×T`, simultaneously `mu_t ≤ u_t` with failure probability at most `delta/2` (hence at most `delta`). Therefore the conditional fixed-shard gain satisfies `mu_t − A_0 ≤ u_t − A_0`.

**Proof sketch.** Conditional on `F_t`, Chernoff inversion gives `P(mu_t>u_t | F_t) ≤ exp(-m·kl(mean(Y)||u_t)) = delta/(2KT)` for each test. Sum these probabilities over the predeclared family. Subtracting the fixed baseline `A_0` preserves the inequality.

The proposition does not establish the assumptions from deterministic cached seeds, does not give a population-HANS guarantee, and does not transfer the bound to a future round or a different selected policy.

## Missing rounds and provenance

- Seed 42 R4: `0` pilots; `no_certified_hypothesis_but_evidence_not_exhausted`. No data were fabricated.
- Seed 42 R6: `0` pilots; `no_certified_hypothesis_but_evidence_not_exhausted`. No data were fabricated.
- Seed 43 R6: `0` pilots; `no_certified_hypothesis_but_evidence_not_exhausted`. No data were fabricated.

Full per-round JSON/CSV and validation provenance are stored under `outputs/stage2_diagnostics/v7_kl_hoeffding/`.
