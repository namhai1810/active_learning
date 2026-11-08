# V7 Cost-Aware Stopping — Final Mathematical Feasibility

> CPU-only offline audit of frozen seeds 42/43. No model inference, teacher call, training, or historical-artifact mutation was performed.

## Executive result

No valid pre-round certified STOP was obtained. The valid online classification is `uncertified_continue` through R8, followed by `budget_stop`; budget exhaustion is not a gain–cost certificate.

The numerical bounds improved, but two assumptions remain unproved: (1) conditional independence/unbiasedness of the 32 fixed-shard pilot training randomizations, and (2) a valid cross-round stability radius `rho` that transfers post-acquisition gain evidence to the next pre-round decision.

Even in the post-acquisition `rho=0` diagnostic, Hoeffding Net-UCB stays positive by `0.157845`–`0.288551`. At the frozen `lambda`, the proven cost LCB would need to increase by at least `$0.789225` (up to `$1.442754` across complete rounds), or the Hoeffding radius would need to shrink by at least `0.182594` in rounds where zero-radius STOP is numerically possible.

## A/B/C before–after summary

| Side | Before | Final audit | Consequence |
|---|---|---|---|
| A — gain | Symmetric Hoeffding radius `0.593443` | Range-aware Hoeffding `0.296721`; empirical Bernstein `0.424124`–`0.437743` | EB is tighter in 0/11 complete rounds; its finite-sample correction is not automatically beneficial. |
| B — cost | Prior v5 universally proven lower bound `$0.020000` | Conditional executable-round LCB `$0.226400` | Larger penalty, but discovery/verification/generation remain zero in the LCB because their positive minimum is state-dependent. |
| C — cross-round | No next-round transfer certificate | Historical max absolute drift `0.084473` is diagnostic only; valid `rho` remains unavailable | Same-round pilots cannot be used before that round; pre-round decisions remain uncertified. |

## A. Variance-aware gain bounds

`X_j = accuracy_after_j - accuracy_before` uses exact support `[-A0, 1-A0]` (width 1). The family-wise allocation is `delta/(K*T) = 0.05/7`. UCBs are intersected with the exact support upper endpoint.

| Seed | Round | m | Mean gain | Variance | Hoeffding r | EB r | H gain-UCB | EB gain-UCB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2 | 32 | 0.016602 | 0.000015 | 0.296721 | 0.426404 | 0.313323 | 0.406250 |
| 42 | 3 | 32 | -0.045410 | 0.000069 | 0.296721 | 0.429041 | 0.203125 | 0.203125 |
| 42 | 4 | 0 | — | — | — | — | — | — |
| 42 | 5 | 32 | -0.026367 | 0.000527 | 0.296721 | 0.437743 | 0.270354 | 0.281250 |
| 42 | 6 | 0 | — | — | — | — | — | — |
| 42 | 7 | 32 | 0.037109 | 0.000374 | 0.296721 | 0.435602 | 0.333831 | 0.343750 |
| 42 | 8 | 32 | 0.016602 | 0.000078 | 0.296721 | 0.429357 | 0.203125 | 0.203125 |
| 43 | 2 | 32 | 0.015625 | 0.000000 | 0.296721 | 0.424124 | 0.312346 | 0.390625 |
| 43 | 3 | 32 | -0.068848 | 0.000486 | 0.296721 | 0.437208 | 0.227874 | 0.343750 |
| 43 | 4 | 32 | 0.005859 | 0.000059 | 0.296721 | 0.428685 | 0.302581 | 0.343750 |
| 43 | 5 | 32 | -0.017578 | 0.000043 | 0.296721 | 0.428030 | 0.218750 | 0.218750 |
| 43 | 6 | 0 | — | — | — | — | — | — |
| 43 | 7 | 32 | 0.070312 | 0.000142 | 0.296721 | 0.431190 | 0.234375 | 0.234375 |
| 43 | 8 | 32 | -0.005371 | 0.000199 | 0.296721 | 0.432487 | 0.291350 | 0.312500 |

Mechanical checks confirm one fixed evaluation shard/baseline, the gain identity, bounded support, the same selected batch/checkpoint, and distinct pilot seeds. They do **not** prove conditional independence of deterministic seeds or population-level unbiasedness. Accordingly these are conditional fixed-shard bounds, not a population theorem.

## B. Protocol-based full-round cost lower bound

The audited lower bound is **$0.226400** per executable 32-example intervention. With `lambda=0.2`, the certified cost penalty is `0.045280` gain units.

| Stage | Included USD | In LCB | Why |
|---|---:|:---:|---|
| discovery | 0.000000 | no | candidate/state dependent; no positive token or GPU-time minimum is frozen |
| verification | 0.000000 | no | may be skipped when no candidates exist; adaptive probe count is state dependent |
| generation_and_qc | 0.000000 | no | generation is skipped when no active hypothesis remains |
| soft_pair_selection | 0.016400 | yes | two scoring passes over at least the acquired examples are mandatory |
| main_lora_training | 0.020000 | yes | one fixed per-run charge; runtime lower bound remains zero |
| epoch_validation_and_best_restore_check | 0.190000 | yes | at least one epoch validation plus one restored-checkpoint evaluation |
| pilot_estimation | 0.000000 | no | post-acquisition estimator overhead is not mandatory in the target V7 round action |

Historical realised costs are reported separately and never substituted for the online LCB. GPU-time projections are also excluded because a projection is not a proven positive runtime lower bound.

| Seed | Round | H Net-UCB (rho=0, post-acq) | EB Net-UCB | H required LCB | H LCB gap | Actual pipeline USD |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2 | 0.268043 | 0.360970 | 1.566615 | 1.340215 | 12.012469 |
| 42 | 3 | 0.157845 | 0.157845 | 1.015625 | 0.789225 | 28.392839 |
| 42 | 4 | — | — | — | — | 1.520650 |
| 42 | 5 | 0.225074 | 0.235970 | 1.351771 | 1.125371 | 18.272579 |
| 42 | 6 | — | — | — | — | 1.155116 |
| 42 | 7 | 0.288551 | 0.298470 | 1.669154 | 1.442754 | 17.316595 |
| 42 | 8 | 0.157845 | 0.157845 | 1.015625 | 0.789225 | 12.293461 |
| 43 | 2 | 0.267066 | 0.345345 | 1.561732 | 1.335332 | 12.095000 |
| 43 | 3 | 0.182594 | 0.298470 | 1.139369 | 0.912969 | 28.478625 |
| 43 | 4 | 0.257301 | 0.298470 | 1.512904 | 1.286504 | 28.844350 |
| 43 | 5 | 0.173470 | 0.173470 | 1.093750 | 0.867350 | 12.157219 |
| 43 | 6 | — | — | — | — | 1.471042 |
| 43 | 7 | 0.189095 | 0.189095 | 1.171875 | 0.945475 | 16.786969 |
| 43 | 8 | 0.246070 | 0.267220 | 1.456752 | 1.230352 | 12.203351 |

These Net-UCB values are post-acquisition diagnostics with `rho=0` for the same intervention. They are not promoted to pre-round certificates.

## C. Cross-round drift and pre-round validity

Observed consecutive-round maximum absolute drift is `0.084473` and maximum signed increase is `0.074707`. Both are retrospective diagnostics from only two dependent trajectories; neither is a valid `rho` calibration.

| Seed | Decision round | Prior pilot round | Gain-UCB at rho=0 | Max rho compatible with STOP | Classification |
|---:|---:|---:|---:|---:|---|
| 42 | 2 | — | — | — | uncertified_continue |
| 42 | 3 | 2 | 0.313323 | -0.268043 | uncertified_continue |
| 42 | 4 | 3 | 0.203125 | -0.157845 | uncertified_continue |
| 42 | 5 | — | — | — | uncertified_continue |
| 42 | 6 | 5 | 0.270354 | -0.225074 | uncertified_continue |
| 42 | 7 | — | — | — | uncertified_continue |
| 42 | 8 | 7 | 0.333831 | -0.288551 | uncertified_continue |
| 43 | 2 | — | — | — | uncertified_continue |
| 43 | 3 | 2 | 0.312346 | -0.267066 | uncertified_continue |
| 43 | 4 | 3 | 0.227874 | -0.182594 | uncertified_continue |
| 43 | 5 | 4 | 0.302581 | -0.257301 | uncertified_continue |
| 43 | 6 | 5 | 0.218750 | -0.173470 | uncertified_continue |
| 43 | 7 | — | — | — | uncertified_continue |
| 43 | 8 | 7 | 0.234375 | -0.189095 | uncertified_continue |

A negative `max rho compatible with STOP` means even the unjustified optimistic choice `rho=0` would not stop; a non-negative value would still require a proved drift bound no larger than that value.

## Decision and savings

| Seed | Earliest certified STOP | Terminal outcome | Certified budget saved | Certified teacher/data cost saved | Certified training cost saved | R8 regret |
|---:|---:|---|---:|---:|---:|---:|
| 42 | — | budget_stop_after_round_8 | 0 | 0.000000 | 0.000000 | 0.000000 |
| 43 | — | budget_stop_after_round_8 | 0 | 0.000000 | 0.000000 | 0.000000 |

Because no certified STOP exists, scientifically attributable savings are zero. Per-round R8 regrets and historical costs in the CSV/JSON are retrospective evaluation fields only.

## Missing evidence and theorem readiness

Confirmed: frozen `lambda/delta/K/T`, complete 32-pilot caches where the evidence gate ran, fixed baseline/shard, gain identity, exact width-one support, family-wise correction, immutable cost rates, and no sealed/future metric in the decision calculation.

Not proved: pilot conditional independence/unbiasedness, a population-level fixed-shard guarantee, a pre-round gain estimator, and a valid cross-round stability radius. Rounds with zero cached pilots were intentionally skipped by historical evidence gates and are not reconstructed.

The implementation and negative feasibility result are sufficient for an experimental/limitations section. They are **not sufficient for a stopping theorem claiming a valid pre-round certified STOP**. Such a theorem needs a predeclared pre-round gain construction and independently justified cross-round stability (or a different theorem that avoids transferring post-acquisition pilots).

## Provenance

- Historical source: `outputs/stage2_runs_v7_minimal_cost_aware_m32_decision_cost_v2`
- Historical artifacts were read-only.
- Sealed HANS/future-round metrics were not used for decisions.
- Accuracy metrics are retrospective diagnostics only.
