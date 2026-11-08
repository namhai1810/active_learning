# V7.1 Pre-Round Cost-Aware Stopping v5 — Offline Replay

> Offline diagnostic only. Historical artifacts are read-only; retrospective accuracy and realised costs never enter the online decision.

## Protocol

- Decision: `STOP` iff `U_R(theta_t) - 0.2 * C_lower(t+1) <= 0`.
- Family confidence budget: delta=0.05, planned risk looks per decision=1.
- `U_R` is a one-sided Clopper–Pearson risk bound after provenance validation.
- Full-round lower bound must cover discovery, verification, generation and training.
- Empirical/projected mean cost is rejected as a lower confidence bound.

## Seed summary

| Seed | Terminal status | Certified before round | Reason | Calls avoidable* | Budget avoidable* | Accuracy regret vs R8* |
|---:|---|---:|---|---:|---:|---:|
| 42 | budget_stop | N/A | maximum_budget_reached_without_certificate | 0 | 0 | 0.000000 |
| 43 | budget_stop | N/A | maximum_budget_reached_without_certificate | 0 | 0 | 0.000000 |

*Retrospective diagnostics only; these values are not used by the certificate.*

## Per-round replay

| Seed | Before R | Risk UCB | Cost LB | Required LB for STOP | LB gap | Actual cost* | Net UCB | Decision | Reason |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 42 | 2 | 0.382030 | 0.020000 | 1.910148 | 1.890148 | 12.012469 | 0.378030 | uncertified_continue | no_valid_stop_certificate |
| 42 | 3 | 0.377811 | 0.020000 | 1.889056 | 1.869056 | 28.392839 | 0.373811 | uncertified_continue | no_valid_stop_certificate |
| 42 | 4 | 0.340603 | 0.020000 | 1.703014 | 1.683014 | 1.520650 | 0.336603 | uncertified_continue | no_valid_stop_certificate |
| 42 | 5 | 0.324485 | 0.020000 | 1.622425 | 1.602425 | 18.272579 | 0.320485 | uncertified_continue | no_valid_stop_certificate |
| 42 | 6 | 0.359650 | 0.020000 | 1.798249 | 1.778249 | 1.155116 | 0.355650 | uncertified_continue | no_valid_stop_certificate |
| 42 | 7 | 0.322362 | 0.020000 | 1.611809 | 1.591809 | 17.316595 | 0.318362 | uncertified_continue | no_valid_stop_certificate |
| 42 | 8 | 0.276785 | 0.020000 | 1.383927 | 1.363927 | 12.293461 | 0.272785 | uncertified_continue | no_valid_stop_certificate |
| 43 | 2 | 0.381186 | 0.020000 | 1.905930 | 1.885930 | 12.095000 | 0.377186 | uncertified_continue | no_valid_stop_certificate |
| 43 | 3 | 0.378233 | 0.020000 | 1.891166 | 1.871166 | 28.478625 | 0.374233 | uncertified_continue | no_valid_stop_certificate |
| 43 | 4 | 0.355421 | 0.020000 | 1.777104 | 1.757104 | 28.844350 | 0.351421 | uncertified_continue | no_valid_stop_certificate |
| 43 | 5 | 0.298969 | 0.020000 | 1.494846 | 1.474846 | 12.157219 | 0.294969 | uncertified_continue | no_valid_stop_certificate |
| 43 | 6 | 0.313013 | 0.020000 | 1.565066 | 1.545066 | 1.471042 | 0.309013 | uncertified_continue | no_valid_stop_certificate |
| 43 | 7 | 0.343992 | 0.020000 | 1.719960 | 1.699960 | 16.786969 | 0.339992 | uncertified_continue | no_valid_stop_certificate |
| 43 | 8 | 0.277213 | 0.020000 | 1.386063 | 1.366063 | 12.203351 | 0.273213 | uncertified_continue | no_valid_stop_certificate |

## Validity interpretation

- Missing or model-selected validation data yields `uncertified_continue`, never STOP.
- Missing/unproven full-round cost lower bounds yield `uncertified_continue`.
- Reaching R8 is reported as `budget_stop`, not as a statistically certified STOP.
- Rows after the first counterfactual STOP are diagnostic-only and are not an online trajectory.
- Reusing a held-out set is accepted only when the manifest attests that training and model selection are independent of it and confidence is allocated across all planned looks.
- The R8 comparison is an offline oracle-style diagnostic and provides no stopping guarantee.

## Minimal V7.1 conclusion rule

Add no theorem or runtime change unless the audit supplies both valid evidence streams. If either stream fails its provenance gate, the scientifically correct result is `uncertified_continue`.
