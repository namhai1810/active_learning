# V7.1 Pre-Round Risk-Ceiling Stopping v5 — Archived

> Historical offline diagnostic only. This risk-ceiling objective is not the
> V7 next-round expected-improvement stopping rule. See
> `archive/v7_cost_aware_experiments/README.md`.

This is a read-only offline experiment. It does not modify V7, checkpoints,
training, cost rates, lambda, or historical artifacts.

## Certificate

Before round `t+1`, the replay computes:

```text
U_pre(t+1) = U_R(theta_t) - 0.2 * C_lower(t+1)
```

It emits `certified_stop` only when `U_pre <= 0` and both evidence streams
pass validation. Otherwise the action is `uncertified_continue`. Finishing R8
without a certificate is `budget_stop`, never a safe STOP.

### Risk evidence gate

`configs/v7_pre_round_v5/risk_evidence.json` must contain records with:

- binomial `errors` and `total` from independent held-out data;
- an explicit dataset role;
- confirmation that the data were not used for training, pilots, early
  stopping, model selection, or sealed final testing;
- confirmation that the model is independent of held-out outcomes;
- the round before which the evidence was available.

The script builds a one-sided Clopper–Pearson UCB and divides `delta=0.05`
over all seven pre-round decisions and `--max-risk-looks`. Ordinary V7
early-stopping validation metrics intentionally fail this gate.

Example schema:

```json
{
  "records": [
    {
      "seed": 42,
      "model_round": 1,
      "errors": 12,
      "total": 900,
      "dataset_role": "independent_heldout",
      "independent_of_training": true,
      "independent_of_pilot_eval": true,
      "not_used_for_early_stopping": true,
      "not_used_for_model_selection": true,
      "not_sealed_test": true,
      "model_independent_of_heldout": true,
      "available_before_round": 2,
      "source": "path/to/read-only/artifact"
    }
  ]
}
```

Do not set these declarations to true unless dataset construction and artifact
provenance support them.

### Cost lower-bound gate

`configs/v7_pre_round_v5/full_round_cost_lower_bounds.json` must provide a
pre-decision lower bound for all of `discovery`, `verification`, `generation`,
and `training`. Accepted bases are deterministic minima, contractual minima,
or a separately proven statistical lower bound. Empirical means, realised
future costs, and mean projections are rejected.

```json
{
  "records": [
    {
      "seed": 42,
      "pre_round": 2,
      "unit": "usd",
      "frozen_pre_decision": true,
      "available_before_round": 2,
      "components": {
        "discovery": {"lower_bound": 0.0, "basis": "deterministic"},
        "verification": {"lower_bound": 0.0, "basis": "deterministic"},
        "generation": {"lower_bound": 0.0, "basis": "deterministic"},
        "training": {"lower_bound": 0.0, "basis": "deterministic"}
      }
    }
  ]
}
```

Zero is valid only when it is a genuine deterministic lower bound. It is not
useful for producing STOP and must not be replaced with an empirical mean.

### Historical diagnostics

`configs/v7_pre_round_v5/historical_trajectory_index.json` contains optional
per-round retrospective fields: `accuracy`, `teacher_calls`, `teacher_cost`,
`total_cost`, and `cumulative_budget`. These values quantify regret and
avoidable resources after replay. They never enter the online equation.

## Outputs

- `outputs/stage2_diagnostics/v7_pre_round_stopping_v5/validation.json`
- `outputs/stage2_diagnostics/v7_pre_round_stopping_v5/replay.json`
- `outputs/stage2_diagnostics/v7_pre_round_stopping_v5/per_round.csv`
- `outputs/stage2_diagnostics/v7_pre_round_stopping_v5/per_seed.csv`
- `reports/v7_minimal_cost_aware/PRE_ROUND_STOPPING_V5.md`

Rows after the first counterfactual STOP are marked diagnostic-only.

## Direct replay from existing seed 42/43 caches

`--bootstrap-cached-artifacts` reads the fixed gain-report checkpoint metrics,
their disjointness manifests, and the frozen per-run tariff from the existing
v2 outputs. It does not write to or modify those sources. The risk metric is
materialized offline but uses only the checkpoint available before the next
round and a predeclared dataset; the report labels this counterfactual status.

The automatic cost lower bound is deliberately weak: only the mandatory LoRA
run tariff is positive. Discovery, verification and generation receive their
universally valid zero lower bound. Actual/projected round costs remain
retrospective and do not enter the certificate.

```bash
CUDA_VISIBLE_DEVICES="" python scripts/replay_v7_pre_round_stopping_v5.py \
  --seeds 42 43 \
  --bootstrap-cached-artifacts
```

## Scientific limitation

The implementation deliberately does not manufacture an independent risk set
or a full-round cost lower bound from historical averages. If current V7
artifacts do not contain those inputs, the expected audit outcome is
`uncertified_continue`. That negative result is the evidence needed before
deciding whether a new theorem or data split is justified.
