# V7 Decision-Aware Adaptive Pilot Allocation v4 — Archived

> Historical offline experiment only. It is not part of the V7 runtime or the
> current next-round expected-net-gain objective. See
> `archive/v7_cost_aware_experiments/README.md`.

This is an independent CPU simulation over cached pilot observations. It does
not alter V7, historical artifacts, cost rates, lambda, delta, or the existing
v2/v2.1 protocols.

## Statistical rule

Pilots are revealed in saved `pilot_index` order at the declared looks:

```text
4 -> 8 -> 16 -> 32
```

For `L` declared looks, equal Bonferroni alpha spending uses:

```text
delta_look = delta / L
b_m = sqrt(log(2*K*T/delta_look) / (2*m))
```

The gain support is `[-A0, 1-A0]`, with width one, where all pilots in a round
must share the same fixed baseline accuracy and 64-example evaluation shard.

At each look:

- `STOP` is certified only when `Gain_UCB - lambda*decision_cost <= 0`;
- `CONTINUE` is certified only when the valid Net-LCB is positive;
- otherwise an allocation heuristic decides whether to buy the next pilot
  block, or returns `uncertified_continue`.

The fixed-32 comparator is a single-look v2.1 bound. The adaptive m=32 bound
is wider because it accounts for all prior looks. Reusing the fixed-m radius
after examining m=4/8/16 would invalidate confidence accounting.

## Allocation heuristic

The heuristic is deliberately separated from the certificate:

1. It checks whether any completion up to the declared cap could produce a
   valid STOP certificate, using only the revealed mean and known support.
2. It bootstraps only revealed pilot gains to estimate the chance of a STOP at
   each remaining declared look.
3. Before any pilot outcome is revealed, it freezes the cost of one pilot from
   the existing run cost config and recipe: one LoRA run over 32 examples, two
   64-example gain evaluations, and at most eight epoch validations over the
   predeclared 900-example stopping subset.
4. For each future look it computes, in explicit units:

   ```text
   expected_benefit_utility = P(STOP at look) * lambda * decision_cost_USD
   information_penalty_utility = additional_pilots * lambda * pilot_cost_USD
   VOI_utility = expected_benefit_utility - information_penalty_utility
   ```

5. It buys only the next scheduled block when the best future-look VOI is
   strictly positive; otherwise it returns `uncertified_continue`.

Bootstrap probability is a sampling heuristic, not a confidence bound. It can
never create STOP. If the frozen pilot GPU rate is unavailable, the simulation
fails closed to `uncertified_continue` and records the missing projection.

The current cost config supplies USD rates, including `0.5 USD/GPU-hour` for
both LoRA training and evaluation. The simulator exposes the corresponding
`frozen_pilot_gpu_second_cost`, but GPU cost is only one component: per-run and
per-item charges remain in `projected_pilot_cost_USD`. USD is converted to
normalized gain utility exactly once through the frozen `lambda=0.2 gain/USD`.

With the currently frozen recipe, validation should reproduce the following
projection without reading any pilot observation:

```text
training GPU projection       = 32 * 0.2 s       = 6.40 s
evaluation GPU projection     = (2*64 + 8*900) * 0.02 s = 146.56 s
frozen GPU rate               = 0.5 / 3600       = 0.000138889 USD/s
projected training cost       = 0.020888889 USD
projected evaluation cost     = 0.803155556 USD
projected total cost / pilot  = 0.824044444 USD
utility penalty / pilot       = lambda * cost    = 0.164808889 gain
```

The per-run and per-item charges make total pilot cost different from merely
multiplying GPU seconds by the GPU rate. This distinction is retained in JSON,
CSV, and Markdown outputs.

Hidden or revealed pilot timings are used only after a simulated decision to
calculate realized GPU-hour savings. They never affect VOI allocation.

## Comparators

- fixed 32 pilots with the original single-look Range-Aware bound;
- adaptive `4 -> 8 -> 16 -> 32`;
- zero-pilot `uncertified_continue` baseline.

Evidence-gated rounds skip pilot simulation. Any round after a counterfactual
certified STOP is marked diagnostic-only.

## Commands

All commands are CPU-only.

### 1. Unit tests and validation

```bash
conda activate active_lr
cd /home/llm/hainn/hainn/active_learning

CUDA_VISIBLE_DEVICES="" python -m pytest \
  tests/test_v7_adaptive_pilot_allocation.py -q

CUDA_VISIBLE_DEVICES="" python scripts/simulate_v7_adaptive_pilot_allocation.py \
  --validate-only \
  --seeds 42 43
```

Inspect validation:

```bash
python -m json.tool \
  outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/validation.json \
  | less
```

Validation must report a frozen USD/GPU-second rate, the complete projected
USD cost per pilot, and currency `USD`. It also checks that the analysis recipe
matches the resolved run config. The code never infers a replacement rate or
runtime from cached/future pilot observations.

### 2. Primary CPU simulation

```bash
CUDA_VISIBLE_DEVICES="" python scripts/simulate_v7_adaptive_pilot_allocation.py \
  --seeds 42 43
```

### 3. Pilot-cap sensitivity

```bash
CUDA_VISIBLE_DEVICES="" python scripts/simulate_v7_adaptive_pilot_allocation.py \
  --seeds 42 43 \
  --sensitivity-only \
  --pilot-caps 8 16 32
```

Each cap has its own predeclared number of looks and Bonferroni allocation; a
fixed-32 radius is never reused for a multi-look schedule.

### 4. Inspect decisions and savings

```bash
sed -n '1,320p' \
  reports/v7_minimal_cost_aware/ADAPTIVE_PILOT_ALLOCATION_V4.md

python -m json.tool \
  outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/simulation.json \
  | less
```

Outputs:

- `outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/simulation.json`
- `outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/simulation.csv`
- `outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/sensitivity.json`
- `outputs/stage2_diagnostics/v7_adaptive_pilot_allocation_v4/sensitivity.csv`
- per-seed JSON/CSV under `seed_42/` and `seed_43/`;
- `reports/v7_minimal_cost_aware/ADAPTIVE_PILOT_ALLOCATION_V4.md`.

## Limitations

- Pilots share a fixed 64-example shard and baseline; the simulation does not
  claim an iid population-level guarantee.
- Conditional independence of pilot training randomization remains an
  assumption.
- Bootstrap VOI is heuristic and not covered by the STOP certificate.
- Cached pilot order is treated as the predeclared reveal order.
- GPU-hour savings are realized historical savings, not a guarantee for a new
  online run.
- The frozen runtime projection is a cost-model assumption, not a measured
  latency forecast. Sensitivity changes pilot caps only; it does not tune the
  rate, lambda, or projection from outcomes.
- Later historical rounds after a simulated STOP are diagnostic only.

