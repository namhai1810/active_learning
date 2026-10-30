# V7 Cost-Aware Safe Stopping

This protocol is an isolated extension of the preserved V7 closed loop. It
writes to `outputs/stage2_runs_v7_cost_aware_safe_stopping/` and never changes
the historical 4x32 ON/OFF artifacts.

## Round protocol

    Discover -> Verify -> Generate + synthetic QC
             -> independent pilot gains -> gain UCB - lambda * C
             -> STOP, or targeted V7 Select -> stateful Train -> Refresh

Only hypotheses certified by the verification LCB are decision candidates.
Unresolved hypotheses remain in the lifecycle archive and may be regenerated
or reverified, but they cannot produce a positive-UCB CONTINUE decision.

For each certified candidate the frozen decision is

    U_t(h) = delta_hat_t(h) + b_t(h) - lambda * C_t(h).

The controller continues with `argmax_h U_t(h)` only when the maximum is
strictly positive. Otherwise it stops. Gain radii use the configured family
wise `K=5`, `T=8`, and `delta=.05`; gain-eval IDs are deterministically split
into disjoint round shards. The main intervention is always 32 examples, with
at most eight rounds and 256 examples.

The existing V7 selector, stateful LoRA tensors, named AdamW state, epoch-level
validation-NLL stopping, patience/min-delta policy, and best-checkpoint restore
remain in force. Pilot adapters never become the main checkpoint, and pilot
training IDs are excluded from the 32-example main acquisition.

Each pilot clones the current LoRA checkpoint and the current named AdamW state,
then applies the same epoch-level validation-NLL early-stopping policy. Its
bounded gain observation is the change in direct-binary HANS classification
risk on a fresh disjoint gain-eval shard. Pilot adapter/optimizer outputs are
diagnostic only and are never committed to the main state chain.

## Cost accounting

`C_t(h)` is built by the existing full-pipeline cost model and is frozen before
the decision. Candidate artifacts retain two phases:

- `incurred_exploration`: measured teacher tokens, generated/verified items,
  pilot runs, and measured pilot GPU time allocated to the candidate;
- `projected_intervention`: configured estimates for the not-yet-run 32-example
  training/evaluation work.

The cumulative realized ledger does not add the candidate total again. It adds
round exploration once and then the realized intervention once. Projection
rates and `lambda_cost` are experimental assumptions and must be calibrated or
ablated before economic claims are made.

The inherited cost schema currently has no human-review/annotation price field,
so the configured `C_t(h)` covers automated LLM, generation/QC, LoRA, and
evaluation resources only. A claim about total real-world monetary cost needs a
separately justified extension of that schema.

## Theorem-assumption boundary

The appendix requires a positive verification margin `gamma` and
`a_t(h) <= gamma/2`. The old 30-probe V7 configuration does not satisfy this
condition for a meaningful margin. The new protocol therefore predeclares one
120-probe look per round with `gamma=.4`, and aborts before a decision if its
mechanically checkable radius/accounting conditions fail.

The code cannot prove that teacher-generated verification probes are
conditionally independent, that the unknown true verification rates obey the
margin separation, or that related LoRA pilot gains are conditionally
independent and unbiased. Every decision writes
`decision/theorem_assumption_audit.json`; even when its mechanical checks pass,
the theorem status is explicitly conditional on these unverified
distributional assumptions. This is a limitation of the current empirical
methodology, not a software error.

The Hoeffding gain radius with `M=1` and only eight pilots is also wide. The
implementation does not clip or replace it with a heuristic threshold, so the
configured procedure may continue to the maximum budget.

## Data firewall and outputs

Neither sealed HANS nor MNLI evaluation is available to verification, pilot
gain estimation, STOP/CONTINUE, acquisition, or training early stopping. They
are evaluated after the adaptive loop is frozen. Final HANS probabilities are
used once to report both Three-way-to-binary and Direct Binary rules.

Important artifacts per seed:

- `refresh/round_NNN/gain/gain_estimates.json` and pilot observations;
- `refresh/round_NNN/decision/{candidates,decision,round_cost}.json`;
- `refresh/round_NNN/decision/theorem_assumption_audit.json`;
- `rounds.jsonl` with decision, costs, optimizer/checkpoint continuity and
  epoch history;
- `cost_aware_stop.json` when STOP occurs before a main update;
- `result.json`, `V7_COST_AWARE_STOPPING.md`, and `.complete`.

Resume reuses only fully materialized refresh, gain, and frozen-decision stage
boundaries. A partially written pilot stage is rejected rather than silently
rerun and double-counted.

## CPU validation

    python scripts/dry_run_v7_cost_aware_stopping.py
    python -m pytest -q tests/test_v7_cost_aware_stopping.py \
      tests/test_cost_aware.py tests/test_multiround.py \
      tests/test_stage2_benchmark.py

## Experiment commands

See `run_v7_cost_aware_safe_stopping.sh`. The launcher requires separate
teacher/student physical GPUs, refuses completed outputs, and supports
`--resume` for validated incomplete runs.
