# Failure-driven active learning — primary LLM method and oracle references

This repository implements MNLI/HANS loading, a Qwen3-1.7B NLI student, HANS
evaluation, and the Base, Random, and entropy-based Uncertainty acquisition
baselines. Random and Uncertainty share one LoRA training recipe. All baseline
runs use a unified local logger with resolved configuration and round metadata.

The primary research path is now distinct from the HANS oracle/reference path:

```text
Qwen HANS predictions -> deterministic five-way split
-> teacher-generated failure candidates -> independent verification + LCB
-> new teacher-generated three-class NLI data -> quality/leakage filtering
-> disjoint synthetic pilots -> gain UCB and cost-aware STOP/CONTINUE
-> one fresh full LoRA intervention (CONTINUE only) -> final held-out HANS
```

`configs/main_common.yaml` deliberately leaves `teacher.name_or_path: null`.
Choose an appropriate local path or Hugging Face model ID before running the
main method. A missing/failed teacher is an error; the code never silently
substitutes the HANS oracle.

## Setup

```bash
conda activate active_lr
python -m pip install -r requirements.txt
```

## Evaluate HANS

```bash
python scripts/evaluate_hans.py --config configs/hans_qwen17b.yaml
```

The run writes `predictions.jsonl` and `metrics.json` beneath the configured
`outputs/<run_name>/` directory. Set `overwrite: true` in the config only when
an existing run should be replaced.

## Test

```bash
python -m pytest -q
```

## Run acquisition baselines

```bash
python scripts/run_baseline.py --method base --config configs/baseline_base.yaml
python scripts/run_baseline.py --method random --config configs/baseline_random.yaml
python scripts/run_baseline.py --method uncertainty --config configs/baseline_uncertainty.yaml
```

Each run saves `config.yaml`, `run.json`, `rounds.jsonl`, `metrics.json`,
`predictions.jsonl`, and `selected_example_ids.json`. Random and Uncertainty
also save their LoRA adapter. Existing non-empty runs are rejected unless the
command includes the explicit `--overwrite` flag.

To compare completed local runs:

```bash
python scripts/summarize_runs.py --outputs outputs
```

For the full CUDA 0 benchmark suite (zero-shot plus Base, Random, and
Uncertainty), activate `active_lr` and run the non-overwriting orchestrator:

```bash
bash run_hans_baselines_cuda0.sh
```

It saves per-stage logs under `logs/hans_baselines_seed42/` and builds a
paper-ready Markdown report, tidy CSV/JSON exports, and SVG vector figures under
`reports/hans_baselines_seed42/`. Completed stages are safely skipped on rerun;
partial run directories cause an explicit failure and are never overwritten.

## Run the primary LLM-based method

Start from an existing HANS evaluation directory containing the unsplit
`predictions.jsonl`. All primary scripts reconstruct and validate the same
12.5/12.5/12.5/12.5/50 split. The compatible main manifest is
`hans_split_v2.json`.

After setting the same explicit teacher configuration in each inherited main
config, run the stages in order:

```bash
python scripts/build_hypotheses.py \
  --config configs/main_hypothesis_discovery.yaml \
  --run-dir outputs/<existing_run>

python scripts/verify_hypotheses.py \
  --config configs/main_verification.yaml \
  --run-dir outputs/<existing_run>

python scripts/generate_synthetic_interventions.py \
  --config configs/main_synthetic_generation.yaml \
  --run-dir outputs/<existing_run>

python scripts/run_main_pilot_gain.py \
  --config configs/main_pilot_gain.yaml \
  --run-dir outputs/<existing_run>

python scripts/run_main_decision.py \
  --config configs/main_cost_aware.yaml \
  --run-dir outputs/<existing_run>
```

An explicit teacher-critic first estimates
`P(student prediction is wrong | premise, hypothesis, prediction, confidence)`.
The critic never receives gold labels, correctness, HANS taxonomy fields, or
non-discovery records. Discovery then uses configured normalization,
logistic/exponential weighting, or threshold sampling; challenging correct
false alarms are the default controls. The hypothesis prompt still receives
only approved fields and never receives HANS `heuristic` or `subcase`.

Verification records are context only. The teacher generates new fixed sets of
one target, one counterfactual/positive control, and one invariant control. A
separate validation call solves their labels and checks their constraints before
the student is loaded. The student is queried once on the frozen accepted set.
Each binary outcome is one only when the target exhibits the predeclared failure
and both controls are correct. Certification additionally requires enough valid
sets and a union-adjusted Hoeffding LCB; its iid interpretation remains
conditional because generated scenarios cannot prove statistical independence.

Synthetic generation uses only `intervention_pool` seeds and emits new complete
counterfactual and invariant pairs (plus optional reasoning contrasts), with
stable `pair_id`, `pair_type`, and member roles. Every member has an explicit
`entailment`, `neutral`, or `contradiction` label; no binary HANS non-entailment
label is converted automatically. A second teacher QC pass and deterministic
checks reject entire pairs for inconsistent labels/semantics, ambiguity, poor
grammar, irrelevance, weak diversity, duplicates, near duplicates, or overlap
with any permitted pre-final split or generated verification probe. `final_test`
is not materialized at this stage. Other held-out/probe text is used only for a
post-generation local leakage veto and never enters a teacher generation prompt.

Pilot synthetic pairs are disjoint across pilots and are never split across
pilot/full selections. Training remains the existing per-example MNLI
cross-entropy; no unsupported contrastive loss is implied. All pilot models begin
from the same checkpoint, and pilot examples are excluded from the full-data
reserve. `run_main_decision.py` freezes
`U(h) = delta_hat(h) + radius(h) - lambda_cost * C(h)` before any full
intervention. STOP performs no training. CONTINUE runs exactly one fresh full
LoRA, reports gain on the preselected `gain_eval` subset, and only then
materializes and evaluates `final_test` with the same HANS evaluator.

Teacher request/response envelopes, prompt versions, deterministic seeds,
selected IDs, QC rejections, gain measurements, and decision artifacts are
saved beneath the run directory. The concentration bounds remain conditional
on their stated boundedness/independence assumptions and are not causal claims.

## Build debug-only HANS oracle hypotheses

The remaining `oracle_hans_*` configs and oracle scripts are debug/reference
experiments, not the primary method. This command reuses an existing run's predictions and deterministically splits
their stable IDs into hypothesis-discovery, verification, intervention-pool,
gain-evaluation, and final-test roles. Only discovery records and metadata are
passed to the oracle generator.

```bash
python scripts/build_oracle_hypotheses.py \
  --config configs/oracle_hans_hypotheses.yaml \
  --run-dir outputs/<existing_run>
```

It adds `hypotheses.json`, `hypothesis_evidence.json`, and
`hans_oracle_split_v2.json` without modifying the older split manifest or any
Milestone 3 artifacts. These are
debug-only candidate hypotheses, not verified or certified failures.

## Verify debug-only HANS oracle hypotheses

```bash
python scripts/verify_oracle_hypotheses.py \
  --config configs/oracle_hans_verification.yaml \
  --run-dir outputs/<existing_run>
```

The verification script reconstructs and validates the versioned five-way split, then
selects fixed non-entailment challenge probes from the verification role before
examining predictions. It writes `verification/probes.jsonl` and
`verification/results.json`. Certification means only that the observed
failure behavior passes the configured reproducibility threshold; it is not a
causal claim and says nothing about intervention value.

## Run targeted pilot gain estimates (oracle debug only)

Milestone 6 assigns HANS examples to five pairwise-disjoint roles:
`hypothesis_discovery`, `verification`, `intervention_pool`, `gain_eval`, and
`final_test`. The manifest is saved as `hans_oracle_split_v2.json`; its
`final_test` IDs are preserved from the earlier 50/50 boundary.

Regenerate the oracle and verification artifacts with their existing commands,
then run only certified hypotheses:

```bash
python scripts/run_pilot_gain.py \
  --config configs/oracle_hans_pilot_gain.yaml \
  --run-dir outputs/<verified_run>
```

Each pilot reloads exactly the same starting Qwen checkpoint, trains a fresh
LoRA intervention on matching `intervention_pool` records, and evaluates risk
before and after on matching `gain_eval` records. It writes
`gain/pilot_observations.jsonl`, `gain/gain_estimates.json`, and isolated pilot
adapters under `gain/pilots/`. The estimate is `mean(risk_before-risk_after)`;
negative values are valid and mean the intervention hurt performance.

The confidence radius assumes independent bounded pilot gains. Fixed HANS data,
related LoRA runs, and unavoidable gain-eval overlap can violate that assumption,
so these oracle-debug intervals are not causal guarantees. Milestone 6 does not
use `final_test` and does not implement UCB or cost-aware stopping.

## Make one cost-aware decision (oracle debug only)

Milestone 7 joins only certified hypotheses to valid pilot-gain estimates. Its
explicit cost is the actual number of matching `intervention_pool` examples
that would be acquired, up to `full_intervention_size`. For each eligible
hypothesis it computes:

```text
J_hat(h) = Delta_hat(h) - lambda_cost * C(h)
U(h)     = Delta_hat(h) + b(h) - lambda_cost * C(h)
```

The command selects `argmax U(h)` and continues only when that maximum is
strictly positive:

```bash
python scripts/run_cost_aware_decision.py \
  --config configs/oracle_hans_cost_aware.yaml \
  --run-dir outputs/<milestone_6_run>
```

It writes `decision/candidates.json` and `decision/decision.json`. A STOP result
does not load the model, acquire examples, or train. A CONTINUE result performs
exactly one fresh full intervention from the same current checkpoint used for
the pilots, saves its isolated adapter under
`decision/full_intervention/<hypothesis_id>/adapter/`, and reports post-decision
risk on `gain_eval` in `decision/post_intervention_metrics.json`. Pilot adapters
are never promoted, and `final_test` is not evaluated.

`lambda_cost` is fixed in the config before selection and must not be tuned on
`final_test` or intervention outcomes. A positive optimistic value does not
guarantee profitability: CONTINUE means only that positive net value cannot yet
be ruled out. Any stopping interpretation also depends on the verification and
gain concentration assumptions, bounded pilot gains, and a correctly specified
cost; this implementation does not itself prove those assumptions or a theorem.

## Run controlled ablations (oracle debug only)

Milestone 8 exposes seven one-round variants through one shared strategy layer
and one runner. All variants read the same checkpoint, five-way split,
hypotheses, verification records, pilot-gain artifacts, intervention budget,
seeds, LoRA recipe, and evaluator. Only the named component changes:

- `full`: LCB verification, gain UCB, acquisition cost, and stopping.
- `no_verification`: accepts every candidate hypothesis.
- `point_verification`: accepts when `nu_hat >= tau_v`.
- `no_gain_ucb`: ranks/stops with `Delta_hat - lambda_cost*C`.
- `no_cost`: ranks/stops with `Delta_hat + b`.
- `random_hypothesis`: seeded uniform selection among accepted hypotheses with
  valid gain estimates; the Full stopping test remains unchanged.
- `fixed_budget`: selects the Full argmax but does not stop merely because its
  optimistic value is non-positive.

Run one variant at a time—the command never launches the matrix automatically:

```bash
python scripts/run_ablation.py \
  --variant full \
  --config configs/oracle_hans_ablation.yaml \
  --run-dir outputs/<milestone_6_run>
```

Replace `full` with `no_verification`, `point_verification`, `no_gain_ucb`,
`no_cost`, `random_hypothesis`, or `fixed_budget` for the other controlled
runs. Each variant writes separately beneath `ablations/<variant>/`; after each
run, `ablations/summary.json` is rebuilt and a compact comparison table is
printed. Existing variant outputs require an explicit `--overwrite` to replace.

The comparison uses `gain_eval` only. `final_test` prediction fields are not
materialized into the ablation strategy or intervention, and no variant is
selected or tuned from final-test performance. Hypotheses without an existing
valid pilot-gain estimate remain in the accepted count but cannot enter the
decision set; obtain comparable pilot estimates before interpreting verification
ablations on a real experiment.

## Validate theoretical assumptions offline (oracle debug only)

Milestone 9 is evaluation-only. It freezes an already-saved Milestone 7
`decision.json` before reading any offline oracle data, then evaluates:

1. verification certification behavior across probe counts;
2. empirical gain-UCB coverage across pilot counts; and
3. when the online action was STOP, the best offline estimated one-step net
   value among robustly valid hypotheses in the existing candidate set.

The full configuration intentionally requires a separate HANS-style JSONL file:

```yaml
reference_data:
  file: /path/to/disjoint_reference.jsonl
  use_precomputed_predictions: false
```

Every row must have a globally disjoint ID and one of two roles:
`verification_reference` or `gain_reference`. The former supplies the large
offline validity reference; the latter is used only to evaluate independent
reference interventions. IDs overlapping any online role are rejected.

After providing that file, the expensive full command is:

```bash
python scripts/run_theory_validation.py \
  --config configs/oracle_hans_theory_validation.yaml \
  --run-dir outputs/<completed_online_run>
```

This full configuration may perform up to 16 online-analysis pilots plus 32
offline reference repetitions per relevant hypothesis, each starting from the
same original checkpoint. It is intentionally not launched automatically.

The small precomputed smoke command is:

```bash
python scripts/run_theory_validation.py \
  --config configs/oracle_hans_theory_validation_smoke.yaml \
  --run-dir outputs/oracle_verification_synthetic_smoke
```

Outputs live under `theory_validation/{verification,gain_ucb,stopping}/` with a
combined `theory_validation/summary.json`. `final_test` is never materialized
into these analyses. The verification reference is an empirical label, the gain
reference approximates rather than equals true gain, and observed coverage or a
passing stop audit supports assumptions only—it does not prove a theorem or
rule out unseen hypotheses, multi-step repairs, or hypothesis interactions.
