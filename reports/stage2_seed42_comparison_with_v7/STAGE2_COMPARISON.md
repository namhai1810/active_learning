# Stage 2 seed-42 comparison including verified v2--v7

This report consolidates all completed Stage 2 results found for `seed=42` and
`budget=128`. Duplicate artifacts are counted once. Unless explicitly marked as
diagnostic or abstained, every trained row uses 128 training examples.

The table reports accuracy as a fraction in `[0,1]`. The corresponding percentage
is obtained by multiplying by 100.

## Complete result table

| Method | Objective | HANS accuracy | HANS non-entailment | MNLI matched | MNLI mismatched | Status |
|---|---|---:|---:|---:|---:|---|
| `zero_shot` | none | 0.638533 | 0.337518 | 0.576566 | 0.579434 | reference; no training |
| `random_synthetic` | CE | 0.644600 | 0.351676 | 0.583902 | 0.585130 | complete synthetic baseline |
| `random_mnli` | CE | 0.652600 | 0.376519 | 0.594906 | 0.597742 | complete cross-source baseline |
| `uncertainty_mnli` | CE | **0.669933** | **0.423000** | **0.610392** | **0.615439** | complete cross-source baseline; best overall |
| `failure_driven_no_verification` | CE | 0.654667 | 0.383732 | 0.593377 | 0.596318 | complete verification ablation |
| `ours_verified_strict_initial` | abstained | 0.638533 | 0.337518 | 0.576566 | 0.579434 | no hypothesis certified; zero-shot fallback; not rankable |
| `ours_verified_target_only_early` | pairwise | 0.647867 | 0.365166 | 0.588691 | 0.591436 | complete |
| `ours_verified_v2_ce` | CE | 0.643800 | 0.348204 | 0.582578 | 0.584317 | complete controlled objective ablation |
| `ours_verified_v2_pairwise` | pairwise | 0.643400 | 0.347669 | 0.582476 | 0.584418 | complete controlled objective ablation |
| `ours_verified_v3_ce` | CE | 0.647533 | 0.361026 | 0.587672 | 0.589402 | complete |
| `ours_verified_v5_ce` | CE | 0.648200 | 0.364766 | 0.588079 | 0.589402 | complete |
| `ours_verified_v6_closed_loop` | CE | 0.638533 | 0.337518 | 0.576566 | 0.579434 | preserved diagnostic; checkpoint continuity failed; not rankable |
| `ours_verified_v7_continuation` | CE | **0.650000** | **0.363296** | **0.586347** | **0.589605** | complete stateful closed-loop continuation; best valid verified result |
| `ours_verified_v7_consolidation` | CE | 0.647200 | 0.355817 | 0.584412 | 0.586554 | complete fresh one-pass consolidation; primary acquisition comparison |

No completed `result.json` was found for a distinct v4 run at `seed=42` and
`budget=128`. Therefore v4 is not reported as an experimental result.

## Ranking by HANS accuracy

After excluding the abstained strict run and the invalid v6 diagnostic, the
ranking is:

1. `uncertainty_mnli`: 66.9933%
2. `failure_driven_no_verification`: 65.4667%
3. `random_mnli`: 65.2600%
4. `ours_verified_v7_continuation`: 65.0000%
5. `ours_verified_v5_ce`: 64.8200%
6. `ours_verified_target_only_early`: 64.7867%
7. `ours_verified_v3_ce`: 64.7533%
8. `ours_verified_v7_consolidation`: 64.7200%
9. `random_synthetic`: 64.4600%
10. `ours_verified_v2_ce`: 64.3800%
11. `ours_verified_v2_pairwise`: 64.3400%
12. `zero_shot`: 63.8533%

## Fairness and checkpoint interpretation

- `ours_verified_v7_continuation` carries LoRA weights and AdamW moments through
  four 32-example rounds, for four optimizer steps in total. It is a valid result
  for the stateful online closed-loop algorithm.
- `ours_verified_v7_consolidation` starts from the original base checkpoint and
  trains once on the frozen set of 128 examples selected by the same v7 closed
  loop, also for four optimizer steps. This is the primary apples-to-apples
  acquisition comparison against the older one-pass baselines.
- The continuation checkpoint controls discovery and verification during the
  shared closed loop; consolidation is trained only after acquisition is frozen.
- `ours_verified_v6_closed_loop` remains in the table to preserve the experimental
  record, but checkpoint continuity failed. Its evaluation equals zero-shot and
  it must not be used as a valid fixed-budget result.
- `ours_verified_strict_initial` abstained and performed no training. Its reported
  metrics are the zero-shot fallback, so it must also be excluded from rankings.
- `random_mnli` and `uncertainty_mnli` train on original MNLI examples. They are
  cross-source baselines and do not have the same data access as the HANS-guided
  LLM-synthetic methods.
- Both v7 branches use exactly the same acquisition trace and the same 128 selected
  synthetic examples. Their difference isolates the training schedule rather than
  the selected data.

## Main empirical finding

The strongest valid verified result is `ours_verified_v7_continuation` at
65.0000% HANS accuracy. Relative to the principal reference points:

| Comparison | HANS difference |
|---|---:|
| v7 continuation minus zero-shot | +1.1467 percentage points |
| v7 continuation minus random synthetic | +0.5400 percentage points |
| v7 continuation minus v5 CE | +0.1800 percentage points |
| v7 continuation minus failure-driven without verification | -0.4667 percentage points |
| v7 continuation minus random MNLI | -0.2600 percentage points |
| v7 continuation minus uncertainty MNLI | **-1.9933 percentage points** |

Thus, the approximately two-point gap discussed in the analysis is exact up to
rounding: `66.9933 - 65.0000 = 1.9933` percentage points.

The gap is considerably larger on HANS non-entailment:

```text
uncertainty_mnli                42.3000%
ours_verified_v7_continuation  36.3296%
difference                       5.9704 percentage points
```

This pattern suggests that the verified synthetic set does not yet cover the
important non-entailment phenomena as effectively as uncertainty sampling from
the original MNLI pool.

## Is LLM-generated data the cause of the gap?

The present results support this hypothesis, but they do not prove that synthetic
generation is the only cause.

### Evidence supporting a source-quality limitation

With the same random acquisition principle, original MNLI is stronger than
LLM-synthetic data:

```text
random_mnli       65.2600%
random_synthetic  64.4600%
difference         0.8000 percentage points
```

This is consistent with original MNLI examples having several advantages:

- human-written language is generally more natural and structurally varied;
- the labels are less exposed to teacher-generation and QC errors;
- the examples contain fewer repeated templates and lexical artifacts;
- the source pool covers a broader range of linguistic phenomena;
- examples can be difficult for the student without being malformed or
  artificially difficult.

LLM-generated examples can pass schema and label QC while still being too easy,
too repetitive, weakly representative of the evaluation distribution, or
unhelpful for changing the student's decision boundary. QC establishes validity;
it does not guarantee training utility.

### Evidence that generation quality is not the only limitation

`failure_driven_no_verification` obtains 65.4667%, which is 0.4667 percentage
points above v7 continuation even though both methods belong to the
failure-directed synthetic setting. This indicates that verification and
selection can also reduce useful coverage by:

- discarding a useful hypothesis because its finite-sample confidence interval is
  unresolved;
- over-concentrating the selected examples on a small subset of failure modes;
- preferring high-NLL or high-entropy examples that are redundant;
- optimizing local hypothesis-specific difficulty instead of global HANS
  coverage.

Moreover, `uncertainty_mnli` combines two advantages at the same time:

1. access to the original human-written MNLI pool; and
2. direct selection of examples on which the current student has high predictive
   uncertainty.

Consequently, the 1.9933-point difference between uncertainty MNLI and v7 cannot
be attributed exclusively to LLM generation. It jointly reflects data source,
label and language quality, diversity, acquisition policy, verification filtering,
and their interaction with training.

The differences above are diagnostic contrasts, not an additive causal
decomposition. For example, the 0.8-point random-source gap must not simply be
subtracted from the 1.9933-point total and interpreted as an exact contribution.

## Experiment required to identify the bottleneck

The cleanest next experiment is a controlled two-arm comparison in which all
components except the source of training examples are fixed:

- **Synthetic arm:** use the verified hypotheses and current selector on the
  LLM-generated candidate pool.
- **MNLI-retrieval arm:** use the same verified hypotheses to retrieve relevant
  examples from the original MNLI pool, then apply the same budget, student
  scoring, diversity rule, objective, number of optimizer steps, and evaluation.

Both arms should use `seed=42`, budget 128, identical base checkpoint, identical
training schedule, and the same sealed evaluation manifests.

Interpretation:

- If the MNLI-retrieval arm approaches `uncertainty_mnli`, synthetic generation is
  the principal bottleneck.
- If both arms remain similarly below `uncertainty_mnli`, the principal bottleneck
  is more likely the hypothesis-verification-selection policy.
- If retrieval improves HANS non-entailment but not overall HANS, the main issue is
  coverage of non-entailment phenomena rather than general data quality.

Additional diagnostics should report label agreement, unique semantic/template
clusters, nearest-neighbor duplication, difficulty distributions, and per-HANS-
heuristic accuracy. These measurements can distinguish noisy generation from
insufficient coverage without changing the primary evaluation protocol.

## Scope of the conclusion

All values in this report come from the single development seed `42`. They are
sufficient for debugging and choosing the next ablation, but not for claiming a
general improvement in a paper. Final claims require the same frozen protocol to
be repeated across the predeclared seeds and summarized with uncertainty across
seeds.
