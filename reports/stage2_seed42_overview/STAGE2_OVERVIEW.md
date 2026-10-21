# Stage 2 comprehensive overview — seed 42, budget 128

This is the single consolidated report for discussion with the supervisor and for preparing the experimental section. It combines the original baselines, verified-flow versions v2--v7, ablations, and every completed published active-learning baseline. All reported final HANS metrics use the same frozen 15,000-example final-test manifest.

> **Scope.** These are development results from one seed. They support debugging and experiment selection, not a paper-level significance or state-of-the-art claim. Multi-seed confirmation remains required.

## Executive summary

- The strongest completed result is `uncertainty_mnli` at **66.9933%** HANS accuracy. It is a mnli baseline and therefore is not an equal-data-access comparison with the verified synthetic flow.
- The strongest valid verified result is `ours_verified_v7_continuation` at **65.0000%**.
- On the controlled static verified-v5 pool, `coreset_synthetic` is numerically best at **65.0533%**, only +0.0533 percentage points (+8 predictions out of 15,000) relative to v7 continuation. This is an effective tie at one seed.
- Every completed published synthetic selector improves over `random_synthetic` (64.4600%), showing that acquisition matters on the verified pool.
- The following published MNLI runs are still pending and are not estimated from partial checkpoints: `margin_mnli`, `coreset_mnli`, `badge_mnli`, `cal_mnli`.

## Paper-ready figures

> **Training-data source note.** Methods named `*_mnli` train on human-annotated MNLI examples and are cross-source baselines, not equal-data-access comparisons with synthetic-data methods. Figure 1 places them below a labeled divider; Figure 2 uses outlined markers. MNLI accuracy on Figure 2's x-axis is an evaluation metric for every method and does not imply MNLI training.

![Sealed HANS final accuracy](figures/figure_1_hans_accuracy.svg)

**Figure 1.** Sealed HANS final accuracy. Published selectors are annotated with paper title, venue, and publication year. Runs trained on human-annotated MNLI examples are grouped below the labeled blue divider. The horizontal axis is explicitly zoomed to make small differences visible; abstained and invalid diagnostic runs are excluded.

![MNLI–HANS trade-off](figures/figure_2_nli_hans_tradeoff.svg)

**Figure 2.** MNLI matched accuracy versus HANS non-entailment accuracy. Outlined markers identify runs trained on human-annotated MNLI examples; all other markers use no MNLI training data. This exposes whether a method improves robustness while preserving general NLI performance.

![Published baseline learning curves](figures/figure_3_literature_learning_curves.svg)

**Figure 3.** Learning curves for the completed published synthetic acquisition baselines, the `failure_driven_no_verification` ablation, and `ours_verified_v7_continuation`. Solid curves use the same 3,750-example post-hoc gain-eval set. The no-verification ablation has only endpoints because it trains once on all 128 examples. The dashed v7 curve uses its predeclared 900-example stopping-eval subset; therefore compare its trend, not its absolute vertical position, with the solid curves. All plotted runs use synthetic training data; the human-annotated MNLI variants are pending. The sealed final-test is evaluated only after training and is never used for acquisition.

![Adaptive verification stopping](figures/figure_4_adaptive_verification_stopping.svg)

**Figure 4.** Probe sets used by adaptive Hoeffding verification compared with a fixed 60-set maximum for every hypothesis. This figure measures verification-stage savings, not total active-learning budget savings.

![Controlled acquisition compute](figures/figure_5_controlled_acquisition_cost.svg)

**Figure 5.** Acquisition-only GPU time on the controlled synthetic setting. It excludes teacher generation, verification, training, and evaluation and therefore must not be presented as total cost.

All figures are SVG vector graphics and can be converted to PDF during paper typesetting without rasterization.

## Complete result table

| Rank | Method | Group | Objective | Train n | HANS | HANS non-ent. | MNLI m/mm | Status |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | `uncertainty_mnli` | MNLI baseline | CE | 128 | 66.9933% | 42.3000% | 61.0392% / 61.5439% | complete cross-source baseline |
| 2 | `failure_driven_no_verification` | ablation | CE | 128 | 65.4667% | 38.3732% | 59.3377% / 59.6318% | complete verification ablation |
| 3 | `random_mnli` | MNLI baseline | CE | 128 | 65.2600% | 37.6519% | 59.4906% / 59.7742% | complete cross-source baseline |
| 4 | `coreset_synthetic` | published synthetic | CE | 128 | 65.0533% | 36.9173% | 59.0117% / 59.2555% | complete; International Conference on Learning Representations (ICLR) |
| 5 | `cal_synthetic` | published synthetic | CE | 128 | 65.0333% | 36.8639% | 59.1034% / 59.2962% | complete; Conference on Empirical Methods in Natural Language Processing (EMNLP) |
| 6 | `ours_verified_v7_continuation` | verified | CE | 128 | 65.0000% | 36.3296% | 58.6347% / 58.9605% | complete stateful closed-loop continuation |
| 7 | `ours_verified_v5_ce` | verified | CE | 128 | 64.8200% | 36.4766% | 58.8079% / 58.9402% | complete |
| 8 | `ours_verified_target_only_early` | verified | pairwise | 128 | 64.7867% | 36.5166% | 58.8691% / 59.1436% | complete early target-only protocol |
| 9 | `ours_verified_v3_ce` | verified | CE | 128 | 64.7533% | 36.1026% | 58.7672% / 58.9402% | complete |
| 10 | `ours_verified_v7_consolidation` | verified | CE | 128 | 64.7200% | 35.5817% | 58.4412% / 58.6554% | complete fresh one-pass consolidation |
| 11 | `margin_synthetic` | published synthetic | CE | 128 | 64.7067% | 35.7420% | 58.5634% / 58.7164% | complete; Journal of Machine Learning Research (JMLR) |
| 12 | `badge_synthetic` | published synthetic | CE | 128 | 64.7000% | 35.7954% | 58.6144% / 58.7673% | complete; International Conference on Learning Representations (ICLR) |
| 13 | `random_synthetic` | synthetic baseline | CE | 128 | 64.4600% | 35.1676% | 58.3902% / 58.5130% | complete synthetic baseline |
| 14 | `ours_verified_v2_ce` | verified | CE | 128 | 64.3800% | 34.8204% | 58.2578% / 58.4317% | complete objective ablation |
| 15 | `ours_verified_v2_pairwise` | verified | pairwise | 128 | 64.3400% | 34.7669% | 58.2476% / 58.4418% | complete objective ablation |
| 16 | `zero_shot` | reference | none | 0 | 63.8533% | 33.7518% | 57.6566% / 57.9434% | reference; no training |
| — | `ours_verified_strict_initial` | verified | none | 0 | 63.8533% | 33.7518% | 57.6566% / 57.9434% | abstained; zero-shot fallback |
| — | `ours_verified_v6_closed_loop` | diagnostic | CE | 0 | 63.8533% | 33.7518% | 57.6566% / 57.9434% | preserved diagnostic; checkpoint continuity failed |

## Controlled published acquisition baselines

The literature methods are not new selectors from this project. They are protocol-compatible reimplementations of published acquisition rules, run with the shared Qwen3-1.7B/LoRA trainer and sealed evaluator. Synthetic variants use the immutable verified-v5 candidate pool; MNLI variants use the original MNLI train pool.

| Method | Publication | Venue | Year | HANS | Acquisition GPU time | Local implementation |
|---|---|---|---:|---:|---:|---|
| `badge_synthetic` | Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds | International Conference on Learning Representations (ICLR) | 2020 | 64.7000% | 62.3 s | `src/baselines/literature/badge.py` |
| `cal_synthetic` | Active Learning by Acquiring Contrastive Examples | Conference on Empirical Methods in Natural Language Processing (EMNLP) | 2021 | 65.0333% | 71.5 s | `src/baselines/literature/cal.py` |
| `coreset_synthetic` | Active Learning for Convolutional Neural Networks: A Core-Set Approach | International Conference on Learning Representations (ICLR) | 2018 | 65.0533% | 28.9 s | `src/baselines/literature/coreset.py` |
| `margin_synthetic` | Support Vector Machine Active Learning with Applications to Text Classification | Journal of Machine Learning Research (JMLR) | 2001 | 64.7067% | 39.4 s | `src/baselines/literature/margin.py` |

The four completed synthetic methods share the same 32-example random warm start; rounds 2--4 use the named published selector. LoRA weights and AdamW state continue across all rounds. Core-set and CAL mainly improve non-entailment accuracy while slightly reducing entailment accuracy relative to v7 continuation.

This static-pool comparison isolates acquisition behavior but does **not** reproduce v7's dynamic candidate trace. The v7 closed loop may retire or add hypotheses and generate fresh candidates after refresh; the published selectors cannot.

## Fairness and interpretation

- `random_mnli`, `uncertainty_mnli`, and published `*_mnli` methods have access to original human-written MNLI candidates. They are cross-source baselines relative to LLM-synthetic methods.
- Published `*_synthetic` methods and `random_synthetic` operate on synthetic candidates. The completed published methods use the same static verified-v5 pool, permitting a controlled selector comparison among themselves.
- `failure_driven_no_verification` is an ablation of the proposed verification component, not an external published method.
- `ours_verified_v7_continuation` is the valid online closed-loop result. `ours_verified_v7_consolidation` retrains once from the base model on the same frozen 128 selected examples.
- `ours_verified_v6_closed_loop` is preserved for auditability but checkpoint continuity failed; it is excluded from ranking. The strict initial run abstained and is also excluded.
- The teacher-token cost stored in each static-pool result is inherited provenance. Because the same pool is reused, that generation cost must not be summed once per selector.

## Budget and stopping evidence

### What is already demonstrated

Adaptive Hoeffding verification used **160/300 probe sets**. It therefore avoided **140 sets (46.7%)** relative to always generating the fixed maximum of 60 sets for each of five hypotheses.

| Hypothesis | Used / maximum | Decision |
|---|---:|---|
| Subject-Object Reversal Misclassification | 20 / 60 | rejected (`hoeffding_ucb_below_threshold`) |
| General-to-Specific Misclassification | 20 / 60 | certified (`hoeffding_lcb_certified`) |
| Contextual Invalidation Misclassification | 60 / 60 | unresolved (`maximum_probe_sets_reached`) |
| Conditional Statement Misclassification | 30 / 60 | certified (`hoeffding_lcb_certified`) |
| Entity Addition Misclassification | 30 / 60 | certified (`hoeffding_lcb_certified`) |

This supports the narrow claim that confidence-bound verification can stop collecting probes once a hypothesis is statistically certified or rejected. It does not by itself prove that the final decision is optimal; correctness still requires repeated seeds and held-out validation.

### What is not yet demonstrated

All valid v3, v5, v7, and published-selector runs trained on the full **128-example** budget. Therefore the current experiment does **not** demonstrate that the complete method uses fewer training examples than its baselines or knows the optimal point at which to stop the whole active-learning loop.

The v5 stopping rule reached the maximum budget. Its final look had negative mean gain and counted as only the first small-gain look, while the predeclared patience was two consecutive looks. Thus stopping at 128 would be a maximum-budget termination, not an adaptive early-stop success.

The modeled v5 cost is lower than the current published synthetic runs partly because v5 evaluates stopping on a reused 900-example subset whereas those runs repeatedly evaluate all 3,750 gain-eval examples. That protocol difference prevents using the current cost totals as evidence that the proposed algorithm itself is cheaper.

To establish an end-to-end stopping claim, predeclare a larger maximum budget (for example 256), allow at least two additional looks after 128, compare adaptive and fixed-budget arms under identical evaluation frequency, and evaluate the sealed final-test only after stopping. Report stopped budget, accuracy regret relative to the best fixed checkpoint, GPU-hours, teacher tokens, and stopping frequency across seeds.

## Cost comparison currently supported

The only reasonably controlled compute comparison currently available is acquisition GPU time on the static verified-synthetic setting:

| Method | Acquisition GPU-s | Training GPU-s | Evaluation GPU-s | Stored modeled total |
|---|---:|---:|---:|---:|
| `coreset_synthetic` | 28.9 | 12.3 | 3378.3 | $7.562 |
| `margin_synthetic` | 39.4 | 12.5 | 3341.6 | $7.558 |
| `badge_synthetic` | 62.3 | 12.1 | 3420.3 | $7.572 |
| `ours_verified_v5_ce` | 64.4 | 12.0 | 3088.3 | $6.191 |
| `cal_synthetic` | 71.5 | 12.6 | 3428.8 | $7.575 |

On acquisition compute, v5 is cheaper than CAL, approximately tied with BADGE, and more expensive than Margin and Core-set. Training cost is effectively the same for all methods because the backbone, LoRA recipe, budget, and optimizer-step count are controlled.

The stored modeled totals are shown for auditability but are **not directly comparable**: v5 uses the 900-example stopping subset and frozen base predictions, whereas the published runs repeatedly evaluate the 3,750-example gain-eval set; published synthetic runs also reuse the already generated v5 pool. Accordingly, the current artifacts do not establish an end-to-end cost advantage over other methods.

There is also no cross-model cost evidence. Every row in the published acquisition comparison uses the same Qwen3-1.7B backbone. No RoBERTa, DeBERTa, BERT, Llama, or other-backbone run has been performed under this protocol, so the report must not claim lower cost than other models.

A valid end-to-end cost comparison requires identical evaluation schedules and separate reporting of: one-time pool construction, verification teacher tokens, acquisition GPU time, training GPU time, evaluation GPU time, peak VRAM, wall time, and amortized versus non-amortized cost. A cross-model claim additionally requires running the same method/budget grid on each backbone.

## Recommended statement for the paper

A defensible current statement is: *Under a controlled static verified-synthetic pool, published acquisition functions and the v7 verified closed loop achieve approximately 65% HANS accuracy at seed 42; Core-set, CAL, and v7 are effectively tied. The verified method provides an auditable hypothesis-certification and closed-loop data-generation mechanism, but the present single-seed experiment does not establish an accuracy advantage over standard acquisition.*

Do not currently claim that v7 is uniformly best. The full-pool published MNLI runs and the remaining predeclared seeds are required before finalizing the experimental claim.

## Reproduction and source artifacts

Machine-readable table: `stage2_overview.csv`

| Method | Source result |
|---|---|
| `badge_synthetic` | `outputs/stage2_runs_literature/experiment_b/badge_synthetic/budget_128/seed_42/result.json` |
| `cal_synthetic` | `outputs/stage2_runs_literature/experiment_b/cal_synthetic/budget_128/seed_42/result.json` |
| `coreset_synthetic` | `outputs/stage2_runs_literature/experiment_b/coreset_synthetic/budget_128/seed_42/result.json` |
| `failure_driven_no_verification` | `outputs/stage2_runs/experiment_a/failure_driven_no_verification/budget_128/seed_42/result.json` |
| `margin_synthetic` | `outputs/stage2_runs_literature/experiment_b/margin_synthetic/budget_128/seed_42/result.json` |
| `ours_verified_strict_initial` | `outputs/stage2_runs_ours_verified/experiment_a/ours_verified/budget_128/seed_42/result.json` |
| `ours_verified_target_only_early` | `outputs/stage2_runs_ours_verified_target_only/experiment_a/ours_verified/budget_128/seed_42/result.json` |
| `ours_verified_v2_ce` | `outputs/stage2_runs_ours_verified_v2_ce/experiment_a/ours_verified/budget_128/seed_42/result.json` |
| `ours_verified_v2_pairwise` | `outputs/stage2_runs_ours_verified_v2_pairwise/experiment_a/ours_verified/budget_128/seed_42/result.json` |
| `ours_verified_v3_ce` | `outputs/stage2_runs_ours_verified_v3_ce/experiment_b/ours_verified/budget_128/seed_42/result.json` |
| `ours_verified_v5_ce` | `outputs/stage2_runs_ours_verified_v5_ce/experiment_b/ours_verified/budget_128/seed_42/result.json` |
| `ours_verified_v6_closed_loop` | `outputs/preserved_checkpoints/ours_verified_v6_closed_loop/seed_42_budget_128_diagnostic_20261019/result.json` |
| `ours_verified_v7_consolidation` | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/consolidation/result.json` |
| `ours_verified_v7_continuation` | `outputs/stage2_runs_ours_verified_v7_dual/experiment_b/ours_verified/budget_128/seed_42/branches/continuation/result.json` |
| `random_mnli` | `outputs/stage2_runs/experiment_a/random_mnli/budget_128/seed_42/result.json` |
| `random_synthetic` | `outputs/stage2_runs/experiment_a/random_synthetic/budget_128/seed_42/result.json` |
| `uncertainty_mnli` | `outputs/stage2_runs/experiment_a/uncertainty_mnli/budget_128/seed_42/result.json` |
| `zero_shot` | `outputs/stage2_runs/experiment_a/zero_shot/budget_128/seed_42/result.json` |

Regenerate this report after new runs complete:

```bash
python scripts/report_stage2_overview.py
```
