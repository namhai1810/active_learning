# Ours verified: training-objective ablation

Seed: `42`; budget: `128`.

Controlled comparison check: **pass** — CE and pairwise training used the same 128 example IDs in the same order.

| Metric | Pairwise | CE | Pairwise − CE |
|---|---:|---:|---:|
| HANS overall | 64.3400% | 64.3800% | -0.0400% |
| HANS entailment | 93.8107% | 93.8373% | -0.0266% |
| HANS non-entailment | 34.7669% | 34.8204% | -0.0534% |
| MNLI matched | 58.2476% | 58.2578% | -0.0102% |
| MNLI mismatched | 58.4418% | 58.4317% | +0.0102% |

Positive deltas favor pairwise training; negative deltas favor CE. This report isolates the training objective, not pool generation or acquisition selection.
