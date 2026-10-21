# Literature active-learning baselines

This directory contains local, protocol-compatible implementations of published
acquisition functions. The algorithms are executed by the repository's shared
Qwen/LoRA trainer and sealed HANS/MNLI evaluator; external repositories are used
as implementation references, not as incomparable end-to-end runners.

| File | Method | Publication | Venue |
|---|---|---|---|
| `margin.py` | Maximum Margin / Breaking Ties | Tong and Koller, *Support Vector Machine Active Learning with Applications to Text Classification* | Journal of Machine Learning Research (JMLR), 2001 |
| `coreset.py` | Greedy k-center Core-set | Sener and Savarese, *Active Learning for Convolutional Neural Networks: A Core-Set Approach* | International Conference on Learning Representations (ICLR), 2018 |
| `badge.py` | BADGE | Ash et al., *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds* | International Conference on Learning Representations (ICLR), 2020 |
| `cal.py` | Contrastive Active Learning (CAL) | Margatina et al., *Active Learning by Acquiring Contrastive Examples* | Conference on Empirical Methods in Natural Language Processing (EMNLP), 2021 |

Protocol notes:

- Every literature method receives the same 32-example random warm start.
- MNLI runs select individual examples from the identical MNLI train pool.
- Controlled synthetic runs select complete two-example pairs from the identical
  verified v5 pool. Pair probability scores use the failure-targeting `base`
  member; pair embeddings are the normalized mean of both members.
- The controlled synthetic runs isolate selector behavior on a static pool.
  They do not reproduce v7's dynamic hypothesis refresh and candidate
  generation trace.
- BADGE uses the published pseudo-label last-layer gradient construction with
  Qwen's three normalized verbalized-label probabilities and contextual pair
  representation.
- CAL uses the published mean `KL(p_neighbor || p_candidate)` over nearest
  already-selected examples; no gold labels enter its acquisition score.
