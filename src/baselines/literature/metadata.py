"""Paper metadata serialized into every literature-baseline result."""

from __future__ import annotations


LITERATURE_METADATA: dict[str, dict[str, str]] = {
    "margin": {
        "paper": "Support Vector Machine Active Learning with Applications to Text Classification",
        "authors": "Tong and Koller",
        "venue": "Journal of Machine Learning Research (JMLR)",
        "year": "2001",
        "implementation": "src/baselines/literature/margin.py",
    },
    "coreset": {
        "paper": "Active Learning for Convolutional Neural Networks: A Core-Set Approach",
        "authors": "Sener and Savarese",
        "venue": "International Conference on Learning Representations (ICLR)",
        "year": "2018",
        "implementation": "src/baselines/literature/coreset.py",
    },
    "badge": {
        "paper": "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds",
        "authors": "Ash et al.",
        "venue": "International Conference on Learning Representations (ICLR)",
        "year": "2020",
        "implementation": "src/baselines/literature/badge.py",
    },
    "cal": {
        "paper": "Active Learning by Acquiring Contrastive Examples",
        "authors": "Margatina et al.",
        "venue": "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
        "year": "2021",
        "implementation": "src/baselines/literature/cal.py",
    },
}
