from __future__ import annotations

import numpy as np

FEATURES = [
    "cardiothoracic_ratio",
    "vascular_prominence",
    "aortic_calcification",
    "lung_texture_entropy",
    "age_scaled",
    "systolic_bp_scaled",
]


def make_dataset(n: int = 1200, seed: int = 27, scanner_shift: float = 0.0):
    """Return synthetic image descriptors, labels, and feature names."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(FEATURES)))
    x[:, :4] += scanner_shift
    interaction = 0.75 * x[:, 0] * np.maximum(x[:, 2], 0)
    logit = (
        1.2 * x[:, 0]
        + 0.85 * x[:, 1]
        + 1.0 * x[:, 2]
        + 0.35 * x[:, 3]
        + 0.65 * x[:, 4]
        + 0.55 * x[:, 5]
        + interaction
        - 1.0
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = rng.binomial(1, prob)
    return x, y, FEATURES
