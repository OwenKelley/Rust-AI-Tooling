"""Shared seeded RNG matching rnumpy::seeded_uniform (Numerical Recipes LCG)."""

from __future__ import annotations

import numpy as np


def seeded_uniform(
    shape: tuple[int, ...] | list[int],
    seed: int,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    state = int(seed) & 0xFFFFFFFFFFFFFFFF
    total = int(np.prod(shape))
    data = np.empty(total, dtype=np.float64)
    span = high - low
    for i in range(total):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
        u = ((state >> 8) & 0xFFFFFF) / float(1 << 24)
        data[i] = low + span * u
    return data.reshape(shape)
