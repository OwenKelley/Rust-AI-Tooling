"""Unit tests: Python reference ops + shared RNG determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core_numerical.numpy_parity.ops import run_op
from core_numerical.numpy_parity.rng import seeded_uniform


def test_seeded_uniform_deterministic():
    a = seeded_uniform((4, 4), 42, -1.0, 1.0)
    b = seeded_uniform((4, 4), 42, -1.0, 1.0)
    np.testing.assert_array_equal(a, b)


def test_seeded_uniform_bounds():
    a = seeded_uniform((1000,), 7, 0.0, 1.0)
    assert float(a.min()) >= 0.0
    assert float(a.max()) < 1.0


@pytest.mark.parametrize(
    "op",
    [
        "zeros",
        "ones",
        "full",
        "arange",
        "linspace",
        "eye",
        "add",
        "add_broadcast",
        "maximum",
        "sin",
        "clip",
        "where",
        "reshape",
        "concatenate",
        "stack",
        "broadcast_to",
        "sum_axis",
        "mean_axis",
        "matmul",
        "dot",
        "sum",
        "mean",
        "var",
        "std",
    ],
)
def test_run_op_finite(op: str):
    checksum = run_op(op, size=32, seed=42)
    assert math.isfinite(checksum)
