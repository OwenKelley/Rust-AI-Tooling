"""Unit tests for sklearn parity ops."""

from __future__ import annotations

import math

import pytest

from core_numerical.sklearn_parity.ops import run_op

OPS = [
    "standard_scaler",
    "label_encoder",
    "train_test_split",
    "linear_regression",
    "logistic_regression",
    "knn_classify",
    "knn_regress",
    "kmeans",
    "metrics_class",
    "metrics_reg",
]


@pytest.mark.parametrize("op", OPS)
def test_run_op_finite(op: str):
    checksum = run_op(op, size=16, seed=42)
    assert math.isfinite(checksum)
