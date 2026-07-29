"""Unit tests for Pandas reference ops."""

from __future__ import annotations

import math

import pytest

from core_numerical.pandas_parity.ops import run_op


@pytest.mark.parametrize(
    "op",
    [
        "construct",
        "select",
        "head",
        "filter_gt",
        "sort_values",
        "dropna",
        "fillna",
        "sum",
        "mean",
        "describe",
        "groupby_sum",
        "merge_inner",
        "merge_left",
        "csv_roundtrip",
        "melt",
        "pivot_sum",
        "rolling_mean",
        "mixed_dtypes",
    ],
)
def test_op_finite_checksum(op: str) -> None:
    checksum = run_op(op, size=16, seed=42)
    assert math.isfinite(checksum)
