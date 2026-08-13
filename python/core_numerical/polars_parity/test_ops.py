"""Unit tests for Polars parity ops."""

from __future__ import annotations

import math

import pytest

from core_numerical.polars_parity.ops import run_op

OPS = [
    "construct",
    "select",
    "filter_gt",
    "with_columns",
    "drop_rename",
    "groupby_sum",
    "join_inner",
    "join_left",
    "sort",
    "head_tail",
    "csv_roundtrip",
    "lazy_filter_select",
]


@pytest.mark.parametrize("op", OPS)
def test_run_op_finite(op: str):
    checksum = run_op(op, size=16, seed=42)
    assert math.isfinite(checksum)
