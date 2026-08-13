"""Unit tests for Arrow parity ops."""

from __future__ import annotations

import math

import pytest

from core_numerical.arrow_parity.ops import run_op


@pytest.mark.parametrize(
    "op",
    [
        "ipc_roundtrip",
        "ipc_file_roundtrip",
        "ipc_write_pyarrow_read",
        "parquet_par1_roundtrip",
    ],
)
def test_run_op_finite(op: str):
    checksum = run_op(op, size=16, seed=42)
    assert math.isfinite(checksum)
