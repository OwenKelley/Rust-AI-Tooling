"""Unit tests for PyTorch reference ops."""

from __future__ import annotations

import math

import pytest

from core_numerical.torch_parity.ops import run_op


@pytest.mark.parametrize(
    "op",
    [
        "zeros",
        "add",
        "mul",
        "matmul",
        "sum",
        "mean",
        "relu",
        "sigmoid",
        "transpose",
        "reshape",
        "linear_forward",
        "mse_loss",
        "train_step",
        "exp",
        "log",
        "pow",
        "clamp",
        "broadcast_add",
        "cat",
        "stack",
        "index_select",
        "softmax",
        "cross_entropy",
        "dropout",
        "sequential_forward",
        "adam_train_step",
        "embedding_forward",
        "layernorm_forward",
        "conv2d_forward",
        "adamw_train_step",
        "steplr",
        "tanh",
        "gelu",
        "batchnorm1d_forward",
        "max_pool2d_forward",
        "flatten_forward",
        "multisteplr",
    ],
)
def test_op_finite_checksum(op: str) -> None:
    checksum = run_op(op, size=16, seed=42)
    assert math.isfinite(checksum)
