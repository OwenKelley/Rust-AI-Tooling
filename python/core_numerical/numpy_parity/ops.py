"""NumPy reference ops used by the Python↔Rust parity harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .rng import seeded_uniform


def diag_dominant(n: int, seed: int, boost: float | None = None):
    a = seeded_uniform((n, n), seed, -1.0, 1.0)
    # Cap boost so det at large n does not overflow float64.
    b = float(n) if boost is None else float(boost)
    b = min(b, 4.0)
    for i in range(n):
        a[i, i] += b
    return a


def symmetric_spd(n: int, seed: int):
    a = seeded_uniform((n, n), seed, -1.0, 1.0)
    a = 0.5 * (a + a.T)
    for i in range(n):
        a[i, i] += float(n)
    return a


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    """
    Build inputs once and return (reference_result, timed_thunk).

    `timed_thunk` must only run the core op (same contract as parity_runner).
    """
    n = size

    if op == "zeros":
        shape = (n, n)

        def thunk() -> np.ndarray:
            return np.zeros(shape, dtype=np.float64)

        return thunk(), thunk

    if op == "ones":
        shape = (n, n)

        def thunk() -> np.ndarray:
            return np.ones(shape, dtype=np.float64)

        return thunk(), thunk

    if op == "full":
        shape = (n, n)

        def thunk() -> np.ndarray:
            return np.full(shape, 3.5, dtype=np.float64)

        return thunk(), thunk

    if op == "arange":
        stop = float(n)

        def thunk() -> np.ndarray:
            return np.arange(0.0, stop, 1.0, dtype=np.float64)

        return thunk(), thunk

    if op == "linspace":

        def thunk() -> np.ndarray:
            return np.linspace(0.0, 1.0, n, dtype=np.float64)

        return thunk(), thunk

    if op == "eye":

        def thunk() -> np.ndarray:
            return np.eye(n, dtype=np.float64)

        return thunk(), thunk

    if op == "add":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.add(a, b), lambda: np.add(a, b)

    if op == "add_broadcast":
        a = seeded_uniform((n, 1), seed, -1.0, 1.0)
        b = seeded_uniform((1, n), seed + 1, -1.0, 1.0)
        return np.add(a, b), lambda: np.add(a, b)

    if op == "subtract":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.subtract(a, b), lambda: np.subtract(a, b)

    if op == "multiply":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.multiply(a, b), lambda: np.multiply(a, b)

    if op == "divide":
        a = seeded_uniform((n, n), seed, 0.5, 1.5)
        b = seeded_uniform((n, n), seed + 1, 0.5, 1.5)
        return np.divide(a, b), lambda: np.divide(a, b)

    if op == "power":
        a = seeded_uniform((n,), seed, 0.5, 1.5)
        b = seeded_uniform((n,), seed + 1, 0.5, 2.0)
        return np.power(a, b), lambda: np.power(a, b)

    if op == "maximum":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.maximum(a, b), lambda: np.maximum(a, b)

    if op == "minimum":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.minimum(a, b), lambda: np.minimum(a, b)

    if op == "greater":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.greater(a, b).astype(np.float64), lambda: np.greater(a, b).astype(np.float64)

    if op == "less":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.less(a, b).astype(np.float64), lambda: np.less(a, b).astype(np.float64)

    if op == "equal":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        b = a.copy()
        return np.equal(a, b).astype(np.float64), lambda: np.equal(a, b).astype(np.float64)

    if op == "not_equal":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.not_equal(a, b).astype(np.float64), lambda: np.not_equal(a, b).astype(np.float64)

    if op == "sqrt":
        a = seeded_uniform((n, n), seed, 0.0, 10.0)
        return np.sqrt(a), lambda: np.sqrt(a)

    if op == "exp":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        return np.exp(a), lambda: np.exp(a)

    if op == "log":
        a = seeded_uniform((n,), seed, 0.1, 10.0)
        return np.log(a), lambda: np.log(a)

    if op == "sin":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.sin(a), lambda: np.sin(a)

    if op == "cos":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.cos(a), lambda: np.cos(a)

    if op == "tan":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        return np.tan(a), lambda: np.tan(a)

    if op == "tanh":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.tanh(a), lambda: np.tanh(a)

    if op == "negative":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.negative(a), lambda: np.negative(a)

    if op == "abs":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.abs(a), lambda: np.abs(a)

    if op == "sign":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.sign(a), lambda: np.sign(a)

    if op == "square":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.square(a), lambda: np.square(a)

    if op == "reciprocal":
        a = seeded_uniform((n, n), seed, 0.5, 1.5)
        return np.reciprocal(a), lambda: np.reciprocal(a)

    if op == "floor":
        a = seeded_uniform((n, n), seed, -5.0, 5.0)
        return np.floor(a), lambda: np.floor(a)

    if op == "ceil":
        a = seeded_uniform((n, n), seed, -5.0, 5.0)
        return np.ceil(a), lambda: np.ceil(a)

    if op == "trunc":
        a = seeded_uniform((n, n), seed, -5.0, 5.0)
        return np.trunc(a), lambda: np.trunc(a)

    if op == "round":
        a = seeded_uniform((n, n), seed, -5.0, 5.0)
        return np.round(a), lambda: np.round(a)

    if op == "clip":
        a = seeded_uniform((n, n), seed, -2.0, 2.0)
        return np.clip(a, -0.5, 0.5), lambda: np.clip(a, -0.5, 0.5)

    if op == "where":
        cond = seeded_uniform((n, n), seed, -1.0, 1.0)
        x = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        y = seeded_uniform((n, n), seed + 2, -1.0, 1.0)
        return np.where(cond, x, y), lambda: np.where(cond, x, y)

    if op == "sum":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.sum(a), lambda: np.sum(a)

    if op == "sum_axis":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.sum(a, axis=0), lambda: np.sum(a, axis=0)

    if op == "mean":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.mean(a), lambda: np.mean(a)

    if op == "mean_axis":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.mean(a, axis=1), lambda: np.mean(a, axis=1)

    if op == "min":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.min(a), lambda: np.min(a)

    if op == "min_axis":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.min(a, axis=0), lambda: np.min(a, axis=0)

    if op == "max":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.max(a), lambda: np.max(a)

    if op == "max_axis":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.max(a, axis=1), lambda: np.max(a, axis=1)

    if op == "var":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.var(a, ddof=0), lambda: np.var(a, ddof=0)

    if op == "std":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.std(a, ddof=0), lambda: np.std(a, ddof=0)

    if op == "argmin":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.argmin(a), lambda: np.argmin(a)

    if op == "argmax":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.argmax(a), lambda: np.argmax(a)

    if op == "cumsum":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.cumsum(a), lambda: np.cumsum(a)

    if op == "cumsum_axis":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.cumsum(a, axis=0), lambda: np.cumsum(a, axis=0)

    if op == "cumprod":
        a = seeded_uniform((n,), seed, 0.5, 1.5)
        return np.cumprod(a), lambda: np.cumprod(a)

    if op == "transpose":
        a = seeded_uniform((n, n + 1), seed, -1.0, 1.0)
        return np.transpose(a), lambda: np.transpose(a)

    if op == "reshape":
        a = seeded_uniform((n * n,), seed, -1.0, 1.0)
        return np.reshape(a, (n, n)), lambda: np.reshape(a, (n, n))

    if op == "reshape_infer":
        a = seeded_uniform((n * n,), seed, -1.0, 1.0)
        return np.reshape(a, (-1, n)), lambda: np.reshape(a, (-1, n))

    if op == "ravel":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.ravel(a), lambda: np.ravel(a)

    if op == "concatenate":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.concatenate((a, b), axis=0), lambda: np.concatenate((a, b), axis=0)

    if op == "stack":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.stack((a, b), axis=0), lambda: np.stack((a, b), axis=0)

    if op == "broadcast_to":
        a = seeded_uniform((1, n), seed, -1.0, 1.0)
        return np.broadcast_to(a, (n, n)).copy(), lambda: np.broadcast_to(a, (n, n)).copy()

    if op == "swapaxes":
        a = seeded_uniform((n, n + 1), seed, -1.0, 1.0)
        return np.swapaxes(a, 0, 1), lambda: np.swapaxes(a, 0, 1)

    if op == "moveaxis":
        a = seeded_uniform((n, n, 2), seed, -1.0, 1.0)
        return np.moveaxis(a, 0, 2), lambda: np.moveaxis(a, 0, 2)

    if op == "matmul":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.matmul(a, b), lambda: np.matmul(a, b)

    if op == "dot":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        b = seeded_uniform((n,), seed + 1, -1.0, 1.0)
        return np.dot(a, b), lambda: np.dot(a, b)

    if op == "trace":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.trace(a), lambda: np.trace(a)

    if op == "norm":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.linalg.norm(a), lambda: np.linalg.norm(a)

    if op == "solve":
        a = diag_dominant(n, seed)
        b = seeded_uniform((n,), seed + 1, -1.0, 1.0)
        return np.linalg.solve(a, b), lambda: np.linalg.solve(a, b)

    if op == "inv":
        a = diag_dominant(n, seed)
        return np.linalg.inv(a), lambda: np.linalg.inv(a)

    if op == "det":
        a = diag_dominant(n, seed)
        return np.linalg.det(a), lambda: np.linalg.det(a)

    if op == "qr":
        a = seeded_uniform((n, n // 2 + 1), seed, -1.0, 1.0)
        q, r = np.linalg.qr(a, mode="reduced")
        return q @ r, lambda: (lambda qr: qr[0] @ qr[1])(np.linalg.qr(a, mode="reduced"))

    if op == "svdvals":
        a = seeded_uniform((n, n // 2 + 1), seed, -1.0, 1.0)
        return np.linalg.svd(a, compute_uv=False), lambda: np.linalg.svd(a, compute_uv=False)

    if op == "eigvalsh":
        a = symmetric_spd(n, seed)
        return np.linalg.eigvalsh(a), lambda: np.linalg.eigvalsh(a)

    if op == "take":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        idx = list(range(0, n, max(2, n // 8)))
        return np.take(a, idx, axis=0), lambda: np.take(a, idx, axis=0)

    if op == "compress":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        cond = a > 0.0
        return np.compress(cond, a), lambda: np.compress(cond, a)

    if op == "slice":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return a[1 : n - 1, :].copy(), lambda: a[1 : n - 1, :].copy()

    if op == "astype_f32":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return a.astype(np.float32).astype(np.float64), lambda: a.astype(np.float32).astype(
            np.float64
        )

    raise ValueError(f"unknown op: {op}")


def checksum(value: Any) -> float:
    if np.isscalar(value):
        return float(value)
    return float(np.sum(value))


def run_op(op: str, size: int, seed: int) -> float:
    """Execute one op and return a scalar checksum."""
    result, _ = prepare(op, size, seed)
    return checksum(result)
