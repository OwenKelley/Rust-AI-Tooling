"""NumPy reference ops used by the Python↔Rust parity harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .rng import seeded_uniform


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

    if op == "sqrt":
        a = seeded_uniform((n, n), seed, 0.0, 10.0)
        return np.sqrt(a), lambda: np.sqrt(a)

    if op == "exp":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        return np.exp(a), lambda: np.exp(a)

    if op == "log":
        a = seeded_uniform((n,), seed, 0.1, 10.0)
        return np.log(a), lambda: np.log(a)

    if op == "negative":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.negative(a), lambda: np.negative(a)

    if op == "abs":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.abs(a), lambda: np.abs(a)

    if op == "sum":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.sum(a), lambda: np.sum(a)

    if op == "mean":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.mean(a), lambda: np.mean(a)

    if op == "min":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.min(a), lambda: np.min(a)

    if op == "max":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        return np.max(a), lambda: np.max(a)

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

    if op == "transpose":
        a = seeded_uniform((n, n + 1), seed, -1.0, 1.0)
        return np.transpose(a), lambda: np.transpose(a)

    if op == "matmul":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        return np.matmul(a, b), lambda: np.matmul(a, b)

    if op == "dot":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        b = seeded_uniform((n,), seed + 1, -1.0, 1.0)
        return np.dot(a, b), lambda: np.dot(a, b)

    raise ValueError(f"unknown op: {op}")


def checksum(value: Any) -> float:
    if np.isscalar(value):
        return float(value)
    return float(np.sum(value))


def run_op(op: str, size: int, seed: int) -> float:
    """Execute one op and return a scalar checksum."""
    result, _ = prepare(op, size, seed)
    return checksum(result)
