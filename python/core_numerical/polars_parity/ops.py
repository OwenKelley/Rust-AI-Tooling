"""Polars parity ops (Python reference)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl

from core_numerical.numpy_parity.rng import seeded_uniform


def _make_df(n: int, seed: int) -> pl.DataFrame:
    a = seeded_uniform((n,), seed, -1.0, 1.0)
    b = []
    for i, v in enumerate(a.tolist()):
        if i % 5 == 0:
            b.append(None)
        else:
            b.append(int(np.floor(v)))
    k = ["x" if i % 2 == 0 else "y" for i in range(n)]
    return pl.DataFrame(
        {
            "a": pl.Series("a", a, dtype=pl.Float64),
            "b": pl.Series("b", b, dtype=pl.Int64),
            "k": pl.Series("k", k, dtype=pl.Utf8),
        }
    )


def _array_checksum(s: pl.Series) -> float:
    total = 0.0
    if s.dtype == pl.Float64:
        for v in s.to_list():
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                total += float(v)
    elif getattr(s.dtype, "is_integer", lambda: False)() or s.dtype in (
        pl.Int64,
        pl.Int32,
        pl.UInt32,
        pl.UInt64,
        pl.UInt8,
    ):
        for v in s.to_list():
            if v is not None:
                total += float(v)
    elif s.dtype == pl.Boolean:
        for v in s.to_list():
            if v is not None:
                total += float(v)
    elif s.dtype in (pl.Utf8, pl.String) or str(s.dtype) in ("Utf8", "String", "str"):
        for v in s.to_list():
            if v is not None:
                b = v.encode("utf-8")
                total += float(len(b) + sum(b))
    else:
        raise TypeError(s.dtype)
    return total


def _checksum_df(df: pl.DataFrame) -> float:
    s = float(df.height + df.width)
    for name in df.columns:
        col = df[name]
        s += float(len(name)) + _array_checksum(col)
    return s


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    n = max(size, 8)
    df = _make_df(n, seed)

    def thunk():
        if op == "construct":
            return _make_df(n, seed)
        if op == "select":
            return df.select(["a", "k"])
        if op == "filter_gt":
            return df.filter(pl.col("a") > 0.0)
        if op == "with_columns":
            return df.with_columns(pl.Series("c", list(range(n)), dtype=pl.Float64))
        if op == "drop_rename":
            return df.drop("b").rename({"a": "alpha"})
        if op == "groupby_sum":
            return df.group_by("k", maintain_order=True).agg(
                pl.col("a").sum().alias("a_sum"),
                pl.col("a").count().alias("a_count"),
            )
        if op == "join_inner":
            right = pl.DataFrame({"k": ["x", "y"], "v": [1.0, 2.0]})
            return df.join(right, on="k", how="inner")
        if op == "join_left":
            right = pl.DataFrame({"k": ["x", "y"], "v": [1.0, 2.0]})
            return df.join(right, on="k", how="left")
        if op == "sort":
            return df.sort("a")
        if op == "head_tail":
            return (df.head(3), df.tail(2))
        if op == "csv_roundtrip":
            import io

            buf = io.StringIO()
            df.write_csv(buf)
            buf.seek(0)
            return pl.read_csv(buf)
        if op == "lazy_filter_select":
            return df.lazy().filter(pl.col("a") > 0.0).select(["a", "k"]).collect()
        raise ValueError(f"unknown op: {op}")

    return thunk(), thunk


def checksum(value: Any) -> float:
    if isinstance(value, pl.DataFrame):
        return _checksum_df(value)
    if isinstance(value, tuple):
        return sum(checksum(v) for v in value)
    raise TypeError(type(value))


def run_op(op: str, size: int, seed: int) -> float:
    result, _ = prepare(op, size, seed)
    return checksum(result)
