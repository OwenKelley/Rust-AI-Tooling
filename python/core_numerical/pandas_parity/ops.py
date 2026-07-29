"""Python pandas reference ops for parity with rpandas."""

from __future__ import annotations

import io
from typing import Any, Callable

import numpy as np
import pandas as pd

from core_numerical.numpy_parity.rng import seeded_uniform


def numeric_frame(n: int, seed: int, ncols: int = 4) -> pd.DataFrame:
    data = seeded_uniform((n, ncols), seed, -1.0, 1.0)
    return pd.DataFrame({f"c{j}": data[:, j] for j in range(ncols)})


def with_nans(df: pd.DataFrame, every: int) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        vals = out[col].to_numpy(dtype=np.float64, copy=True)
        vals[::every] = np.nan
        out[col] = vals
    return out


def frame_with_group_key(n: int, seed: int) -> pd.DataFrame:
    df = numeric_frame(n, seed, ncols=3)
    g = np.floor((df["c0"].to_numpy() + 1.0) * 2.0)
    g = np.clip(g, 0.0, 3.0)
    df = df.copy()
    df["g"] = g
    return df


def merge_frames(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = max(n, 4)
    half = n // 2
    left = pd.DataFrame(
        {
            "k": np.arange(n, dtype=np.float64),
            "v": seeded_uniform((n,), seed, -1.0, 1.0),
        }
    )
    right = pd.DataFrame(
        {
            "k": np.arange(half, half + n, dtype=np.float64),
            "w": seeded_uniform((n,), seed + 1, -1.0, 1.0),
        }
    )
    return left, right


def pivot_source(n: int, seed: int) -> pd.DataFrame:
    n = max(n, 8)
    vals = seeded_uniform((n,), seed, -1.0, 1.0)
    return pd.DataFrame(
        {
            "i": np.array([float(r % 4) for r in range(n)], dtype=np.float64),
            "c": [f"g{r % 3}" for r in range(n)],
            "v": vals,
        }
    )


def mixed_frame(n: int, seed: int) -> pd.DataFrame:
    f = seeded_uniform((n,), seed, -1.0, 1.0)
    return pd.DataFrame(
        {
            "f": f,
            "i": np.floor(f * 10.0).astype(np.int64),
            "b": f > 0.0,
        }
    )


def frame_checksum(df: pd.DataFrame) -> float:
    """Match rpandas::DataFrame::checksum — nrows + ncols + col checksums."""
    s = float(df.shape[0] + df.shape[1])
    for col in df.columns:
        vals = df[col]
        if pd.api.types.is_numeric_dtype(vals):
            arr = vals.to_numpy(dtype=np.float64)
            s += float(np.nansum(arr))
        else:
            # string: sum of lengths of non-null
            for v in vals:
                if pd.isna(v):
                    continue
                s += float(len(str(v)))
    return s


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    n = size

    if op == "construct":
        result = numeric_frame(n, seed, 4)

        def thunk() -> pd.DataFrame:
            return numeric_frame(n, seed, 4)

        return result, thunk

    if op == "select":
        df = numeric_frame(n, seed, 4)
        result = df[["c0", "c2"]]

        def thunk() -> pd.DataFrame:
            return df[["c0", "c2"]]

        return result, thunk

    if op == "head":
        df = numeric_frame(n, seed, 3)
        k = max(n // 4, 1)
        result = df.head(k)

        def thunk() -> pd.DataFrame:
            return df.head(k)

        return result, thunk

    if op == "filter_gt":
        df = numeric_frame(n, seed, 3)
        result = df[df["c0"] > 0.0]

        def thunk() -> pd.DataFrame:
            return df[df["c0"] > 0.0]

        return result, thunk

    if op == "sort_values":
        df = numeric_frame(n, seed, 3)
        result = df.sort_values("c0", ascending=True)

        def thunk() -> pd.DataFrame:
            return df.sort_values("c0", ascending=True)

        return result, thunk

    if op == "dropna":
        df = with_nans(numeric_frame(n, seed, 3), 7)
        result = df.dropna(how="any")

        def thunk() -> pd.DataFrame:
            return df.dropna(how="any")

        return result, thunk

    if op == "fillna":
        df = with_nans(numeric_frame(n, seed, 3), 7)
        result = df.fillna(0.0)

        def thunk() -> pd.DataFrame:
            return df.fillna(0.0)

        return result, thunk

    if op == "sum":
        df = numeric_frame(n, seed, 4)
        # pandas sum returns Series; wrap as 1-row frame like Rust
        s = df.sum(numeric_only=True)
        result = pd.DataFrame([s.to_dict()])

        def thunk() -> pd.DataFrame:
            ss = df.sum(numeric_only=True)
            return pd.DataFrame([ss.to_dict()])

        return result, thunk

    if op == "mean":
        df = numeric_frame(n, seed, 4)
        s = df.mean(numeric_only=True)
        result = pd.DataFrame([s.to_dict()])

        def thunk() -> pd.DataFrame:
            ss = df.mean(numeric_only=True)
            return pd.DataFrame([ss.to_dict()])

        return result, thunk

    if op == "describe":
        df = numeric_frame(n, seed, 3)
        result = df.describe()

        def thunk() -> pd.DataFrame:
            return df.describe()

        return result, thunk

    if op == "groupby_sum":
        df = frame_with_group_key(n, seed)
        # sort=False → first-appearance order; named agg matches Rust c1_sum
        result = (
            df.groupby("g", sort=False)
            .agg(c1_sum=("c1", "sum"))
            .reset_index()
        )

        def thunk() -> pd.DataFrame:
            return (
                df.groupby("g", sort=False)
                .agg(c1_sum=("c1", "sum"))
                .reset_index()
            )

        return result, thunk

    if op == "merge_inner":
        left, right = merge_frames(n, seed)
        result = pd.merge(left, right, on="k", how="inner")

        def thunk() -> pd.DataFrame:
            return pd.merge(left, right, on="k", how="inner")

        return result, thunk

    if op == "merge_left":
        left, right = merge_frames(n, seed)
        result = pd.merge(left, right, on="k", how="left")

        def thunk() -> pd.DataFrame:
            return pd.merge(left, right, on="k", how="left")

        return result, thunk

    if op == "csv_roundtrip":
        df = numeric_frame(n, seed, ncols=3)
        text = df.to_csv(index=False)
        result = pd.read_csv(io.StringIO(text))

        def thunk() -> pd.DataFrame:
            t = df.to_csv(index=False)
            return pd.read_csv(io.StringIO(t))

        return result, thunk

    if op == "melt":
        df = numeric_frame(n, seed, ncols=3)
        result = pd.melt(df, id_vars=["c0"], value_vars=["c1", "c2"])

        def thunk() -> pd.DataFrame:
            return pd.melt(df, id_vars=["c0"], value_vars=["c1", "c2"])

        return result, thunk

    if op == "pivot_sum":
        df = pivot_source(n, seed)
        result = pd.pivot_table(
            df, index="i", columns="c", values="v", aggfunc="sum", sort=False
        ).reset_index()
        # flatten column names (pandas may use MultiIndex)
        result.columns = [str(c) if c != "i" else "i" for c in result.columns.to_list()]

        def thunk() -> pd.DataFrame:
            out = pd.pivot_table(
                df, index="i", columns="c", values="v", aggfunc="sum", sort=False
            ).reset_index()
            out.columns = [str(c) if c != "i" else "i" for c in out.columns.to_list()]
            return out

        return result, thunk

    if op == "rolling_mean":
        df = numeric_frame(n, seed, ncols=1)
        window = 5
        s = df["c0"].rolling(window).mean()
        result = pd.DataFrame({"c0": s})

        def thunk() -> pd.DataFrame:
            return pd.DataFrame({"c0": df["c0"].rolling(window).mean()})

        return result, thunk

    if op == "mixed_dtypes":
        result = mixed_frame(n, seed)

        def thunk() -> pd.DataFrame:
            return mixed_frame(n, seed)

        return result, thunk

    raise ValueError(f"unknown op: {op}")


def checksum(value: Any) -> float:
    if isinstance(value, pd.DataFrame):
        return frame_checksum(value)
    if isinstance(value, pd.Series):
        return frame_checksum(value.to_frame())
    raise TypeError(f"unsupported checksum type: {type(value)}")


def run_op(op: str, size: int, seed: int) -> float:
    result, _ = prepare(op, size, seed)
    return checksum(result)
