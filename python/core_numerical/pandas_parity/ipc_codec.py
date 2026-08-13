"""Pure-Python RPIC IPC codec matching rpandas::ipc (Arrow-inspired v1)."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pandas as pd

MAGIC = b"RPIC"
VERSION = 1
DTYPE_F64 = 0
DTYPE_I64 = 1
DTYPE_BOOL = 2
DTYPE_UTF8 = 3


def to_ipc_bytes(df: pd.DataFrame) -> bytes:
    out = bytearray()
    out += MAGIC
    out += struct.pack("<I", VERSION)
    out += struct.pack("<I", df.shape[1])
    out += struct.pack("<Q", df.shape[0])
    nrows = df.shape[0]
    for name in df.columns:
        nb = str(name).encode("utf-8")
        out += struct.pack("<I", len(nb))
        out += nb
        col = df[name]
        if pd.api.types.is_bool_dtype(col) and not pd.api.types.is_float_dtype(col):
            out.append(DTYPE_BOOL)
            vals = col.to_numpy()
            nulls = pd.isna(col).to_numpy()
            for i in range(nrows):
                out.append(1 if (not nulls[i] and bool(vals[i])) else 0)
            for i in range(nrows):
                out.append(1 if nulls[i] else 0)
        elif pd.api.types.is_integer_dtype(col):
            out.append(DTYPE_I64)
            vals = col.to_numpy()
            nulls = pd.isna(col).to_numpy()
            for i in range(nrows):
                v = 0 if nulls[i] else int(vals[i])
                out += struct.pack("<q", v)
            for i in range(nrows):
                out.append(1 if nulls[i] else 0)
        elif pd.api.types.is_float_dtype(col):
            out.append(DTYPE_F64)
            vals = col.to_numpy(dtype=np.float64)
            for i in range(nrows):
                out += struct.pack("<d", float(vals[i]))
        else:
            out.append(DTYPE_UTF8)
            for i in range(nrows):
                v = col.iloc[i]
                if pd.isna(v):
                    out += struct.pack("<I", 0xFFFFFFFF)
                else:
                    b = str(v).encode("utf-8")
                    out += struct.pack("<I", len(b))
                    out += b
    return bytes(out)


def read_ipc_bytes(data: bytes) -> pd.DataFrame:
    off = 0
    assert data[0:4] == MAGIC
    off = 4
    (version,) = struct.unpack_from("<I", data, off)
    off += 4
    assert version == VERSION
    (ncols,) = struct.unpack_from("<I", data, off)
    off += 4
    (nrows,) = struct.unpack_from("<Q", data, off)
    off += 8
    columns: dict[str, Any] = {}
    for _ in range(ncols):
        (nlen,) = struct.unpack_from("<I", data, off)
        off += 4
        name = data[off : off + nlen].decode("utf-8")
        off += nlen
        dtype = data[off]
        off += 1
        if dtype == DTYPE_F64:
            vals = np.empty(nrows, dtype=np.float64)
            for i in range(nrows):
                (vals[i],) = struct.unpack_from("<d", data, off)
                off += 8
            columns[name] = vals
        elif dtype == DTYPE_I64:
            vals = np.empty(nrows, dtype=np.int64)
            for i in range(nrows):
                (vals[i],) = struct.unpack_from("<q", data, off)
                off += 8
            nulls = np.frombuffer(data, dtype=np.uint8, count=nrows, offset=off) != 0
            off += nrows
            s = pd.Series(vals, dtype="Int64")
            s = s.mask(nulls)
            columns[name] = s
        elif dtype == DTYPE_BOOL:
            vals = np.frombuffer(data, dtype=np.uint8, count=nrows, offset=off) != 0
            off += nrows
            nulls = np.frombuffer(data, dtype=np.uint8, count=nrows, offset=off) != 0
            off += nrows
            s = pd.Series(vals, dtype="boolean")
            s = s.mask(nulls)
            columns[name] = s
        elif dtype == DTYPE_UTF8:
            vals: list[str | None] = []
            for _i in range(nrows):
                (n,) = struct.unpack_from("<I", data, off)
                off += 4
                if n == 0xFFFFFFFF:
                    vals.append(None)
                else:
                    vals.append(data[off : off + n].decode("utf-8"))
                    off += n
            columns[name] = vals
        else:
            raise ValueError(f"unknown dtype {dtype}")
    return pd.DataFrame(columns)
