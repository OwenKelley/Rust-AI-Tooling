"""Arrow / Parquet parity ops (PyArrow reference)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from core_numerical.numpy_parity.rng import seeded_uniform


def _make_table(n: int, seed: int) -> pa.Table:
    a = seeded_uniform((n,), seed, -1.0, 1.0)
    b = []
    for i, v in enumerate(a.tolist()):
        if i % 5 == 0:
            b.append(None)
        else:
            b.append(int(np.floor(v)))
    c = [None if i % 7 == 0 else f"s{i}" for i in range(n)]
    return pa.table(
        {
            "a": pa.array(a, type=pa.float64()),
            "b": pa.array(b, type=pa.int64()),
            "c": pa.array(c, type=pa.string()),
        }
    )


def _checksum_table(t: pa.Table) -> float:
    s = float(t.num_rows + t.num_columns)
    a = t.column(0).combine_chunks()
    s += float(np.nansum(a.to_numpy(zero_copy_only=False)))
    b = t.column(1).combine_chunks()
    for v in b.to_pylist():
        if v is not None:
            s += float(v)
    c = t.column(2).combine_chunks()
    for v in c.to_pylist():
        if v is not None:
            s += float(len(v) + sum(v.encode("utf-8")))
    return s


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    n = max(size, 8)
    if op in (
        "ipc_roundtrip",
        "ipc_read",
        "ipc_write_pyarrow_read",
        "ipc_file_roundtrip",
        "parquet_roundtrip",
        "parquet_par1_roundtrip",
    ):
        table = _make_table(n, seed)

        def thunk():
            if op == "ipc_roundtrip":
                sink = pa.BufferOutputStream()
                with ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
                buf = sink.getvalue()
                return ipc.open_stream(buf).read_all()
            if op == "ipc_read":
                sink = pa.BufferOutputStream()
                with ipc.new_stream(sink, table.schema) as writer:
                    writer.write_table(table)
                return ipc.open_stream(sink.getvalue()).read_all()
            if op == "ipc_file_roundtrip":
                sink = pa.BufferOutputStream()
                with ipc.new_file(sink, table.schema) as writer:
                    writer.write_table(table)
                return ipc.open_file(sink.getvalue()).read_all()
            if op == "ipc_write_pyarrow_read":
                return table
            import io

            import pyarrow.parquet as pq

            bio = io.BytesIO()
            pq.write_table(table, bio, compression="NONE", use_dictionary=False)
            bio.seek(0)
            return pq.read_table(bio)

        return thunk(), thunk

    raise ValueError(f"unknown op: {op}")


def checksum(value: Any) -> float:
    if isinstance(value, pa.Table):
        return _checksum_table(value)
    raise TypeError(type(value))


def run_op(op: str, size: int, seed: int) -> float:
    result, _ = prepare(op, size, seed)
    return checksum(result)
