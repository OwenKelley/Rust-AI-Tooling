# Core Numerical — Arrow / Parquet parity (Python ↔ Rust)

Maps **PyArrow**-shaped APIs onto the Rust crate `rarrow` (`std` only FlatBuffers
IPC + parquet containers), with correctness and speed checks.

## Layout

| Path | Role |
|------|------|
| `crates/rarrow` | Arrow arrays, `RecordBatch`, IPC stream/file, parquet |
| `crates/parity_runner` (`arrow_parity_runner`) | Times an op and prints JSON |
| `python/core_numerical/arrow_parity` | PyArrow reference + comparison harness |
| `rpandas::{dataframe_to_record_batch,record_batch_to_dataframe}` | Frame ↔ batch bridge |

## API map

| Python (`pyarrow`) | Rust (`rarrow`) |
|--------------------|-----------------|
| `pa.array` / Table columns | `rarrow::{Float64Array,Int64Array,BooleanArray,StringArray,ListFloat64Array,DictionaryUtf8Array}` + `Array::TimestampNs` |
| `pa.schema` / `Field` | `rarrow::{Schema,Field,DataType}` (`TimestampNs`, `ListFloat64`, `DictionaryUtf8`) |
| `pa.RecordBatch` / `Table` | `rarrow::RecordBatch` |
| `pa.ipc.new_stream` / `open_stream` | `rarrow::{write_ipc_stream,read_ipc_stream}` |
| `pa.ipc.new_file` / `open_file` | `rarrow::{write_ipc_file,read_ipc_file}` |
| `pq.write_table` / `read_table` | `rarrow::{write_parquet,read_parquet}` (`PAR1` default; `RPQT` via `write_parquet_rpqt`) |

**Interop notes:** Rust IPC **stream** and **file** bytes are accepted by PyArrow
(`open_stream` / `open_file`). Rust also reads PyArrow-produced streams/files for
the supported dtype subset (`f64` / `i64` / `bool` / `utf8`, plus IPC `timestamp[ns]`,
`list<float64>`, and dictionary utf8). `write_parquet` emits Apache `PAR1` (PLAIN /
UNCOMPRESSED) for the scalar subset (dictionary densifies to utf8); nested lists are
IPC-only for now. `read_parquet` auto-detects `PAR1` vs legacy `RPQT`.

## Setup

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
cargo build -p parity_runner --bin arrow_parity_runner --release
cargo test -p rarrow
cd python
pip install -e .
pytest core_numerical/arrow_parity
```

## Compare

```powershell
cargo build -p parity_runner --bin arrow_parity_runner --release
cd python
python -m core_numerical.arrow_parity.compare --size 64 --iters 20
```

## Pass criteria

- Parity checksums within tolerance vs PyArrow-constructed tables
- `ipc_write_pyarrow_read` confirms PyArrow can read Rust stream bytes
- Docs updated when the surface grows

## Next

Phase 2 Track A optional dtypes are landed. See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md).
