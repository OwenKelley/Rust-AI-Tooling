# Core Numerical — Phase 2 roadmap

Follow-on plan after Phase 1 (NumPy / SciPy / Pandas / RusTorch / `rarrow` v1).
Keep **local/`std` only**, extend existing parity harnesses, and land docs when
an item ships. Phase 1 archive: [`ROADMAP.md`](ROADMAP.md).

Work top-down within each track. Prefer finishing Arrow/Parquet interop before
deep Polars work (Polars sits on Arrow-shaped columns).

## Pass criteria (all slices)

- Parity checksums within op-specific tolerances vs Python
- Compare harness entry in the matching `*_parity` package
- Docs API map updated when an item lands (`ARROW.md`, new `POLARS.md`, …)

---

## Track A — Arrow / Parquet hardening (`rarrow`)

Close the gaps left from `rarrow` v1 ([`ARROW.md`](ARROW.md)).

| Status | Item |
|--------|------|
| done | Harden `write_ipc_stream` FlatBuffers so **PyArrow** `open_stream` accepts Rust bytes |
| done | Roundtrip parity: Rust write → PyArrow read, and PyArrow write → Rust read |
| done | Arrow IPC **file** format (`ARROW1` magic + footer), not only stream |
| done | Apache Parquet **`PAR1` write** (PLAIN / UNCOMPRESSED; f64 / i64 / bool / utf8) readable by `pyarrow.parquet` |
| done | Apache Parquet **`PAR1` read** for the same dtype subset |
| done | Keep or deprecate RPQT: auto-detect `RPQT` vs `PAR1` in `read_parquet`; `write_parquet_rpqt` for legacy |
| done | Optional: `timestamp[ns]`, nested `list<float64>` v1 (IPC; Parquet list deferred) |
| done | Optional: dictionary-encoded utf8 (IPC DictionaryBatch; Parquet densifies to utf8) |

**Exit:** `arrow_parity` ops include `ipc_write_pyarrow_read` and `parquet_par1_*`; [`ARROW.md`](ARROW.md) drops the “prefer read-only for exchange” note.

---

## Track B — Polars-shaped API (`rpolars`)

New crate + harness, built on `rarrow` / `rnumpy` (not a Polars FFI wrap).

| Status | Item |
|--------|------|
| done | Scaffold `crates/rpolars` + workspace member + `POLARS.md` |
| done | `DataFrame` / `Series` construction from columns / `rarrow::RecordBatch` |
| done | Select / with_columns / drop / rename |
| done | Filter (`expr` v1: col comparisons + and/or) |
| done | Group-by aggregations (`sum` / `mean` / `count` / `min` / `max`) |
| done | Joins (inner / left) on one or more keys |
| done | Sort / slice / head / tail |
| done | CSV read/write (local `std` CSV) |
| done | Lazy frame v1: plan + collect for filter/select/groupby only |
| done | Parity package `python/core_numerical/polars_parity` + `polars_parity_runner` |

**Exit:** Core eager API parity green; lazy is optional stretch in the same track.

---

## Track C — Broader Python AI/ML map

Phased Rust mirrors of high-value rows from [`python-ai-ml-tooling.md`](../python-ai-ml-tooling.md).
Do **not** boil the ocean; one library family at a time with a thin API + parity.

### C1 — Classical ML (`rsklearn`)

| Status | Item |
|--------|------|
| done | Train/test split, `StandardScaler`, `LabelEncoder` |
| done | Linear / logistic regression (GD or closed form where trivial) |
| done | k-NN classify/regress (brute force v1) |
| done | k-means |
| done | Metrics: accuracy, precision/recall/F1, MSE/MAE/R² |
| done | Parity harness vs scikit-learn on small synthetic sets |

### C2 — Tokenizers / embeddings bridge

| Status | Item |
|--------|------|
| done | Thin `rtokenizers`-shaped API (Whitespace + BPE/WordPiece surface) |
| done | Simple bag-of-words / hashing vectorizer on `rnumpy` for classical pipelines |

### C3 — Serving / interchange (after C1)

| Status | Item |
|--------|------|
| done | ONNX-ish JSON model dump for Linear / Logistic / StandardScaler ([`ONNXISH.md`](ONNXISH.md)) |
| done | Minimal HTTP/JSON inference sketch (`rsklearn` example `serve_onnxish`, `std::net`) |

### Explicitly deferred (inventory only)

Dask, CuPy, Numba, JAX, TF, Lightning, DeepSpeed, HF full Transformers stack,
XGBoost/LightGBM, LangChain, vLLM, experiment trackers, cloud feature stores.
Revisit after C1–C2 prove the parity pattern.

---

## Suggested order

1. **A** (Arrow write + `PAR1`) — unblocks Polars IO and cross-runtime exchange  
2. **B** (rpolars eager core) — next dataframe surface after Pandas  
3. **C1** (sklearn-shaped) — first non-dataframe ML library slice  
4. **C2 / C3** as capacity allows  

## Current focus

**Phase 2 complete** for tracks A–C3 on this roadmap. Further work is deferred
inventory or new Phase 3 planning.
