# Core Numerical — Polars parity (Python ↔ Rust)

Maps **Polars**-shaped APIs onto the Rust crate `rpolars` (`std` only, on
`rarrow` arrays), with correctness and speed checks.

## Layout

| Path | Role |
|------|------|
| `crates/rpolars` | Eager (+ lazy v1) DataFrame / Series / Expr |
| `crates/parity_runner` (`polars_parity_runner`) | Times an op and prints JSON |
| `python/core_numerical/polars_parity` | Polars reference + comparison harness |

## API map

| Python (`polars`) | Rust (`rpolars`) |
|-------------------|------------------|
| `pl.Series` / `pl.DataFrame` | `rpolars::{Series,DataFrame}` |
| `df.select` / `drop` / `rename` / `with_columns` | same method names |
| `df.filter(pl.col("a") > 0)` | `df.filter(&col("a").gt(lit_f64(0.0)))` |
| `df.group_by(...).agg(...)` | `df.groupby(...).agg(&[("c", Agg::Sum)])` |
| `df.join(..., how="inner"\|"left")` | `df.join(..., JoinHow::Inner\|Left)` |
| `df.sort` / `head` / `tail` / `slice` | same |
| `df.write_csv` / `pl.read_csv` | `write_csv` / `read_csv` |
| `df.lazy().filter(...).select(...).collect()` | `df.lazy()...collect()` |
| `RecordBatch` bridge | `DataFrame::{from_record_batch,to_record_batch}` |

## Setup

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
cargo build -p parity_runner --bin polars_parity_runner --release
cargo test -p rpolars
cd python
pip install -e ".[ ]"  # needs polars
pytest core_numerical/polars_parity
```

## Compare

```powershell
cargo build -p parity_runner --bin polars_parity_runner --release
cd python
python -m core_numerical.polars_parity.compare --size 64 --iters 20
```

## Pass criteria

- Parity checksums within tolerance vs Polars
- Harness entry in `polars_parity`
- Docs updated when the surface grows

## Next

See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md) Track C1 (`rsklearn`) after eager Polars is green.
