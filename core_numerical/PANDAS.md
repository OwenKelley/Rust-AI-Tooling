# Core Numerical — Pandas parity (Python ↔ Rust)

Third slice of the 1:1 Python→Rust translation work. Maps common **Pandas**
tabular APIs onto the Rust crate `rpandas` (built on `rnumpy`), with correctness
and speed checks against Pandas.

## Layout

| Path | Role |
|------|------|
| `crates/rpandas` | Rust Pandas-like API (`Series`, `DataFrame`, ops, groupby, merge, io, reshape, rolling) |
| `crates/parity_runner` (`pandas_parity_runner`) | Times an op and prints JSON |
| `python/core_numerical/pandas_parity` | Pandas reference + comparison harness |

## API map

### Construction / selection

| Python | Rust |
|--------|------|
| `pd.Series(data, name=)` | `Series::from_f64` / `from_i64` / `from_bool` / `from_str` |
| `pd.DataFrame({...})` | `DataFrame::from_columns` |
| `DataFrame` from matrix | `DataFrame::from_numeric` |
| `df[col]` / `df[[cols]]` | `df.column` / `df.select` |
| `df[name] = ...` | `df.with_column` |
| `df.head` / `tail` | `df.head` / `tail` |

### Frame ops

| Python | Rust |
|--------|------|
| `df[mask]` / `df[df.c > t]` | `rpandas::{filter,filter_gt}` |
| `df.sort_values` | `rpandas::sort_values` |
| `df.dropna` / `fillna` | `rpandas::{dropna,fillna}` |
| `df.describe` (numeric) | `rpandas::describe` |
| `df.sum` / `mean` (axis=0) | `rpandas::{sum,mean}` |

### Group / join / reshape / window / IO

| Python | Rust |
|--------|------|
| `df.groupby(key).agg(...)` | `rpandas::groupby_agg` |
| `pd.merge(..., how='inner'/'left'/'right'/'outer')` | `rpandas::{merge,merge_on,MergeHow}` |
| `pd.merge(..., on=[k1,k2])` | `rpandas::merge_on` |
| `pd.melt` | `rpandas::melt` |
| `pd.pivot_table` | `rpandas::pivot_table` |
| `Series.map` / `apply` (f64→f64) | `Series::{map_f64,apply_f64}` / `rpandas::{map_f64,apply_f64}` |
| `Series.astype('category')` codes | `rpandas::{Categorical,categorical_codes}` |
| `pd.date_range` / `DatetimeIndex` | `rpandas::{date_range,DatetimeIndex,Freq}` (naive epoch ns) |
| `df.set_index(dt)` / `df.index` | `df.set_index` / `rpandas::Index::{Range,Datetime}` |
| `df.resample('D'/'h').mean/sum` | `rpandas::{resample_mean,resample_sum}` or `resample_*_index` when index is datetime |
| `pd.read_csv` / `df.to_csv` | `rpandas::{read_csv,read_csv_str,to_csv,to_csv_string}` |
| Arrow-inspired IPC / Parquet alias | `rpandas::{to_ipc_bytes,read_ipc_bytes,to_parquet_bytes,…}` (RPIC v1; not full Arrow/Parquet) |

**Dtypes:** `f64` (NaN = missing), `i64` / `bool` / UTF-8 (null masks). Time series v1:
naive `datetime64[ns]` via epoch nanoseconds; freqs `'h'` and `'D'` only. Categoricals v1:
lexicographically sorted categories + integer codes (`-1` null). No MultiIndex yet.

## Setup

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --bin pandas_parity_runner --release
cargo test -p rpandas

cd python
pip install -e .
pytest core_numerical/pandas_parity
```

## Compare

```powershell
cargo build -p parity_runner --bin pandas_parity_runner --release
cd python
python -m core_numerical.pandas_parity.compare --size 64 --iters 20
```

## Pass criteria

- **Parity:** checksums within op-specific tolerances
- **Speed:** `python_median_ns / rust_median_ns`

## Next

See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md) (IPC write↔PyArrow; Apache `PAR1`).
