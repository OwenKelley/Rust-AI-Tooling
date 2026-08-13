# Core Numerical — Phase 1 roadmap (complete)

Practical slices after the closed `rustorch` TORCH.md items 1–7.
**Phase 1 is complete.** Active plan: [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md).

Work was top-down within each crate; local/`std` only; parity harnesses extended
as items landed.

## 1. NumPy / `rnumpy` — linalg & indexing depth

| Status | Item |
|--------|------|
| done | Full SVD: `np.linalg.svd` → `(U, S, Vh)` (reduced) + parity |
| done | General eigen values: `np.linalg.eigvals` → `(real, imag)` + parity |
| done | General eigen vectors: `np.linalg.eig` → `((wr,wi),(vr,vi))` + parity |
| done | Richer fancy indexing (`boolean_index`, `fancy_index_2d`, `take_along_axis`) + parity |
| done | Broader dtype coverage: `f32` / `i64` / `bool` companions + `astype_*` + parity |
| done | O(1) views for `expand_dims` / `squeeze` / `index_axis` (select-like) + parity |

## 2. SciPy / `rscipy` — sparse & signal

| Status | Item |
|--------|------|
| done | `sparse.linalg`: `spsolve`, `cg` (CSR; spsolve densifies for v1) |
| done | Signal: `butter` / `filtfilt`, `welch` / `stft` (onesided mag) |
| done | Fuller continuous distributions: `uniform` / `expon` / `laplace` / `logistic` (pdf/cdf/ppf) |
| done | More continuous distributions: `t` / `chi2` / `gamma` / `beta` (pdf/cdf/ppf) |
| done | Discrete distributions: `poisson` / `binom` (pmf/cdf) |
| done | `dblquad` + `solve_ivp_rk23` (Bogacki–Shampine) + parity |

## 3. Pandas / `rpandas` — time series & Arrow

| Status | Item |
|--------|------|
| done | `DatetimeIndex` + `date_range` + `resample_mean` / `resample_sum` (`h`/`D`, left bins) |
| done | Multi-key `merge_on` + richer `MergeHow` (`inner`/`left`/`right`/`outer`) |
| done | `Series.apply` / `map` (`map_f64` / `apply_f64` elementwise) |
| done | Categoricals v1 (`Categorical` + `categorical_codes`, sorted cats) |
| done | Arrow-inspired IPC / Parquet alias (`to_ipc_bytes` / `read_ipc_bytes`, RPIC v1) |
| done | Embed `DatetimeIndex` on `DataFrame.index` (`Index` enum + `set_index` / `resample_*_index`) |
| done | `rarrow` v1: arrays / `RecordBatch` / Arrow IPC stream (+ read PyArrow streams) / RPQT parquet + `rpandas` bridge + parity |

## Pass criteria (all slices)

- Parity checksums within op-specific tolerances vs Python
- Compare harness entry in the matching `*_parity` package
- Docs API map updated when an item lands

## Next

See [`ROADMAP_PHASE2.md`](ROADMAP_PHASE2.md) (Arrow/Parquet hardening → `rpolars` → sklearn-shaped / later tooling).
