# Core Numerical — next roadmap

Practical next slices after the closed `rustorch` TORCH.md items 1–7.
Work top-down within each crate; keep local/`std` only and extend existing
parity harnesses.

## 1. NumPy / `rnumpy` — linalg & indexing depth

| Status | Item |
|--------|------|
| done | Full SVD: `np.linalg.svd` → `(U, S, Vh)` (reduced) + parity |
| done | General eigen values: `np.linalg.eigvals` → `(real, imag)` + parity |
| done | General eigen vectors: `np.linalg.eig` → `((wr,wi),(vr,vi))` + parity |
| done | Richer fancy indexing (`boolean_index`, `fancy_index_2d`, `take_along_axis`) + parity |
| pending | Broader dtype coverage beyond f64 primary + f32 cast companion |
| pending | Prefer O(1) views where NumPy does (`select`-like paths still copying) |

## 2. SciPy / `rscipy` — sparse & signal

| Status | Item |
|--------|------|
| done | `sparse.linalg`: `spsolve`, `cg` (CSR; spsolve densifies for v1) |
| done | Signal: `butter` / `filtfilt`, `welch` / `stft` (onesided mag) |
| pending | Fuller distributions beyond current specials |
| pending | `dblquad` / more ODE methods beyond RK45 |

## 3. Pandas / `rpandas` — time series & Arrow

| Status | Item |
|--------|------|
| pending | `DatetimeIndex` + `resample` |
| pending | Multi-key `join` / richer `merge` |
| pending | `Series.apply` / `map` |
| pending | Parquet / Arrow interop (with a future `rarrow` or via `std` + simple IPC) |
| pending | Categoricals |

## 4. Later crates (not started)

- **Polars**-shaped API (`rpolars`) and/or **PyArrow** (`rarrow`), as noted in [`README.md`](README.md)
- Remainder of [`python-ai-ml-tooling.md`](../python-ai-ml-tooling.md) (sklearn, HF, serving, …)

## Pass criteria (all slices)

- Parity checksums within op-specific tolerances vs Python
- Compare harness entry in the matching `*_parity` package
- Docs API map updated when an item lands

## Current focus

**2 → Fuller distributions** in `rscipy`, or **3 → Pandas DatetimeIndex / resample**.
