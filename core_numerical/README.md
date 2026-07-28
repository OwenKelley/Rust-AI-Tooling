# Core Numerical — NumPy parity (Python ↔ Rust)

First slice of the 1:1 Python→Rust translation work. This directory pair maps
common **NumPy** APIs onto the Rust crate `rnumpy`, then checks both **correctness**
and **speed**.

## Layout

| Path | Role |
|------|------|
| `crates/rnumpy` | Rust NumPy-like API (`zeros`, `add`, `matmul`, …) |
| `crates/parity_runner` | Release binary that times an op and prints JSON |
| `python/core_numerical/numpy_parity` | NumPy reference + comparison harness |
| `python/core_numerical/numpy_parity/compare.py` | Side-by-side parity + speedup table |

## API map (initial)

| Python (`numpy`) | Rust (`rnumpy`) |
|------------------|-----------------|
| `np.zeros(shape)` | `rnumpy::zeros(&[…])` |
| `np.ones(shape)` | `rnumpy::ones(&[…])` |
| `np.full(shape, v)` | `rnumpy::full(&[…], v)` |
| `np.arange(start, stop, step)` | `rnumpy::arange(start, stop, step)` |
| `np.linspace(start, stop, num)` | `rnumpy::linspace(start, stop, num)` |
| `np.eye(n)` | `rnumpy::eye(n)` |
| `np.add` / `subtract` / `multiply` / `divide` | `rnumpy::{add,subtract,multiply,divide}` |
| `np.power` / `sqrt` / `exp` / `log` | `rnumpy::{power,sqrt,exp,log}` |
| `np.negative` / `abs` | `rnumpy::{negative,abs}` |
| `np.sum` / `mean` / `min` / `max` | `rnumpy::{sum,mean,min,max}` |
| `np.var` / `std` (ddof=0) | `rnumpy::{var,std}` |
| `np.argmin` / `argmax` | `rnumpy::{argmin,argmax}` |
| `np.transpose` | `rnumpy::transpose` |
| `np.matmul` / `np.dot` | `rnumpy::{matmul,dot}` |

Shared inputs use the same LCG (`seeded_uniform`) on both sides so checksums
are comparable without depending on NumPy's RNG.

## Setup

Rust must be on your PATH (`%USERPROFILE%\.cargo\bin`). In a **new** PowerShell after installing rustup:

```powershell
# If `cargo` is still not found in this session:
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --release
cargo test -p rnumpy

# Python
cd python
pip install -e .
pytest
```

## Compare results + speed

```powershell
cargo build -p parity_runner --release
cd python
python -m core_numerical.numpy_parity.compare --size 256 --iters 50
```

Useful flags:

- `--ops add matmul sum` — subset of ops
- `--size 1024` — problem size
- `--json-out ..\results\numpy_parity.json` — save machine-readable report

Optional Rust-only microbench:

```powershell
cargo bench -p rnumpy
```

## Pass criteria

- **Parity:** Python and Rust checksums agree within `rtol=1e-7`, `atol=1e-8`
- **Speed:** reported as `python_median_ns / rust_median_ns` (higher ⇒ Rust faster)

## Speed notes

Performance work prefers **in-house `std` kernels** over extra crates:

- release `lto = fat`, `codegen-units = 1`, `target-cpu=native`
- contiguous slice scans for `min`/`max`/`argmin`/`argmax`
- `crates/rnumpy/src/gemm.rs`: blocked + `std::thread` parallel GEMM / dot
  (no rayon, no OpenBLAS/MKL)

`ndarray` remains only as the array **container** (still pulls `matrixmultiply`
transitively, but `matmul`/`dot` do not call into it). Replacing that storage
with a pure `Vec`+shape type is the next dependency-reduction step.

`transpose` still materializes an owned array; NumPy often returns an O(1) view.

## Next in Core Numerical

After NumPy coverage deepens (broadcasting, advanced indexing, `np.linalg.*`),
mirror the same harness pattern for SciPy, Pandas, Polars, and PyArrow.
