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
| `np.add` / `subtract` / `multiply` / `divide` | `rnumpy::{add,subtract,multiply,divide}` (broadcasting) |
| `np.sign` / `square` / `reciprocal` / `floor` / `ceil` / `trunc` / `round` | `rnumpy::{sign,square,reciprocal,floor,ceil,trunc,round}` |
| `np.greater` / `less` / `equal` / `not_equal` | `rnumpy::{greater,less,equal,not_equal}` (float 0/1 masks) |
| `np.cumsum` / `cumprod` (+ axis) | `rnumpy::{cumsum,cumsum_axis,cumprod}` |
| `np.reshape` (incl. `-1`) | `rnumpy::{reshape,reshape_infer}` |
| `np.swapaxes` / `moveaxis` | `rnumpy::{swapaxes,moveaxis}` |
| `np.transpose` / `swapaxes` | O(1) strided views (`transpose_view` / `swapaxes_view`) |
| `a[start:stop]` slicing | `rnumpy::slice_array` / `NdArray::slice` |
| `np.take` / `np.compress` | `rnumpy::{take,compress}` |
| `np.linalg.qr` / `svd` (values) / `eigvalsh` | `rnumpy::{qr,svdvals,eigvalsh}` |
| `astype(float32)` | `NdArray::astype_f32` / `NdArrayF32` |
| `np.linalg.solve` / `inv` / `det` | `rnumpy::{solve,inv,det}` |
| `np.sqrt` / `exp` / `log` / `sin` / `cos` / `tan` / `tanh` | `rnumpy::{sqrt,exp,log,sin,cos,tan,tanh}` |
| `np.negative` / `abs` / `clip` | `rnumpy::{negative,abs,clip}` |
| `np.where` | `rnumpy::where_` |
| `np.sum` / `mean` / `min` / `max` | `rnumpy::{sum,mean,min,max}` |
| `np.sum/mean/min/max(..., axis=)` | `rnumpy::{sum_axis,mean_axis,min_axis,max_axis}` |
| `np.var` / `std` (ddof=0) | `rnumpy::{var,std}` |
| `np.argmin` / `argmax` | `rnumpy::{argmin,argmax}` |
| `np.transpose` / `reshape` / `ravel` | `rnumpy::{transpose,reshape,ravel}` |
| `np.concatenate` / `stack` / `broadcast_to` | `rnumpy::{concatenate,stack,broadcast_to}` |
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

Timing for Rust ops is covered by `parity_runner` (and the Python compare harness);
there is no separate Criterion bench crate.

## Pass criteria

- **Parity:** Python and Rust checksums agree within `rtol=1e-7`, `atol=1e-8`
- **Speed:** reported as `python_median_ns / rust_median_ns` (higher ⇒ Rust faster)

## Speed notes

Performance work prefers **in-house `std` kernels** over extra crates:

- release `lto = fat`, `codegen-units = 1`, `target-cpu=native`
- contiguous slice scans for `min`/`max`/`argmin`/`argmax`
- `crates/rnumpy/src/gemm.rs`: in-house GEMM (`std` only)
  - pack each 8-col B panel once, then AVX2+FMA 4×8 microkernel
  - `std::thread` row parallelism only above a high flop threshold
    (avoids spawn-inside-column-loop overhead)
  - Goto A/B packing reserved for very large shapes
  - no rayon / OpenBLAS / MKL

Array storage is a local contiguous `NdArray` (`Vec<f64>` + shape) in
`crates/rnumpy/src/array.rs` — no `ndarray` / BLAS crate dependency.

`transpose` still materializes an owned array; NumPy often returns an O(1) view.

## Next in Core Numerical

Still ahead for deeper NumPy parity: full SVD with U/Vh, general (non-symmetric)
eigen, richer fancy indexing, and broader dtype coverage beyond f32/f64 cast.
After that, mirror the harness for SciPy, Pandas, Polars, and PyArrow.
