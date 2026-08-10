# Core Numerical — SciPy parity (Python ↔ Rust)

Second slice of the 1:1 Python→Rust translation work. Maps common **SciPy**
APIs onto the Rust crate `rscipy` (built on `rnumpy`), with correctness and
speed checks against SciPy.

## Layout

| Path | Role |
|------|------|
| `crates/rscipy` | Rust SciPy-like API (`special`, `linalg`, `optimize`, `stats`, `sparse`, `fft`, `signal`, `integrate`) |
| `crates/parity_runner` (`scipy_parity_runner`) | Times an op and prints JSON |
| `python/core_numerical/scipy_parity` | SciPy reference + comparison harness |

## API map

### `scipy.special` → `rscipy::special`

| Python | Rust |
|--------|------|
| `scipy.special.erf` / `erfc` | `rscipy::{erf,erfc}` |
| `scipy.special.gamma` / `gammaln` | `rscipy::{gamma,gammaln}` |
| `scipy.special.expit` / `logit` | `rscipy::{expit,logit}` |
| `scipy.special.logsumexp` | `rscipy::logsumexp` |
| `scipy.special.softmax` | `rscipy::softmax` |
| `scipy.special.i0` | `rscipy::i0` |
| `scipy.special.ndtr` / `ndtri` | `rscipy::{ndtr,ndtri}` |

### `scipy.linalg` → `rscipy::linalg`

| Python | Rust |
|--------|------|
| `scipy.linalg.lu` | `rscipy::lu` → `(P, L, U)` |
| `scipy.linalg.lu_factor` | `rscipy::lu_factor` → `(lu, piv)` |
| `scipy.linalg.cholesky` | `rscipy::cholesky` |
| `scipy.linalg.solve_triangular` | `rscipy::solve_triangular` |
| `scipy.linalg.lstsq` | `rscipy::lstsq` (incl. singular values) |
| `scipy.linalg.norm` | `rscipy::{norm,norm_ord}` |
| `scipy.linalg.expm` | `rscipy::expm` |

### `scipy.optimize` → `rscipy::optimize`

| Python | Rust |
|--------|------|
| `minimize(..., method='Nelder-Mead')` | `rscipy::minimize_nelder_mead` |
| `minimize(..., method='L-BFGS-B')` | `rscipy::minimize_lbfgsb` |
| `least_squares` | `rscipy::least_squares` (Levenberg–Marquardt) |

### `scipy.stats` → `rscipy::stats`

| Python | Rust |
|--------|------|
| `stats.norm.pdf` / `cdf` / `ppf` | `rscipy::{norm_pdf,norm_cdf,norm_ppf}` |
| `stats.entropy` | `rscipy::entropy` |
| `stats.zscore` | `rscipy::zscore` |
| `stats.rankdata` | `rscipy::rankdata` |
| `stats.pearsonr` / `spearmanr` | `rscipy::{pearsonr,spearmanr}` |
| `stats.ttest_ind` | `rscipy::ttest_ind` |
| `stats.skew` / `kurtosis` / `sem` | `rscipy::{skew,kurtosis,sem}` |

### `scipy.sparse` → `rscipy::sparse`

| Python | Rust |
|--------|------|
| `sparse.csr_matrix` / `csc_matrix` | `rscipy::{CsrMatrix,CscMatrix}` |
| `csr_matrix(dense)` / `toarray` | `rscipy::{csr_from_dense,csr_to_dense}` |
| `sparse.eye(..., format='csr')` | `rscipy::eye_csr` |
| `A @ x` / `A @ B` (CSR) | `rscipy::{csr_matvec,csr_matmat}` |
| `A.T` / `A.tocsc()` | `rscipy::{csr_transpose,csr_to_csc}` |
| `A + B` (CSR) | `rscipy::csr_add` |
| `sparse.linalg.norm(A)` | `rscipy::csr_frobenius_norm` |
| `sparse.linalg.spsolve` / `cg` | `rscipy::{spsolve,cg}` |

### `scipy.fft` → `rscipy::fft`

| Python | Rust |
|--------|------|
| `scipy.fft.fft` / `ifft` | `rscipy::{fft,ifft}` (complex as `[n,2]`) |
| `scipy.fft.rfft` / `irfft` | `rscipy::{rfft,irfft}` |
| `scipy.fft.fftfreq` / `rfftfreq` | `rscipy::{fftfreq,rfftfreq}` |
| `scipy.fft.fft2` | `rscipy::fft2` |

### `scipy.signal` → `rscipy::signal`

| Python | Rust |
|--------|------|
| `signal.convolve` / `fftconvolve` | `rscipy::{convolve,fftconvolve}` |
| `signal.correlate` | `rscipy::correlate` |
| `signal.windows.hann` / `hamming` / `blackman` | `rscipy::{hann,hamming,blackman}` |
| `signal.detrend` | `rscipy::detrend` |

### `scipy.integrate` → `rscipy::integrate`

| Python | Rust |
|--------|------|
| `integrate.trapezoid` | `rscipy::trapezoid` |
| `integrate.simpson` | `rscipy::simpson` |
| `integrate.cumulative_trapezoid` | `rscipy::cumulative_trapezoid` |
| `integrate.quad` | `rscipy::quad` (adaptive Simpson) |
| `integrate.solve_ivp(..., method='RK45')` | `rscipy::solve_ivp_rk45` |

## Setup

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --bin scipy_parity_runner --release
cargo test -p rscipy

cd python
pip install -e .
pytest core_numerical/scipy_parity
```

## Compare

```powershell
cargo build -p parity_runner --bin scipy_parity_runner --release
cd python
python -m core_numerical.scipy_parity.compare --size 64 --iters 20
```

## Pass criteria

- **Parity:** checksums within op-specific tolerances (tighter for algebra; looser for specials/optimizers)
- **Speed:** `python_median_ns / rust_median_ns`

## Next

Filters (`butter`/`filtfilt`), STFT/welch; fuller distributions; `dblquad` / more ODE methods.
