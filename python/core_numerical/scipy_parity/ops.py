"""SciPy reference ops used by the Python↔Rust parity harness."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import fft as sp_fft
from scipy import integrate as sp_integrate
from scipy import linalg, optimize, signal as sp_signal
from scipy import sparse, special, stats
from scipy.sparse import linalg as sparselinalg
from scipy.signal import windows as sp_windows

from core_numerical.numpy_parity.rng import seeded_uniform


def symmetric_spd(n: int, seed: int):
    a = seeded_uniform((n, n), seed, -1.0, 1.0)
    a = 0.5 * (a + a.T)
    for i in range(n):
        a[i, i] += float(n)
    return a


def diag_dominant(n: int, seed: int):
    a = seeded_uniform((n, n), seed, -1.0, 1.0)
    boost = min(float(n), 4.0)
    for i in range(n):
        a[i, i] += boost
    return a


def rosenbrock(x: np.ndarray) -> float:
    return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    a, b = float(x[0]), float(x[1])
    return np.array(
        [-2.0 * (1.0 - a) - 400.0 * a * (b - a * a), 200.0 * (b - a * a)],
        dtype=np.float64,
    )


def csr_from_threshold(a: np.ndarray, thresh: float = 0.5):
    """Match rscipy::csr_from_threshold — keep |v| >= thresh."""
    mask = np.abs(a) >= thresh
    return sparse.csr_matrix(a * mask)


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    """Build inputs once and return (reference_result, timed_thunk)."""
    n = size

    if op == "erf":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return special.erf(a)

        return thunk(), thunk

    if op == "erfc":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return special.erfc(a)

        return thunk(), thunk

    if op == "gamma":
        a = seeded_uniform((n,), seed, 0.2, 8.0)

        def thunk():
            return special.gamma(a)

        return thunk(), thunk

    if op == "gammaln":
        a = seeded_uniform((n,), seed, 0.2, 20.0)

        def thunk():
            return special.gammaln(a)

        return thunk(), thunk

    if op == "expit":
        a = seeded_uniform((n,), seed, -5.0, 5.0)

        def thunk():
            return special.expit(a)

        return thunk(), thunk

    if op == "logit":
        a = seeded_uniform((n,), seed, 0.05, 0.95)

        def thunk():
            return special.logit(a)

        return thunk(), thunk

    if op == "logsumexp":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return special.logsumexp(a)

        return thunk(), thunk

    if op == "softmax":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return special.softmax(a)

        return thunk(), thunk

    if op == "i0":
        a = seeded_uniform((n,), seed, 0.0, 5.0)

        def thunk():
            return special.i0(a)

        return thunk(), thunk

    if op == "ndtr":
        a = seeded_uniform((n,), seed, -3.0, 3.0)

        def thunk():
            return special.ndtr(a)

        return thunk(), thunk

    if op == "ndtri":
        a = seeded_uniform((n,), seed, 0.05, 0.95)

        def thunk():
            return special.ndtri(a)

        return thunk(), thunk

    if op == "lu":
        a = diag_dominant(n, seed)

        def thunk():
            return linalg.lu(a)

        return thunk(), thunk

    if op == "lu_factor":
        a = diag_dominant(n, seed)

        def thunk():
            return linalg.lu_factor(a)

        return thunk(), thunk

    if op == "cholesky":
        a = symmetric_spd(n, seed)

        def thunk():
            return linalg.cholesky(a, lower=True)

        return thunk(), thunk

    if op == "solve_triangular":
        a = symmetric_spd(n, seed)
        lower = linalg.cholesky(a, lower=True)
        b = seeded_uniform((n,), seed + 1, -1.0, 1.0)

        def thunk():
            return linalg.solve_triangular(lower, b, lower=True)

        return thunk(), thunk

    if op == "lstsq":
        m = 2 * n
        a = seeded_uniform((m, n), seed, -1.0, 1.0)
        b = seeded_uniform((m,), seed + 1, -1.0, 1.0)

        def thunk():
            return linalg.lstsq(a, b)

        return thunk(), thunk

    if op == "norm":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)

        def thunk():
            return linalg.norm(a)

        return thunk(), thunk

    if op == "norm_1":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)

        def thunk():
            return linalg.norm(a, ord=1)

        return thunk(), thunk

    if op == "norm_inf":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)

        def thunk():
            return linalg.norm(a, ord=np.inf)

        return thunk(), thunk

    if op == "expm":
        en = max(2, min(n, 6))
        a = seeded_uniform((en, en), seed, -0.5, 0.5)

        def thunk():
            return linalg.expm(a)

        return thunk(), thunk

    if op == "nelder_mead":
        x0 = np.array([-1.2, 1.0], dtype=np.float64)

        def thunk():
            return optimize.minimize(
                rosenbrock, x0, method="Nelder-Mead", tol=1e-8
            )

        return thunk(), thunk

    if op == "lbfgsb":
        x0 = np.array([-1.2, 1.0], dtype=np.float64)

        def thunk():
            return optimize.minimize(
                rosenbrock,
                x0,
                method="L-BFGS-B",
                jac=rosenbrock_grad,
                bounds=[(-2.0, 2.0), (-2.0, 2.0)],
            )

        return thunk(), thunk

    if op == "least_squares":
        xs = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        ys = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float64)

        def resid(p):
            return p[0] * xs + p[1] - ys

        def jac(p):
            return np.column_stack([xs, np.ones_like(xs)])

        def thunk():
            return optimize.least_squares(resid, [0.0, 0.0], jac=jac)

        return thunk(), thunk

    if op == "norm_pdf":
        a = seeded_uniform((n,), seed, -3.0, 3.0)

        def thunk():
            return stats.norm.pdf(a)

        return thunk(), thunk

    if op == "norm_cdf":
        a = seeded_uniform((n,), seed, -3.0, 3.0)

        def thunk():
            return stats.norm.cdf(a)

        return thunk(), thunk

    if op == "norm_ppf":
        a = seeded_uniform((n,), seed, 0.05, 0.95)

        def thunk():
            return stats.norm.ppf(a)

        return thunk(), thunk

    if op == "entropy":
        a = seeded_uniform((n,), seed, 0.1, 2.0)

        def thunk():
            return stats.entropy(a)

        return thunk(), thunk

    if op == "zscore":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return stats.zscore(a, ddof=0)

        return thunk(), thunk

    if op == "rankdata":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return stats.rankdata(a, method="average")

        return thunk(), thunk

    if op == "pearsonr":
        x = seeded_uniform((n,), seed, -1.0, 1.0)
        y = seeded_uniform((n,), seed + 1, -1.0, 1.0)

        def thunk():
            return stats.pearsonr(x, y)

        return thunk(), thunk

    if op == "spearmanr":
        x = seeded_uniform((n,), seed, -1.0, 1.0)
        y = seeded_uniform((n,), seed + 1, -1.0, 1.0)

        def thunk():
            return stats.spearmanr(x, y)

        return thunk(), thunk

    if op == "ttest_ind":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        b = seeded_uniform((n,), seed + 1, -1.0, 1.0)

        def thunk():
            return stats.ttest_ind(a, b, equal_var=True)

        return thunk(), thunk

    if op == "skew":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return stats.skew(a, bias=True)

        return thunk(), thunk

    if op == "kurtosis":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return stats.kurtosis(a, fisher=True, bias=True)

        return thunk(), thunk

    if op == "sem":
        a = seeded_uniform((n,), seed, -2.0, 2.0)

        def thunk():
            return stats.sem(a, ddof=1)

        return thunk(), thunk

    if op == "csr_from_dense":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)

        def thunk():
            return csr_from_threshold(a, 0.5)

        return thunk(), thunk

    if op == "csr_matvec":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        csr = csr_from_threshold(a, 0.5)
        x = seeded_uniform((n,), seed + 1, -1.0, 1.0)

        def thunk():
            return csr @ x

        return thunk(), thunk

    if op == "csr_matmat":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        csr = csr_from_threshold(a, 0.5)
        b = seeded_uniform((n, 8), seed + 1, -1.0, 1.0)

        def thunk():
            return csr @ b

        return thunk(), thunk

    if op == "csr_transpose":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        csr = csr_from_threshold(a, 0.5)

        def thunk():
            return csr.transpose().tocsc()

        return thunk(), thunk

    if op == "csr_add":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        ca = csr_from_threshold(a, 0.5)
        cb = csr_from_threshold(b, 0.5)

        def thunk():
            return ca + cb

        return (ca + cb).toarray(), thunk

    if op == "csr_eye":
        def thunk():
            return sparse.eye(n, format="csr")

        return sparse.eye(n, format="csr").toarray(), thunk

    if op == "csr_norm":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        csr = csr_from_threshold(a, 0.5)

        def thunk():
            return sparselinalg.norm(csr)

        return thunk(), thunk

    if op == "csr_to_csc":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        csr = csr_from_threshold(a, 0.5)

        def thunk():
            return csr.tocsc()

        return thunk(), thunk

    if op == "fft":
        a = seeded_uniform((n,), seed, -1.0, 1.0)

        def thunk():
            return sp_fft.fft(a)

        return thunk(), thunk

    if op == "ifft":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        spec = sp_fft.fft(a)

        def thunk():
            return sp_fft.ifft(spec)

        return thunk(), thunk

    if op == "rfft":
        a = seeded_uniform((n,), seed, -1.0, 1.0)

        def thunk():
            return sp_fft.rfft(a)

        return thunk(), thunk

    if op == "irfft":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        spec = sp_fft.rfft(a)

        def thunk():
            return sp_fft.irfft(spec, n=n)

        return thunk(), thunk

    if op == "fftfreq":

        def thunk():
            return sp_fft.fftfreq(n, d=1.0)

        return thunk(), thunk

    if op == "convolve":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        v = seeded_uniform((17,), seed + 1, -1.0, 1.0)

        def thunk():
            return sp_signal.convolve(a, v, mode="full")

        return thunk(), thunk

    if op == "fftconvolve":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        v = seeded_uniform((17,), seed + 1, -1.0, 1.0)

        def thunk():
            return sp_signal.fftconvolve(a, v, mode="full")

        return thunk(), thunk

    if op == "correlate":
        a = seeded_uniform((n,), seed, -1.0, 1.0)
        v = seeded_uniform((17,), seed + 1, -1.0, 1.0)

        def thunk():
            return sp_signal.correlate(a, v, mode="full")

        return thunk(), thunk

    if op == "hann":

        def thunk():
            return sp_windows.hann(n, sym=True)

        return thunk(), thunk

    if op == "hamming":

        def thunk():
            return sp_windows.hamming(n, sym=True)

        return thunk(), thunk

    if op == "blackman":

        def thunk():
            return sp_windows.blackman(n, sym=True)

        return thunk(), thunk

    if op == "detrend":
        a = seeded_uniform((n,), seed, -1.0, 1.0)

        def thunk():
            return sp_signal.detrend(a, type="linear")

        return thunk(), thunk

    if op == "trapezoid":
        y = seeded_uniform((n,), seed, -1.0, 1.0)

        def thunk():
            return sp_integrate.trapezoid(y, dx=1.0)

        return thunk(), thunk

    if op == "simpson":
        y = seeded_uniform((n,), seed, -1.0, 1.0)

        def thunk():
            return sp_integrate.simpson(y, dx=1.0)

        return thunk(), thunk

    if op == "cumulative_trapezoid":
        y = seeded_uniform((n,), seed, -1.0, 1.0)

        def thunk():
            return sp_integrate.cumulative_trapezoid(y, dx=1.0, initial=0.0)

        return thunk(), thunk

    if op == "quad":

        def thunk():
            return sp_integrate.quad(lambda x: np.exp(-(x * x)), 0.0, 1.0)

        return thunk(), thunk

    if op == "solve_ivp":
        n_pts = max(n // 4, 11)
        t_eval = np.arange(n_pts, dtype=np.float64) * 0.1
        tf = float(t_eval[-1])

        def fun(_t, y):
            return [y[1], -y[0]]

        def thunk():
            return sp_integrate.solve_ivp(
                fun,
                (0.0, tf),
                [1.0, 0.0],
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
            )

        return thunk(), thunk

    raise ValueError(f"unknown op: {op}")


def checksum(value: Any) -> float:
    if hasattr(value, "y") and hasattr(value, "t") and hasattr(value, "success"):
        # OdeResult from solve_ivp
        return float(np.sum(value.y))
    # SciPy SignificanceResult / TtestResult / PearsonRResult (often tuple subclasses)
    if hasattr(value, "statistic") and hasattr(value, "pvalue"):
        return float(value.statistic) + float(value.pvalue)
    if hasattr(value, "correlation") and hasattr(value, "pvalue"):
        return float(value.correlation) + float(value.pvalue)
    if isinstance(value, tuple) and len(value) == 2 and all(
        np.isscalar(v) or (isinstance(v, float)) for v in value
    ):
        # quad returns (integral, err) — checksum integral only
        return float(value[0])
    if hasattr(value, "x") and hasattr(value, "fun"):
        # OptimizeResult / LeastSquaresResult
        xsum = float(np.sum(value.x))
        fun = value.fun
        if np.ndim(fun) == 0:
            return xsum + float(fun)
        # least_squares stores residual vector in `.fun`
        return xsum + 0.5 * float(np.sum(np.asarray(fun, dtype=np.float64) ** 2))
    if sparse.issparse(value):
        # Match Rust checksums for sparse ops
        if value.format == "csr":
            dense = value.toarray()
            return float(np.sum(dense) + value.nnz)
        if value.format == "csc":
            return float(
                np.sum(value.data)
                + np.sum(value.indices.astype(np.float64))
                + np.sum(value.indptr.astype(np.float64))
            )
        dense = value.toarray()
        return float(np.sum(dense))
    if isinstance(value, tuple):
        if len(value) == 4:
            # lstsq: solution x only (Rust uses R-diag σ estimates)
            return float(np.sum(value[0]))
        if len(value) == 2:
            # lu_factor / pearsonr / spearmanr
            a, b = value
            return float(np.sum(a) + np.sum(b))
        return float(sum(np.sum(x) for x in value))
    if np.isscalar(value):
        return float(value)
    if np.iscomplexobj(value):
        return float(np.sum(np.real(value)) + np.sum(np.imag(value)))
    return float(np.sum(value))


def run_op(op: str, size: int, seed: int) -> float:
    result, _ = prepare(op, size, seed)
    return checksum(result)
