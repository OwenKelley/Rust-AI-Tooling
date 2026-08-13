"""Python ↔ Rust SciPy parity + speed comparison harness."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .ops import checksum, prepare


OPS = [
    "erf",
    "erfc",
    "gamma",
    "gammaln",
    "expit",
    "logit",
    "logsumexp",
    "softmax",
    "i0",
    "ndtr",
    "ndtri",
    "lu",
    "lu_factor",
    "cholesky",
    "solve_triangular",
    "lstsq",
    "norm",
    "norm_1",
    "norm_inf",
    "expm",
    "nelder_mead",
    "lbfgsb",
    "least_squares",
    "norm_pdf",
    "norm_cdf",
    "norm_ppf",
    "uniform_pdf",
    "uniform_cdf",
    "uniform_ppf",
    "expon_pdf",
    "expon_cdf",
    "expon_ppf",
    "laplace_pdf",
    "laplace_cdf",
    "laplace_ppf",
    "logistic_pdf",
    "logistic_cdf",
    "logistic_ppf",
    "t_pdf",
    "t_cdf",
    "t_ppf",
    "chi2_pdf",
    "chi2_cdf",
    "chi2_ppf",
    "gamma_pdf",
    "gamma_cdf",
    "gamma_ppf",
    "beta_pdf",
    "beta_cdf",
    "beta_ppf",
    "poisson_pmf",
    "poisson_cdf",
    "binom_pmf",
    "binom_cdf",
    "entropy",
    "zscore",
    "rankdata",
    "pearsonr",
    "spearmanr",
    "ttest_ind",
    "skew",
    "kurtosis",
    "sem",
    "csr_from_dense",
    "csr_matvec",
    "csr_matmat",
    "csr_transpose",
    "csr_add",
    "csr_eye",
    "csr_norm",
    "csr_to_csc",
    "spsolve",
    "cg",
    "butter",
    "filtfilt",
    "welch",
    "stft",
    "fft",
    "ifft",
    "rfft",
    "irfft",
    "fftfreq",
    "convolve",
    "fftconvolve",
    "correlate",
    "hann",
    "hamming",
    "blackman",
    "detrend",
    "trapezoid",
    "simpson",
    "cumulative_trapezoid",
    "quad",
    "dblquad",
    "solve_ivp",
    "solve_ivp_rk23",
]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_bin() -> Path:
    target_root = Path(
        __import__("os").environ.get("CARGO_TARGET_DIR", REPO_ROOT / "target")
    )
    candidates = [
        target_root / "release" / "scipy_parity_runner.exe",
        target_root / "release" / "scipy_parity_runner",
        REPO_ROOT / "target" / "release" / "scipy_parity_runner.exe",
        REPO_ROOT / "target" / "release" / "scipy_parity_runner",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DEFAULT_BIN = _default_bin()


def median_ns(samples: list[int]) -> int:
    s = sorted(samples)
    n = len(s)
    if n == 0:
        return 0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def time_python(op: str, size: int, seed: int, iters: int, warmup: int) -> dict:
    result, thunk = prepare(op, size, seed)
    cs = checksum(result)

    for _ in range(warmup):
        thunk()

    samples: list[int] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        thunk()
        samples.append(time.perf_counter_ns() - t0)

    return {
        "language": "python",
        "op": op,
        "size": size,
        "iters": iters,
        "warmup": warmup,
        "seed": seed,
        "median_ns": median_ns(samples),
        "mean_ns": statistics.fmean(samples),
        "min_ns": min(samples),
        "max_ns": max(samples),
        "checksum": cs,
    }


def time_rust(
    bin_path: Path, op: str, size: int, seed: int, iters: int, warmup: int
) -> dict:
    cmd = [
        str(bin_path),
        "--op",
        op,
        "--size",
        str(size),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def nearly_equal(
    a: float | None, b: float | None, rtol: float = 1e-7, atol: float = 1e-8
) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if not math.isfinite(a) or not math.isfinite(b):
        return math.isnan(a) and math.isnan(b) or (
            math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0)
        )
    return abs(a - b) <= atol + rtol * abs(b)


def tolerances(op: str) -> tuple[float, float]:
    if op in (
        "erf",
        "erfc",
        "ndtr",
        "i0",
        "norm_pdf",
        "norm_cdf",
        "norm_ppf",
        "uniform_pdf",
        "uniform_cdf",
        "uniform_ppf",
        "expon_pdf",
        "expon_cdf",
        "expon_ppf",
        "laplace_pdf",
        "laplace_cdf",
        "laplace_ppf",
        "logistic_pdf",
        "logistic_cdf",
        "logistic_ppf",
        "t_pdf",
        "t_cdf",
        "t_ppf",
        "chi2_pdf",
        "chi2_cdf",
        "chi2_ppf",
        "gamma_pdf",
        "gamma_cdf",
        "gamma_ppf",
        "beta_pdf",
        "beta_cdf",
        "beta_ppf",
        "poisson_pmf",
        "poisson_cdf",
        "binom_pmf",
        "binom_cdf",
    ):
        return 2e-5, 2e-5
    if op in ("expm",):
        return 1e-5, 1e-6
    if op in ("nelder_mead", "lbfgsb", "least_squares"):
        return 1e-5, 1e-6
    if op in ("fft", "ifft", "rfft", "irfft", "fftconvolve"):
        return 1e-6, 1e-7
    if op in ("quad", "dblquad", "solve_ivp", "solve_ivp_rk23"):
        return 1e-4, 1e-5
    if op in ("pearsonr", "spearmanr", "ttest_ind"):
        return 1e-6, 1e-7
    if op in ("lu_factor",):
        return 1e-7, 1e-8
    return 1e-7, 1e-8


def compare_one(
    bin_path: Path, op: str, size: int, seed: int, iters: int, warmup: int
) -> dict:
    rtol, atol = tolerances(op)
    py = time_python(op, size, seed, iters, warmup)
    rs = time_rust(bin_path, op, size, seed, iters, warmup)
    ok = nearly_equal(py["checksum"], rs["checksum"], rtol=rtol, atol=atol)
    speedup = py["median_ns"] / rs["median_ns"] if rs["median_ns"] else float("inf")
    return {
        "op": op,
        "size": size,
        "parity_ok": ok,
        "python_checksum": py["checksum"],
        "rust_checksum": rs["checksum"],
        "python_median_ns": py["median_ns"],
        "rust_median_ns": rs["median_ns"],
        "speedup_rust_vs_python": speedup,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", nargs="*", default=OPS, help="Ops to compare")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bin",
        type=Path,
        default=DEFAULT_BIN,
        help="Path to scipy_parity_runner binary",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write full JSON results",
    )
    args = parser.parse_args(argv)

    if not args.bin.exists():
        print(
            f"scipy_parity_runner not found at {args.bin}\n"
            f"Build it with: cargo build -p parity_runner --bin scipy_parity_runner --release",
            file=sys.stderr,
        )
        return 2

    rows = []
    failures = 0
    print(
        f"{'op':<18} {'parity':<8} {'py_ns':>12} {'rs_ns':>12} {'speedup':>10}"
    )
    print("-" * 64)
    for op in args.ops:
        row = compare_one(
            args.bin, op, args.size, args.seed, args.iters, args.warmup
        )
        rows.append(row)
        if not row["parity_ok"]:
            failures += 1
        print(
            f"{row['op']:<18} "
            f"{'OK' if row['parity_ok'] else 'FAIL':<8} "
            f"{row['python_median_ns']:>12} "
            f"{row['rust_median_ns']:>12} "
            f"{row['speedup_rust_vs_python']:>10.2f}x"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    if failures:
        print(f"\n{failures} parity failure(s)", file=sys.stderr)
        return 1
    print(f"\nAll {len(rows)} ops matched within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
