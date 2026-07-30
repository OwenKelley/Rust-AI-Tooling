"""Python ↔ Rust PyTorch parity + speed comparison harness."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .ops import prepare, run_op


OPS = [
    "zeros",
    "add",
    "mul",
    "matmul",
    "sum",
    "mean",
    "relu",
    "sigmoid",
    "transpose",
    "reshape",
    "linear_forward",
    "mse_loss",
    "train_step",
    "exp",
    "log",
    "pow",
    "clamp",
    "broadcast_add",
    "cat",
    "stack",
    "index_select",
    "softmax",
    "cross_entropy",
    "dropout",
    "sequential_forward",
    "adam_train_step",
    "embedding_forward",
    "layernorm_forward",
    "conv2d_forward",
    "adamw_train_step",
    "steplr",
    "tanh",
    "gelu",
    "batchnorm1d_forward",
    "max_pool2d_forward",
    "flatten_forward",
    "multisteplr",
    "batchnorm2d_forward",
    "avg_pool2d_forward",
    "cosineannealinglr",
    "dataloader_epoch",
]

REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_bin() -> Path:
    target_root = Path(
        __import__("os").environ.get("CARGO_TARGET_DIR", REPO_ROOT / "target")
    )
    candidates = [
        target_root / "release" / "torch_parity_runner.exe",
        target_root / "release" / "torch_parity_runner",
        REPO_ROOT / "target" / "release" / "torch_parity_runner.exe",
        REPO_ROOT / "target" / "release" / "torch_parity_runner",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def median_ns(samples: list[float]) -> float:
    return float(statistics.median(samples))


def time_python(op: str, size: int, seed: int, iters: int, warmup: int) -> dict:
    result, thunk = prepare(op, size, seed)
    from .ops import checksum

    cs = float(result) if isinstance(result, (float, int)) else checksum(result)
    for _ in range(warmup):
        thunk()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        thunk()
        samples.append(float(time.perf_counter_ns() - t0))
    return {
        "language": "python",
        "op": op,
        "checksum": cs,
        "median_ns": median_ns(samples),
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
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def nearly_equal(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def tolerances(op: str) -> tuple[float, float]:
    if op in ("exp", "log", "pow"):
        return 1e-3, 1e-4
    if op in (
        "train_step",
        "adam_train_step",
        "adamw_train_step",
        "linear_forward",
        "sequential_forward",
        "embedding_forward",
        "layernorm_forward",
        "batchnorm1d_forward",
        "batchnorm2d_forward",
        "conv2d_forward",
        "max_pool2d_forward",
        "avg_pool2d_forward",
        "mse_loss",
        "cross_entropy",
        "matmul",
        "sigmoid",
        "softmax",
        "gelu",
        "tanh",
        "dataloader_epoch",
    ):
        return 1e-4, 1e-5
    return 1e-5, 1e-6


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
        "parity_ok": ok,
        "python_checksum": py["checksum"],
        "rust_checksum": rs["checksum"],
        "python_median_ns": py["median_ns"],
        "rust_median_ns": rs["median_ns"],
        "speedup_rust_vs_python": speedup,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", nargs="*", default=OPS)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bin", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)
    bin_path = args.bin or _default_bin()
    if not bin_path.exists():
        print(f"missing rust binary: {bin_path}", file=sys.stderr)
        print(
            "build with: cargo build -p parity_runner --bin torch_parity_runner --release",
            file=sys.stderr,
        )
        return 2

    rows = []
    print(f"{'op':<18} {'parity':<8} {'py_ns':>12} {'rs_ns':>12} {'speedup':>10}")
    print("-" * 64)
    failures = 0
    for op in args.ops:
        row = compare_one(
            bin_path, op, args.size, args.seed, args.iters, args.warmup
        )
        rows.append(row)
        status = "OK" if row["parity_ok"] else "FAIL"
        if not row["parity_ok"]:
            failures += 1
            print(
                f"{row['op']:<18} {status:<8} {row['python_median_ns']:12.0f} "
                f"{row['rust_median_ns']:12.0f} {row['speedup_rust_vs_python']:9.2f}x"
            )
            print(
                f"  checksums py={row['python_checksum']!r} rs={row['rust_checksum']!r}"
            )
        else:
            print(
                f"{row['op']:<18} {status:<8} {row['python_median_ns']:12.0f} "
                f"{row['rust_median_ns']:12.0f} {row['speedup_rust_vs_python']:9.2f}x"
            )

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if failures:
        print(f"\n{failures} parity failure(s)")
        return 1
    print(f"\nAll {len(rows)} ops matched within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
