"""Python ↔ Rust rtokenizers parity harness."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from .ops import checksum, prepare

OPS = ["whitespace", "bpe", "wordpiece"]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _default_bin() -> Path:
    target_root = Path(
        __import__("os").environ.get("CARGO_TARGET_DIR", REPO_ROOT / "target")
    )
    for path in [
        target_root / "release" / "tokenizers_parity_runner.exe",
        target_root / "release" / "tokenizers_parity_runner",
        REPO_ROOT / "target" / "release" / "tokenizers_parity_runner.exe",
        REPO_ROOT / "target" / "release" / "tokenizers_parity_runner",
    ]:
        if path.exists():
            return path
    raise FileNotFoundError("tokenizers_parity_runner not found; build with cargo")


def time_python(op: str, size: int, seed: int, iters: int, warmup: int) -> dict:
    result, thunk = prepare(op, size, seed)
    for _ in range(warmup):
        thunk()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        thunk()
        samples.append(time.perf_counter_ns() - t0)
    return {"checksum": checksum(result), "median_ns": statistics.median(samples)}


def time_rust(bin_path: Path, op: str, size: int, seed: int, iters: int, warmup: int) -> dict:
    cmd = [
        str(bin_path),
        "--op",
        op,
        "--size",
        str(size),
        "--seed",
        str(seed),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def nearly_equal(a: float, b: float, rtol: float = 1e-7, atol: float = 1e-8) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--ops", nargs="*", default=OPS)
    p.add_argument("--bin", type=Path, default=None)
    args = p.parse_args()
    bin_path = args.bin or _default_bin()
    rows = []
    print(f"{'op':22} {'parity':8} {'py_ns':>12} {'rs_ns':>12} {'speedup':>10}")
    print("-" * 68)
    for op in args.ops:
        py = time_python(op, args.size, args.seed, args.iters, args.warmup)
        rs = time_rust(bin_path, op, args.size, args.seed, args.iters, args.warmup)
        ok = nearly_equal(py["checksum"], rs["checksum"])
        speedup = py["median_ns"] / rs["median_ns"] if rs["median_ns"] else float("inf")
        rows.append(ok)
        status = "OK" if ok else "FAIL"
        print(
            f"{op:22} {status:8} {py['median_ns']:12.0f} {rs['median_ns']:12.0f} {speedup:10.2f}x"
        )
        if not ok:
            print(f"  checksums py={py['checksum']} rs={rs['checksum']}")
    n_ok = sum(1 for r in rows if r)
    print()
    print(f"{n_ok}/{len(rows)} ops matched within tolerance.")
    if n_ok != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
