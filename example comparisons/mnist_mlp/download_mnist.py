"""Download MNIST IDX files into ./data (next to this script)."""

from __future__ import annotations

import gzip
import hashlib
import sys
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Mirror used widely by tutorials / torchvision-compatible layout.
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / name
    raw_path = DATA_DIR / name.replace(".gz", "")
    if raw_path.exists() and raw_path.stat().st_size > 0:
        return raw_path
    url = f"{BASE}/{name}"
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as src, open(raw_path, "wb") as dst:
        dst.write(src.read())
    gz_path.unlink(missing_ok=True)
    return raw_path


def main() -> int:
    for name in FILES:
        path = _download(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        print(f"ok {path.name}  sha256={digest}…  bytes={path.stat().st_size}")
    print(f"MNIST ready under {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
