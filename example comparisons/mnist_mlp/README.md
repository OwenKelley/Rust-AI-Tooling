# MNIST MLP — PyTorch vs rtorch

Train a small MLP on MNIST with matching Python and Rust programs, then compare
**wall-clock** train+eval time on CPU.

## Model (both sides)

```
Flatten 784
→ Linear(784, 128) + ReLU
→ Linear(128, 10)
→ CrossEntropyLoss
→ Adam(lr=1e-3)
```

Defaults: `epochs=25`, `batch_size=128`, `seed=42`, full train/test splits.

## Layout

| Path | Role |
|------|------|
| `data/` | MNIST IDX files (downloaded once) |
| `download_mnist.py` | Fetches + decompresses IDX into `data/` |
| `python/train_mnist.py` | PyTorch training loop |
| `rust/` | rtorch crate (`cargo run --release`) |
| `run_compare.ps1` | Download (if needed), run both, print timings |

## Setup

```powershell
# From repo root
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$env:CARGO_TARGET_DIR = "target/example_comparisons"

# MNIST (~11 MB compressed); writes into this folder's data/
python "example comparisons/mnist_mlp/download_mnist.py"

# Optional: torch for the Python side
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run one side

```powershell
# Python
python "example comparisons/mnist_mlp/python/train_mnist.py" --epochs 25

# Rust (release)
cargo run --release --manifest-path "example comparisons/mnist_mlp/rust/Cargo.toml"
```

## Compare wall clock

```powershell
powershell -File "example comparisons/mnist_mlp/run_compare.ps1"
```

Optional: `-Epochs 5 -BatchSize 128`

## How to read the result

- Both load the **same IDX files**, use the **same batch size / epochs / seed** for
  shuffling, and print the same metrics (`train_loss`, `val_acc`, `wall_sec`).
- Weight init uses each library’s RNG (not bit-identical), so accuracy curves may
  differ slightly; the comparison target is **runtime for a fixed training recipe**.
- This is CPU-only. It is closer to a real loop than `SPEED.md` microbenchmarks
  (tiny ops / Python dispatch), and is the better check that rtorch’s train stack
  is actually fast.
