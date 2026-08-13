# MNIST MLP — PyTorch vs RusTorch (naive + fast)

Train a small MLP on MNIST and compare **wall-clock** train+eval time on CPU.

Two Rust modes:

| Mode | Flag | Meaning |
|------|------|---------|
| **naive** | `--mode naive` | 1:1 translation of the Python trainer (`Linear` → `ReLU` → `Linear`, `CrossEntropyLoss`, `zero_grad` + `step`, same LCG index shuffle + row gather) |
| **fast** | `--mode fast` | Same data recipe; fused helpers only (`forward_relu`, `forward_cross_entropy`, `step_and_zero_grad`) |

Both modes share the same shuffle/gather path as Python so the gap is API fusion, not a different I/O strategy.

## Model (all sides)

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
| `rust/` | RusTorch crate (`--mode naive|fast`) |
| `run_compare.ps1` | Download (if needed), run all three, print timings |

## Setup

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
$env:CARGO_TARGET_DIR = "target/example_comparisons"

python "example comparisons/mnist_mlp/download_mnist.py"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run one side

```powershell
python "example comparisons/mnist_mlp/python/train_mnist.py" --epochs 25

# Save a checkpoint (state_dict + train metadata) for reuse elsewhere:
python "example comparisons/mnist_mlp/python/train_mnist.py" --epochs 25 --save path\to\mlp_mnist.pt

cargo run --release --manifest-path "example comparisons/mnist_mlp/rust/Cargo.toml" -- --mode naive
cargo run --release --manifest-path "example comparisons/mnist_mlp/rust/Cargo.toml" -- --mode fast

# RusTorch: save weights / run inference-only
cargo run --release --manifest-path "example comparisons/mnist_mlp/rust/Cargo.toml" -- `
  --mode naive --epochs 25 --seed 0 --save path\to\seed_00.bin
cargo run --release --manifest-path "example comparisons/mnist_mlp/rust/Cargo.toml" -- `
  --infer --mode naive --checkpoint path\to\seed_00.bin --passes 50
```

## Compare wall clock

```powershell
powershell -File "example comparisons/mnist_mlp/run_compare.ps1"
```

Optional: `-Epochs 5 -BatchSize 128`

## How to read the result

- **naive vs PyTorch**: same call shape; shows default-library speed for a line-by-line port.
- **fast vs PyTorch**: upper bound when using rustorch train helpers.
- Weight init is not bit-identical across libraries; the target metric is **runtime**.
- CPU-only; closer to a real train loop than `SPEED.md` microbenchmarks.
