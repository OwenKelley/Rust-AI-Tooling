# Core Numerical — PyTorch parity (Python ↔ Rust)

Deep-learning slice of the 1:1 Python→Rust translation work. Maps common
**PyTorch** APIs onto the Rust crate `rtorch` (CPU `f32`, local/`std` only),
with correctness and speed checks against PyTorch.

## Layout

| Path | Role |
|------|------|
| `crates/rtorch` | Tensor + autograd + `nn` / optim |
| `crates/parity_runner` (`torch_parity_runner`) | Times an op and prints JSON |
| `python/core_numerical/torch_parity` | PyTorch reference + comparison harness |

## API map (v1 + Phase 2 + nn/optim slice)

| Python | Rust |
|--------|------|
| `torch.zeros/ones/full` / seeded uniforms | `rtorch::{zeros,ones,full,seeded_uniform,randn}` |
| `+ - * /` (broadcast), `matmul` | `rtorch::{add,sub,mul,div,matmul}` |
| `exp` / `log` / `pow` / `neg` / `abs` / `clamp` | `rtorch::{exp,log,pow,neg,abs,clamp}` |
| `sum` / `mean` | `rtorch::{sum,mean}` |
| `reshape` / `t` | `rtorch::{reshape,transpose}` |
| `torch.cat` / `stack` / `index_select` | `rtorch::{cat,stack,index_select}` |
| `F.relu` / `sigmoid` / `tanh` / `gelu` (tanh approx) / `softmax` / `log_softmax` | `rtorch::{relu,sigmoid,tanh,gelu,softmax,log_softmax}` |
| `F.dropout` / `F.max_pool2d` | `rtorch::{dropout,max_pool2d}` / `nn::{Dropout,MaxPool2d}` |
| `nn.Linear` / `Sequential` / `Embedding` / `Flatten` | `rtorch::{Linear,Sequential,Embedding,Flatten}` |
| `nn.LayerNorm` / `BatchNorm1d` / `Conv2d` (NCHW, s=1, p=0) | `rtorch::{LayerNorm,BatchNorm1d,Conv2d}` |
| `nn.MSELoss` / `CrossEntropyLoss` | `rtorch::{MSELoss,CrossEntropyLoss}` |
| `loss.backward()` | `Tensor::backward` |
| `optim.SGD` / `Adam` / `AdamW` | `rtorch::{SGD,Adam,AdamW}` |
| `lr_scheduler.StepLR` / `MultiStepLR` | `rtorch::{StepLR,MultiStepLR}` |
| `tensor.detach` / `torch.no_grad` | `Tensor::detach` / `rtorch::no_grad` |

**CPU `f32` only.** Dynamic reverse-mode autograd. No CUDA, DataLoader, or `torch.compile`.

## Setup

```powershell
$env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"

cargo build -p parity_runner --bin torch_parity_runner --release
cargo test -p rtorch

cd python
pip install -e .
pytest core_numerical/torch_parity
```

## Compare

```powershell
cargo build -p parity_runner --bin torch_parity_runner --release
cd python
python -m core_numerical.torch_parity.compare --size 64 --iters 20
```

## Roadmap (later phases)

1. ~~**Tensor completeness**~~ — Phase 2 landed. Still open: dtypes, in-place ops, views/slicing sugar
2. **Autograd depth** — `create_graph`, custom `Function`, gradcheck
3. ~~**`nn` modules**~~ — Embedding, LayerNorm, Conv2d, BatchNorm1d, MaxPool2d, Flatten, GELU/Tanh landed. Still open: BatchNorm2d, RNN/Transformer blocks, …
4. ~~**Optimizers**~~ — Adam / AdamW / StepLR / MultiStepLR landed. Still open: more schedulers, `state_dict`
5. **Data** — Dataset / DataLoader; bridges from `rpandas` / `rnumpy`
6. **Devices / perf** — device enum, optional GPU, fused kernels, AMP
7. **Ecosystem** — reference models, ONNX path, HF/Lightning notes

## Pass criteria

- **Parity:** checksums within op-specific tolerances (looser for train / matmul / transcendental)
- **Speed:** `python_median_ns / rust_median_ns`

## Performance notes

Hot-path fixes already in tree:

- Ops borrow `Vec<f32>` in place (no per-op full-buffer clone via `Tensor::data()`)
- Elementwise add/mul/sub/div and ReLU use AVX2 when available
- Sigmoid uses AVX2/FMA polynomial exp
- `matmul` uses packed 4×8 f32 microkernels (AVX2/FMA)

Remaining gaps vs PyTorch at large `n`: owning tensors always copy on `reshape`/`transpose`/`broadcast` expand; large GEMM still trails MKL/OpenBLAS. Conv2d is naive nested loops.
