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
| `tensor.add_` / `mul_` / `sub_` / `relu_` / `zero_` / `fill_` | `rtorch::{add_,mul_,sub_,relu_,zero_,fill_}` |
| `torch.narrow` / `select` | `rtorch::{narrow,select}` (owned copies) |
| `exp` / `log` / `pow` / `neg` / `abs` / `clamp` | `rtorch::{exp,log,pow,neg,abs,clamp}` |
| `sum` / `mean` | `rtorch::{sum,mean}` |
| `reshape` / `t` | `rtorch::{reshape,transpose}` |
| `torch.cat` / `stack` / `index_select` | `rtorch::{cat,stack,index_select}` |
| `F.relu` / `leaky_relu` / `silu` / `sigmoid` / `tanh` / `gelu` / `softmax` | `rtorch::{relu,leaky_relu,silu,sigmoid,tanh,gelu,softmax}` |
| `F.scaled_dot_product_attention` | `rtorch::scaled_dot_product_attention` (+ `_masked`) |
| `nn.Transformer.generate_square_subsequent_mask` | `rtorch::generate_square_subsequent_mask` |
| `nn.Linear` / `Sequential` / `Embedding` / `Flatten` / `GRU` / `LSTM` / `MultiheadAttention` | `rtorch::{Linear,Sequential,Embedding,Flatten,GRU,LSTM,MultiheadAttention}` |
| `nn.TransformerEncoderLayer` / `TransformerEncoder` | `rtorch::{TransformerEncoderLayer,TransformerEncoder}` |
| `nn.TransformerDecoderLayer` / `TransformerDecoder` | `rtorch::{TransformerDecoderLayer,TransformerDecoder}` |
| `nn.LayerNorm` / `BatchNorm1d` / `BatchNorm2d` / `Conv2d` | `rtorch::{LayerNorm,BatchNorm1d,BatchNorm2d,Conv2d}` |
| `nn.AdaptiveAvgPool2d` / Max/Avg pool | `rtorch::{AdaptiveAvgPool2d,MaxPool2d,AvgPool2d}` |
| `nn.MSELoss` / `CrossEntropyLoss` | `rtorch::{MSELoss,CrossEntropyLoss}` |
| `loss.backward()` | `Tensor::backward` |
| `optim.SGD` / `Adam` / `AdamW` | `rtorch::{SGD,Adam,AdamW}` |
| `Adam.state_dict` / `load_state_dict` | `Adam::{state_dict,load_state_dict}` / `AdamStateDict` |
| `StepLR` / `MultiStepLR` / `CosineAnnealingLR` | `rtorch::{StepLR,MultiStepLR,CosineAnnealingLR}` |
| `state_dict` / `load_state_dict` (params) | `rtorch::{state_dict,load_state_dict}` |
| `TensorDataset` / `DataLoader` | `rtorch::{TensorDataset,DataLoader}` |
| `tensor.detach` / `torch.no_grad` | `Tensor::detach` / `rtorch::no_grad` |

**CPU `f32` only.** Dynamic reverse-mode autograd. No CUDA or `torch.compile`.

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

1. ~~**Tensor completeness**~~ — Phase 2 + in-place (`add_`/`mul_`/`relu_`/…) + `narrow`/`select` (owned) landed. Still open: dtypes, true strided views
2. **Autograd depth** — `create_graph`, custom `Function`, gradcheck
3. ~~**`nn` modules**~~ — Conv/Norm/Pool/Embedding/GRU/LSTM/MHA/Transformer Encoder+Decoder (+ causal mask) landed. Still open: key_padding_mask, nested tensors, …
4. ~~**Optimizers**~~ — Adam family + schedulers + param/optim `state_dict` landed
5. ~~**Data (first slice)**~~ — `TensorDataset` / `DataLoader` landed. Still open: samplers, collate, `rpandas`/`rnumpy` bridges
6. **Devices / perf** — device enum, optional GPU, fused kernels, AMP
7. **Ecosystem** — reference models, ONNX path, HF/Lightning notes

## Pass criteria

- **Parity:** checksums within op-specific tolerances (looser for train / matmul / transcendental)
- **Speed:** `python_median_ns / rust_median_ns`

## Performance notes

Hot-path fixes already in tree:

- Ops borrow `Vec<f32>` in place (no per-op full-buffer clone via `Tensor::data()`)
- Elementwise add/mul/sub/div and ReLU / `relu_` use AVX2 when available
- Broadcast binary ops fuse `(M,N)⊕(N,)` without materializing a full expand
- `cat` uses contiguous `copy_from_slice` chunks; transpose uses larger tiles + no zero-fill
- Sigmoid / tanh / gelu / silu use AVX2/FMA polynomial exp (tanh via `2*sigmoid(2x)-1`)
- `matmul` uses packed 4×8 f32 microkernels (AVX2/FMA)

Remaining gaps vs PyTorch at large `n`: owning tensors always copy on `reshape`/`transpose`/`broadcast` expand; large GEMM still trails MKL/OpenBLAS. Conv2d is naive nested loops.
