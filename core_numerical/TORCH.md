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
| `torch.narrow` / `select` | `rtorch::{narrow,select}` (narrow is a strided view; select materializes) |
| `exp` / `log` / `pow` / `neg` / `abs` / `clamp` | `rtorch::{exp,log,pow,neg,abs,clamp}` |
| `sum` / `mean` | `rtorch::{sum,mean}` |
| `reshape` / `t` | `rtorch::{reshape,transpose}` |
| `torch.cat` / `stack` / `index_select` | `rtorch::{cat,stack,index_select}` |
| `F.relu` / `leaky_relu` / `silu` / `sigmoid` / `tanh` / `gelu` / `softmax` | `rtorch::{relu,leaky_relu,silu,sigmoid,tanh,gelu,softmax}` |
| `F.scaled_dot_product_attention` | `rtorch::scaled_dot_product_attention` (+ `_masked`) |
| `nn.Transformer.generate_square_subsequent_mask` | `rtorch::generate_square_subsequent_mask` |
| `nn.Linear` / `Sequential` / `Embedding` / `Flatten` / `GRU` / `LSTM` / `MultiheadAttention` | `rtorch::{Linear,Sequential,Embedding,Flatten,GRU,LSTM,MultiheadAttention}` (`forward_qkv_masked`) |
| `nn.TransformerEncoderLayer` / `TransformerEncoder` | `rtorch::{TransformerEncoderLayer,TransformerEncoder}` |
| `nn.TransformerDecoderLayer` / `TransformerDecoder` | `rtorch::{TransformerDecoderLayer,TransformerDecoder}` |
| `nn.LayerNorm` / `BatchNorm1d` / `BatchNorm2d` / `Conv2d` | `rtorch::{LayerNorm,BatchNorm1d,BatchNorm2d,Conv2d}` |
| `nn.AdaptiveAvgPool2d` / Max/Avg pool | `rtorch::{AdaptiveAvgPool2d,MaxPool2d,AvgPool2d}` |
| `nn.MSELoss` / `CrossEntropyLoss` | `rtorch::{MSELoss,CrossEntropyLoss}` |
| `loss.backward()` / `backward(create_graph=True)` | `Tensor::backward` / `backward_with` |
| `torch.autograd.grad` / `gradcheck` | `rtorch::{grad,gradcheck_max_error}` |
| `torch.autograd.Function` | `rtorch::{apply_function,FunctionCtx,square_function}` |
| `tensor.device` / `.to` / `.cpu` | `Tensor::{device,to,cpu}` / `rtorch::Device` (`Cpu` / `Cuda` stub) |
| `F.relu(F.linear(...))` fused | `rtorch::fused_linear_relu` |
| `torch.amp.GradScaler` / `autocast` | `rtorch::{GradScaler,autocast}` (f32 scale/unscale; no FP16 kernels) |
| `tensor.dtype` / `.float` / `.double` / `.long` / `.bool` / `.to(dtype)` | `Tensor::{dtype,float,double,long,bool_,to_dtype}` / `rtorch::Dtype` (`Float32`/`Float64`/`Int64`/`Bool`) |
| `torch.nested.nested_tensor` / `to_padded_tensor` | `rtorch::{nested_tensor,NestedTensor::to_padded_tensor}` |
| `torch.from_numpy` / `tensor.numpy` | `rtorch::{from_numpy,to_numpy}` / `Tensor::numpy` (via `rnumpy`) |
| `torch.tensor(df.values)` / `DataFrame(tensor)` | `rtorch::{from_dataframe,to_dataframe}` (via `rpandas`) |
| `optim.SGD` / `Adam` / `AdamW` | `rtorch::{SGD,Adam,AdamW}` |
| `Adam.state_dict` / `load_state_dict` | `Adam::{state_dict,load_state_dict}` / `AdamStateDict` |
| `StepLR` / `MultiStepLR` / `CosineAnnealingLR` | `rtorch::{StepLR,MultiStepLR,CosineAnnealingLR}` |
| `state_dict` / `load_state_dict` (params) | `rtorch::{state_dict,load_state_dict}` |
| `TensorDataset` / `DataLoader` / `SequentialSampler` / `RandomSampler` / `default_collate` | `rtorch::{TensorDataset,DataLoader,SequentialSampler,RandomSampler,default_collate}` |
| `tensor.detach` / `torch.no_grad` | `Tensor::detach` / `rtorch::no_grad` |

**CPU storage:** Float32/Float64 share `f32` buffers (Float64 is a dtype tag). Int64 and Bool use typed `i64` / `u8` (0/1) buffers. Dynamic reverse-mode autograd. No CUDA or `torch.compile`.

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

1. ~~**Tensor completeness**~~ — Phase 2 + in-place + `narrow`/`select` + `Dtype::{Float32,Float64,Int64,Bool}` / `.dtype()` / `.to_dtype()` / `.float()` / `.double()` / `.long()` / `.bool_()` + true strided views (`transpose`/`reshape`/`narrow` share storage when legal; hot-path ops materialize via `contiguous()`). Int64/Bool use typed buffers (`TensorStorage::{I64,Bool}`); Float64 remains f32-backed tagging. Zero-copy views are F32-only in this slice (non-F32 reshape/transpose/narrow materialize).
2. ~~**Autograd depth**~~ — `grad` / `create_graph` (…/pools / CrossEntropy / Chunk·narrow) + full LayerNorm Hessian via differentiable op composition + `gradcheck_max_error` + custom `Function` landed
3. ~~**`nn` modules**~~ — Conv/Norm/Pool/Embedding/GRU/LSTM/MHA/Transformer Encoder+Decoder (+ causal / key_padding masks) + `NestedTensor` / `to_padded_tensor` landed
4. ~~**Optimizers**~~ — Adam family + schedulers + param/optim `state_dict` landed
5. ~~**Data**~~ — `TensorDataset` / `DataLoader` + samplers + `default_collate` + `rnumpy`/`rpandas` bridges (`from_numpy` / `to_numpy` / `from_dataframe` / `to_dataframe`) landed
6. ~~**Devices / perf**~~ — `Device::{Cpu,Cuda}` (CUDA API-only; `.to(cuda)` panics) + `fused_linear_relu` + `GradScaler` / `autocast` scaffolding (CPU f32; no FP16 kernels) landed
7. ~~**Ecosystem**~~ — `examples/reference_mlp.rs` + notes below (not a full ONNX/HF runtime)

## Ecosystem

Reference model:

```powershell
cargo run -p rtorch --example reference_mlp --release
```

Linear → ReLU → Linear with CrossEntropy + Adam for a few steps; also exercises `fused_linear_relu` under `no_grad`.

**ONNX (conceptual):** export by walking `state_dict()` keys (`weight`/`bias` per module) into an ONNX graph of Gemm/Relu/Softmax nodes; rtorch does not ship an ONNX writer. Keep parameter shapes identical to PyTorch (`Linear`: `[out,in]`).

**Hugging Face:** map `from_pretrained` weight names onto `state_dict` / `load_state_dict` (e.g. `encoder.layer.*.linear1.weight` → nested `Sequential` / custom module names). Transpose rules match PyTorch (`nn.Linear` weight layout).

**Lightning-style loop:** use `TensorDataset` + `DataLoader`, `Module::forward`, `loss.backward()`, optimizer `step` / `zero_grad`, optional `GradScaler` around the loss. Schedulers (`StepLR`, …) wrap the optimizer the same way as PyTorch.

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

Remaining gaps vs PyTorch at large `n`: broadcast expand still materializes; large GEMM still trails MKL/OpenBLAS. Conv2d is naive nested loops. Views are zero-copy when legal (`transpose`/`reshape`/`narrow`); non-contiguous inputs are gathered at op boundaries.
