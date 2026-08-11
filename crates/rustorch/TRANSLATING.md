# PyTorch → RusTorch translation guide

Cheat sheet for porting common PyTorch (Python) code to **RusTorch** (crate `rustorch`).

**Scope today:** CPU, primarily `f32`, PyTorch-shaped APIs. Not every PyTorch API exists; prefer the tables below over guessing from names alone.

**Import style**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```rust
use rustorch::{
    add, matmul, mean, mul, no_grad, relu, sum, zeros, Adam, CrossEntropyLoss, Linear, Module,
    ReLU, Tensor,
};
// Or pull from modules: rustorch::nn::*, rustorch::ops::*, rustorch::functional::*
```

---

## Mental model differences

| Topic | PyTorch | RusTorch |
|-------|---------|--------|
| Ownership | Tensors are references / GC | Pass `&Tensor`; clones share storage via `Rc` |
| Ops | Methods + operators (`a + b`, `x.relu()`) | Free functions (`add(&a, &b)`, `relu(&x)`) |
| Grad flag | `requires_grad=True` on ctor / `.requires_grad_()` | `requires_grad: bool` on ctor, or `t.set_requires_grad(true)` |
| Devices | `.to("cuda")` etc. | CPU only (`Device::Cpu`) |
| Dtypes | many | Float32 primary; some Int64/Bool for indexing / masks |
| Modules | `nn.Module` subclass | Struct + `Module` trait (`forward`, `parameters`) |
| Class labels | `LongTensor` | `&[usize]` for `CrossEntropyLoss` / `cross_entropy` |
| RNG | global / generator | Explicit `seed: u64` on many constructors |
| `no_grad` | `with torch.no_grad():` | `no_grad(\|\| { ... })` |

---

## Tensors: create and inspect

| PyTorch | RusTorch |
|---------|--------|
| `torch.zeros(2, 3)` | `zeros(&[2, 3], false)` |
| `torch.ones(2, 3, requires_grad=True)` | `ones(&[2, 3], true)` |
| `torch.full((2, 3), 0.5)` | `full(&[2, 3], 0.5, false)` |
| `torch.randn(2, 3)` | `randn(&[2, 3], seed, false)` |
| `torch.tensor([1., 2., 3.])` | `Tensor::from_vec(vec![1., 2., 3.], &[3], false)` |
| `x.shape` / `x.size()` | `x.shape()` → `Vec<usize>` |
| `x.numel()` | `x.numel()` |
| `x.item()` | `x.item()` |
| `x.requires_grad_(True)` | `x.set_requires_grad(true)` |
| `x.detach()` | `x.detach()` |
| `x.data` / `.tolist()` | `x.data()` → `Vec<f32>`, or `x.with_data(|s| ...)` |
| `x.grad` | `x.grad()` → `Option<Vec<f32>>` |

```python
x = torch.randn(4, 8, requires_grad=True)
y = x.sum()
y.backward()
g = x.grad
```

```rust
let x = randn(&[4, 8], 0, true);
let y = sum(&x);
y.backward();
let g = x.grad(); // Option<Vec<f32>>
```

---

## Elementwise and reductions

| PyTorch | RusTorch |
|---------|--------|
| `a + b` | `a + b` / `&a + &b` (or `add(&a, &b)`) |
| `a - b` | `a - b` (or `sub(&a, &b)`) |
| `a * b` | `a * b` **elementwise** (or `mul`; use `a.matmul(&b)` for matmul) |
| `a / b` | `a / b` (or `div(&a, &b)`) |
| `-a` | `-a` / `-&a` (or `neg(&a)`) |
| `a += b` | `a += b` (or `add_(&a, &b)`) |
| `a -= b` / `a *= b` | `a -= b` / `a *= b` |
| `a.abs()` | `a.abs()` (or `abs(&a)`) |
| `a.exp()` / `a.log()` | `a.exp()` / `a.log()` |
| `a.pow(b)` | `a.pow(&b)` (or `pow(&a, &b)`) |
| `a.clamp(lo, hi)` | `clamp(&a, lo, hi)` |
| `a.sum()` | `a.sum()` (or `sum(&a)`) |
| `a.mean()` | `a.mean()` (or `mean(&a)`) |
| `a.zero_()` | `zero_(&a)` |
| `a.fill_(v)` | `fill_(&a, v)` |
| `F.relu(a)` / `a.relu_()` | `relu(&a)` / `relu_(&a)` |

Broadcasting is supported for binary ops when shapes are compatible (PyTorch-like rules).

Scalar–tensor forms like `a + 1.0` are not overloaded yet; use `add(&a, &full(&a.shape(), 1.0, false))` or similar.

---

## Linear algebra and shape ops

| PyTorch | RusTorch |
|---------|--------|
| `a @ b` / `torch.matmul(a, b)` | `a.matmul(&b)` (or `matmul(&a, &b)`) |
| `torch.bmm(a, b)` | `a.bmm(&b)` (or `bmm(&a, &b)`) |
| `a.t()` / `a.transpose(0, 1)` (2D) | `a.t()` / `a.transpose()` |
| `a.reshape(...)` / `a.view(...)` | `a.reshape(&[...])` / `a.view(&[...])` |
| `a.permute(0, 2, 1)` | `permute(&a, &[0, 2, 1])` |
| `torch.cat([a, b], dim=0)` | `cat(&[&a, &b], 0)` |
| `torch.stack([a, b], dim=0)` | `stack(&[&a, &b], 0)` |
| `torch.chunk(a, 2, dim=0)` | `chunk(&a, 2, 0)` |
| `a.narrow(0, start, len)` | `narrow(&a, 0, start, len)` |
| `a.select(0, i)` | `select(&a, 0, i)` |
| `a.index_select(0, idx)` | `index_select(&a, 0, &indices)` where `indices: &[usize]` |

**Batch row gather (train loops):**

| PyTorch | RusTorch |
|---------|--------|
| `x[batch_idx]` (2D rows) | `gather_rows(&x, &batch_idx)` |

`gather_rows` skips building an autograd node when `x` does not require grad (typical for inputs).

---

## `torch.nn.functional` → `rustorch::functional`

| PyTorch | RusTorch |
|---------|--------|
| `F.relu(x)` | `relu(&x)` |
| `F.leaky_relu(x, 0.01)` | `leaky_relu(&x, 0.01)` |
| `F.sigmoid(x)` | `sigmoid(&x)` |
| `F.tanh(x)` | `tanh(&x)` |
| `F.gelu(x)` | `gelu(&x)` |
| `F.silu(x)` | `silu(&x)` |
| `F.softmax(x, dim=-1)` (2D) | `softmax(&x)` |
| `F.log_softmax(x, dim=-1)` (2D) | `log_softmax(&x)` |
| `F.linear(x, w, b)` | `linear(&x, &w, Some(&b))` |
| `F.mse_loss(pred, target)` | `mse_loss(&pred, &target)` |
| `F.cross_entropy(logits, y)` | `cross_entropy(&logits, &y_usize)` |
| `F.dropout(x, p, training)` | `dropout(&x, p, train, seed)` |
| `F.scaled_dot_product_attention(q,k,v)` | `scaled_dot_product_attention(&q, &k, &v)` |

---

## `torch.nn` modules

Construct modules, keep them in a struct, implement `Module` (or call `.forward` directly).

| PyTorch | RusTorch |
|---------|--------|
| `nn.Linear(in_f, out_f)` | `Linear::new(in_f, out_f, true, seed)` |
| `nn.ReLU()` | `ReLU` |
| `nn.LeakyReLU(0.01)` | `LeakyReLU::new(0.01)` |
| `nn.Sigmoid()` / `nn.Tanh()` | `Sigmoid` / `Tanh` |
| `nn.GELU()` / `nn.SiLU()` | `GELU` / `SiLU` |
| `nn.Softmax(dim=-1)` | `Softmax` |
| `nn.Dropout(p)` | `Dropout::new(p, seed)` |
| `nn.Embedding(n, d)` | `Embedding::...` |
| `nn.LayerNorm(d)` | `LayerNorm::...` |
| `nn.BatchNorm1d` / `2d` | `BatchNorm1d` / `BatchNorm2d` |
| `nn.Conv2d(...)` | `Conv2d::...` |
| `nn.MaxPool2d` / `AvgPool2d` | `MaxPool2d` / `AvgPool2d` |
| `nn.AdaptiveAvgPool2d` | `AdaptiveAvgPool2d` |
| `nn.Flatten()` | `Flatten` |
| `nn.GRU` / `nn.LSTM` | `GRU` / `LSTM` |
| `nn.MultiheadAttention` | `MultiheadAttention` |
| `nn.TransformerEncoder` / `Decoder` | `TransformerEncoder` / `TransformerDecoder` (+ layers) |
| `nn.Sequential(...)` | `Sequential { modules: vec![...] }` |
| `nn.ModuleList` | `ModuleList` |
| `nn.MSELoss()` | `MSELoss` |
| `nn.CrossEntropyLoss()` | `CrossEntropyLoss` |

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
```

```rust
struct Mlp {
    fc1: Linear,
    relu: ReLU,
    fc2: Linear,
}

impl Mlp {
    fn new(seed: u64) -> Self {
        Self {
            fc1: Linear::new(784, 128, true, seed),
            relu: ReLU,
            fc2: Linear::new(128, 10, true, seed + 100),
        }
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.relu.forward(&self.fc1.forward(x)))
    }
    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }
}
```

**Loss call shape**

```python
loss = nn.CrossEntropyLoss()(logits, y)  # y: LongTensor
```

```rust
let loss = CrossEntropyLoss.forward(&logits, &y); // y: &[usize] or Vec<usize>
```

---

## Optimizers and LR schedules

| PyTorch | RusTorch |
|---------|--------|
| `optim.SGD(params, lr=...)` | `SGD::new(params, lr)` |
| `optim.Adam(params, lr=...)` | `Adam::new(params, lr)` |
| `optim.AdamW(params, lr=..., weight_decay=...)` | `AdamW::new(...)` |
| `opt.zero_grad()` | `opt.zero_grad()` |
| `opt.step()` | `opt.step()` / `opt.step()` (`&mut self` for Adam) |
| `StepLR` / `MultiStepLR` / `CosineAnnealingLR` | `StepLR` / `MultiStepLR` / `CosineAnnealingLR` |

```python
opt.zero_grad(set_to_none=True)
loss.backward()
opt.step()
```

```rust
opt.zero_grad();
loss.backward();
opt.step();
// Or fused:
// opt.step_and_zero_grad();
```

---

## Autograd

| PyTorch | RusTorch |
|---------|--------|
| `loss.backward()` | `loss.backward()` (scalar) |
| `torch.autograd.grad(y, x, create_graph=True)` | `grad(&y, &[&x], true)` |
| `with torch.no_grad():` | `no_grad(\|\| { ... })` |
| `torch.set_grad_enabled(False)` | `set_grad_enabled(false)` |
| custom `torch.autograd.Function` | `apply_function(inputs, forward, backward)` |

```python
with torch.no_grad():
    pred = model(x)
```

```rust
let pred = no_grad(|| model.forward(&x));
```

---

## Data utilities

| PyTorch | RusTorch |
|---------|--------|
| `TensorDataset(x, y)` | `TensorDataset::new(x, y)` |
| `DataLoader(..., shuffle=True)` | `DataLoader::new(&ds, batch_size, true, seed)` |
| `SequentialSampler` / `RandomSampler` | `SequentialSampler` / `RandomSampler` |
| `default_collate` | `default_collate` |

For hand-rolled MNIST-style loops, prefer `gather_rows` + an index permutation (same idea as `x[batch_idx]`).

---

## Optional train helpers (not required for a 1:1 port)

These are **opt-in** speed helpers. A line-by-line port can ignore them; see also the [MNIST compare](../../example%20comparisons/mnist_mlp/README.md) (`naive` vs `fast`).

| Pattern | Helper |
|---------|--------|
| `relu(linear(x))` | `Linear::forward_relu(&x)` / `fused_linear_relu` |
| `cross_entropy(linear(h), y)` | `Linear::forward_cross_entropy(&h, &y)` / `linear_cross_entropy` |
| `zero_grad` + `step` | `Adam::step_and_zero_grad()` |
| Row batching without grad | `gather_rows` |

---

## End-to-end train step

```python
model.train()
opt.zero_grad()
logits = model(xb)
loss = loss_fn(logits, yb)
loss.backward()
opt.step()
print(loss.item())
```

```rust
opt.zero_grad();
let logits = model.forward(&xb);
let loss = loss_fn.forward(&logits, &yb);
loss.backward();
opt.step();
println!("{}", loss.item());
```

---

## Interop (NumPy / DataFrames)

| PyTorch / ecosystem | RusTorch |
|---------------------|--------|
| from NumPy array | `from_numpy` / `from_numpy_f32` |
| to NumPy | `to_numpy` / `to_numpy_f32` |
| pandas-like frame | `from_dataframe` / `to_dataframe` (via `rpandas`) |

---

## Common pitfalls

1. **Borrow checker:** keep intermediates in bindings (`let h = ...; let out = ...`) so references live long enough.
2. **Targets type:** class indices are `&[usize]`, not an integer `Tensor`.
3. **Seeds:** `Linear::new` / `randn` need an explicit seed; ports are not bit-identical to PyTorch init.
4. **Scalar backward only (v1):** `backward()` expects a 0-dim / single-element loss.
5. **No CUDA:** stay on CPU or keep PyTorch for GPU.
6. **Softmax dims:** current softmax/log_softmax/cross_entropy paths are oriented around 2D `(N, C)`.
7. **In-place vs out-of-place:** `add` vs `add_`; in-place on tensors that share storage / sit on the autograd tape can be unsafe in the same ways as PyTorch.

---

## Where to look next

- Crate overview: [`README.md`](README.md)
- End-to-end wall-clock example: [`example comparisons/mnist_mlp/`](../../example%20comparisons/mnist_mlp/)
- Public exports: [`src/lib.rs`](src/lib.rs)
