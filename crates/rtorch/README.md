# rtorch

PyTorch-shaped tensors + autograd for Rust (CPU `f32`).

## Features

| Feature | Default | Purpose |
|---------|---------|---------|
| `parallel` | on | Rayon pool for mid/large GEMMs |

```toml
rtorch = { path = "...", default-features = true }           # portable + parallel
rtorch = { path = "...", default-features = false }          # serial GEMM only
```

GEMM always uses the pure-Rust [`matrixmultiply`](https://crates.io/crates/matrixmultiply) crate (no system BLAS required). That keeps the library usable on any machine without Fortran/OpenBLAS install steps.

## Train-oriented helpers

- `fused_linear_relu` / `Linear::forward_relu` — one tape node for Linear+ReLU
- `Adam::step_and_zero_grad` — update + clear grads in one pass
- `gather_rows` — fast batch gather when inputs do not need grad
