# Agent notes — Rust AI Tooling

## RusTorch (`rustorch`)

PyTorch-like tensors + autograd for Rust (CPU `f32`).

| Doc | Use |
|-----|-----|
| [`crates/rustorch/TRANSLATING.md`](crates/rustorch/TRANSLATING.md) | **Start here** — PyTorch → RusTorch API map |
| [`crates/rustorch/README.md`](crates/rustorch/README.md) | Crate overview, Cargo features, train helpers |
| [`crates/rustorch/src/lib.rs`](crates/rustorch/src/lib.rs) | Public exports |
| [`example comparisons/mnist_mlp/`](example%20comparisons/mnist_mlp/) | End-to-end train compare (naive vs fast) |

**Cursor:** project rule `.cursor/rules/rustorch.mdc` applies when editing RusTorch-related paths.

### Quick rules

- Crate name in code: `rustorch` (branding: RusTorch).
- `*` = elementwise; use `.matmul(&other)` for matrix multiply.
- `CrossEntropyLoss` / `cross_entropy` take `&[usize]` labels.
- Prefer reading TRANSLATING.md over guessing from PyTorch names.

## Other stacks in this repo

- NumPy slice: `crates/rnumpy`, `core_numerical/`
- SciPy: `crates/rscipy`
- Pandas-like: `crates/rpandas`
- Inventory: [`python-ai-ml-tooling.md`](python-ai-ml-tooling.md)
