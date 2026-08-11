# Example comparisons (Python vs Rust)

Side-by-side projects that exercise **PyTorch** and **rustorch** on the same task,
with the same architecture, hyperparameters, and data so wall-clock differences
are easier to interpret than microbenchmarks.

| Project | Task |
|---------|------|
| [`mnist_mlp/`](mnist_mlp/) | MLP classifier on MNIST (train + validation) |

Each project has mirrored `python/` and `rust/` entrypoints plus a small runner
that prints wall times for both.
