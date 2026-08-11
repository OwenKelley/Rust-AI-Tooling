"""MNIST MLP train/val — PyTorch side (1:1 with rust/src/main.rs)."""

from __future__ import annotations

import argparse
import struct
import time
from pathlib import Path

import torch
import torch.nn as nn

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Match Rust Linear::new LCG-ish init is not identical; architecture/hparams are.
HIDDEN = 128
LR = 1e-3
DEFAULT_EPOCHS = 25
DEFAULT_BATCH = 128
DEFAULT_SEED = 42


def read_idx_images(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    magic, n, rows, cols = struct.unpack(">IIII", raw[:16])
    assert magic == 2051, magic
    pixels = torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).clone()
    x = pixels.view(n, rows * cols).float().div_(255.0)
    return x


def read_idx_labels(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    magic, n = struct.unpack(">II", raw[:8])
    assert magic == 2049, magic
    y = torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).clone().long()
    assert y.numel() == n
    return y


def lcg_shuffle(n: int, seed: int) -> list[int]:
    """Fisher–Yates with the same LCG as the Rust trainer."""
    idx = list(range(n))
    state = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(n - 1, 0, -1):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
        j = (state >> 8) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def shuffle_epoch_inplace(x: torch.Tensor, y: torch.Tensor, seed: int) -> None:
    """Apply the LCG Fisher–Yates permutation once per epoch (contiguous batches)."""
    order = lcg_shuffle(x.shape[0], seed)
    x.copy_(x[order])
    y.copy_(y[order])


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def run_epoch_train(
    model: MLP,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    seed: int,
) -> float:
    model.train()
    shuffle_epoch_inplace(x, y, seed)
    n = x.shape[0]
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end]
        yb = y[start:end]
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def run_eval(model: MLP, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += yb.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="MNIST MLP (PyTorch)")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    x_train = read_idx_images(args.data_dir / "train-images-idx3-ubyte").to(device)
    y_train = read_idx_labels(args.data_dir / "train-labels-idx1-ubyte").to(device)
    x_test = read_idx_images(args.data_dir / "t10k-images-idx3-ubyte").to(device)
    y_test = read_idx_labels(args.data_dir / "t10k-labels-idx1-ubyte").to(device)

    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    last_train_loss = 0.0
    last_val_acc = 0.0
    for epoch in range(args.epochs):
        last_train_loss = run_epoch_train(
            model,
            opt,
            loss_fn,
            x_train,
            y_train,
            args.batch_size,
            args.seed + epoch,
        )
        last_val_acc = run_eval(model, x_test, y_test, args.batch_size)
        print(
            f"epoch={epoch} train_loss={last_train_loss:.6f} val_acc={last_val_acc:.4f}",
            flush=True,
        )
    wall = time.perf_counter() - t0
    print(
        f"RESULT backend=pytorch wall_sec={wall:.4f} "
        f"train_loss={last_train_loss:.6f} val_acc={last_val_acc:.4f} "
        f"epochs={args.epochs} batch_size={args.batch_size}",
        flush=True,
    )


if __name__ == "__main__":
    main()
