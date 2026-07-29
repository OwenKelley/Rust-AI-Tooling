"""Python torch reference ops for parity with rtorch."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core_numerical.numpy_parity.rng import seeded_uniform as seeded_uniform_f64


def seeded_uniform(
    shape: tuple[int, ...] | list[int],
    seed: int,
    low: float = 0.0,
    high: float = 1.0,
) -> torch.Tensor:
    arr = seeded_uniform_f64(shape, seed, low, high).astype(np.float32)
    return torch.from_numpy(arr)


def checksum(t: torch.Tensor) -> float:
    x = t.detach().float().cpu().numpy().reshape(-1)
    return float(np.sum(x[np.isfinite(x)]))


def make_linear(in_f: int, out_f: int, seed: int) -> nn.Linear:
    layer = nn.Linear(in_f, out_f, bias=True)
    with torch.no_grad():
        layer.weight.copy_(seeded_uniform((out_f, in_f), seed, -0.5, 0.5))
        layer.bias.copy_(seeded_uniform((out_f,), seed + 1, -0.1, 0.1))
    return layer


def train_once(n: int, seed: int, steps: int) -> float:
    in_f, hidden = 4, 8
    x = seeded_uniform((n, in_f), seed, -1.0, 1.0)
    y = seeded_uniform((n, 1), seed + 1, -1.0, 1.0)
    l1 = make_linear(in_f, hidden, seed + 10)
    l2 = make_linear(hidden, 1, seed + 20)
    opt = torch.optim.SGD(list(l1.parameters()) + list(l2.parameters()), lr=0.05)
    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        pred = l2(F.relu(l1(x)))
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    n = size

    if op == "zeros":
        result = torch.zeros(n, n, dtype=torch.float32)

        def thunk() -> torch.Tensor:
            return torch.zeros(n, n, dtype=torch.float32)

        return result, thunk

    if op == "add":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        result = a + b

        def thunk() -> torch.Tensor:
            return a + b

        return result, thunk

    if op == "mul":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        result = a * b

        def thunk() -> torch.Tensor:
            return a * b

        return result, thunk

    if op == "matmul":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        result = a @ b

        def thunk() -> torch.Tensor:
            return a @ b

        return result, thunk

    if op == "sum":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = a.sum()

        def thunk() -> torch.Tensor:
            return a.sum()

        return result, thunk

    if op == "mean":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = a.mean()

        def thunk() -> torch.Tensor:
            return a.mean()

        return result, thunk

    if op == "relu":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = F.relu(a)

        def thunk() -> torch.Tensor:
            return F.relu(a)

        return result, thunk

    if op == "sigmoid":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = torch.sigmoid(a)

        def thunk() -> torch.Tensor:
            return torch.sigmoid(a)

        return result, thunk

    if op == "transpose":
        a = seeded_uniform((n, max(n // 2, 1)), seed, -1.0, 1.0)
        result = a.t()

        def thunk() -> torch.Tensor:
            return a.t()

        return result, thunk

    if op == "reshape":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = a.reshape(n * n)

        def thunk() -> torch.Tensor:
            return a.reshape(n * n)

        return result, thunk

    if op == "linear_forward":
        batch = max(min(n, 32), 4)
        x = seeded_uniform((batch, 8), seed, -1.0, 1.0)
        layer = make_linear(8, 4, seed + 3)
        result = layer(x)

        def thunk() -> torch.Tensor:
            return layer(x)

        return result, thunk

    if op == "mse_loss":
        a = seeded_uniform((n, 4), seed, -1.0, 1.0)
        b = seeded_uniform((n, 4), seed + 1, -1.0, 1.0)
        result = F.mse_loss(a, b)

        def thunk() -> torch.Tensor:
            return F.mse_loss(a, b)

        return result, thunk

    if op == "train_step":
        batch = max(min(n, 32), 8)
        result = train_once(batch, seed, 5)

        def thunk() -> float:
            return train_once(batch, seed, 5)

        return result, thunk

    if op == "exp":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = torch.exp(a)

        def thunk() -> torch.Tensor:
            return torch.exp(a)

        return result, thunk

    if op == "log":
        a = seeded_uniform((n, n), seed, 0.1, 2.0)
        result = torch.log(a)

        def thunk() -> torch.Tensor:
            return torch.log(a)

        return result, thunk

    if op == "pow":
        a = seeded_uniform((n, n), seed, 0.1, 2.0)
        b = seeded_uniform((n, n), seed + 1, 0.5, 2.0)
        result = torch.pow(a, b)

        def thunk() -> torch.Tensor:
            return torch.pow(a, b)

        return result, thunk

    if op == "clamp":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = torch.clamp(a, -0.5, 0.5)

        def thunk() -> torch.Tensor:
            return torch.clamp(a, -0.5, 0.5)

        return result, thunk

    if op == "broadcast_add":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n,), seed + 1, -1.0, 1.0)
        result = a + b

        def thunk() -> torch.Tensor:
            return a + b

        return result, thunk

    if op == "cat":
        w = max(n // 2, 1)
        a = seeded_uniform((n, w), seed, -1.0, 1.0)
        b = seeded_uniform((n, w), seed + 1, -1.0, 1.0)
        result = torch.cat([a, b], dim=1)

        def thunk() -> torch.Tensor:
            return torch.cat([a, b], dim=1)

        return result, thunk

    if op == "stack":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        b = seeded_uniform((n, n), seed + 1, -1.0, 1.0)
        result = torch.stack([a, b], dim=0)

        def thunk() -> torch.Tensor:
            return torch.stack([a, b], dim=0)

        return result, thunk

    if op == "index_select":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        k = max(n // 2, 1)
        state = seed + 7
        idxs = []
        for _ in range(k):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
            idxs.append(int((state >> 8) % n))
        index = torch.tensor(idxs, dtype=torch.long)
        result = torch.index_select(a, 1, index)

        def thunk() -> torch.Tensor:
            return torch.index_select(a, 1, index)

        return result, thunk

    if op == "softmax":
        classes = max(min(n, 16), 4)
        a = seeded_uniform((n, classes), seed, -1.0, 1.0)
        result = torch.softmax(a, dim=-1)

        def thunk() -> torch.Tensor:
            return torch.softmax(a, dim=-1)

        return result, thunk

    if op == "cross_entropy":
        batch = max(min(n, 32), 8)
        classes = 4
        a = seeded_uniform((batch, classes), seed, -1.0, 1.0)
        state = seed + 3
        target_list = []
        for _ in range(batch):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
            target_list.append(int((state >> 8) % classes))
        target = torch.tensor(target_list, dtype=torch.long)
        result = F.cross_entropy(a, target)

        def thunk() -> torch.Tensor:
            return F.cross_entropy(a, target)

        return result, thunk

    if op == "dropout":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        p = 0.25
        scale = 1.0 / (1.0 - p)
        state = seed + 9
        flat = a.reshape(-1)
        mask = torch.empty_like(flat)
        for i in range(flat.numel()):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
            u = ((state >> 8) & 0xFFFFFF) / float(1 << 24)
            mask[i] = scale if u >= p else 0.0
        mask = mask.reshape(a.shape)
        result = a * mask

        def thunk() -> torch.Tensor:
            return a * mask

        return result, thunk

    if op == "sequential_forward":
        batch = max(min(n, 32), 4)
        x = seeded_uniform((batch, 8), seed, -1.0, 1.0)
        l1 = make_linear(8, 16, seed + 3)
        l2 = make_linear(16, 4, seed + 5)
        model = nn.Sequential(l1, nn.ReLU(), l2)
        result = model(x)

        def thunk() -> torch.Tensor:
            return model(x)

        return result, thunk

    if op == "adam_train_step":
        batch = max(min(n, 32), 8)
        result = adam_train_once(batch, seed, 5)

        def thunk() -> float:
            return adam_train_once(batch, seed, 5)

        return result, thunk

    if op == "embedding_forward":
        vocab = max(min(n, 32), 8)
        dim = 8
        n_idx = max(min(n, 16), 4)
        weight = seeded_uniform((vocab, dim), seed, -0.5, 0.5)
        emb = nn.Embedding(vocab, dim)
        with torch.no_grad():
            emb.weight.copy_(weight)
        state = seed + 7
        idxs = []
        for _ in range(n_idx):
            state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
            idxs.append(int((state >> 8) % vocab))
        index = torch.tensor(idxs, dtype=torch.long)
        result = emb(index)

        def thunk() -> torch.Tensor:
            return emb(index)

        return result, thunk

    if op == "layernorm_forward":
        batch = max(min(n, 32), 4)
        c = 8
        x = seeded_uniform((batch, c), seed, -1.0, 1.0)
        ln = nn.LayerNorm(c, eps=1e-5)
        with torch.no_grad():
            ln.weight.copy_(seeded_uniform((c,), seed + 1, 0.5, 1.5))
            ln.bias.copy_(seeded_uniform((c,), seed + 2, -0.1, 0.1))
        result = ln(x)

        def thunk() -> torch.Tensor:
            return ln(x)

        return result, thunk

    if op == "conv2d_forward":
        batch = max(min(n, 4), 2)
        spatial = max(min(n, 8), 4)
        cin, cout, k = 2, 3, 3
        x = seeded_uniform((batch, cin, spatial, spatial), seed, -1.0, 1.0)
        conv = nn.Conv2d(cin, cout, k, bias=True)
        with torch.no_grad():
            conv.weight.copy_(seeded_uniform((cout, cin, k, k), seed + 1, -0.2, 0.2))
            conv.bias.copy_(seeded_uniform((cout,), seed + 2, -0.1, 0.1))
        result = conv(x)

        def thunk() -> torch.Tensor:
            return conv(x)

        return result, thunk

    if op == "adamw_train_step":
        batch = max(min(n, 32), 8)
        result = adamw_train_once(batch, seed, 5)

        def thunk() -> float:
            return adamw_train_once(batch, seed, 5)

        return result, thunk

    if op == "steplr":
        result = steplr_once(5)

        def thunk() -> float:
            return steplr_once(5)

        return result, thunk

    if op == "tanh":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = torch.tanh(a)

        def thunk() -> torch.Tensor:
            return torch.tanh(a)

        return result, thunk

    if op == "gelu":
        a = seeded_uniform((n, n), seed, -1.0, 1.0)
        result = F.gelu(a, approximate="tanh")

        def thunk() -> torch.Tensor:
            return F.gelu(a, approximate="tanh")

        return result, thunk

    if op == "batchnorm1d_forward":
        batch = max(min(n, 32), 4)
        c = 8
        x = seeded_uniform((batch, c), seed, -1.0, 1.0)
        bn = nn.BatchNorm1d(c, eps=1e-5, momentum=0.1)
        with torch.no_grad():
            bn.weight.copy_(seeded_uniform((c,), seed + 1, 0.5, 1.5))
            bn.bias.copy_(seeded_uniform((c,), seed + 2, -0.1, 0.1))
        bn.train()
        result = bn(x)

        def thunk() -> torch.Tensor:
            return bn(x)

        return result, thunk

    if op == "max_pool2d_forward":
        batch = max(min(n, 4), 2)
        spatial = max(min(n, 8), 4)
        x = seeded_uniform((batch, 2, spatial, spatial), seed, -1.0, 1.0)
        result = F.max_pool2d(x, kernel_size=2, stride=2)

        def thunk() -> torch.Tensor:
            return F.max_pool2d(x, kernel_size=2, stride=2)

        return result, thunk

    if op == "flatten_forward":
        batch = max(min(n, 4), 2)
        spatial = max(min(n, 8), 4)
        x = seeded_uniform((batch, 2, spatial, spatial), seed, -1.0, 1.0)
        flat = nn.Flatten()
        result = flat(x)

        def thunk() -> torch.Tensor:
            return flat(x)

        return result, thunk

    if op == "multisteplr":
        result = multisteplr_once(6)

        def thunk() -> float:
            return multisteplr_once(6)

        return result, thunk

    raise ValueError(f"unknown op: {op}")


def adam_train_once(n: int, seed: int, steps: int) -> float:
    in_f, hidden, classes = 4, 8, 3
    x = seeded_uniform((n, in_f), seed, -1.0, 1.0)
    state = seed + 1
    target_list = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
        target_list.append(int((state >> 8) % classes))
    target = torch.tensor(target_list, dtype=torch.long)
    l1 = make_linear(in_f, hidden, seed + 10)
    l2 = make_linear(hidden, classes, seed + 20)
    opt = torch.optim.Adam(list(l1.parameters()) + list(l2.parameters()), lr=0.05)
    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(l2(F.relu(l1(x))), target)
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def adamw_train_once(n: int, seed: int, steps: int) -> float:
    in_f, hidden, classes = 4, 8, 3
    x = seeded_uniform((n, in_f), seed, -1.0, 1.0)
    state = seed + 1
    target_list = []
    for _ in range(n):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFFFFFFFFFF
        target_list.append(int((state >> 8) % classes))
    target = torch.tensor(target_list, dtype=torch.long)
    l1 = make_linear(in_f, hidden, seed + 10)
    l2 = make_linear(hidden, classes, seed + 20)
    opt = torch.optim.AdamW(
        list(l1.parameters()) + list(l2.parameters()), lr=0.05, weight_decay=0.01
    )
    last = 0.0
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(l2(F.relu(l1(x))), target)
        loss.backward()
        opt.step()
        last = float(loss.item())
    return last


def steplr_once(steps: int) -> float:
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(steps):
        opt.step()
        sched.step()
    return float(opt.param_groups[0]["lr"])


def multisteplr_once(steps: int) -> float:
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2, 4], gamma=0.5)
    for _ in range(steps):
        opt.step()
        sched.step()
    return float(opt.param_groups[0]["lr"])


def run_op(op: str, size: int, seed: int) -> float:
    result, _ = prepare(op, size, seed)
    if isinstance(result, (float, int)):
        return float(result)
    return checksum(result)
