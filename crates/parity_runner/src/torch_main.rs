//! CLI used by the Python PyTorch comparison harness.

use std::env;
use std::process;
use std::time::Instant;

use rtorch::{
    add, cat, clamp, cross_entropy, dropout, exp, gelu, index_select, log, matmul, max_pool2d, mean,
    mul, pow, relu, reshape, seeded_uniform, sigmoid, softmax, stack, sum, tanh, transpose, zeros,
    Adam, AdamW, BatchNorm1d, Conv2d, CrossEntropyLoss, Embedding, Flatten, LayerNorm, Linear,
    Module, MultiStepLR, MSELoss, ReLU, SGD, Sequential, StepLR,
};

#[derive(Debug, Clone)]
enum Op {
    Zeros,
    Add,
    Mul,
    Matmul,
    Sum,
    Mean,
    Relu,
    Sigmoid,
    Transpose,
    Reshape,
    LinearForward,
    MseLoss,
    TrainStep,
    Exp,
    Log,
    Pow,
    Clamp,
    BroadcastAdd,
    Cat,
    Stack,
    IndexSelect,
    Softmax,
    CrossEntropy,
    Dropout,
    SequentialForward,
    AdamTrainStep,
    EmbeddingForward,
    LayerNormForward,
    Conv2dForward,
    AdamWTrainStep,
    StepLr,
    Tanh,
    Gelu,
    BatchNorm1dForward,
    MaxPool2dForward,
    FlattenForward,
    MultiStepLr,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "zeros" => Self::Zeros,
            "add" => Self::Add,
            "mul" => Self::Mul,
            "matmul" => Self::Matmul,
            "sum" => Self::Sum,
            "mean" => Self::Mean,
            "relu" => Self::Relu,
            "sigmoid" => Self::Sigmoid,
            "transpose" => Self::Transpose,
            "reshape" => Self::Reshape,
            "linear_forward" => Self::LinearForward,
            "mse_loss" => Self::MseLoss,
            "train_step" => Self::TrainStep,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "pow" => Self::Pow,
            "clamp" => Self::Clamp,
            "broadcast_add" => Self::BroadcastAdd,
            "cat" => Self::Cat,
            "stack" => Self::Stack,
            "index_select" => Self::IndexSelect,
            "softmax" => Self::Softmax,
            "cross_entropy" => Self::CrossEntropy,
            "dropout" => Self::Dropout,
            "sequential_forward" => Self::SequentialForward,
            "adam_train_step" => Self::AdamTrainStep,
            "embedding_forward" => Self::EmbeddingForward,
            "layernorm_forward" => Self::LayerNormForward,
            "conv2d_forward" => Self::Conv2dForward,
            "adamw_train_step" => Self::AdamWTrainStep,
            "steplr" => Self::StepLr,
            "tanh" => Self::Tanh,
            "gelu" => Self::Gelu,
            "batchnorm1d_forward" => Self::BatchNorm1dForward,
            "max_pool2d_forward" => Self::MaxPool2dForward,
            "flatten_forward" => Self::FlattenForward,
            "multisteplr" => Self::MultiStepLr,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Zeros => "zeros",
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Matmul => "matmul",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Relu => "relu",
            Self::Sigmoid => "sigmoid",
            Self::Transpose => "transpose",
            Self::Reshape => "reshape",
            Self::LinearForward => "linear_forward",
            Self::MseLoss => "mse_loss",
            Self::TrainStep => "train_step",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Pow => "pow",
            Self::Clamp => "clamp",
            Self::BroadcastAdd => "broadcast_add",
            Self::Cat => "cat",
            Self::Stack => "stack",
            Self::IndexSelect => "index_select",
            Self::Softmax => "softmax",
            Self::CrossEntropy => "cross_entropy",
            Self::Dropout => "dropout",
            Self::SequentialForward => "sequential_forward",
            Self::AdamTrainStep => "adam_train_step",
            Self::EmbeddingForward => "embedding_forward",
            Self::LayerNormForward => "layernorm_forward",
            Self::Conv2dForward => "conv2d_forward",
            Self::AdamWTrainStep => "adamw_train_step",
            Self::StepLr => "steplr",
            Self::Tanh => "tanh",
            Self::Gelu => "gelu",
            Self::BatchNorm1dForward => "batchnorm1d_forward",
            Self::MaxPool2dForward => "max_pool2d_forward",
            Self::FlattenForward => "flatten_forward",
            Self::MultiStepLr => "multisteplr",
        }
    }
}

struct Args {
    op: Op,
    size: usize,
    iters: usize,
    warmup: usize,
    seed: u64,
}

fn usage() -> ! {
    eprintln!(
        "Usage: torch_parity_runner --op <name> [--size N] [--iters N] [--warmup N] [--seed N]"
    );
    process::exit(2);
}

fn parse_args() -> Result<Args, String> {
    let mut op = None;
    let mut size = 64usize;
    let mut iters = 20usize;
    let mut warmup = 3usize;
    let mut seed = 42u64;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--op" => {
                let v = args.next().ok_or("missing --op value")?;
                op = Some(Op::parse(&v)?);
            }
            "--size" => {
                size = args
                    .next()
                    .ok_or("missing --size")?
                    .parse()
                    .map_err(|_| "bad --size")?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .ok_or("missing --iters")?
                    .parse()
                    .map_err(|_| "bad --iters")?;
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .ok_or("missing --warmup")?
                    .parse()
                    .map_err(|_| "bad --warmup")?;
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or("missing --seed")?
                    .parse()
                    .map_err(|_| "bad --seed")?;
            }
            "-h" | "--help" => usage(),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }
    Ok(Args {
        op: op.ok_or("missing --op")?,
        size,
        iters,
        warmup,
        seed,
    })
}

fn median_u64(xs: &[u64]) -> u64 {
    let mut v = xs.to_vec();
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2
    }
}

fn make_linear(in_f: usize, out_f: usize, seed: u64) -> Linear {
    let w = seeded_uniform(&[out_f, in_f], seed, -0.5, 0.5);
    let b = seeded_uniform(&[out_f], seed + 1, -0.1, 0.1);
    Linear::from_params(w, Some(b))
}

fn train_once(n: usize, seed: u64, steps: usize) -> f64 {
    let in_f = 4usize;
    let hidden = 8usize;
    let x = seeded_uniform(&[n, in_f], seed, -1.0, 1.0);
    let y = seeded_uniform(&[n, 1], seed + 1, -1.0, 1.0);
    let l1 = make_linear(in_f, hidden, seed + 10);
    let l2 = make_linear(hidden, 1, seed + 20);
    let relu_m = ReLU;
    let loss_fn = MSELoss;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let opt = SGD::new(params, 0.05);
    let mut last = 0.0f64;
    for _ in 0..steps {
        opt.zero_grad();
        let h = relu_m.forward(&l1.forward(&x));
        let pred = l2.forward(&h);
        let loss = loss_fn.forward(&pred, &y);
        loss.backward();
        opt.step();
        last = loss.item() as f64;
    }
    last
}

fn make_indices(n: usize, seed: u64) -> Vec<usize> {
    let k = (n / 2).max(1);
    let mut state = seed;
    let mut out = Vec::with_capacity(k);
    for _ in 0..k {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % n);
    }
    out
}

fn make_class_targets(n: usize, classes: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % classes);
    }
    out
}

fn adam_train_once(n: usize, seed: u64, steps: usize) -> f64 {
    let in_f = 4usize;
    let hidden = 8usize;
    let classes = 3usize;
    let x = seeded_uniform(&[n, in_f], seed, -1.0, 1.0);
    let target = make_class_targets(n, classes, seed + 1);
    let l1 = make_linear(in_f, hidden, seed + 10);
    let l2 = make_linear(hidden, classes, seed + 20);
    let relu_m = ReLU;
    let loss_fn = CrossEntropyLoss;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut opt = Adam::new(params, 0.05);
    let mut last = 0.0f64;
    for _ in 0..steps {
        opt.zero_grad();
        let h = relu_m.forward(&l1.forward(&x));
        let logits = l2.forward(&h);
        let loss = loss_fn.forward(&logits, &target);
        loss.backward();
        opt.step();
        last = loss.item() as f64;
    }
    last
}

fn adamw_train_once(n: usize, seed: u64, steps: usize) -> f64 {
    let in_f = 4usize;
    let hidden = 8usize;
    let classes = 3usize;
    let x = seeded_uniform(&[n, in_f], seed, -1.0, 1.0);
    let target = make_class_targets(n, classes, seed + 1);
    let l1 = make_linear(in_f, hidden, seed + 10);
    let l2 = make_linear(hidden, classes, seed + 20);
    let relu_m = ReLU;
    let loss_fn = CrossEntropyLoss;
    let mut params = l1.parameters();
    params.extend(l2.parameters());
    let mut opt = AdamW::new(params, 0.05, 0.01);
    let mut last = 0.0f64;
    for _ in 0..steps {
        opt.zero_grad();
        let h = relu_m.forward(&l1.forward(&x));
        let logits = l2.forward(&h);
        let loss = loss_fn.forward(&logits, &target);
        loss.backward();
        opt.step();
        last = loss.item() as f64;
    }
    last
}

fn steplr_once(steps: usize) -> f64 {
    let mut lr = 0.1f32;
    let mut sched = StepLR::new(&mut lr, 2, 0.5);
    for _ in 0..steps {
        sched.step();
    }
    lr as f64
}

fn multisteplr_once(steps: usize) -> f64 {
    let mut lr = 0.1f32;
    let mut sched = MultiStepLR::new(&mut lr, vec![2, 4], 0.5);
    for _ in 0..steps {
        sched.step();
    }
    lr as f64
}

fn make_emb_indices(n: usize, vocab: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push(((state >> 8) as usize) % vocab);
    }
    out
}

fn run_op(op: &Op, n: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    match op {
        Op::Zeros => {
            let t = zeros(&[n, n], false);
            let checksum = t.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(zeros(&[n, n], false));
                }),
            )
        }
        Op::Add => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = add(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(add(&a, &b));
                }),
            )
        }
        Op::Mul => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = mul(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mul(&a, &b));
                }),
            )
        }
        Op::Matmul => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = matmul(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(matmul(&a, &b));
                }),
            )
        }
        Op::Sum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = sum(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sum(&a));
                }),
            )
        }
        Op::Mean => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = mean(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mean(&a));
                }),
            )
        }
        Op::Relu => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = relu(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(relu(&a));
                }),
            )
        }
        Op::Sigmoid => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = sigmoid(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sigmoid(&a));
                }),
            )
        }
        Op::Transpose => {
            let a = seeded_uniform(&[n, (n / 2).max(1)], seed, -1.0, 1.0);
            let checksum = transpose(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(transpose(&a));
                }),
            )
        }
        Op::Reshape => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = reshape(&a, &[n * n]).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(reshape(&a, &[n * n]));
                }),
            )
        }
        Op::LinearForward => {
            let batch = n.min(32).max(4);
            let x = seeded_uniform(&[batch, 8], seed, -1.0, 1.0);
            let layer = make_linear(8, 4, seed + 3);
            let checksum = layer.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(layer.forward(&x));
                }),
            )
        }
        Op::MseLoss => {
            let a = seeded_uniform(&[n, 4], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, 4], seed + 1, -1.0, 1.0);
            let checksum = MSELoss.forward(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(MSELoss.forward(&a, &b));
                }),
            )
        }
        Op::TrainStep => {
            let batch = n.min(32).max(8);
            let checksum = train_once(batch, seed, 5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(train_once(batch, seed, 5));
                }),
            )
        }
        Op::Exp => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = exp(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(exp(&a));
                }),
            )
        }
        Op::Log => {
            let a = seeded_uniform(&[n, n], seed, 0.1, 2.0);
            let checksum = log(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(log(&a));
                }),
            )
        }
        Op::Pow => {
            let a = seeded_uniform(&[n, n], seed, 0.1, 2.0);
            let b = seeded_uniform(&[n, n], seed + 1, 0.5, 2.0);
            let checksum = pow(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(pow(&a, &b));
                }),
            )
        }
        Op::Clamp => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = clamp(&a, -0.5, 0.5).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(clamp(&a, -0.5, 0.5));
                }),
            )
        }
        Op::BroadcastAdd => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = add(&a, &b).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(add(&a, &b));
                }),
            )
        }
        Op::Cat => {
            let w = (n / 2).max(1);
            let a = seeded_uniform(&[n, w], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, w], seed + 1, -1.0, 1.0);
            let checksum = cat(&[&a, &b], 1).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cat(&[&a, &b], 1));
                }),
            )
        }
        Op::Stack => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = stack(&[&a, &b], 0).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(stack(&[&a, &b], 0));
                }),
            )
        }
        Op::IndexSelect => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let idx = make_indices(n, seed + 7);
            let checksum = index_select(&a, 1, &idx).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(index_select(&a, 1, &idx));
                }),
            )
        }
        Op::Softmax => {
            let classes = n.min(16).max(4);
            let a = seeded_uniform(&[n, classes], seed, -1.0, 1.0);
            let checksum = softmax(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(softmax(&a));
                }),
            )
        }
        Op::CrossEntropy => {
            let batch = n.min(32).max(8);
            let classes = 4usize;
            let a = seeded_uniform(&[batch, classes], seed, -1.0, 1.0);
            let target = make_class_targets(batch, classes, seed + 3);
            let checksum = cross_entropy(&a, &target).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cross_entropy(&a, &target));
                }),
            )
        }
        Op::Dropout => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = dropout(&a, 0.25, true, seed + 9).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dropout(&a, 0.25, true, seed + 9));
                }),
            )
        }
        Op::SequentialForward => {
            let batch = n.min(32).max(4);
            let x = seeded_uniform(&[batch, 8], seed, -1.0, 1.0);
            let l1 = make_linear(8, 16, seed + 3);
            let l2 = make_linear(16, 4, seed + 5);
            let model = Sequential::new(vec![
                Box::new(l1),
                Box::new(ReLU),
                Box::new(l2),
            ]);
            let checksum = model.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(model.forward(&x));
                }),
            )
        }
        Op::AdamTrainStep => {
            let batch = n.min(32).max(8);
            let checksum = adam_train_once(batch, seed, 5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(adam_train_once(batch, seed, 5));
                }),
            )
        }
        Op::EmbeddingForward => {
            let vocab = n.min(32).max(8);
            let dim = 8usize;
            let n_idx = n.min(16).max(4);
            let weight = seeded_uniform(&[vocab, dim], seed, -0.5, 0.5);
            let emb = Embedding::from_params(weight);
            let idx = make_emb_indices(n_idx, vocab, seed + 7);
            let checksum = emb.forward_indices(&idx).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(emb.forward_indices(&idx));
                }),
            )
        }
        Op::LayerNormForward => {
            let batch = n.min(32).max(4);
            let c = 8usize;
            let x = seeded_uniform(&[batch, c], seed, -1.0, 1.0);
            let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
            let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
            let ln = LayerNorm::from_params(w, b, 1e-5);
            let checksum = ln.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ln.forward(&x));
                }),
            )
        }
        Op::Conv2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let cin = 2usize;
            let cout = 3usize;
            let k = 3usize;
            let x = seeded_uniform(&[batch, cin, spatial, spatial], seed, -1.0, 1.0);
            let w = seeded_uniform(&[cout, cin, k, k], seed + 1, -0.2, 0.2);
            let b = seeded_uniform(&[cout], seed + 2, -0.1, 0.1);
            let conv = Conv2d::from_params(w, Some(b));
            let checksum = conv.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(conv.forward(&x));
                }),
            )
        }
        Op::AdamWTrainStep => {
            let batch = n.min(32).max(8);
            let checksum = adamw_train_once(batch, seed, 5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(adamw_train_once(batch, seed, 5));
                }),
            )
        }
        Op::StepLr => {
            let checksum = steplr_once(5);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(steplr_once(5));
                }),
            )
        }
        Op::Tanh => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = tanh(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(tanh(&a));
                }),
            )
        }
        Op::Gelu => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = gelu(&a).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(gelu(&a));
                }),
            )
        }
        Op::BatchNorm1dForward => {
            let batch = n.min(32).max(4);
            let c = 8usize;
            let x = seeded_uniform(&[batch, c], seed, -1.0, 1.0);
            let w = seeded_uniform(&[c], seed + 1, 0.5, 1.5);
            let b = seeded_uniform(&[c], seed + 2, -0.1, 0.1);
            let bn = BatchNorm1d::from_params(w, b, 1e-5, 0.1);
            let checksum = bn.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(bn.forward(&x));
                }),
            )
        }
        Op::MaxPool2dForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
            let checksum = max_pool2d(&x, 2, 2).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(max_pool2d(&x, 2, 2));
                }),
            )
        }
        Op::FlattenForward => {
            let batch = n.min(4).max(2);
            let spatial = n.min(8).max(4);
            let x = seeded_uniform(&[batch, 2, spatial, spatial], seed, -1.0, 1.0);
            let flat = Flatten::default();
            let checksum = flat.forward(&x).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(flat.forward(&x));
                }),
            )
        }
        Op::MultiStepLr => {
            let checksum = multisteplr_once(6);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(multisteplr_once(6));
                }),
            )
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            usage();
        }
    };
    if args.iters == 0 {
        eprintln!("error: iters must be > 0");
        process::exit(1);
    }

    let (checksum, mut thunk) = run_op(&args.op, args.size, args.seed);
    for _ in 0..args.warmup {
        thunk();
    }
    let mut samples = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        thunk();
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    let mean_ns = samples.iter().map(|&x| x as f64).sum::<f64>() / samples.len() as f64;
    println!(
        "{{\n  \"language\": \"rust\",\n  \"op\": \"{}\",\n  \"size\": {},\n  \"iters\": {},\n  \"warmup\": {},\n  \"seed\": {},\n  \"median_ns\": {},\n  \"mean_ns\": {:.6},\n  \"min_ns\": {},\n  \"max_ns\": {},\n  \"checksum\": {:.17e}\n}}",
        args.op.as_str(),
        args.size,
        args.iters,
        args.warmup,
        args.seed,
        median_u64(&samples),
        mean_ns,
        samples.iter().min().unwrap(),
        samples.iter().max().unwrap(),
        checksum
    );
}
