//! MNIST MLP — RusTorch side.
//!
//! `--mode naive`  1:1 translation of `python/train_mnist.py` (default module API)
//! `--mode fast`   fused helpers / train-path opts (fused Linear+ReLU/CE, …)

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rustorch::{
    gather_rows, no_grad, Adam, CrossEntropyLoss, Linear, Module, ReLU, Tensor,
};

const HIDDEN: usize = 128;
const LR: f32 = 1e-3;
const DEFAULT_EPOCHS: usize = 25;
const DEFAULT_BATCH: usize = 128;
const DEFAULT_SEED: u64 = 42;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Naive,
    Fast,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Mode::Naive => "naive",
            Mode::Fast => "fast",
        }
    }

    fn backend_tag(self) -> &'static str {
        match self {
            Mode::Naive => "rustorch_naive",
            Mode::Fast => "rustorch_fast",
        }
    }
}

fn data_dir_default() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("data")
}

fn read_u32_be(buf: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

fn read_idx_images(path: &Path) -> Tensor {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let magic = read_u32_be(&raw, 0);
    assert_eq!(magic, 2051, "bad image magic");
    let n = read_u32_be(&raw, 4) as usize;
    let rows = read_u32_be(&raw, 8) as usize;
    let cols = read_u32_be(&raw, 12) as usize;
    let flat = rows * cols;
    assert_eq!(raw.len(), 16 + n * flat);
    let mut data = Vec::with_capacity(n * flat);
    for &b in &raw[16..] {
        data.push(b as f32 / 255.0);
    }
    Tensor::from_vec(data, &[n, flat], false)
}

fn read_idx_labels(path: &Path) -> Vec<usize> {
    let raw = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let magic = read_u32_be(&raw, 0);
    assert_eq!(magic, 2049, "bad label magic");
    let n = read_u32_be(&raw, 4) as usize;
    assert_eq!(raw.len(), 8 + n);
    raw[8..].iter().map(|&b| b as usize).collect()
}

/// Same LCG Fisher–Yates as the Python trainer.
fn lcg_shuffle(n: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let j = ((state >> 8) as usize) % (i + 1);
        idx.swap(i, j);
    }
    idx
}

struct Mlp {
    fc1: Linear,
    relu: ReLU,
    fc2: Linear,
}

impl Mlp {
    fn new(seed: u64) -> Self {
        Self {
            fc1: Linear::new(784, HIDDEN, true, seed),
            relu: ReLU,
            fc2: Linear::new(HIDDEN, 10, true, seed + 100),
        }
    }

    /// Naive / PyTorch-shaped: separate Linear → ReLU → Linear.
    fn forward_naive(&self, x: &Tensor) -> Tensor {
        let h = self.relu.forward(&self.fc1.forward(x));
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }
}

fn batch_labels(labels: &[usize], indices: &[usize]) -> Vec<usize> {
    indices.iter().map(|&i| labels[i]).collect()
}

fn argmax_correct(logits: &Tensor, yb: &[usize]) -> usize {
    let shape = logits.shape();
    let (n, c) = (shape[0], shape[1]);
    logits.with_data(|data| {
        let mut correct = 0usize;
        for i in 0..n {
            let row = &data[i * c..(i + 1) * c];
            let mut best = 0usize;
            let mut best_v = row[0];
            for (j, &v) in row.iter().enumerate().skip(1) {
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            if best == yb[i] {
                correct += 1;
            }
        }
        correct
    })
}

fn run_epoch_train_naive(
    model: &Mlp,
    opt: &mut Adam,
    loss_fn: &CrossEntropyLoss,
    x: &Tensor,
    y: &[usize],
    batch_size: usize,
    seed: u64,
) -> f32 {
    let order = lcg_shuffle(y.len(), seed);
    let mut total_loss = 0.0f32;
    let mut n_batches = 0usize;
    let mut start = 0usize;
    while start < order.len() {
        let end = (start + batch_size).min(order.len());
        let batch_idx = &order[start..end];
        let xb = gather_rows(x, batch_idx);
        let yb = batch_labels(y, batch_idx);

        opt.zero_grad();
        let logits = model.forward_naive(&xb);
        let loss = loss_fn.forward(&logits, &yb);
        loss.backward();
        opt.step();

        total_loss += loss.item();
        n_batches += 1;
        start = end;
    }
    total_loss / n_batches.max(1) as f32
}

fn run_epoch_train_fast(
    model: &Mlp,
    opt: &mut Adam,
    x: &Tensor,
    y: &[usize],
    batch_size: usize,
    seed: u64,
) -> f32 {
    // Same shuffle/gather recipe as naive + Python; speed comes from fused
    // Linear+ReLU / Linear+CE and step_and_zero_grad (not a different data path).
    let order = lcg_shuffle(y.len(), seed);
    let mut total_loss = 0.0f32;
    let mut n_batches = 0usize;
    let mut start = 0usize;
    while start < order.len() {
        let end = (start + batch_size).min(order.len());
        let batch_idx = &order[start..end];
        let xb = gather_rows(x, batch_idx);
        let yb = batch_labels(y, batch_idx);

        let h = model.fc1.forward_relu(&xb);
        let loss = model.fc2.forward_cross_entropy(&h, &yb);
        loss.backward();
        opt.step_and_zero_grad();

        total_loss += loss.item();
        n_batches += 1;
        start = end;
    }
    total_loss / n_batches.max(1) as f32
}

fn run_eval_naive(model: &Mlp, x: &Tensor, y: &[usize], batch_size: usize) -> f64 {
    no_grad(|| {
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut start = 0usize;
        while start < y.len() {
            let end = (start + batch_size).min(y.len());
            let batch_idx: Vec<usize> = (start..end).collect();
            let xb = gather_rows(x, &batch_idx);
            let yb = &y[start..end];
            let logits = model.forward_naive(&xb);
            correct += argmax_correct(&logits, yb);
            total += end - start;
            start = end;
        }
        correct as f64 / total.max(1) as f64
    })
}

fn run_eval_fast(model: &Mlp, x: &Tensor, y: &[usize], batch_size: usize) -> f64 {
    no_grad(|| {
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut start = 0usize;
        while start < y.len() {
            let end = (start + batch_size).min(y.len());
            let batch_idx: Vec<usize> = (start..end).collect();
            let xb = gather_rows(x, &batch_idx);
            let yb = &y[start..end];
            let h = model.fc1.forward_relu(&xb);
            let logits = model.fc2.forward(&h);
            correct += argmax_correct(&logits, yb);
            total += end - start;
            start = end;
        }
        correct as f64 / total.max(1) as f64
    })
}

fn parse_args() -> (Mode, usize, usize, u64, PathBuf) {
    let mut mode = Mode::Naive;
    let mut epochs = DEFAULT_EPOCHS;
    let mut batch_size = DEFAULT_BATCH;
    let mut seed = DEFAULT_SEED;
    let mut data_dir = data_dir_default();
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => {
                i += 1;
                mode = match args[i].as_str() {
                    "naive" => Mode::Naive,
                    "fast" => Mode::Fast,
                    other => panic!("--mode must be naive|fast, got {other}"),
                };
            }
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().expect("--epochs");
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().expect("--batch-size");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("--seed");
            }
            "--data-dir" => {
                i += 1;
                data_dir = PathBuf::from(&args[i]);
            }
            other => panic!("unknown arg: {other}"),
        }
        i += 1;
    }
    (mode, epochs, batch_size, seed, data_dir)
}

fn main() {
    let (mode, epochs, batch_size, seed, data_dir) = parse_args();

    let x_train = read_idx_images(&data_dir.join("train-images-idx3-ubyte"));
    let y_train = read_idx_labels(&data_dir.join("train-labels-idx1-ubyte"));
    let x_test = read_idx_images(&data_dir.join("t10k-images-idx3-ubyte"));
    let y_test = read_idx_labels(&data_dir.join("t10k-labels-idx1-ubyte"));

    let model = Mlp::new(seed);
    let mut opt = Adam::new(model.parameters(), LR);
    let loss_fn = CrossEntropyLoss;

    let t0 = Instant::now();
    let mut last_train_loss = 0.0f32;
    let mut last_val_acc = 0.0f64;
    for epoch in 0..epochs {
        last_train_loss = match mode {
            Mode::Naive => run_epoch_train_naive(
                &model,
                &mut opt,
                &loss_fn,
                &x_train,
                &y_train,
                batch_size,
                seed + epoch as u64,
            ),
            Mode::Fast => run_epoch_train_fast(
                &model,
                &mut opt,
                &x_train,
                &y_train,
                batch_size,
                seed + epoch as u64,
            ),
        };
        last_val_acc = match mode {
            Mode::Naive => run_eval_naive(&model, &x_test, &y_test, batch_size),
            Mode::Fast => run_eval_fast(&model, &x_test, &y_test, batch_size),
        };
        println!(
            "epoch={epoch} train_loss={last_train_loss:.6} val_acc={last_val_acc:.4} mode={}",
            mode.as_str()
        );
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "RESULT backend={} wall_sec={wall:.4} train_loss={last_train_loss:.6} \
         val_acc={last_val_acc:.4} epochs={epochs} batch_size={batch_size} mode={}",
        mode.backend_tag(),
        mode.as_str()
    );
}
