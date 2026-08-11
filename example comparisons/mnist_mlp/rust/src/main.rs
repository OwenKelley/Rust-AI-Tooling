//! MNIST MLP train/val — rtorch side (1:1 with python/train_mnist.py).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use rtorch::{narrow, no_grad, shuffle_rows_inplace, Adam, Linear, Module, Tensor};

const HIDDEN: usize = 128;
const LR: f32 = 1e-3;
const DEFAULT_EPOCHS: usize = 25;
const DEFAULT_BATCH: usize = 128;
const DEFAULT_SEED: u64 = 42;

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

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(seed: u64) -> Self {
        Self {
            fc1: Linear::new(784, HIDDEN, true, seed),
            fc2: Linear::new(HIDDEN, 10, true, seed + 100),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }
}

#[derive(Default, Clone, Copy)]
struct PhaseTimers {
    shuffle: f64,
    forward: f64,
    loss: f64,
    backward: f64,
    step: f64,
    eval: f64,
    batches: usize,
}

fn run_epoch_train(
    model: &Mlp,
    opt: &mut Adam,
    x: &Tensor,
    y: &mut [usize],
    batch_size: usize,
    seed: u64,
    profile: bool,
    timers: &mut PhaseTimers,
) -> f32 {
    let t_sh = Instant::now();
    shuffle_rows_inplace(x, y, seed);
    if profile {
        timers.shuffle += t_sh.elapsed().as_secs_f64();
    }

    let n = y.len();
    let mut total_loss = 0.0f32;
    let mut n_batches = 0usize;
    let mut start = 0usize;
    while start < n {
        let end = (start + batch_size).min(n);
        let len = end - start;
        let xb = narrow(x, 0, start, len);
        let yb = &y[start..end];

        let t1 = Instant::now();
        let h = model.fc1.forward_relu(&xb);
        if profile {
            timers.forward += t1.elapsed().as_secs_f64();
        }

        let t2 = Instant::now();
        let loss = model.fc2.forward_cross_entropy(&h, yb);
        if profile {
            timers.loss += t2.elapsed().as_secs_f64();
        }

        let t3 = Instant::now();
        loss.backward();
        if profile {
            timers.backward += t3.elapsed().as_secs_f64();
        }

        let t4 = Instant::now();
        opt.step_and_zero_grad();
        if profile {
            timers.step += t4.elapsed().as_secs_f64();
        }

        total_loss += loss.item();
        n_batches += 1;
        start = end;
    }
    if profile {
        timers.batches += n_batches;
    }
    total_loss / n_batches.max(1) as f32
}

fn run_eval(model: &Mlp, x: &Tensor, y: &[usize], batch_size: usize) -> f64 {
    no_grad(|| {
        let mut correct = 0usize;
        let mut total = 0usize;
        let mut start = 0usize;
        while start < y.len() {
            let end = (start + batch_size).min(y.len());
            let len = end - start;
            let xb = narrow(x, 0, start, len);
            let yb = &y[start..end];
            let h = model.fc1.forward_relu(&xb);
            let logits = model.fc2.forward(&h);
            let shape = logits.shape();
            let (n, c) = (shape[0], shape[1]);
            let batch_correct = logits.with_data(|data| {
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
            });
            correct += batch_correct;
            total += n;
            start = end;
        }
        correct as f64 / total.max(1) as f64
    })
}

fn parse_args() -> (usize, usize, u64, PathBuf, bool) {
    let mut epochs = DEFAULT_EPOCHS;
    let mut batch_size = DEFAULT_BATCH;
    let mut seed = DEFAULT_SEED;
    let mut data_dir = data_dir_default();
    let mut profile = false;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
            "--profile" => {
                profile = true;
            }
            other => panic!("unknown arg: {other}"),
        }
        i += 1;
    }
    (epochs, batch_size, seed, data_dir, profile)
}

fn main() {
    let (epochs, batch_size, seed, data_dir, profile) = parse_args();

    let x_train = read_idx_images(&data_dir.join("train-images-idx3-ubyte"));
    let mut y_train = read_idx_labels(&data_dir.join("train-labels-idx1-ubyte"));
    let x_test = read_idx_images(&data_dir.join("t10k-images-idx3-ubyte"));
    let y_test = read_idx_labels(&data_dir.join("t10k-labels-idx1-ubyte"));

    let model = Mlp::new(seed);
    let mut opt = Adam::new(model.parameters(), LR);

    let t0 = Instant::now();
    let mut last_train_loss = 0.0f32;
    let mut last_val_acc = 0.0f64;
    let mut timers = PhaseTimers::default();
    for epoch in 0..epochs {
        last_train_loss = run_epoch_train(
            &model,
            &mut opt,
            &x_train,
            &mut y_train,
            batch_size,
            seed + epoch as u64,
            profile,
            &mut timers,
        );
        let te = Instant::now();
        last_val_acc = run_eval(&model, &x_test, &y_test, batch_size);
        if profile {
            timers.eval += te.elapsed().as_secs_f64();
        }
        println!(
            "epoch={epoch} train_loss={last_train_loss:.6} val_acc={last_val_acc:.4}"
        );
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "RESULT backend=rtorch wall_sec={wall:.4} train_loss={last_train_loss:.6} \
         val_acc={last_val_acc:.4} epochs={epochs} batch_size={batch_size}"
    );
    if profile {
        let train =
            timers.shuffle + timers.forward + timers.loss + timers.backward + timers.step;
        println!(
            "PROFILE batches={} shuffle_s={:.4} forward_s={:.4} loss_s={:.4} \
             backward_s={:.4} step_s={:.4} eval_s={:.4} train_sum_s={:.4} wall_s={:.4}",
            timers.batches,
            timers.shuffle,
            timers.forward,
            timers.loss,
            timers.backward,
            timers.step,
            timers.eval,
            train,
            wall
        );
        let denom = train.max(1e-9);
        println!(
            "PROFILE_PCT shuffle={:.1}% forward={:.1}% loss={:.1}% backward={:.1}% \
             step={:.1}% (of train phases)",
            100.0 * timers.shuffle / denom,
            100.0 * timers.forward / denom,
            100.0 * timers.loss / denom,
            100.0 * timers.backward / denom,
            100.0 * timers.step / denom
        );
    }
}
