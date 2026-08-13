//! MNIST MLP — RusTorch side.
//!
//! `--mode naive`  1:1 translation of `python/train_mnist.py` (default module API)
//! `--mode fast`   fused helpers / train-path opts (fused Linear+ReLU/CE, …)
//!
//! Training: optional `--save PATH` writes a portable weight checkpoint.
//! Inference: `--infer --checkpoint PATH` loads weights and runs timed forward passes.

use std::env;
use std::fs;
use std::io::Read;
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
const CKPT_MAGIC: u32 = 0x5254_4D4C; // "RTML"

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

fn run_inference_passes(model: &Mlp, mode: Mode, x: &Tensor, batch_size: usize, passes: usize) {
    no_grad(|| {
        for _ in 0..passes {
            let mut start = 0usize;
            while start < x.shape()[0] {
                let end = (start + batch_size).min(x.shape()[0]);
                let batch_idx: Vec<usize> = (start..end).collect();
                let xb = gather_rows(x, &batch_idx);
                match mode {
                    Mode::Naive => {
                        let _ = model.forward_naive(&xb);
                    }
                    Mode::Fast => {
                        let h = model.fc1.forward_relu(&xb);
                        let _ = model.fc2.forward(&h);
                    }
                }
                start = end;
            }
        }
    });
}

fn write_f32_le(out: &mut Vec<u8>, values: &[f32]) {
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn read_f32_le(buf: &[u8], off: &mut usize, n: usize) -> Vec<f32> {
    let need = n * 4;
    assert!(
        *off + need <= buf.len(),
        "checkpoint truncated at offset {}",
        *off
    );
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut b = [0u8; 4];
        b.copy_from_slice(&buf[*off..*off + 4]);
        *off += 4;
        out.push(f32::from_le_bytes(b));
    }
    out
}

/// Portable RusTorch MLP checkpoint (little-endian f32 weights).
fn save_checkpoint(
    path: &Path,
    model: &Mlp,
    seed: u64,
    epochs: usize,
    batch_size: usize,
    train_loss: f32,
    val_acc: f64,
    mode: Mode,
) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).ok();
    }
    let w1 = model.fc1.weight.data();
    let b1 = model
        .fc1
        .bias
        .as_ref()
        .expect("fc1 bias")
        .data();
    let w2 = model.fc2.weight.data();
    let b2 = model
        .fc2
        .bias
        .as_ref()
        .expect("fc2 bias")
        .data();
    assert_eq!(w1.len(), HIDDEN * 784);
    assert_eq!(b1.len(), HIDDEN);
    assert_eq!(w2.len(), 10 * HIDDEN);
    assert_eq!(b2.len(), 10);

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&CKPT_MAGIC.to_le_bytes());
    bytes.extend_from_slice(&(HIDDEN as u32).to_le_bytes());
    bytes.extend_from_slice(&(seed as u64).to_le_bytes());
    bytes.extend_from_slice(&(epochs as u32).to_le_bytes());
    bytes.extend_from_slice(&(batch_size as u32).to_le_bytes());
    bytes.extend_from_slice(&train_loss.to_le_bytes());
    bytes.extend_from_slice(&val_acc.to_le_bytes());
    let mode_tag: u32 = match mode {
        Mode::Naive => 0,
        Mode::Fast => 1,
    };
    bytes.extend_from_slice(&mode_tag.to_le_bytes());
    write_f32_le(&mut bytes, &w1);
    write_f32_le(&mut bytes, &b1);
    write_f32_le(&mut bytes, &w2);
    write_f32_le(&mut bytes, &b2);
    fs::write(path, &bytes).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));

    let meta = format!(
        "{{\n  \"backend\": \"rustorch\",\n  \"mode\": \"{}\",\n  \"hidden\": {HIDDEN},\n  \
         \"seed\": {seed},\n  \"epochs\": {epochs},\n  \"batch_size\": {batch_size},\n  \
         \"train_loss\": {train_loss},\n  \"val_acc\": {val_acc}\n}}\n",
        mode.as_str()
    );
    let meta_path = path.with_extension("json");
    fs::write(&meta_path, meta).ok();
    println!("saved checkpoint -> {}", path.display());
}

fn load_checkpoint(path: &Path) -> (Mlp, u64, usize, f64) {
    let mut file = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut off = 0usize;
    let magic = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
    off += 4;
    assert_eq!(magic, CKPT_MAGIC, "bad rustorch checkpoint magic");
    let hidden = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    assert_eq!(hidden, HIDDEN);
    let seed = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    off += 8;
    let epochs = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
    off += 4;
    off += 4; // batch_size
    off += 4; // train_loss
    let val_acc = f64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    off += 8;
    off += 4; // mode tag

    let w1 = read_f32_le(&buf, &mut off, HIDDEN * 784);
    let b1 = read_f32_le(&buf, &mut off, HIDDEN);
    let w2 = read_f32_le(&buf, &mut off, 10 * HIDDEN);
    let b2 = read_f32_le(&buf, &mut off, 10);

    let model = Mlp {
        fc1: Linear::from_params(
            Tensor::from_vec(w1, &[HIDDEN, 784], true),
            Some(Tensor::from_vec(b1, &[HIDDEN], true)),
        ),
        relu: ReLU,
        fc2: Linear::from_params(
            Tensor::from_vec(w2, &[10, HIDDEN], true),
            Some(Tensor::from_vec(b2, &[10], true)),
        ),
    };
    (model, seed, epochs, val_acc)
}

struct Cli {
    infer: bool,
    mode: Mode,
    epochs: usize,
    batch_size: usize,
    seed: u64,
    data_dir: PathBuf,
    save: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    passes: usize,
}

fn parse_args() -> Cli {
    let mut cli = Cli {
        infer: false,
        mode: Mode::Naive,
        epochs: DEFAULT_EPOCHS,
        batch_size: DEFAULT_BATCH,
        seed: DEFAULT_SEED,
        data_dir: data_dir_default(),
        save: None,
        checkpoint: None,
        passes: 50,
    };
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--infer" => cli.infer = true,
            "--mode" => {
                i += 1;
                cli.mode = match args[i].as_str() {
                    "naive" => Mode::Naive,
                    "fast" => Mode::Fast,
                    other => panic!("--mode must be naive|fast, got {other}"),
                };
            }
            "--epochs" => {
                i += 1;
                cli.epochs = args[i].parse().expect("--epochs");
            }
            "--batch-size" => {
                i += 1;
                cli.batch_size = args[i].parse().expect("--batch-size");
            }
            "--seed" => {
                i += 1;
                cli.seed = args[i].parse().expect("--seed");
            }
            "--data-dir" => {
                i += 1;
                cli.data_dir = PathBuf::from(&args[i]);
            }
            "--save" => {
                i += 1;
                cli.save = Some(PathBuf::from(&args[i]));
            }
            "--checkpoint" => {
                i += 1;
                cli.checkpoint = Some(PathBuf::from(&args[i]));
            }
            "--passes" => {
                i += 1;
                cli.passes = args[i].parse().expect("--passes");
            }
            other => panic!("unknown arg: {other}"),
        }
        i += 1;
    }
    cli
}

fn main() {
    let cli = parse_args();
    let x_test = read_idx_images(&cli.data_dir.join("t10k-images-idx3-ubyte"));
    let y_test = read_idx_labels(&cli.data_dir.join("t10k-labels-idx1-ubyte"));

    if cli.infer {
        let ckpt = cli
            .checkpoint
            .as_ref()
            .expect("--infer requires --checkpoint");
        let (model, seed, epochs, train_val_acc) = load_checkpoint(ckpt);
        let val_acc = match cli.mode {
            Mode::Naive => run_eval_naive(&model, &x_test, &y_test, cli.batch_size),
            Mode::Fast => run_eval_fast(&model, &x_test, &y_test, cli.batch_size),
        };
        // Warmup
        run_inference_passes(&model, cli.mode, &x_test, cli.batch_size, 1);
        let t0 = Instant::now();
        run_inference_passes(&model, cli.mode, &x_test, cli.batch_size, cli.passes);
        let wall = t0.elapsed().as_secs_f64();
        println!(
            "RESULT backend={} phase=infer wall_sec={wall:.6} val_acc={val_acc:.4} \
             train_val_acc={train_val_acc:.4} seed={seed} epochs={epochs} \
             passes={} batch_size={} mode={} checkpoint={}",
            cli.mode.backend_tag(),
            cli.passes,
            cli.batch_size,
            cli.mode.as_str(),
            ckpt.display()
        );
        return;
    }

    let x_train = read_idx_images(&cli.data_dir.join("train-images-idx3-ubyte"));
    let y_train = read_idx_labels(&cli.data_dir.join("train-labels-idx1-ubyte"));

    let model = Mlp::new(cli.seed);
    let mut opt = Adam::new(model.parameters(), LR);
    let loss_fn = CrossEntropyLoss;

    let t0 = Instant::now();
    let mut last_train_loss = 0.0f32;
    let mut last_val_acc = 0.0f64;
    for epoch in 0..cli.epochs {
        last_train_loss = match cli.mode {
            Mode::Naive => run_epoch_train_naive(
                &model,
                &mut opt,
                &loss_fn,
                &x_train,
                &y_train,
                cli.batch_size,
                cli.seed + epoch as u64,
            ),
            Mode::Fast => run_epoch_train_fast(
                &model,
                &mut opt,
                &x_train,
                &y_train,
                cli.batch_size,
                cli.seed + epoch as u64,
            ),
        };
        last_val_acc = match cli.mode {
            Mode::Naive => run_eval_naive(&model, &x_test, &y_test, cli.batch_size),
            Mode::Fast => run_eval_fast(&model, &x_test, &y_test, cli.batch_size),
        };
        println!(
            "epoch={epoch} train_loss={last_train_loss:.6} val_acc={last_val_acc:.4} mode={}",
            cli.mode.as_str()
        );
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "RESULT backend={} wall_sec={wall:.4} train_loss={last_train_loss:.6} \
         val_acc={last_val_acc:.4} epochs={} batch_size={} mode={}",
        cli.mode.backend_tag(),
        cli.epochs,
        cli.batch_size,
        cli.mode.as_str()
    );

    if let Some(path) = &cli.save {
        save_checkpoint(
            path,
            &model,
            cli.seed,
            cli.epochs,
            cli.batch_size,
            last_train_loss,
            last_val_acc,
            cli.mode,
        );
    }
}
