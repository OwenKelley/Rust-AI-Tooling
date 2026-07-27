//! CLI used by the Python comparison harness.
//!
//! Emits one JSON object to stdout with timing stats and a checksum so the
//! Python side can verify numerical agreement and compare speed.
//!
//! Timing covers only the core op (inputs are prepared once beforehand).

use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use rnumpy::{
    abs, add, arange, argmax, argmin, divide, dot, exp, eye, full, linspace, log, matmul, max,
    mean, min, multiply, negative, ones, power, seeded_uniform, sqrt,
    std as np_std, subtract, sum, transpose, var, zeros, NdArray,
};
use serde::Serialize;

#[derive(Debug, Clone, ValueEnum)]
#[value(rename_all = "snake_case")]
enum Op {
    Zeros,
    Ones,
    Full,
    Arange,
    Linspace,
    Eye,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Sqrt,
    Exp,
    Log,
    Negative,
    Abs,
    Sum,
    Mean,
    Min,
    Max,
    Var,
    Std,
    Argmin,
    Argmax,
    Transpose,
    Matmul,
    Dot,
}

#[derive(Parser, Debug)]
#[command(name = "parity_runner")]
struct Args {
    /// Operation to run (NumPy name, snake_case).
    #[arg(long, value_enum)]
    op: Op,

    /// Leading dimension / vector length used to build inputs.
    #[arg(long, default_value_t = 256)]
    size: usize,

    /// Timed iterations.
    #[arg(long, default_value_t = 50)]
    iters: usize,

    /// Warmup iterations (not timed).
    #[arg(long, default_value_t = 5)]
    warmup: usize,

    /// RNG seed for shared inputs.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Serialize)]
struct Report {
    language: &'static str,
    op: String,
    size: usize,
    iters: usize,
    warmup: usize,
    seed: u64,
    median_ns: u64,
    mean_ns: f64,
    min_ns: u64,
    max_ns: u64,
    /// Scalar checksum for parity (sum of result values, or the scalar itself).
    checksum: f64,
}

fn median_ns(mut samples: Vec<u64>) -> u64 {
    samples.sort_unstable();
    let n = samples.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        samples[n / 2]
    } else {
        let a = samples[n / 2 - 1];
        let b = samples[n / 2];
        a / 2 + b / 2 + (a % 2 + b % 2) / 2
    }
}

fn checksum_array(a: &NdArray) -> f64 {
    a.sum()
}

fn op_name(op: &Op) -> String {
    format!("{:?}", op)
        .chars()
        .enumerate()
        .flat_map(|(i, c)| {
            if i > 0 && c.is_uppercase() {
                vec!['_', c.to_ascii_lowercase()]
            } else {
                vec![c.to_ascii_lowercase()]
            }
        })
        .collect()
}

/// Prepare inputs and return (checksum, timed closure that only runs the op).
fn run_op(op: &Op, size: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    let n = size;
    match op {
        Op::Zeros => {
            let shape = [n, n];
            let checksum = checksum_array(&zeros(&shape));
            (checksum, Box::new(move || {
                std::hint::black_box(zeros(&shape));
            }))
        }
        Op::Ones => {
            let shape = [n, n];
            let checksum = checksum_array(&ones(&shape));
            (checksum, Box::new(move || {
                std::hint::black_box(ones(&shape));
            }))
        }
        Op::Full => {
            let shape = [n, n];
            let checksum = checksum_array(&full(&shape, 3.5));
            (checksum, Box::new(move || {
                std::hint::black_box(full(&shape, 3.5));
            }))
        }
        Op::Arange => {
            let stop = n as f64;
            let checksum = checksum_array(&arange(0.0, stop, 1.0));
            (checksum, Box::new(move || {
                std::hint::black_box(arange(0.0, stop, 1.0));
            }))
        }
        Op::Linspace => {
            let checksum = checksum_array(&linspace(0.0, 1.0, n));
            (checksum, Box::new(move || {
                std::hint::black_box(linspace(0.0, 1.0, n));
            }))
        }
        Op::Eye => {
            let checksum = checksum_array(&eye(n));
            (checksum, Box::new(move || {
                std::hint::black_box(eye(n));
            }))
        }
        Op::Add => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&add(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(add(&a, &b));
            }))
        }
        Op::Subtract => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&subtract(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(subtract(&a, &b));
            }))
        }
        Op::Multiply => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&multiply(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(multiply(&a, &b));
            }))
        }
        Op::Divide => {
            let a = seeded_uniform(&[n, n], seed, 0.5, 1.5);
            let b = seeded_uniform(&[n, n], seed + 1, 0.5, 1.5);
            let checksum = checksum_array(&divide(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(divide(&a, &b));
            }))
        }
        Op::Power => {
            let a = seeded_uniform(&[n], seed, 0.5, 1.5);
            let b = seeded_uniform(&[n], seed + 1, 0.5, 2.0);
            let checksum = checksum_array(&power(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(power(&a, &b));
            }))
        }
        Op::Sqrt => {
            let a = seeded_uniform(&[n, n], seed, 0.0, 10.0);
            let checksum = checksum_array(&sqrt(&a));
            (checksum, Box::new(move || {
                std::hint::black_box(sqrt(&a));
            }))
        }
        Op::Exp => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&exp(&a));
            (checksum, Box::new(move || {
                std::hint::black_box(exp(&a));
            }))
        }
        Op::Log => {
            let a = seeded_uniform(&[n], seed, 0.1, 10.0);
            let checksum = checksum_array(&log(&a));
            (checksum, Box::new(move || {
                std::hint::black_box(log(&a));
            }))
        }
        Op::Negative => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&negative(&a));
            (checksum, Box::new(move || {
                std::hint::black_box(negative(&a));
            }))
        }
        Op::Abs => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&abs(&a));
            (checksum, Box::new(move || {
                std::hint::black_box(abs(&a));
            }))
        }
        Op::Sum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = sum(&a);
            (checksum, Box::new(move || {
                std::hint::black_box(sum(&a));
            }))
        }
        Op::Mean => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = mean(&a);
            (checksum, Box::new(move || {
                std::hint::black_box(mean(&a));
            }))
        }
        Op::Min => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = min(&a);
            (checksum, Box::new(move || {
                std::hint::black_box(min(&a));
            }))
        }
        Op::Max => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = max(&a);
            (checksum, Box::new(move || {
                std::hint::black_box(max(&a));
            }))
        }
        Op::Var => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = var(&a);
            (checksum, Box::new(move || {
                std::hint::black_box(var(&a));
            }))
        }
        Op::Std => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = np_std(&a);
            (checksum, Box::new(move || {
                std::hint::black_box(np_std(&a));
            }))
        }
        Op::Argmin => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = argmin(&a) as f64;
            (checksum, Box::new(move || {
                std::hint::black_box(argmin(&a));
            }))
        }
        Op::Argmax => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = argmax(&a) as f64;
            (checksum, Box::new(move || {
                std::hint::black_box(argmax(&a));
            }))
        }
        Op::Transpose => {
            let a = seeded_uniform(&[n, n + 1], seed, -1.0, 1.0);
            let checksum = checksum_array(&transpose(&a));
            (checksum, Box::new(move || {
                std::hint::black_box(transpose(&a));
            }))
        }
        Op::Matmul => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&matmul(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(matmul(&a, &b));
            }))
        }
        Op::Dot => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&dot(&a, &b));
            (checksum, Box::new(move || {
                std::hint::black_box(dot(&a, &b));
            }))
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.iters == 0 {
        bail!("iters must be > 0");
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
    let report = Report {
        language: "rust",
        op: op_name(&args.op),
        size: args.size,
        iters: args.iters,
        warmup: args.warmup,
        seed: args.seed,
        median_ns: median_ns(samples.clone()),
        mean_ns,
        min_ns: *samples.iter().min().unwrap(),
        max_ns: *samples.iter().max().unwrap(),
        checksum,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
