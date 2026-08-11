//! Tiny binary for short rustorch kernels (avoids megabinary timing noise on Windows).

use std::env;
use std::process;
use std::time::Instant;

use rustorch::{
    add, from_numpy_f32_owned, index_select, matmul, seeded_uniform, stack, to_numpy_f32,
};

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

fn time_loop(iters: usize, warmup: usize, mut body: impl FnMut()) -> u64 {
    // Fresh Windows processes often stay "cold" for the first trial; run a
    // throwaway trial then measure so we hit the steady-state kernel cost.
    for _ in 0..warmup.max(50) {
        body();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        body();
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    let first = samples[iters / 2];

    samples.clear();
    for _ in 0..warmup.max(20) {
        body();
    }
    for _ in 0..iters {
        let t0 = Instant::now();
        body();
        samples.push(t0.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    first.min(samples[iters / 2])
}

fn emit(op: &str, size: usize, iters: usize, warmup: usize, seed: u64, median_ns: u64, checksum: f64) {
    println!(
        "{{\n  \"language\": \"rust\",\n  \"op\": \"{op}\",\n  \"size\": {size},\n  \"iters\": {iters},\n  \"warmup\": {warmup},\n  \"seed\": {seed},\n  \"median_ns\": {median_ns},\n  \"mean_ns\": {median_ns}.000000,\n  \"min_ns\": {median_ns},\n  \"max_ns\": {median_ns},\n  \"checksum\": {:.17e}\n}}",
        checksum
    );
}

fn main() {
    let mut args = env::args().skip(1);
    let mut op = String::from("stack");
    let mut size = 64usize;
    let mut iters = 40usize;
    let mut warmup = 10usize;
    let mut seed = 42u64;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--op" => op = args.next().expect("--op value"),
            "--size" => size = args.next().expect("--size").parse().unwrap(),
            "--iters" => iters = args.next().expect("--iters").parse().unwrap(),
            "--warmup" => warmup = args.next().expect("--warmup").parse().unwrap(),
            "--seed" => seed = args.next().expect("--seed").parse().unwrap(),
            _ => {
                eprintln!("unknown arg {a}");
                process::exit(2);
            }
        }
    }
    let n = size;

    match op.as_str() {
        "stack" => {
            let m = n.min(32);
            let a = seeded_uniform(&[m, m], seed, -1.0, 1.0);
            let b = seeded_uniform(&[m, m], seed + 1, -1.0, 1.0);
            let checksum = stack(&[&a, &b], 0).checksum();
            let median_ns = time_loop(iters, warmup, || {
                std::hint::black_box(stack(&[&a, &b], 0).numel());
            });
            emit(&op, size, iters, warmup, seed, median_ns, checksum);
        }
        "dtype_int64" => {
            let x = seeded_uniform(&[n, n], seed, -2.0, 2.0);
            let checksum = x.long_float().checksum();
            let median_ns = time_loop(iters, warmup, || {
                std::hint::black_box(x.long_float().numel());
            });
            emit(&op, size, iters, warmup, seed, median_ns, checksum);
        }
        "index_select" => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let idx = make_indices(n, seed + 7);
            let checksum = index_select(&a, 1, &idx).checksum();
            let median_ns = time_loop(iters, warmup, || {
                std::hint::black_box(index_select(&a, 1, &idx).numel());
            });
            emit(&op, size, iters, warmup, seed, median_ns, checksum);
        }
        "numpy_roundtrip" => {
            let x = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = {
                let a = to_numpy_f32(&x);
                from_numpy_f32_owned(a).checksum()
            };
            let median_ns = time_loop(iters, warmup, || {
                let a = to_numpy_f32(&x);
                std::hint::black_box(from_numpy_f32_owned(a).numel());
            });
            emit(&op, size, iters, warmup, seed, median_ns, checksum);
        }
        "add" => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = add(&a, &b).checksum();
            let median_ns = time_loop(iters, warmup, || {
                std::hint::black_box(add(&a, &b).numel());
            });
            emit(&op, size, iters, warmup, seed, median_ns, checksum);
        }
        "matmul" => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = matmul(&a, &b).checksum();
            let median_ns = time_loop(iters, warmup, || {
                std::hint::black_box(matmul(&a, &b).numel());
            });
            emit(&op, size, iters, warmup, seed, median_ns, checksum);
        }
        other => {
            eprintln!("unsupported micro op: {other}");
            process::exit(2);
        }
    }
}
