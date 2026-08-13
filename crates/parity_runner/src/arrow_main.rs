//! CLI for Arrow / Parquet parity harness.

use std::env;
use std::process;
use std::time::Instant;

use rarrow::{
    batch_from_columns, read_ipc_file, read_ipc_stream, read_parquet, write_ipc_file,
    write_ipc_stream, write_parquet, Array, Float64Array, Int64Array, StringArray,
};
use rnumpy::seeded_uniform;

#[derive(Debug, Clone)]
enum Op {
    IpcRoundtrip,
    IpcRead,
    IpcWritePyarrowRead,
    IpcFileRoundtrip,
    ParquetPar1Roundtrip,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "ipc_roundtrip" => Self::IpcRoundtrip,
            "ipc_read" => Self::IpcRead,
            "ipc_write_pyarrow_read" => Self::IpcWritePyarrowRead,
            "ipc_file_roundtrip" => Self::IpcFileRoundtrip,
            "parquet_roundtrip" | "parquet_par1_roundtrip" => Self::ParquetPar1Roundtrip,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::IpcRoundtrip => "ipc_roundtrip",
            Self::IpcRead => "ipc_read",
            Self::IpcWritePyarrowRead => "ipc_write_pyarrow_read",
            Self::IpcFileRoundtrip => "ipc_file_roundtrip",
            Self::ParquetPar1Roundtrip => "parquet_par1_roundtrip",
        }
    }
}

struct Args {
    op: Op,
    size: usize,
    seed: u64,
    iters: usize,
    warmup: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut op = None;
    let mut size = 64usize;
    let mut seed = 42u64;
    let mut iters = 20usize;
    let mut warmup = 3usize;
    let mut args = env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--op" => {
                op = Some(Op::parse(
                    &args.next().ok_or_else(|| "--op needs value".to_string())?,
                )?);
            }
            "--size" => {
                size = args
                    .next()
                    .ok_or_else(|| "--size needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("size: {e}"))?;
            }
            "--seed" => {
                seed = args
                    .next()
                    .ok_or_else(|| "--seed needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("seed: {e}"))?;
            }
            "--iters" => {
                iters = args
                    .next()
                    .ok_or_else(|| "--iters needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("iters: {e}"))?;
            }
            "--warmup" => {
                warmup = args
                    .next()
                    .ok_or_else(|| "--warmup needs value".to_string())?
                    .parse()
                    .map_err(|e| format!("warmup: {e}"))?;
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(Args {
        op: op.ok_or_else(|| "missing --op".to_string())?,
        size,
        seed,
        iters,
        warmup,
    })
}

fn make_batch(n: usize, seed: u64) -> rarrow::RecordBatch {
    let a = seeded_uniform(&[n], seed, -1.0, 1.0);
    let vals: Vec<f64> = a.as_slice().unwrap().to_vec();
    let mut i64s = Vec::with_capacity(n);
    let mut nulls = Vec::with_capacity(n);
    for (i, &v) in vals.iter().enumerate() {
        if i % 5 == 0 {
            i64s.push(0);
            nulls.push(true);
        } else {
            i64s.push(v.floor() as i64);
            nulls.push(false);
        }
    }
    let strings: Vec<Option<String>> = (0..n)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else {
                Some(format!("s{i}"))
            }
        })
        .collect();
    batch_from_columns(vec![
        (
            "a".into(),
            Array::Float64(Float64Array {
                values: vals,
                nulls: vec![false; n],
            }),
        ),
        (
            "b".into(),
            Array::Int64(Int64Array {
                values: i64s,
                nulls,
            }),
        ),
        ("c".into(), Array::Utf8(StringArray { values: strings })),
    ])
}

fn run_op(op: &Op, n: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    match op {
        Op::IpcRoundtrip
        | Op::IpcRead
        | Op::IpcWritePyarrowRead
        | Op::IpcFileRoundtrip
        | Op::ParquetPar1Roundtrip => {
            let batch = make_batch(n, seed);
            let checksum = batch.checksum();
            let kind = match op {
                Op::ParquetPar1Roundtrip => 2u8,
                Op::IpcFileRoundtrip => 1u8,
                Op::IpcWritePyarrowRead => {
                    let path = std::env::temp_dir().join("rarrow_parity_ipc.stream");
                    let _ = std::fs::write(&path, write_ipc_stream(&batch));
                    0u8
                }
                _ => 0u8,
            };
            (
                checksum,
                Box::new(move || match kind {
                    2 => {
                        let bytes = write_parquet(&batch);
                        std::hint::black_box(read_parquet(&bytes));
                    }
                    1 => {
                        let bytes = write_ipc_file(&batch);
                        std::hint::black_box(read_ipc_file(&bytes));
                    }
                    _ => {
                        let bytes = write_ipc_stream(&batch);
                        std::hint::black_box(read_ipc_stream(&bytes));
                    }
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
            process::exit(2);
        }
    };
    let (checksum, mut thunk) = run_op(&args.op, args.size.max(8), args.seed);
    for _ in 0..args.warmup {
        thunk();
    }
    let mut times = Vec::with_capacity(args.iters);
    for _ in 0..args.iters {
        let t0 = Instant::now();
        thunk();
        times.push(t0.elapsed().as_nanos() as u64);
    }
    times.sort_unstable();
    let median = times[times.len() / 2];
    println!(
        "{{\"op\":\"{}\",\"checksum\":{checksum},\"median_ns\":{median},\"iters\":{}}}",
        args.op.as_str(),
        args.iters
    );
}
