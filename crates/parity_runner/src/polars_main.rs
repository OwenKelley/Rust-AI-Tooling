//! CLI for Polars-shaped parity harness.

use std::env;
use std::process;
use std::time::Instant;

use rnumpy::seeded_uniform;
use rpolars::{
    col, lit_f64, read_csv_str, write_csv_string, Agg, DataFrame, JoinHow, Series,
};

#[derive(Debug, Clone)]
enum Op {
    Construct,
    Select,
    FilterGt,
    WithColumns,
    DropRename,
    GroupbySum,
    JoinInner,
    JoinLeft,
    Sort,
    HeadTail,
    CsvRoundtrip,
    LazyFilterSelect,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "construct" => Self::Construct,
            "select" => Self::Select,
            "filter_gt" => Self::FilterGt,
            "with_columns" => Self::WithColumns,
            "drop_rename" => Self::DropRename,
            "groupby_sum" => Self::GroupbySum,
            "join_inner" => Self::JoinInner,
            "join_left" => Self::JoinLeft,
            "sort" => Self::Sort,
            "head_tail" => Self::HeadTail,
            "csv_roundtrip" => Self::CsvRoundtrip,
            "lazy_filter_select" => Self::LazyFilterSelect,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Construct => "construct",
            Self::Select => "select",
            Self::FilterGt => "filter_gt",
            Self::WithColumns => "with_columns",
            Self::DropRename => "drop_rename",
            Self::GroupbySum => "groupby_sum",
            Self::JoinInner => "join_inner",
            Self::JoinLeft => "join_left",
            Self::Sort => "sort",
            Self::HeadTail => "head_tail",
            Self::CsvRoundtrip => "csv_roundtrip",
            Self::LazyFilterSelect => "lazy_filter_select",
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

fn make_df(n: usize, seed: u64) -> DataFrame {
    let a = seeded_uniform(&[n], seed, -1.0, 1.0);
    let vals: Vec<f64> = a.as_slice().unwrap().to_vec();
    let b: Vec<Option<i64>> = vals
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i % 5 == 0 {
                None
            } else {
                Some(v.floor() as i64)
            }
        })
        .collect();
    let k: Vec<Option<String>> = (0..n)
        .map(|i| Some(if i % 2 == 0 { "x" } else { "y" }.into()))
        .collect();
    DataFrame::new(vec![
        Series::from_f64("a", vals),
        Series::from_i64("b", b),
        Series::from_utf8("k", k),
    ])
}

fn run_op(op: &Op, n: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    let df = make_df(n.max(8), seed);
    match op {
        Op::Construct => {
            let checksum = df.checksum();
            let n = n.max(8);
            let seed = seed;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(make_df(n, seed));
                }),
            )
        }
        Op::Select => {
            let out = df.select(&["a", "k"]);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.select(&["a", "k"]));
                }),
            )
        }
        Op::FilterGt => {
            let pred = col("a").gt(lit_f64(0.0));
            let out = df.filter(&pred);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.filter(&col("a").gt(lit_f64(0.0))));
                }),
            )
        }
        Op::WithColumns => {
            let extra = Series::from_f64(
                "c",
                (0..df.height()).map(|i| i as f64).collect(),
            );
            let out = df.with_columns(vec![extra.clone()]);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.with_columns(vec![extra.clone()]));
                }),
            )
        }
        Op::DropRename => {
            let out = df.drop(&["b"]).rename(&[("a", "alpha")]);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.drop(&["b"]).rename(&[("a", "alpha")]));
                }),
            )
        }
        Op::GroupbySum => {
            let out = df.groupby(&["k"]).agg(&[("a", Agg::Sum), ("a", Agg::Count)]);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.groupby(&["k"]).agg(&[("a", Agg::Sum), ("a", Agg::Count)]));
                }),
            )
        }
        Op::JoinInner | Op::JoinLeft => {
            let right = DataFrame::new(vec![
                Series::from_utf8("k", vec![Some("x".into()), Some("y".into())]),
                Series::from_f64("v", vec![1.0, 2.0]),
            ]);
            let how = if matches!(op, Op::JoinInner) {
                JoinHow::Inner
            } else {
                JoinHow::Left
            };
            let out = df.join(&right, &["k"], how);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.join(&right, &["k"], how));
                }),
            )
        }
        Op::Sort => {
            let out = df.sort(&["a"], false);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.sort(&["a"], false));
                }),
            )
        }
        Op::HeadTail => {
            let out = df.head(3);
            let checksum = out.checksum() + df.tail(2).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.head(3));
                    std::hint::black_box(df.tail(2));
                }),
            )
        }
        Op::CsvRoundtrip => {
            let csv = write_csv_string(&df);
            let back = read_csv_str(&csv);
            let checksum = back.checksum();
            (
                checksum,
                Box::new(move || {
                    let csv = write_csv_string(&df);
                    std::hint::black_box(read_csv_str(&csv));
                }),
            )
        }
        Op::LazyFilterSelect => {
            let out = df
                .clone()
                .lazy()
                .filter(col("a").gt(lit_f64(0.0)))
                .select(&["a", "k"])
                .collect();
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(
                        df.clone()
                            .lazy()
                            .filter(col("a").gt(lit_f64(0.0)))
                            .select(&["a", "k"])
                            .collect(),
                    );
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
    let (checksum, mut thunk) = run_op(&args.op, args.size, args.seed);
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
