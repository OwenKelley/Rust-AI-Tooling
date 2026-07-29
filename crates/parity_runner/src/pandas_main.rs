//! CLI used by the Python Pandas comparison harness.
//!
//! Same contract as other parity runners: prepare once, time the core op, emit JSON.

use std::env;
use std::process;
use std::time::Instant;

use rnumpy::{seeded_uniform, NdArray};
use rpandas::{
    describe, dropna, fillna, filter_gt, groupby_agg, mean, melt, merge, pivot_table, read_csv_str,
    rolling_mean, rolling_mean_frame, sort_values, sum, to_csv_string, Agg, Column, DataFrame,
    MergeHow,
};

#[derive(Debug, Clone)]
enum Op {
    Construct,
    Select,
    Head,
    FilterGt,
    SortValues,
    Dropna,
    Fillna,
    Sum,
    Mean,
    Describe,
    GroupbySum,
    MergeInner,
    MergeLeft,
    CsvRoundtrip,
    Melt,
    PivotSum,
    RollingMean,
    MixedDtypes,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "construct" => Self::Construct,
            "select" => Self::Select,
            "head" => Self::Head,
            "filter_gt" => Self::FilterGt,
            "sort_values" => Self::SortValues,
            "dropna" => Self::Dropna,
            "fillna" => Self::Fillna,
            "sum" => Self::Sum,
            "mean" => Self::Mean,
            "describe" => Self::Describe,
            "groupby_sum" => Self::GroupbySum,
            "merge_inner" => Self::MergeInner,
            "merge_left" => Self::MergeLeft,
            "csv_roundtrip" => Self::CsvRoundtrip,
            "melt" => Self::Melt,
            "pivot_sum" => Self::PivotSum,
            "rolling_mean" => Self::RollingMean,
            "mixed_dtypes" => Self::MixedDtypes,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Construct => "construct",
            Self::Select => "select",
            Self::Head => "head",
            Self::FilterGt => "filter_gt",
            Self::SortValues => "sort_values",
            Self::Dropna => "dropna",
            Self::Fillna => "fillna",
            Self::Sum => "sum",
            Self::Mean => "mean",
            Self::Describe => "describe",
            Self::GroupbySum => "groupby_sum",
            Self::MergeInner => "merge_inner",
            Self::MergeLeft => "merge_left",
            Self::CsvRoundtrip => "csv_roundtrip",
            Self::Melt => "melt",
            Self::PivotSum => "pivot_sum",
            Self::RollingMean => "rolling_mean",
            Self::MixedDtypes => "mixed_dtypes",
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
        "Usage: pandas_parity_runner --op <name> [--size N] [--iters N] [--warmup N] [--seed N]"
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
    checksum: f64,
}

impl Report {
    fn to_json(&self) -> String {
        format!(
            "{{\n  \"language\": \"{}\",\n  \"op\": \"{}\",\n  \"size\": {},\n  \"iters\": {},\n  \"warmup\": {},\n  \"seed\": {},\n  \"median_ns\": {},\n  \"mean_ns\": {:.6},\n  \"min_ns\": {},\n  \"max_ns\": {},\n  \"checksum\": {:.17e}\n}}",
            self.language,
            self.op,
            self.size,
            self.iters,
            self.warmup,
            self.seed,
            self.median_ns,
            self.mean_ns,
            self.min_ns,
            self.max_ns,
            self.checksum
        )
    }
}

fn numeric_frame(n: usize, seed: u64, ncols: usize) -> DataFrame {
    let data = seeded_uniform(&[n, ncols], seed, -1.0, 1.0);
    let names: Vec<String> = (0..ncols).map(|j| format!("c{j}")).collect();
    let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    DataFrame::from_numeric(&name_refs, &data)
}

fn with_nans(df: DataFrame, every: usize) -> DataFrame {
    if every == 0 {
        return df;
    }
    let names: Vec<String> = df.column_names().into_iter().map(str::to_string).collect();
    let mut out = df;
    for name in names {
        let xs = out.float_slice(&name);
        let data: Vec<f64> = xs
            .into_iter()
            .enumerate()
            .map(|(i, x)| if i % every == 0 { f64::NAN } else { x })
            .collect();
        out = out.with_column(name, Column::Float64(NdArray::from_vec(data)));
    }
    out
}

fn frame_with_group_key(n: usize, seed: u64) -> DataFrame {
    let df = numeric_frame(n, seed, 3);
    let c0 = df.float_slice("c0");
    // Discrete groups in {0,1,2,3}
    let g: Vec<f64> = c0
        .iter()
        .map(|&x| ((x + 1.0) * 2.0).floor().clamp(0.0, 3.0))
        .collect();
    df.with_column("g", Column::Float64(NdArray::from_vec(g)))
}

fn merge_frames(n: usize, seed: u64) -> (DataFrame, DataFrame) {
    let n = n.max(4);
    let half = n / 2;
    // Left keys 0..n-1, right keys half..(half+n)
    let left_k: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let left_v = seeded_uniform(&[n], seed, -1.0, 1.0);
    let right_k: Vec<f64> = (0..n).map(|i| (i + half) as f64).collect();
    let right_w = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
    let left = DataFrame::from_columns(vec![
        ("k".into(), Column::Float64(NdArray::from_vec(left_k))),
        (
            "v".into(),
            Column::Float64(NdArray::from_vec(left_v.to_contiguous().as_slice().unwrap().to_vec())),
        ),
    ]);
    let right = DataFrame::from_columns(vec![
        ("k".into(), Column::Float64(NdArray::from_vec(right_k))),
        (
            "w".into(),
            Column::Float64(NdArray::from_vec(right_w.to_contiguous().as_slice().unwrap().to_vec())),
        ),
    ]);
    (left, right)
}

fn pivot_source(n: usize, seed: u64) -> DataFrame {
    let n = n.max(8);
    let vals = seeded_uniform(&[n], seed, -1.0, 1.0);
    let vals_c = vals.to_contiguous();
    let vs = vals_c.as_slice().unwrap();
    let mut i = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let nulls = vec![false; n];
    let mut v = Vec::with_capacity(n);
    for r in 0..n {
        i.push((r % 4) as f64);
        c.push(format!("g{}", r % 3));
        v.push(vs[r]);
    }
    DataFrame::from_columns(vec![
        ("i".into(), Column::Float64(NdArray::from_vec(i))),
        (
            "c".into(),
            Column::Utf8 {
                values: c,
                nulls,
            },
        ),
        ("v".into(), Column::Float64(NdArray::from_vec(v))),
    ])
}

fn mixed_frame(n: usize, seed: u64) -> DataFrame {
    let f = seeded_uniform(&[n], seed, -1.0, 1.0);
    let f_c = f.to_contiguous();
    let fs = f_c.as_slice().unwrap();
    let mut ints = Vec::with_capacity(n);
    let mut bools = Vec::with_capacity(n);
    let nulls_i = vec![false; n];
    let nulls_b = vec![false; n];
    for r in 0..n {
        ints.push((fs[r] * 10.0).floor() as i64);
        bools.push(fs[r] > 0.0);
    }
    DataFrame::from_columns(vec![
        (
            "f".into(),
            Column::Float64(NdArray::from_vec(fs.to_vec())),
        ),
        (
            "i".into(),
            Column::Int64 {
                values: ints,
                nulls: nulls_i,
            },
        ),
        (
            "b".into(),
            Column::Bool {
                values: bools,
                nulls: nulls_b,
            },
        ),
    ])
}

fn run_op(op: &Op, n: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    match op {
        Op::Construct => {
            let checksum = numeric_frame(n, seed, 4).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(numeric_frame(n, seed, 4));
                }),
            )
        }
        Op::Select => {
            let df = numeric_frame(n, seed, 4);
            let checksum = df.select(&["c0", "c2"]).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.select(&["c0", "c2"]));
                }),
            )
        }
        Op::Head => {
            let df = numeric_frame(n, seed, 3);
            let k = (n / 4).max(1);
            let checksum = df.head(k).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(df.head(k));
                }),
            )
        }
        Op::FilterGt => {
            let df = numeric_frame(n, seed, 3);
            let checksum = filter_gt(&df, "c0", 0.0).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(filter_gt(&df, "c0", 0.0));
                }),
            )
        }
        Op::SortValues => {
            let df = numeric_frame(n, seed, 3);
            let checksum = sort_values(&df, "c0", true).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sort_values(&df, "c0", true));
                }),
            )
        }
        Op::Dropna => {
            let df = with_nans(numeric_frame(n, seed, 3), 7);
            let checksum = dropna(&df, "any").checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dropna(&df, "any"));
                }),
            )
        }
        Op::Fillna => {
            let df = with_nans(numeric_frame(n, seed, 3), 7);
            let checksum = fillna(&df, 0.0, None).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(fillna(&df, 0.0, None));
                }),
            )
        }
        Op::Sum => {
            let df = numeric_frame(n, seed, 4);
            let checksum = sum(&df).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sum(&df));
                }),
            )
        }
        Op::Mean => {
            let df = numeric_frame(n, seed, 4);
            let checksum = mean(&df).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mean(&df));
                }),
            )
        }
        Op::Describe => {
            let df = numeric_frame(n, seed, 3);
            let checksum = describe(&df).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(describe(&df));
                }),
            )
        }
        Op::GroupbySum => {
            let df = frame_with_group_key(n, seed);
            let out = groupby_agg(&df, "g", &[("c1", Agg::Sum)]);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(groupby_agg(&df, "g", &[("c1", Agg::Sum)]));
                }),
            )
        }
        Op::MergeInner => {
            let (left, right) = merge_frames(n, seed);
            let checksum = merge(&left, &right, "k", MergeHow::Inner).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(merge(&left, &right, "k", MergeHow::Inner));
                }),
            )
        }
        Op::MergeLeft => {
            let (left, right) = merge_frames(n, seed);
            let checksum = merge(&left, &right, "k", MergeHow::Left).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(merge(&left, &right, "k", MergeHow::Left));
                }),
            )
        }
        Op::CsvRoundtrip => {
            let df = numeric_frame(n, seed, 3);
            let text = to_csv_string(&df);
            let checksum = read_csv_str(&text).checksum();
            (
                checksum,
                Box::new(move || {
                    let t = to_csv_string(&df);
                    std::hint::black_box(read_csv_str(&t));
                }),
            )
        }
        Op::Melt => {
            let df = numeric_frame(n, seed, 3);
            let checksum = melt(&df, &["c0"], &["c1", "c2"]).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(melt(&df, &["c0"], &["c1", "c2"]));
                }),
            )
        }
        Op::PivotSum => {
            let df = pivot_source(n, seed);
            let checksum = pivot_table(&df, "i", "c", "v", Agg::Sum).checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(pivot_table(&df, "i", "c", "v", Agg::Sum));
                }),
            )
        }
        Op::RollingMean => {
            let df = numeric_frame(n, seed, 1);
            let window = 5usize;
            let col = rolling_mean(&df, "c0", window);
            let out = DataFrame::from_columns(vec![("c0".into(), col)]);
            let checksum = out.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(rolling_mean_frame(&df, window));
                }),
            )
        }
        Op::MixedDtypes => {
            let df = mixed_frame(n, seed);
            let checksum = df.checksum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mixed_frame(n, seed));
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
    let report = Report {
        language: "rust",
        op: args.op.as_str().to_string(),
        size: args.size,
        iters: args.iters,
        warmup: args.warmup,
        seed: args.seed,
        median_ns: median_u64(&samples),
        mean_ns,
        min_ns: *samples.iter().min().unwrap(),
        max_ns: *samples.iter().max().unwrap(),
        checksum,
    };
    println!("{}", report.to_json());
}
