//! CLI for scikit-learn-shaped parity harness.

use std::env;
use std::process;
use std::time::Instant;

use rnumpy::{seeded_uniform, NdArray};
use rsklearn::{
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, train_test_split, CountVectorizer, HashingVectorizer, KMeans,
    KNeighborsClassifier, KNeighborsRegressor, LabelEncoder, LinearRegression, LogisticRegression,
    StandardScaler,
};

#[derive(Debug, Clone)]
enum Op {
    StandardScaler,
    LabelEncoder,
    TrainTestSplit,
    LinearRegression,
    LogisticRegression,
    KnnClassify,
    KnnRegress,
    KMeans,
    MetricsClass,
    MetricsReg,
    HashingVectorizer,
    CountVectorizer,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "standard_scaler" => Self::StandardScaler,
            "label_encoder" => Self::LabelEncoder,
            "train_test_split" => Self::TrainTestSplit,
            "linear_regression" => Self::LinearRegression,
            "logistic_regression" => Self::LogisticRegression,
            "knn_classify" => Self::KnnClassify,
            "knn_regress" => Self::KnnRegress,
            "kmeans" => Self::KMeans,
            "metrics_class" => Self::MetricsClass,
            "metrics_reg" => Self::MetricsReg,
            "hashing_vectorizer" => Self::HashingVectorizer,
            "count_vectorizer" => Self::CountVectorizer,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::StandardScaler => "standard_scaler",
            Self::LabelEncoder => "label_encoder",
            Self::TrainTestSplit => "train_test_split",
            Self::LinearRegression => "linear_regression",
            Self::LogisticRegression => "logistic_regression",
            Self::KnnClassify => "knn_classify",
            Self::KnnRegress => "knn_regress",
            Self::KMeans => "kmeans",
            Self::MetricsClass => "metrics_class",
            Self::MetricsReg => "metrics_reg",
            Self::HashingVectorizer => "hashing_vectorizer",
            Self::CountVectorizer => "count_vectorizer",
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

fn make_x(n: usize, d: usize, seed: u64) -> NdArray {
    seeded_uniform(&[n, d], seed, -1.0, 1.0)
}

fn make_docs(n: usize, seed: u64) -> Vec<String> {
    let words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "cat", "bird"];
    let mut state = seed | 1;
    (0..n)
        .map(|i| {
            let mut parts = Vec::new();
            for _ in 0..(3 + (i % 3)) {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                parts.push(words[(state as usize) % words.len()]);
            }
            parts.join(" ")
        })
        .collect()
}

fn checksum_array(a: &NdArray) -> f64 {
    let n = a.len();
    let mut s = n as f64;
    for i in 0..n {
        s += a.get_flat(i);
    }
    s
}

fn checksum_f64(v: &[f64]) -> f64 {
    v.len() as f64 + v.iter().sum::<f64>()
}

fn checksum_i64(v: &[i64]) -> f64 {
    v.len() as f64 + v.iter().map(|&x| x as f64).sum::<f64>()
}

fn run_op(op: &Op, n: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    let n = n.max(16);
    let d = 3usize;
    match op {
        Op::StandardScaler => {
            let x = make_x(n, d, seed);
            let mut sc = StandardScaler::new();
            let out = sc.fit_transform(&x);
            let checksum = checksum_array(&out);
            (
                checksum,
                Box::new(move || {
                    let mut sc = StandardScaler::new();
                    std::hint::black_box(sc.fit_transform(&x));
                }),
            )
        }
        Op::LabelEncoder => {
            let labels: Vec<String> = (0..n)
                .map(|i| match i % 3 {
                    0 => "a".into(),
                    1 => "b".into(),
                    _ => "c".into(),
                })
                .collect();
            let mut enc = LabelEncoder::new();
            let codes = enc.fit_transform(&labels);
            let checksum = checksum_i64(&codes);
            (
                checksum,
                Box::new(move || {
                    let mut enc = LabelEncoder::new();
                    std::hint::black_box(enc.fit_transform(&labels));
                }),
            )
        }
        Op::TrainTestSplit => {
            let x = make_x(n, d, seed);
            let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
            // shuffle=false for exact sklearn match
            let (xtr, xte, ytr, yte) = train_test_split(&x, &y, 0.25, seed, false);
            let checksum = checksum_array(&xtr)
                + checksum_array(&xte)
                + checksum_f64(&ytr)
                + checksum_f64(&yte);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(train_test_split(&x, &y, 0.25, seed, false));
                }),
            )
        }
        Op::LinearRegression => {
            let x = make_x(n, d, seed);
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    1.0 + 2.0 * x.get(&[i, 0]) + 3.0 * x.get(&[i, 1]) - 0.5 * x.get(&[i, 2])
                })
                .collect();
            let mut lr = LinearRegression::new();
            lr.fit(&x, &y);
            let pred = lr.predict(&x);
            let checksum = checksum_f64(&lr.coef_) + lr.intercept_ + checksum_f64(&pred);
            (
                checksum,
                Box::new(move || {
                    let mut lr = LinearRegression::new();
                    lr.fit(&x, &y);
                    std::hint::black_box(lr.predict(&x));
                }),
            )
        }
        Op::LogisticRegression => {
            let x = make_x(n, d, seed);
            let y: Vec<f64> = (0..n)
                .map(|i| if x.get(&[i, 0]) + x.get(&[i, 1]) > 0.0 { 1.0 } else { 0.0 })
                .collect();
            let mut lr = LogisticRegression::new();
            lr.lr = 0.5;
            lr.max_iter = 800;
            lr.fit(&x, &y);
            let pred = lr.predict(&x);
            let y_i: Vec<i64> = y.iter().map(|&v| v as i64).collect();
            let checksum = accuracy_score(&y_i, &pred);
            (
                checksum,
                Box::new(move || {
                    let mut lr = LogisticRegression::new();
                    lr.lr = 0.5;
                    lr.max_iter = 800;
                    lr.fit(&x, &y);
                    std::hint::black_box(lr.predict(&x));
                }),
            )
        }
        Op::KnnClassify => {
            let x = make_x(n, d, seed);
            let y: Vec<i64> = (0..n).map(|i| (i % 3) as i64).collect();
            let mut knn = KNeighborsClassifier::new(3);
            knn.fit(&x, &y);
            let pred = knn.predict(&x);
            let checksum = checksum_i64(&pred);
            (
                checksum,
                Box::new(move || {
                    let mut knn = KNeighborsClassifier::new(3);
                    knn.fit(&x, &y);
                    std::hint::black_box(knn.predict(&x));
                }),
            )
        }
        Op::KnnRegress => {
            let x = make_x(n, d, seed);
            let y: Vec<f64> = (0..n).map(|i| x.get(&[i, 0]) + 0.1 * i as f64).collect();
            let mut knn = KNeighborsRegressor::new(3);
            knn.fit(&x, &y);
            let pred = knn.predict(&x);
            let checksum = checksum_f64(&pred);
            (
                checksum,
                Box::new(move || {
                    let mut knn = KNeighborsRegressor::new(3);
                    knn.fit(&x, &y);
                    std::hint::black_box(knn.predict(&x));
                }),
            )
        }
        Op::KMeans => {
            let x = make_x(n, d, seed);
            let mut km = KMeans::new(3);
            km.random_state = seed;
            km.fit(&x);
            // Inertia (permutation-invariant).
            let centers = km.cluster_centers_.as_ref().unwrap();
            let mut inertia = 0.0;
            for i in 0..n {
                let c = km.labels_[i] as usize;
                for j in 0..d {
                    let diff = x.get(&[i, j]) - centers.get(&[c, j]);
                    inertia += diff * diff;
                }
            }
            let checksum = inertia;
            (
                checksum,
                Box::new(move || {
                    let mut km = KMeans::new(3);
                    km.random_state = seed;
                    std::hint::black_box(km.fit(&x));
                }),
            )
        }
        Op::MetricsClass => {
            let y_true: Vec<i64> = (0..n).map(|i| (i % 2) as i64).collect();
            let y_pred: Vec<i64> = (0..n)
                .map(|i| if i % 5 == 0 { 1 - y_true[i] } else { y_true[i] })
                .collect();
            let checksum = accuracy_score(&y_true, &y_pred)
                + precision_score(&y_true, &y_pred, 1)
                + recall_score(&y_true, &y_pred, 1)
                + f1_score(&y_true, &y_pred, 1);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(accuracy_score(&y_true, &y_pred));
                    std::hint::black_box(f1_score(&y_true, &y_pred, 1));
                }),
            )
        }
        Op::MetricsReg => {
            let y_true: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
            let y_pred: Vec<f64> = y_true.iter().map(|&v| v + 0.05).collect();
            let checksum = mean_squared_error(&y_true, &y_pred)
                + mean_absolute_error(&y_true, &y_pred)
                + r2_score(&y_true, &y_pred);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mean_squared_error(&y_true, &y_pred));
                    std::hint::black_box(r2_score(&y_true, &y_pred));
                }),
            )
        }
        Op::HashingVectorizer => {
            let docs = make_docs(n, seed);
            let hv = HashingVectorizer::new(64);
            let out = hv.transform(&docs);
            let checksum = checksum_array(&out);
            (
                checksum,
                Box::new(move || {
                    let hv = HashingVectorizer::new(64);
                    std::hint::black_box(hv.transform(&docs));
                }),
            )
        }
        Op::CountVectorizer => {
            let docs = make_docs(n, seed);
            let mut cv = CountVectorizer::new();
            let out = cv.fit_transform(&docs);
            let checksum = checksum_array(&out);
            (
                checksum,
                Box::new(move || {
                    let mut cv = CountVectorizer::new();
                    std::hint::black_box(cv.fit_transform(&docs));
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
