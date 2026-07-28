//! CLI used by the Python comparison harness.
//!
//! Emits one JSON object to stdout with timing stats and a checksum so the
//! Python side can verify numerical agreement and compare speed.
//!
//! Timing covers only the core op (inputs are prepared once beforehand).
//!
//! Argument parsing and JSON emission use only `std` (no clap/serde/anyhow).

use std::env;
use std::process;
use std::time::Instant;

use rnumpy::{
    abs, add, arange, argmax, argmin, broadcast_to, ceil, clip, compress, concatenate, cos, cumprod,
    cumsum, cumsum_axis, det, divide, dot, eigvalsh, equal, exp, eye, floor, full, greater, inv,
    less, linspace, log, matmul, max, max_axis, maximum, mean, mean_axis, min, min_axis, minimum,
    moveaxis, multiply, negative, norm, not_equal, ones, power, qr, ravel, reciprocal, reshape,
    reshape_infer, round, seeded_uniform, sign, sin, slice_array, solve, sqrt, square, stack,
    std as np_std, subtract, sum, sum_axis, svdvals, swapaxes, take, tan, tanh, trace, transpose,
    trunc, var, where_, zeros, AxisSlice, NdArray,
};

#[derive(Debug, Clone)]
enum Op {
    Zeros,
    Ones,
    Full,
    Arange,
    Linspace,
    Eye,
    Add,
    AddBroadcast,
    Subtract,
    Multiply,
    Divide,
    Power,
    Maximum,
    Minimum,
    Greater,
    Less,
    Equal,
    NotEqual,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tan,
    Tanh,
    Negative,
    Abs,
    Sign,
    Square,
    Reciprocal,
    Floor,
    Ceil,
    Trunc,
    Round,
    Clip,
    Where,
    Sum,
    SumAxis,
    Mean,
    MeanAxis,
    Min,
    MinAxis,
    Max,
    MaxAxis,
    Var,
    Std,
    Argmin,
    Argmax,
    Cumsum,
    CumsumAxis,
    Cumprod,
    Transpose,
    Reshape,
    ReshapeInfer,
    Ravel,
    Concatenate,
    Stack,
    BroadcastTo,
    Swapaxes,
    Moveaxis,
    Matmul,
    Dot,
    Trace,
    Norm,
    Solve,
    Inv,
    Det,
    Qr,
    Svdvals,
    Eigvalsh,
    Take,
    Compress,
    Slice,
    AstypeF32,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "zeros" => Self::Zeros,
            "ones" => Self::Ones,
            "full" => Self::Full,
            "arange" => Self::Arange,
            "linspace" => Self::Linspace,
            "eye" => Self::Eye,
            "add" => Self::Add,
            "add_broadcast" => Self::AddBroadcast,
            "subtract" => Self::Subtract,
            "multiply" => Self::Multiply,
            "divide" => Self::Divide,
            "power" => Self::Power,
            "maximum" => Self::Maximum,
            "minimum" => Self::Minimum,
            "greater" => Self::Greater,
            "less" => Self::Less,
            "equal" => Self::Equal,
            "not_equal" => Self::NotEqual,
            "sqrt" => Self::Sqrt,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tan" => Self::Tan,
            "tanh" => Self::Tanh,
            "negative" => Self::Negative,
            "abs" => Self::Abs,
            "sign" => Self::Sign,
            "square" => Self::Square,
            "reciprocal" => Self::Reciprocal,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "trunc" => Self::Trunc,
            "round" => Self::Round,
            "clip" => Self::Clip,
            "where" => Self::Where,
            "sum" => Self::Sum,
            "sum_axis" => Self::SumAxis,
            "mean" => Self::Mean,
            "mean_axis" => Self::MeanAxis,
            "min" => Self::Min,
            "min_axis" => Self::MinAxis,
            "max" => Self::Max,
            "max_axis" => Self::MaxAxis,
            "var" => Self::Var,
            "std" => Self::Std,
            "argmin" => Self::Argmin,
            "argmax" => Self::Argmax,
            "cumsum" => Self::Cumsum,
            "cumsum_axis" => Self::CumsumAxis,
            "cumprod" => Self::Cumprod,
            "transpose" => Self::Transpose,
            "reshape" => Self::Reshape,
            "reshape_infer" => Self::ReshapeInfer,
            "ravel" => Self::Ravel,
            "concatenate" => Self::Concatenate,
            "stack" => Self::Stack,
            "broadcast_to" => Self::BroadcastTo,
            "swapaxes" => Self::Swapaxes,
            "moveaxis" => Self::Moveaxis,
            "matmul" => Self::Matmul,
            "dot" => Self::Dot,
            "trace" => Self::Trace,
            "norm" => Self::Norm,
            "solve" => Self::Solve,
            "inv" => Self::Inv,
            "det" => Self::Det,
            "qr" => Self::Qr,
            "svdvals" => Self::Svdvals,
            "eigvalsh" => Self::Eigvalsh,
            "take" => Self::Take,
            "compress" => Self::Compress,
            "slice" => Self::Slice,
            "astype_f32" => Self::AstypeF32,
            other => {
                return Err(format!(
                    "unknown op '{other}' (expected snake_case NumPy name, e.g. add, matmul)"
                ))
            }
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Zeros => "zeros",
            Self::Ones => "ones",
            Self::Full => "full",
            Self::Arange => "arange",
            Self::Linspace => "linspace",
            Self::Eye => "eye",
            Self::Add => "add",
            Self::AddBroadcast => "add_broadcast",
            Self::Subtract => "subtract",
            Self::Multiply => "multiply",
            Self::Divide => "divide",
            Self::Power => "power",
            Self::Maximum => "maximum",
            Self::Minimum => "minimum",
            Self::Greater => "greater",
            Self::Less => "less",
            Self::Equal => "equal",
            Self::NotEqual => "not_equal",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Tanh => "tanh",
            Self::Negative => "negative",
            Self::Abs => "abs",
            Self::Sign => "sign",
            Self::Square => "square",
            Self::Reciprocal => "reciprocal",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Trunc => "trunc",
            Self::Round => "round",
            Self::Clip => "clip",
            Self::Where => "where",
            Self::Sum => "sum",
            Self::SumAxis => "sum_axis",
            Self::Mean => "mean",
            Self::MeanAxis => "mean_axis",
            Self::Min => "min",
            Self::MinAxis => "min_axis",
            Self::Max => "max",
            Self::MaxAxis => "max_axis",
            Self::Var => "var",
            Self::Std => "std",
            Self::Argmin => "argmin",
            Self::Argmax => "argmax",
            Self::Cumsum => "cumsum",
            Self::CumsumAxis => "cumsum_axis",
            Self::Cumprod => "cumprod",
            Self::Transpose => "transpose",
            Self::Reshape => "reshape",
            Self::ReshapeInfer => "reshape_infer",
            Self::Ravel => "ravel",
            Self::Concatenate => "concatenate",
            Self::Stack => "stack",
            Self::BroadcastTo => "broadcast_to",
            Self::Swapaxes => "swapaxes",
            Self::Moveaxis => "moveaxis",
            Self::Matmul => "matmul",
            Self::Dot => "dot",
            Self::Trace => "trace",
            Self::Norm => "norm",
            Self::Solve => "solve",
            Self::Inv => "inv",
            Self::Det => "det",
            Self::Qr => "qr",
            Self::Svdvals => "svdvals",
            Self::Eigvalsh => "eigvalsh",
            Self::Take => "take",
            Self::Compress => "compress",
            Self::Slice => "slice",
            Self::AstypeF32 => "astype_f32",
        }
    }
}

fn diag_dominant(n: usize, seed: u64) -> NdArray {
    let mut a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    // Cap boost so det at large n does not overflow float64.
    let boost = (n as f64).min(4.0);
    for i in 0..n {
        a[[i, i]] += boost;
    }
    a
}

fn symmetric_spd(n: usize, seed: u64) -> NdArray {
    let mut a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
        a[[i, i]] += n as f64;
    }
    a
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
        "Usage: parity_runner --op <name> [--size N] [--iters N] [--warmup N] [--seed N]\n\
         \n\
         Required:\n\
           --op <snake_case>   Operation (e.g. add, matmul, sum)\n\
         \n\
         Optional (defaults):\n\
           --size 256\n\
           --iters 50\n\
           --warmup 5\n\
           --seed 42"
    );
    process::exit(2);
}

fn parse_args() -> Result<Args, String> {
    let mut op: Option<Op> = None;
    let mut size: usize = 256;
    let mut iters: usize = 50;
    let mut warmup: usize = 5;
    let mut seed: u64 = 42;

    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for '{flag}'"))?;
        match flag.as_str() {
            "--op" => op = Some(Op::parse(&value)?),
            "--size" => {
                size = value
                    .parse()
                    .map_err(|_| format!("invalid --size '{value}'"))?
            }
            "--iters" => {
                iters = value
                    .parse()
                    .map_err(|_| format!("invalid --iters '{value}'"))?
            }
            "--warmup" => {
                warmup = value
                    .parse()
                    .map_err(|_| format!("invalid --warmup '{value}'"))?
            }
            "--seed" => {
                seed = value
                    .parse()
                    .map_err(|_| format!("invalid --seed '{value}'"))?
            }
            "--help" | "-h" => usage(),
            other => return Err(format!("unknown argument '{other}'")),
        }
    }

    let op = op.ok_or_else(|| "missing required --op".to_string())?;
    Ok(Args {
        op,
        size,
        iters,
        warmup,
        seed,
    })
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
    /// Scalar checksum for parity (sum of result values, or the scalar itself).
    checksum: f64,
}

fn json_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn format_json_f64(x: f64) -> String {
    if x.is_finite() {
        // Ensure a decimal form JSON numbers accept (avoid bare integers looking wrong).
        let s = format!("{x}");
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{s}.0")
        }
    } else if x.is_nan() {
        // Parity harness expects a number; NaN is not valid JSON — emit null.
        "null".to_string()
    } else if x.is_sign_positive() {
        "null".to_string() // +inf
    } else {
        "null".to_string() // -inf
    }
}

fn report_to_pretty_json(r: &Report) -> String {
    format!(
        "{{\n  \"language\": {},\n  \"op\": {},\n  \"size\": {},\n  \"iters\": {},\n  \"warmup\": {},\n  \"seed\": {},\n  \"median_ns\": {},\n  \"mean_ns\": {},\n  \"min_ns\": {},\n  \"max_ns\": {},\n  \"checksum\": {}\n}}",
        json_escape_string(r.language),
        json_escape_string(&r.op),
        r.size,
        r.iters,
        r.warmup,
        r.seed,
        r.median_ns,
        format_json_f64(r.mean_ns),
        r.min_ns,
        r.max_ns,
        format_json_f64(r.checksum),
    )
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

/// Prepare inputs and return (checksum, timed closure that only runs the op).
fn run_op(op: &Op, size: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    let n = size;
    match op {
        Op::Zeros => {
            let shape = [n, n];
            let checksum = checksum_array(&zeros(&shape));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(zeros(&shape));
                }),
            )
        }
        Op::Ones => {
            let shape = [n, n];
            let checksum = checksum_array(&ones(&shape));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ones(&shape));
                }),
            )
        }
        Op::Full => {
            let shape = [n, n];
            let checksum = checksum_array(&full(&shape, 3.5));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(full(&shape, 3.5));
                }),
            )
        }
        Op::Arange => {
            let stop = n as f64;
            let checksum = checksum_array(&arange(0.0, stop, 1.0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(arange(0.0, stop, 1.0));
                }),
            )
        }
        Op::Linspace => {
            let checksum = checksum_array(&linspace(0.0, 1.0, n));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(linspace(0.0, 1.0, n));
                }),
            )
        }
        Op::Eye => {
            let checksum = checksum_array(&eye(n));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(eye(n));
                }),
            )
        }
        Op::Add => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&add(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(add(&a, &b));
                }),
            )
        }
        Op::AddBroadcast => {
            let a = seeded_uniform(&[n, 1], seed, -1.0, 1.0);
            let b = seeded_uniform(&[1, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&add(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(add(&a, &b));
                }),
            )
        }
        Op::Subtract => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&subtract(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(subtract(&a, &b));
                }),
            )
        }
        Op::Multiply => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&multiply(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(multiply(&a, &b));
                }),
            )
        }
        Op::Divide => {
            let a = seeded_uniform(&[n, n], seed, 0.5, 1.5);
            let b = seeded_uniform(&[n, n], seed + 1, 0.5, 1.5);
            let checksum = checksum_array(&divide(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(divide(&a, &b));
                }),
            )
        }
        Op::Power => {
            let a = seeded_uniform(&[n], seed, 0.5, 1.5);
            let b = seeded_uniform(&[n], seed + 1, 0.5, 2.0);
            let checksum = checksum_array(&power(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(power(&a, &b));
                }),
            )
        }
        Op::Maximum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&maximum(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(maximum(&a, &b));
                }),
            )
        }
        Op::Minimum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&minimum(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(minimum(&a, &b));
                }),
            )
        }
        Op::Greater => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&greater(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(greater(&a, &b));
                }),
            )
        }
        Op::Less => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&less(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(less(&a, &b));
                }),
            )
        }
        Op::Equal => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let b = a.clone();
            let checksum = checksum_array(&equal(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(equal(&a, &b));
                }),
            )
        }
        Op::NotEqual => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&not_equal(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(not_equal(&a, &b));
                }),
            )
        }
        Op::Sqrt => {
            let a = seeded_uniform(&[n, n], seed, 0.0, 10.0);
            let checksum = checksum_array(&sqrt(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sqrt(&a));
                }),
            )
        }
        Op::Exp => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&exp(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(exp(&a));
                }),
            )
        }
        Op::Log => {
            let a = seeded_uniform(&[n], seed, 0.1, 10.0);
            let checksum = checksum_array(&log(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(log(&a));
                }),
            )
        }
        Op::Sin => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&sin(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sin(&a));
                }),
            )
        }
        Op::Cos => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&cos(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cos(&a));
                }),
            )
        }
        Op::Tan => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&tan(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(tan(&a));
                }),
            )
        }
        Op::Tanh => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&tanh(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(tanh(&a));
                }),
            )
        }
        Op::Negative => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&negative(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(negative(&a));
                }),
            )
        }
        Op::Abs => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&abs(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(abs(&a));
                }),
            )
        }
        Op::Sign => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&sign(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sign(&a));
                }),
            )
        }
        Op::Square => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&square(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(square(&a));
                }),
            )
        }
        Op::Reciprocal => {
            let a = seeded_uniform(&[n, n], seed, 0.5, 1.5);
            let checksum = checksum_array(&reciprocal(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(reciprocal(&a));
                }),
            )
        }
        Op::Floor => {
            let a = seeded_uniform(&[n, n], seed, -5.0, 5.0);
            let checksum = checksum_array(&floor(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(floor(&a));
                }),
            )
        }
        Op::Ceil => {
            let a = seeded_uniform(&[n, n], seed, -5.0, 5.0);
            let checksum = checksum_array(&ceil(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ceil(&a));
                }),
            )
        }
        Op::Trunc => {
            let a = seeded_uniform(&[n, n], seed, -5.0, 5.0);
            let checksum = checksum_array(&trunc(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(trunc(&a));
                }),
            )
        }
        Op::Round => {
            let a = seeded_uniform(&[n, n], seed, -5.0, 5.0);
            let checksum = checksum_array(&round(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(round(&a));
                }),
            )
        }
        Op::Clip => {
            let a = seeded_uniform(&[n, n], seed, -2.0, 2.0);
            let checksum = checksum_array(&clip(&a, -0.5, 0.5));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(clip(&a, -0.5, 0.5));
                }),
            )
        }
        Op::Where => {
            let cond = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let x = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let y = seeded_uniform(&[n, n], seed + 2, -1.0, 1.0);
            let checksum = checksum_array(&where_(&cond, &x, &y));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(where_(&cond, &x, &y));
                }),
            )
        }
        Op::Sum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = sum(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sum(&a));
                }),
            )
        }
        Op::SumAxis => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&sum_axis(&a, 0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sum_axis(&a, 0));
                }),
            )
        }
        Op::Mean => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = mean(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mean(&a));
                }),
            )
        }
        Op::MeanAxis => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&mean_axis(&a, 1));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(mean_axis(&a, 1));
                }),
            )
        }
        Op::Min => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = min(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(min(&a));
                }),
            )
        }
        Op::MinAxis => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&min_axis(&a, 0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(min_axis(&a, 0));
                }),
            )
        }
        Op::Max => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = max(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(max(&a));
                }),
            )
        }
        Op::MaxAxis => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&max_axis(&a, 1));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(max_axis(&a, 1));
                }),
            )
        }
        Op::Var => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = var(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(var(&a));
                }),
            )
        }
        Op::Std => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = np_std(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(np_std(&a));
                }),
            )
        }
        Op::Argmin => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = argmin(&a) as f64;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(argmin(&a));
                }),
            )
        }
        Op::Argmax => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = argmax(&a) as f64;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(argmax(&a));
                }),
            )
        }
        Op::Cumsum => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&cumsum(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cumsum(&a));
                }),
            )
        }
        Op::CumsumAxis => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&cumsum_axis(&a, 0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cumsum_axis(&a, 0));
                }),
            )
        }
        Op::Cumprod => {
            let a = seeded_uniform(&[n], seed, 0.5, 1.5);
            let checksum = checksum_array(&cumprod(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cumprod(&a));
                }),
            )
        }
        Op::Transpose => {
            let a = seeded_uniform(&[n, n + 1], seed, -1.0, 1.0);
            let checksum = checksum_array(&transpose(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(transpose(&a));
                }),
            )
        }
        Op::Reshape => {
            let a = seeded_uniform(&[n * n], seed, -1.0, 1.0);
            let shape = [n, n];
            let checksum = checksum_array(&reshape(&a, &shape));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(reshape(&a, &shape));
                }),
            )
        }
        Op::ReshapeInfer => {
            let a = seeded_uniform(&[n * n], seed, -1.0, 1.0);
            let shape = [-1isize, n as isize];
            let checksum = checksum_array(&reshape_infer(&a, &shape));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(reshape_infer(&a, &shape));
                }),
            )
        }
        Op::Ravel => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&ravel(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ravel(&a));
                }),
            )
        }
        Op::Concatenate => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&concatenate(&[&a, &b], 0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(concatenate(&[&a, &b], 0));
                }),
            )
        }
        Op::Stack => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&stack(&[&a, &b], 0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(stack(&[&a, &b], 0));
                }),
            )
        }
        Op::BroadcastTo => {
            let a = seeded_uniform(&[1, n], seed, -1.0, 1.0);
            let shape = [n, n];
            let checksum = checksum_array(&broadcast_to(&a, &shape));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(broadcast_to(&a, &shape));
                }),
            )
        }
        Op::Swapaxes => {
            let a = seeded_uniform(&[n, n + 1], seed, -1.0, 1.0);
            let checksum = checksum_array(&swapaxes(&a, 0, 1));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(swapaxes(&a, 0, 1));
                }),
            )
        }
        Op::Moveaxis => {
            let a = seeded_uniform(&[n, n, 2], seed, -1.0, 1.0);
            let checksum = checksum_array(&moveaxis(&a, 0, 2));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(moveaxis(&a, 0, 2));
                }),
            )
        }
        Op::Matmul => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&matmul(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(matmul(&a, &b));
                }),
            )
        }
        Op::Dot => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&dot(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(dot(&a, &b));
                }),
            )
        }
        Op::Trace => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = trace(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(trace(&a));
                }),
            )
        }
        Op::Norm => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = norm(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(norm(&a));
                }),
            )
        }
        Op::Solve => {
            let a = diag_dominant(n, seed);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&solve(&a, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(solve(&a, &b));
                }),
            )
        }
        Op::Inv => {
            let a = diag_dominant(n, seed);
            let checksum = checksum_array(&inv(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(inv(&a));
                }),
            )
        }
        Op::Det => {
            let a = diag_dominant(n, seed);
            let checksum = det(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(det(&a));
                }),
            )
        }
        Op::Qr => {
            let a = seeded_uniform(&[n, n / 2 + 1], seed, -1.0, 1.0);
            let (q, r) = qr(&a);
            let checksum = checksum_array(&matmul(&q, &r));
            (
                checksum,
                Box::new(move || {
                    let (q, r) = qr(&a);
                    std::hint::black_box(matmul(&q, &r));
                }),
            )
        }
        Op::Svdvals => {
            let a = seeded_uniform(&[n, n / 2 + 1], seed, -1.0, 1.0);
            let checksum = checksum_array(&svdvals(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(svdvals(&a));
                }),
            )
        }
        Op::Eigvalsh => {
            let a = symmetric_spd(n, seed);
            let checksum = checksum_array(&eigvalsh(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(eigvalsh(&a));
                }),
            )
        }
        Op::Take => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let idx: Vec<usize> = (0..n).step_by(2.max(n / 8)).collect();
            let checksum = checksum_array(&take(&a, &idx, Some(0)));
            let idx2 = idx.clone();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(take(&a, &idx2, Some(0)));
                }),
            )
        }
        Op::Compress => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let cond = greater(&a, &zeros(&[n]));
            let checksum = checksum_array(&compress(&cond, &a, None));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(compress(&cond, &a, None));
                }),
            )
        }
        Op::Slice => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let specs = [
                AxisSlice::new(Some(1), Some((n as isize) - 1), 1),
                AxisSlice::all(),
            ];
            let checksum = checksum_array(&slice_array(&a, &specs));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(slice_array(&a, &specs));
                }),
            )
        }
        Op::AstypeF32 => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = checksum_array(&a.astype_f32().astype_f64());
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(a.astype_f32().astype_f64());
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
        median_ns: median_ns(samples.clone()),
        mean_ns,
        min_ns: *samples.iter().min().unwrap(),
        max_ns: *samples.iter().max().unwrap(),
        checksum,
    };

    println!("{}", report_to_pretty_json(&report));
}
