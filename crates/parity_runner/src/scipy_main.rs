//! CLI used by the Python SciPy comparison harness.
//!
//! Same contract as `parity_runner`: prepare once, time the core op, emit JSON.

use std::env;
use std::process;
use std::time::Instant;

use rnumpy::{seeded_uniform, NdArray};
use rscipy::{
    blackman, butter, cg, cholesky, convolve, correlate, csr_add, csr_frobenius_norm,
    csr_from_dense, csr_from_threshold, csr_matmat, csr_matvec, csr_to_csc, csr_to_dense,
    csr_transpose, cumulative_trapezoid, detrend, entropy, erf, erfc, expit, expm, eye_csr, fft,
    fftconvolve, fftfreq, filtfilt, gamma, gammaln, hamming, hann, i0, ifft, irfft, kurtosis,
    least_squares, logit, logsumexp, lstsq, lu, lu_factor, minimize_lbfgsb, minimize_nelder_mead,
    ndtr, ndtri, norm, norm_cdf, norm_ord, norm_pdf, norm_ppf, pearsonr, quad, rankdata, rfft, sem,
    simpson, skew, softmax, solve_ivp_rk45, solve_triangular, spearmanr, spsolve, stft, trapezoid,
    ttest_ind, welch, zscore, NormOrd,
};

#[derive(Debug, Clone)]
enum Op {
    Erf,
    Erfc,
    Gamma,
    Gammaln,
    Expit,
    Logit,
    Logsumexp,
    Softmax,
    I0,
    Ndtr,
    Ndtri,
    Lu,
    LuFactor,
    Cholesky,
    SolveTriangular,
    Lstsq,
    Norm,
    Norm1,
    NormInf,
    Expm,
    NelderMead,
    Lbfgsb,
    LeastSquares,
    NormPdf,
    NormCdf,
    NormPpf,
    Entropy,
    Zscore,
    Rankdata,
    Pearsonr,
    Spearmanr,
    TtestInd,
    Skew,
    Kurtosis,
    Sem,
    CsrFromDense,
    CsrMatvec,
    CsrMatmat,
    CsrTranspose,
    CsrAdd,
    CsrEye,
    CsrNorm,
    CsrToCsc,
    Spsolve,
    Cg,
    Butter,
    Filtfilt,
    Welch,
    Stft,
    Fft,
    Ifft,
    Rfft,
    Irfft,
    Fftfreq,
    Convolve,
    Fftconvolve,
    Correlate,
    Hann,
    Hamming,
    Blackman,
    Detrend,
    Trapezoid,
    Simpson,
    CumulativeTrapezoid,
    Quad,
    SolveIvp,
}

impl Op {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "erf" => Self::Erf,
            "erfc" => Self::Erfc,
            "gamma" => Self::Gamma,
            "gammaln" => Self::Gammaln,
            "expit" => Self::Expit,
            "logit" => Self::Logit,
            "logsumexp" => Self::Logsumexp,
            "softmax" => Self::Softmax,
            "i0" => Self::I0,
            "ndtr" => Self::Ndtr,
            "ndtri" => Self::Ndtri,
            "lu" => Self::Lu,
            "lu_factor" => Self::LuFactor,
            "cholesky" => Self::Cholesky,
            "solve_triangular" => Self::SolveTriangular,
            "lstsq" => Self::Lstsq,
            "norm" => Self::Norm,
            "norm_1" => Self::Norm1,
            "norm_inf" => Self::NormInf,
            "expm" => Self::Expm,
            "nelder_mead" => Self::NelderMead,
            "lbfgsb" => Self::Lbfgsb,
            "least_squares" => Self::LeastSquares,
            "norm_pdf" => Self::NormPdf,
            "norm_cdf" => Self::NormCdf,
            "norm_ppf" => Self::NormPpf,
            "entropy" => Self::Entropy,
            "zscore" => Self::Zscore,
            "rankdata" => Self::Rankdata,
            "pearsonr" => Self::Pearsonr,
            "spearmanr" => Self::Spearmanr,
            "ttest_ind" => Self::TtestInd,
            "skew" => Self::Skew,
            "kurtosis" => Self::Kurtosis,
            "sem" => Self::Sem,
            "csr_from_dense" => Self::CsrFromDense,
            "csr_matvec" => Self::CsrMatvec,
            "csr_matmat" => Self::CsrMatmat,
            "csr_transpose" => Self::CsrTranspose,
            "csr_add" => Self::CsrAdd,
            "csr_eye" => Self::CsrEye,
            "csr_norm" => Self::CsrNorm,
            "csr_to_csc" => Self::CsrToCsc,
            "spsolve" => Self::Spsolve,
            "cg" => Self::Cg,
            "butter" => Self::Butter,
            "filtfilt" => Self::Filtfilt,
            "welch" => Self::Welch,
            "stft" => Self::Stft,
            "fft" => Self::Fft,
            "ifft" => Self::Ifft,
            "rfft" => Self::Rfft,
            "irfft" => Self::Irfft,
            "fftfreq" => Self::Fftfreq,
            "convolve" => Self::Convolve,
            "fftconvolve" => Self::Fftconvolve,
            "correlate" => Self::Correlate,
            "hann" => Self::Hann,
            "hamming" => Self::Hamming,
            "blackman" => Self::Blackman,
            "detrend" => Self::Detrend,
            "trapezoid" => Self::Trapezoid,
            "simpson" => Self::Simpson,
            "cumulative_trapezoid" => Self::CumulativeTrapezoid,
            "quad" => Self::Quad,
            "solve_ivp" => Self::SolveIvp,
            other => return Err(format!("unknown op '{other}'")),
        })
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Erf => "erf",
            Self::Erfc => "erfc",
            Self::Gamma => "gamma",
            Self::Gammaln => "gammaln",
            Self::Expit => "expit",
            Self::Logit => "logit",
            Self::Logsumexp => "logsumexp",
            Self::Softmax => "softmax",
            Self::I0 => "i0",
            Self::Ndtr => "ndtr",
            Self::Ndtri => "ndtri",
            Self::Lu => "lu",
            Self::LuFactor => "lu_factor",
            Self::Cholesky => "cholesky",
            Self::SolveTriangular => "solve_triangular",
            Self::Lstsq => "lstsq",
            Self::Norm => "norm",
            Self::Norm1 => "norm_1",
            Self::NormInf => "norm_inf",
            Self::Expm => "expm",
            Self::NelderMead => "nelder_mead",
            Self::Lbfgsb => "lbfgsb",
            Self::LeastSquares => "least_squares",
            Self::NormPdf => "norm_pdf",
            Self::NormCdf => "norm_cdf",
            Self::NormPpf => "norm_ppf",
            Self::Entropy => "entropy",
            Self::Zscore => "zscore",
            Self::Rankdata => "rankdata",
            Self::Pearsonr => "pearsonr",
            Self::Spearmanr => "spearmanr",
            Self::TtestInd => "ttest_ind",
            Self::Skew => "skew",
            Self::Kurtosis => "kurtosis",
            Self::Sem => "sem",
            Self::CsrFromDense => "csr_from_dense",
            Self::CsrMatvec => "csr_matvec",
            Self::CsrMatmat => "csr_matmat",
            Self::CsrTranspose => "csr_transpose",
            Self::CsrAdd => "csr_add",
            Self::CsrEye => "csr_eye",
            Self::CsrNorm => "csr_norm",
            Self::CsrToCsc => "csr_to_csc",
            Self::Spsolve => "spsolve",
            Self::Cg => "cg",
            Self::Butter => "butter",
            Self::Filtfilt => "filtfilt",
            Self::Welch => "welch",
            Self::Stft => "stft",
            Self::Fft => "fft",
            Self::Ifft => "ifft",
            Self::Rfft => "rfft",
            Self::Irfft => "irfft",
            Self::Fftfreq => "fftfreq",
            Self::Convolve => "convolve",
            Self::Fftconvolve => "fftconvolve",
            Self::Correlate => "correlate",
            Self::Hann => "hann",
            Self::Hamming => "hamming",
            Self::Blackman => "blackman",
            Self::Detrend => "detrend",
            Self::Trapezoid => "trapezoid",
            Self::Simpson => "simpson",
            Self::CumulativeTrapezoid => "cumulative_trapezoid",
            Self::Quad => "quad",
            Self::SolveIvp => "solve_ivp",
        }
    }
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

fn diag_dominant(n: usize, seed: u64) -> NdArray {
    let mut a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
    let boost = (n as f64).min(4.0);
    for i in 0..n {
        a[[i, i]] += boost;
    }
    a
}

fn rosenbrock(x: &[f64]) -> f64 {
    (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
}

fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
    let (a, b) = (x[0], x[1]);
    vec![
        -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
        200.0 * (b - a * a),
    ]
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
        "Usage: scipy_parity_runner --op <name> [--size N] [--iters N] [--warmup N] [--seed N]"
    );
    process::exit(2);
}

fn parse_args() -> Result<Args, String> {
    let mut op: Option<Op> = None;
    let mut size: usize = 64;
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

fn json_f64(v: f64) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    let s = format!("{v}");
    if s.contains('e') || s.contains('E') || s.contains('.') {
        s
    } else {
        format!("{v:.1}")
    }
}

impl Report {
    fn to_json(&self) -> String {
        format!(
            "{{\"language\":{},\"op\":{},\"size\":{},\"iters\":{},\"warmup\":{},\"seed\":{},\
\"median_ns\":{},\"mean_ns\":{},\"min_ns\":{},\"max_ns\":{},\"checksum\":{}}}",
            json_escape_string(self.language),
            json_escape_string(&self.op),
            self.size,
            self.iters,
            self.warmup,
            self.seed,
            self.median_ns,
            json_f64(self.mean_ns),
            self.min_ns,
            self.max_ns,
            json_f64(self.checksum),
        )
    }
}

fn median_u64(samples: &[u64]) -> u64 {
    let mut s = samples.to_vec();
    s.sort_unstable();
    let n = s.len();
    if n == 0 {
        return 0;
    }
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2
    }
}

fn checksum_array(a: &NdArray) -> f64 {
    a.sum()
}

fn run_op(op: &Op, size: usize, seed: u64) -> (f64, Box<dyn FnMut()>) {
    let n = size;
    match op {
        Op::Erf => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = checksum_array(&erf(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(erf(&a));
                }),
            )
        }
        Op::Erfc => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = checksum_array(&erfc(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(erfc(&a));
                }),
            )
        }
        Op::Gamma => {
            let a = seeded_uniform(&[n], seed, 0.2, 8.0);
            let checksum = checksum_array(&gamma(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(gamma(&a));
                }),
            )
        }
        Op::Gammaln => {
            let a = seeded_uniform(&[n], seed, 0.2, 20.0);
            let checksum = checksum_array(&gammaln(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(gammaln(&a));
                }),
            )
        }
        Op::Expit => {
            let a = seeded_uniform(&[n], seed, -5.0, 5.0);
            let checksum = checksum_array(&expit(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(expit(&a));
                }),
            )
        }
        Op::Logit => {
            let a = seeded_uniform(&[n], seed, 0.05, 0.95);
            let checksum = checksum_array(&logit(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(logit(&a));
                }),
            )
        }
        Op::Logsumexp => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = logsumexp(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(logsumexp(&a));
                }),
            )
        }
        Op::Softmax => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = checksum_array(&softmax(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(softmax(&a));
                }),
            )
        }
        Op::I0 => {
            let a = seeded_uniform(&[n], seed, 0.0, 5.0);
            let checksum = checksum_array(&i0(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(i0(&a));
                }),
            )
        }
        Op::Ndtr => {
            let a = seeded_uniform(&[n], seed, -3.0, 3.0);
            let checksum = checksum_array(&ndtr(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ndtr(&a));
                }),
            )
        }
        Op::Ndtri => {
            let a = seeded_uniform(&[n], seed, 0.05, 0.95);
            let checksum = checksum_array(&ndtri(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ndtri(&a));
                }),
            )
        }
        Op::Lu => {
            let a = diag_dominant(n, seed);
            let (p, l, u) = lu(&a);
            let checksum = checksum_array(&p) + checksum_array(&l) + checksum_array(&u);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(lu(&a));
                }),
            )
        }
        Op::LuFactor => {
            let a = diag_dominant(n, seed);
            let (lu_m, piv) = lu_factor(&a);
            let checksum = checksum_array(&lu_m) + checksum_array(&piv);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(lu_factor(&a));
                }),
            )
        }
        Op::Cholesky => {
            let a = symmetric_spd(n, seed);
            let checksum = checksum_array(&cholesky(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cholesky(&a));
                }),
            )
        }
        Op::SolveTriangular => {
            let a = symmetric_spd(n, seed);
            let l = cholesky(&a);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&solve_triangular(&l, &b, true));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(solve_triangular(&l, &b, true));
                }),
            )
        }
        Op::Lstsq => {
            let m = 2 * n;
            let a = seeded_uniform(&[m, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[m], seed + 1, -1.0, 1.0);
            // Checksum solution only (singular values are R-diag estimates).
            let (x, _, _, _) = lstsq(&a, &b);
            let checksum = checksum_array(&x);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(lstsq(&a, &b));
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
        Op::Norm1 => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = norm_ord(&a, NormOrd::One);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(norm_ord(&a, NormOrd::One));
                }),
            )
        }
        Op::NormInf => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let checksum = norm_ord(&a, NormOrd::Inf);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(norm_ord(&a, NormOrd::Inf));
                }),
            )
        }
        Op::Expm => {
            let en = n.min(6).max(2);
            let a = seeded_uniform(&[en, en], seed, -0.5, 0.5);
            let checksum = checksum_array(&expm(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(expm(&a));
                }),
            )
        }
        Op::NelderMead => {
            let x0 = [-1.2, 1.0];
            let r = minimize_nelder_mead(rosenbrock, &x0, 2000, 1e-8, 1e-8);
            let checksum = r.x.iter().sum::<f64>() + r.fun;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(minimize_nelder_mead(rosenbrock, &x0, 2000, 1e-8, 1e-8));
                }),
            )
        }
        Op::Lbfgsb => {
            let x0 = [-1.2, 1.0];
            let bounds = [(-2.0, 2.0), (-2.0, 2.0)];
            let r = minimize_lbfgsb(rosenbrock, rosenbrock_grad, &x0, &bounds, 2000, 10, 1e-6);
            let checksum = r.x.iter().sum::<f64>() + r.fun;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(minimize_lbfgsb(
                        rosenbrock,
                        rosenbrock_grad,
                        &x0,
                        &bounds,
                        2000,
                        10,
                        1e-6,
                    ));
                }),
            )
        }
        Op::LeastSquares => {
            let xs = [0.0, 1.0, 2.0, 3.0];
            let ys = [3.0, 5.0, 7.0, 9.0];
            let resid = move |p: &[f64]| {
                xs.iter()
                    .zip(ys.iter())
                    .map(|(&x, &y)| p[0] * x + p[1] - y)
                    .collect::<Vec<_>>()
            };
            let jac = move |_p: &[f64]| {
                let mut j = Vec::with_capacity(8);
                for &x in &xs {
                    j.push(x);
                    j.push(1.0);
                }
                j
            };
            let r = least_squares(resid, jac, &[0.0, 0.0], 4, 50, 1e-12, 1e-12, 1e-12);
            let checksum = r.x.iter().sum::<f64>() + r.fun;
            (
                checksum,
                Box::new(move || {
                    let resid = |p: &[f64]| {
                        xs.iter()
                            .zip(ys.iter())
                            .map(|(&x, &y)| p[0] * x + p[1] - y)
                            .collect::<Vec<_>>()
                    };
                    let jac = |_p: &[f64]| {
                        let mut j = Vec::with_capacity(8);
                        for &x in &xs {
                            j.push(x);
                            j.push(1.0);
                        }
                        j
                    };
                    std::hint::black_box(least_squares(
                        resid, jac, &[0.0, 0.0], 4, 50, 1e-12, 1e-12, 1e-12,
                    ));
                }),
            )
        }
        Op::NormPdf => {
            let a = seeded_uniform(&[n], seed, -3.0, 3.0);
            let checksum = checksum_array(&norm_pdf(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(norm_pdf(&a));
                }),
            )
        }
        Op::NormCdf => {
            let a = seeded_uniform(&[n], seed, -3.0, 3.0);
            let checksum = checksum_array(&norm_cdf(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(norm_cdf(&a));
                }),
            )
        }
        Op::NormPpf => {
            let a = seeded_uniform(&[n], seed, 0.05, 0.95);
            let checksum = checksum_array(&norm_ppf(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(norm_ppf(&a));
                }),
            )
        }
        Op::Entropy => {
            let a = seeded_uniform(&[n], seed, 0.1, 2.0);
            let checksum = entropy(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(entropy(&a));
                }),
            )
        }
        Op::Zscore => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = checksum_array(&zscore(&a, 0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(zscore(&a, 0));
                }),
            )
        }
        Op::Rankdata => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = checksum_array(&rankdata(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(rankdata(&a));
                }),
            )
        }
        Op::Pearsonr => {
            let x = seeded_uniform(&[n], seed, -1.0, 1.0);
            let y = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let (r, p) = pearsonr(&x, &y);
            let checksum = r + p;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(pearsonr(&x, &y));
                }),
            )
        }
        Op::Spearmanr => {
            let x = seeded_uniform(&[n], seed, -1.0, 1.0);
            let y = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let (r, p) = spearmanr(&x, &y);
            let checksum = r + p;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(spearmanr(&x, &y));
                }),
            )
        }
        Op::TtestInd => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let r = ttest_ind(&a, &b);
            let checksum = r.statistic + r.pvalue;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ttest_ind(&a, &b));
                }),
            )
        }
        Op::Skew => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = skew(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(skew(&a));
                }),
            )
        }
        Op::Kurtosis => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = kurtosis(&a);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(kurtosis(&a));
                }),
            )
        }
        Op::Sem => {
            let a = seeded_uniform(&[n], seed, -2.0, 2.0);
            let checksum = sem(&a, 1);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(sem(&a, 1));
                }),
            )
        }
        Op::CsrFromDense => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let csr = csr_from_threshold(&a, 0.5);
            let checksum = checksum_array(&csr_to_dense(&csr)) + csr.nnz() as f64;
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_from_threshold(&a, 0.5));
                }),
            )
        }
        Op::CsrMatvec => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let csr = csr_from_threshold(&a, 0.5);
            let x = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&csr_matvec(&csr, &x));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_matvec(&csr, &x));
                }),
            )
        }
        Op::CsrMatmat => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let csr = csr_from_threshold(&a, 0.5);
            let b = seeded_uniform(&[n, 8], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&csr_matmat(&csr, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_matmat(&csr, &b));
                }),
            )
        }
        Op::CsrTranspose => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let csr = csr_from_threshold(&a, 0.5);
            let at = csr_transpose(&csr);
            let checksum = at.data.iter().sum::<f64>()
                + at.indices.iter().map(|&i| i as f64).sum::<f64>()
                + at.indptr.iter().map(|&i| i as f64).sum::<f64>();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_transpose(&csr));
                }),
            )
        }
        Op::CsrAdd => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let b = seeded_uniform(&[n, n], seed + 1, -1.0, 1.0);
            let ca = csr_from_threshold(&a, 0.5);
            let cb = csr_from_threshold(&b, 0.5);
            let checksum = checksum_array(&csr_to_dense(&csr_add(&ca, &cb)));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_add(&ca, &cb));
                }),
            )
        }
        Op::CsrEye => {
            let checksum = checksum_array(&csr_to_dense(&eye_csr(n)));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(eye_csr(n));
                }),
            )
        }
        Op::CsrNorm => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let csr = csr_from_threshold(&a, 0.5);
            let checksum = csr_frobenius_norm(&csr);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_frobenius_norm(&csr));
                }),
            )
        }
        Op::CsrToCsc => {
            let a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            let csr = csr_from_threshold(&a, 0.5);
            let csc = csr_to_csc(&csr);
            let checksum = csc.data.iter().sum::<f64>()
                + csc.indices.iter().map(|&i| i as f64).sum::<f64>()
                + csc.indptr.iter().map(|&i| i as f64).sum::<f64>();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(csr_to_csc(&csr));
                }),
            )
        }
        Op::Spsolve => {
            let mut a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            for i in 0..n {
                a[[i, i]] += (n as f64) + 1.0;
            }
            let csr = csr_from_dense(&a);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&spsolve(&csr, &b));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(spsolve(&csr, &b));
                }),
            )
        }
        Op::Cg => {
            let mut a = seeded_uniform(&[n, n], seed, -1.0, 1.0);
            for i in 0..n {
                for j in 0..i {
                    let v = 0.5 * (a[[i, j]] + a[[j, i]]);
                    a[[i, j]] = v;
                    a[[j, i]] = v;
                }
                a[[i, i]] += (n as f64) + 1.0;
            }
            let csr = csr_from_dense(&a);
            let b = seeded_uniform(&[n], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&cg(&csr, &b, 1e-10, Some(n * 20)));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cg(&csr, &b, 1e-10, Some(n * 20)));
                }),
            )
        }
        Op::Butter => {
            let (b, a) = butter(4, 0.2, "lowpass");
            let mut packed = b.as_slice().unwrap().to_vec();
            packed.extend_from_slice(a.as_slice().unwrap());
            let checksum = checksum_array(&NdArray::from_vec(packed));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(butter(4, 0.2, "lowpass"));
                }),
            )
        }
        Op::Filtfilt => {
            let (b, a) = butter(4, 0.15, "lowpass");
            let x = seeded_uniform(&[n.max(64)], seed, -1.0, 1.0);
            let checksum = checksum_array(&filtfilt(&b, &a, &x));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(filtfilt(&b, &a, &x));
                }),
            )
        }
        Op::Welch => {
            let m = n.max(256);
            let x = seeded_uniform(&[m], seed, -1.0, 1.0);
            let nperseg = 64usize;
            let (_f, pxx) = welch(&x, 1.0, nperseg, Some(32));
            let checksum = checksum_array(&pxx);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(welch(&x, 1.0, nperseg, Some(32)));
                }),
            )
        }
        Op::Stft => {
            let m = n.max(256);
            let x = seeded_uniform(&[m], seed, -1.0, 1.0);
            let nperseg = 64usize;
            let (_f, _t, z) = stft(&x, 1.0, nperseg, Some(32));
            let checksum = checksum_array(&z);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(stft(&x, 1.0, nperseg, Some(32)));
                }),
            )
        }
        Op::Fft => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&fft(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(fft(&a));
                }),
            )
        }
        Op::Ifft => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let spec = fft(&a);
            let checksum = checksum_array(&ifft(&spec));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(ifft(&spec));
                }),
            )
        }
        Op::Rfft => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&rfft(&a));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(rfft(&a));
                }),
            )
        }
        Op::Irfft => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let spec = rfft(&a);
            let checksum = checksum_array(&irfft(&spec, Some(n)));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(irfft(&spec, Some(n)));
                }),
            )
        }
        Op::Fftfreq => {
            let checksum = checksum_array(&fftfreq(n, 1.0));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(fftfreq(n, 1.0));
                }),
            )
        }
        Op::Convolve => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let v = seeded_uniform(&[17], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&convolve(&a, &v, "full"));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(convolve(&a, &v, "full"));
                }),
            )
        }
        Op::Fftconvolve => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let v = seeded_uniform(&[17], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&fftconvolve(&a, &v, "full"));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(fftconvolve(&a, &v, "full"));
                }),
            )
        }
        Op::Correlate => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let v = seeded_uniform(&[17], seed + 1, -1.0, 1.0);
            let checksum = checksum_array(&correlate(&a, &v, "full"));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(correlate(&a, &v, "full"));
                }),
            )
        }
        Op::Hann => {
            let checksum = checksum_array(&hann(n, true));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(hann(n, true));
                }),
            )
        }
        Op::Hamming => {
            let checksum = checksum_array(&hamming(n, true));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(hamming(n, true));
                }),
            )
        }
        Op::Blackman => {
            let checksum = checksum_array(&blackman(n, true));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(blackman(n, true));
                }),
            )
        }
        Op::Detrend => {
            let a = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&detrend(&a, "linear"));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(detrend(&a, "linear"));
                }),
            )
        }
        Op::Trapezoid => {
            let y = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = trapezoid(&y, None, 1.0);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(trapezoid(&y, None, 1.0));
                }),
            )
        }
        Op::Simpson => {
            let y = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = simpson(&y, None, 1.0);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(simpson(&y, None, 1.0));
                }),
            )
        }
        Op::CumulativeTrapezoid => {
            let y = seeded_uniform(&[n], seed, -1.0, 1.0);
            let checksum = checksum_array(&cumulative_trapezoid(&y, None, 1.0, Some(0.0)));
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(cumulative_trapezoid(&y, None, 1.0, Some(0.0)));
                }),
            )
        }
        Op::Quad => {
            let (checksum, _) = quad(|x| (-x * x).exp(), 0.0, 1.0, 1e-10);
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(quad(|x| (-x * x).exp(), 0.0, 1.0, 1e-10));
                }),
            )
        }
        Op::SolveIvp => {
            // Harmonic oscillator: y'' = -y → [y, v], y'=v, v'=-y
            let n_pts = (n / 4).max(11);
            let t_eval: Vec<f64> = (0..n_pts).map(|i| i as f64 * 0.1).collect();
            let tf = t_eval[t_eval.len() - 1];
            let r = solve_ivp_rk45(
                |_t, y| vec![y[1], -y[0]],
                (0.0, tf),
                &[1.0, 0.0],
                &t_eval,
                1e-6,
                1e-9,
            );
            let checksum = r.y_sum();
            (
                checksum,
                Box::new(move || {
                    std::hint::black_box(solve_ivp_rk45(
                        |_t, y| vec![y[1], -y[0]],
                        (0.0, tf),
                        &[1.0, 0.0],
                        &t_eval,
                        1e-6,
                        1e-9,
                    ));
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
