//! Statistics — mirrors common `scipy.stats` entry points used in AI/ML.
//!
//! Local implementations only. Continuous distributions reuse `special::ndtr` /
//! `ndtri` where applicable.

use crate::special::{gammaln_scalar, ndtr_scalar, ndtri_scalar};
use rnumpy::NdArray;

fn contig(a: &NdArray) -> NdArray {
    a.to_contiguous()
}

fn mean_slice(xs: &[f64]) -> f64 {
    assert!(!xs.is_empty(), "stats: empty sample");
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn var_slice(xs: &[f64], ddof: usize) -> f64 {
    let n = xs.len();
    assert!(n > ddof, "stats: need n > ddof");
    let m = mean_slice(xs);
    let mut s = 0.0;
    for &x in xs {
        let d = x - m;
        s += d * d;
    }
    s / (n - ddof) as f64
}

/// Standard normal PDF (loc=0, scale=1).
pub fn norm_pdf_scalar(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1/sqrt(2π)
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Normal PDF with location/scale — `scipy.stats.norm.pdf(x, loc, scale)`.
pub fn norm_pdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "norm.pdf: scale must be > 0");
    norm_pdf_scalar((x - loc) / scale) / scale
}

/// Standard normal CDF.
pub fn norm_cdf_scalar(x: f64) -> f64 {
    ndtr_scalar(x)
}

/// Normal CDF with location/scale.
pub fn norm_cdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "norm.cdf: scale must be > 0");
    ndtr_scalar((x - loc) / scale)
}

/// Standard normal PPF (quantile).
pub fn norm_ppf_scalar(p: f64) -> f64 {
    ndtri_scalar(p)
}

/// Normal PPF with location/scale.
pub fn norm_ppf_scalar_ls(p: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "norm.ppf: scale must be > 0");
    loc + scale * ndtri_scalar(p)
}

/// `scipy.stats.norm.pdf` (elementwise, loc=0, scale=1).
pub fn norm_pdf(a: &NdArray) -> NdArray {
    map_contig(a, norm_pdf_scalar)
}

/// `scipy.stats.norm.cdf` (elementwise, loc=0, scale=1).
pub fn norm_cdf(a: &NdArray) -> NdArray {
    map_contig(a, norm_cdf_scalar)
}

/// `scipy.stats.norm.ppf` (elementwise, loc=0, scale=1).
pub fn norm_ppf(a: &NdArray) -> NdArray {
    map_contig(a, norm_ppf_scalar)
}

// --- uniform (scipy.stats.uniform, loc=0, scale=1 → U[0,1]) ---

/// `scipy.stats.uniform.pdf(x, loc=0, scale=1)`.
pub fn uniform_pdf_scalar(x: f64) -> f64 {
    if (0.0..=1.0).contains(&x) {
        1.0
    } else {
        0.0
    }
}

pub fn uniform_pdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "uniform.pdf: scale must be > 0");
    uniform_pdf_scalar((x - loc) / scale) / scale
}

/// `scipy.stats.uniform.cdf(x, loc=0, scale=1)`.
pub fn uniform_cdf_scalar(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else if x > 1.0 {
        1.0
    } else {
        x
    }
}

pub fn uniform_cdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "uniform.cdf: scale must be > 0");
    uniform_cdf_scalar((x - loc) / scale)
}

/// `scipy.stats.uniform.ppf(p, loc=0, scale=1)`.
pub fn uniform_ppf_scalar(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "uniform.ppf: p in [0,1]");
    p
}

pub fn uniform_ppf_scalar_ls(p: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "uniform.ppf: scale must be > 0");
    loc + scale * uniform_ppf_scalar(p)
}

pub fn uniform_pdf(a: &NdArray) -> NdArray {
    map_contig(a, uniform_pdf_scalar)
}

pub fn uniform_cdf(a: &NdArray) -> NdArray {
    map_contig(a, uniform_cdf_scalar)
}

pub fn uniform_ppf(a: &NdArray) -> NdArray {
    map_contig(a, uniform_ppf_scalar)
}

// --- expon (scipy.stats.expon, loc=0, scale=1) ---

/// `scipy.stats.expon.pdf(x)` standard form.
pub fn expon_pdf_scalar(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        (-x).exp()
    }
}

pub fn expon_pdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "expon.pdf: scale must be > 0");
    expon_pdf_scalar((x - loc) / scale) / scale
}

pub fn expon_cdf_scalar(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        -(-x).exp_m1() // 1 - exp(-x)
    }
}

pub fn expon_cdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "expon.cdf: scale must be > 0");
    expon_cdf_scalar((x - loc) / scale)
}

pub fn expon_ppf_scalar(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "expon.ppf: p in [0,1]");
    if p == 1.0 {
        return f64::INFINITY;
    }
    -(-p).ln_1p() // -ln(1-p)
}

pub fn expon_ppf_scalar_ls(p: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "expon.ppf: scale must be > 0");
    loc + scale * expon_ppf_scalar(p)
}

pub fn expon_pdf(a: &NdArray) -> NdArray {
    map_contig(a, expon_pdf_scalar)
}

pub fn expon_cdf(a: &NdArray) -> NdArray {
    map_contig(a, expon_cdf_scalar)
}

pub fn expon_ppf(a: &NdArray) -> NdArray {
    map_contig(a, expon_ppf_scalar)
}

// --- laplace (scipy.stats.laplace, loc=0, scale=1) ---

pub fn laplace_pdf_scalar(x: f64) -> f64 {
    0.5 * (-x.abs()).exp()
}

pub fn laplace_pdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "laplace.pdf: scale must be > 0");
    laplace_pdf_scalar((x - loc) / scale) / scale
}

pub fn laplace_cdf_scalar(x: f64) -> f64 {
    if x < 0.0 {
        0.5 * x.exp()
    } else {
        1.0 - 0.5 * (-x).exp()
    }
}

pub fn laplace_cdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "laplace.cdf: scale must be > 0");
    laplace_cdf_scalar((x - loc) / scale)
}

pub fn laplace_ppf_scalar(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "laplace.ppf: p in [0,1]");
    if p < 0.5 {
        (2.0 * p).ln()
    } else {
        -((2.0 * (1.0 - p)).ln())
    }
}

pub fn laplace_ppf_scalar_ls(p: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "laplace.ppf: scale must be > 0");
    loc + scale * laplace_ppf_scalar(p)
}

pub fn laplace_pdf(a: &NdArray) -> NdArray {
    map_contig(a, laplace_pdf_scalar)
}

pub fn laplace_cdf(a: &NdArray) -> NdArray {
    map_contig(a, laplace_cdf_scalar)
}

pub fn laplace_ppf(a: &NdArray) -> NdArray {
    map_contig(a, laplace_ppf_scalar)
}

// --- logistic (scipy.stats.logistic, loc=0, scale=1) ---

pub fn logistic_pdf_scalar(x: f64) -> f64 {
    // Stable form of e^{-|x|} / (1 + e^{-|x|})^2
    let e = (-x.abs()).exp();
    let den = 1.0 + e;
    e / (den * den)
}

pub fn logistic_pdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "logistic.pdf: scale must be > 0");
    logistic_pdf_scalar((x - loc) / scale) / scale
}

pub fn logistic_cdf_scalar(x: f64) -> f64 {
    // 1 / (1 + exp(-x))
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

pub fn logistic_cdf_scalar_ls(x: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "logistic.cdf: scale must be > 0");
    logistic_cdf_scalar((x - loc) / scale)
}

pub fn logistic_ppf_scalar(p: f64) -> f64 {
    assert!((0.0..=1.0).contains(&p), "logistic.ppf: p in [0,1]");
    // logit(p) = ln(p/(1-p))
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    (p / (1.0 - p)).ln()
}

pub fn logistic_ppf_scalar_ls(p: f64, loc: f64, scale: f64) -> f64 {
    assert!(scale > 0.0, "logistic.ppf: scale must be > 0");
    loc + scale * logistic_ppf_scalar(p)
}

pub fn logistic_pdf(a: &NdArray) -> NdArray {
    map_contig(a, logistic_pdf_scalar)
}

pub fn logistic_cdf(a: &NdArray) -> NdArray {
    map_contig(a, logistic_cdf_scalar)
}

pub fn logistic_ppf(a: &NdArray) -> NdArray {
    map_contig(a, logistic_ppf_scalar)
}

// --- Student-t (scipy.stats.t, loc=0, scale=1) ---

/// `scipy.stats.t.pdf(x, df)`.
pub fn t_pdf_scalar(x: f64, df: f64) -> f64 {
    assert!(df > 0.0, "t.pdf: df must be > 0");
    let half = 0.5 * df;
    let log_c = gammaln_scalar(half + 0.5)
        - gammaln_scalar(half)
        - 0.5 * (df.ln() + std::f64::consts::PI.ln());
    log_c.exp() * (1.0 + x * x / df).powf(-(df + 1.0) * 0.5)
}

/// `scipy.stats.t.cdf(x, df)`.
pub fn t_cdf_scalar(x: f64, df: f64) -> f64 {
    assert!(df > 0.0, "t.cdf: df must be > 0");
    if x == 0.0 {
        return 0.5;
    }
    let z = df / (df + x * x);
    let ib = betainc_reg(0.5 * df, 0.5, z);
    if x > 0.0 {
        1.0 - 0.5 * ib
    } else {
        0.5 * ib
    }
}

/// `scipy.stats.t.ppf(p, df)`.
pub fn t_ppf_scalar(p: f64, df: f64) -> f64 {
    assert!(df > 0.0, "t.ppf: df must be > 0");
    assert!((0.0..=1.0).contains(&p), "t.ppf: p in [0,1]");
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }
    invert_monotonic(p, -1e3, 1e3, |x| t_cdf_scalar(x, df))
}

pub fn t_pdf(a: &NdArray, df: f64) -> NdArray {
    map_contig(a, |x| t_pdf_scalar(x, df))
}

pub fn t_cdf(a: &NdArray, df: f64) -> NdArray {
    map_contig(a, |x| t_cdf_scalar(x, df))
}

pub fn t_ppf(a: &NdArray, df: f64) -> NdArray {
    map_contig(a, |p| t_ppf_scalar(p, df))
}

// --- chi-square (scipy.stats.chi2) ---

/// `scipy.stats.chi2.pdf(x, df)`.
pub fn chi2_pdf_scalar(x: f64, df: f64) -> f64 {
    assert!(df > 0.0, "chi2.pdf: df must be > 0");
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return if df == 2.0 {
            0.5
        } else if df < 2.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }
    let half = 0.5 * df;
    (-gammaln_scalar(half) - half * std::f64::consts::LN_2 + (half - 1.0) * x.ln() - 0.5 * x).exp()
}

/// `scipy.stats.chi2.cdf(x, df)`.
pub fn chi2_cdf_scalar(x: f64, df: f64) -> f64 {
    assert!(df > 0.0, "chi2.cdf: df must be > 0");
    if x <= 0.0 {
        return 0.0;
    }
    gammainc_reg(0.5 * df, 0.5 * x)
}

/// `scipy.stats.chi2.ppf(p, df)`.
pub fn chi2_ppf_scalar(p: f64, df: f64) -> f64 {
    assert!(df > 0.0, "chi2.ppf: df must be > 0");
    assert!((0.0..=1.0).contains(&p), "chi2.ppf: p in [0,1]");
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_monotonic(p, 0.0, (df + 40.0) * 4.0, |x| chi2_cdf_scalar(x, df))
}

pub fn chi2_pdf(a: &NdArray, df: f64) -> NdArray {
    map_contig(a, |x| chi2_pdf_scalar(x, df))
}

pub fn chi2_cdf(a: &NdArray, df: f64) -> NdArray {
    map_contig(a, |x| chi2_cdf_scalar(x, df))
}

pub fn chi2_ppf(a: &NdArray, df: f64) -> NdArray {
    map_contig(a, |p| chi2_ppf_scalar(p, df))
}

// --- gamma (scipy.stats.gamma, a=shape, loc=0, scale=1) ---

/// `scipy.stats.gamma.pdf(x, a)`.
pub fn gamma_pdf_scalar(x: f64, shape: f64) -> f64 {
    assert!(shape > 0.0, "gamma.pdf: a must be > 0");
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return if shape == 1.0 {
            1.0
        } else if shape < 1.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }
    (-gammaln_scalar(shape) + (shape - 1.0) * x.ln() - x).exp()
}

/// `scipy.stats.gamma.cdf(x, a)`.
pub fn gamma_cdf_scalar(x: f64, shape: f64) -> f64 {
    assert!(shape > 0.0, "gamma.cdf: a must be > 0");
    if x <= 0.0 {
        return 0.0;
    }
    gammainc_reg(shape, x)
}

/// `scipy.stats.gamma.ppf(p, a)`.
pub fn gamma_ppf_scalar(p: f64, shape: f64) -> f64 {
    assert!(shape > 0.0, "gamma.ppf: a must be > 0");
    assert!((0.0..=1.0).contains(&p), "gamma.ppf: p in [0,1]");
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    invert_monotonic(p, 0.0, (shape + 40.0) * 4.0, |x| gamma_cdf_scalar(x, shape))
}

pub fn gamma_pdf_shape(a: &NdArray, shape: f64) -> NdArray {
    map_contig(a, |x| gamma_pdf_scalar(x, shape))
}

pub fn gamma_cdf_shape(a: &NdArray, shape: f64) -> NdArray {
    map_contig(a, |x| gamma_cdf_scalar(x, shape))
}

pub fn gamma_ppf_shape(a: &NdArray, shape: f64) -> NdArray {
    map_contig(a, |p| gamma_ppf_scalar(p, shape))
}

// --- beta (scipy.stats.beta, loc=0, scale=1) ---

/// `scipy.stats.beta.pdf(x, a, b)`.
pub fn beta_pdf_scalar(x: f64, a: f64, b: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "beta.pdf: a,b must be > 0");
    if x < 0.0 || x > 1.0 {
        return 0.0;
    }
    if x == 0.0 || x == 1.0 {
        // Boundary behavior matches SciPy for a,b > 1 → 0.
        if (x == 0.0 && a < 1.0) || (x == 1.0 && b < 1.0) {
            return f64::INFINITY;
        }
        if (x == 0.0 && a == 1.0) || (x == 1.0 && b == 1.0) {
            // fall through
        } else if (x == 0.0 && a > 1.0) || (x == 1.0 && b > 1.0) {
            return 0.0;
        }
    }
    let log_b = gammaln_scalar(a) + gammaln_scalar(b) - gammaln_scalar(a + b);
    ( (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - log_b ).exp()
}

/// `scipy.stats.beta.cdf(x, a, b)`.
pub fn beta_cdf_scalar(x: f64, a: f64, b: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "beta.cdf: a,b must be > 0");
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    betainc_reg(a, b, x)
}

/// `scipy.stats.beta.ppf(p, a, b)`.
pub fn beta_ppf_scalar(p: f64, a: f64, b: f64) -> f64 {
    assert!(a > 0.0 && b > 0.0, "beta.ppf: a,b must be > 0");
    assert!((0.0..=1.0).contains(&p), "beta.ppf: p in [0,1]");
    if p == 0.0 {
        return 0.0;
    }
    if p == 1.0 {
        return 1.0;
    }
    invert_monotonic(p, 0.0, 1.0, |x| beta_cdf_scalar(x, a, b))
}

pub fn beta_pdf(arr: &NdArray, a: f64, b: f64) -> NdArray {
    map_contig(arr, |x| beta_pdf_scalar(x, a, b))
}

pub fn beta_cdf(arr: &NdArray, a: f64, b: f64) -> NdArray {
    map_contig(arr, |x| beta_cdf_scalar(x, a, b))
}

pub fn beta_ppf(arr: &NdArray, a: f64, b: f64) -> NdArray {
    map_contig(arr, |p| beta_ppf_scalar(p, a, b))
}

// --- poisson (scipy.stats.poisson) ---

/// `scipy.stats.poisson.pmf(k, mu)`.
pub fn poisson_pmf_scalar(k: f64, mu: f64) -> f64 {
    assert!(mu >= 0.0, "poisson.pmf: mu must be >= 0");
    if k < 0.0 || k.fract() != 0.0 {
        return 0.0;
    }
    let kk = k as i64;
    if mu == 0.0 {
        return if kk == 0 { 1.0 } else { 0.0 };
    }
    // exp(k*ln(mu) - mu - gammaln(k+1))
    (k * mu.ln() - mu - gammaln_scalar(k + 1.0)).exp()
}

/// `scipy.stats.poisson.cdf(k, mu)` (right-continuous on integers).
pub fn poisson_cdf_scalar(k: f64, mu: f64) -> f64 {
    assert!(mu >= 0.0, "poisson.cdf: mu must be >= 0");
    if k < 0.0 {
        return 0.0;
    }
    let kk = k.floor() as i64;
    let mut s = 0.0;
    for i in 0..=kk {
        s += poisson_pmf_scalar(i as f64, mu);
    }
    s.clamp(0.0, 1.0)
}

pub fn poisson_pmf(a: &NdArray, mu: f64) -> NdArray {
    map_contig(a, |k| poisson_pmf_scalar(k, mu))
}

pub fn poisson_cdf(a: &NdArray, mu: f64) -> NdArray {
    map_contig(a, |k| poisson_cdf_scalar(k, mu))
}

// --- binom (scipy.stats.binom) ---

/// `scipy.stats.binom.pmf(k, n, p)`.
pub fn binom_pmf_scalar(k: f64, n: f64, p: f64) -> f64 {
    assert!(n >= 0.0 && n.fract() == 0.0, "binom.pmf: n must be non-neg int");
    assert!((0.0..=1.0).contains(&p), "binom.pmf: p in [0,1]");
    let nn = n as i64;
    if k < 0.0 || k > n || k.fract() != 0.0 {
        return 0.0;
    }
    let kk = k as i64;
    if p == 0.0 {
        return if kk == 0 { 1.0 } else { 0.0 };
    }
    if p == 1.0 {
        return if kk == nn { 1.0 } else { 0.0 };
    }
    // C(n,k) * p^k * (1-p)^(n-k)
    let log_c = gammaln_scalar(n + 1.0) - gammaln_scalar(k + 1.0) - gammaln_scalar(n - k + 1.0);
    (log_c + kk as f64 * p.ln() + (nn - kk) as f64 * (1.0 - p).ln()).exp()
}

/// `scipy.stats.binom.cdf(k, n, p)`.
pub fn binom_cdf_scalar(k: f64, n: f64, p: f64) -> f64 {
    assert!(n >= 0.0 && n.fract() == 0.0, "binom.cdf: n must be non-neg int");
    assert!((0.0..=1.0).contains(&p), "binom.cdf: p in [0,1]");
    if k < 0.0 {
        return 0.0;
    }
    if k >= n {
        return 1.0;
    }
    let kk = k.floor() as i64;
    let mut s = 0.0;
    for i in 0..=kk {
        s += binom_pmf_scalar(i as f64, n, p);
    }
    s.clamp(0.0, 1.0)
}

pub fn binom_pmf(a: &NdArray, n: f64, p: f64) -> NdArray {
    map_contig(a, |k| binom_pmf_scalar(k, n, p))
}

pub fn binom_cdf(a: &NdArray, n: f64, p: f64) -> NdArray {
    map_contig(a, |k| binom_cdf_scalar(k, n, p))
}

/// Invert a rising CDF on `[lo, hi]` by bisection (with mild bracket expansion).
fn invert_monotonic(p: f64, mut lo: f64, mut hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    // Expand bracket if needed (skip non-finite expansion for [0,1] beta).
    for _ in 0..60 {
        let flo = cdf(lo);
        let fhi = cdf(hi);
        if flo <= p && p <= fhi {
            break;
        }
        if flo > p {
            let width = (hi - lo).abs().max(1.0);
            lo -= width;
        }
        if fhi < p {
            let width = (hi - lo).abs().max(1.0);
            hi += width;
        }
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Lower regularized incomplete gamma P(a,x) = γ(a,x)/Γ(a).
fn gammainc_reg(a: f64, x: f64) -> f64 {
    assert!(a > 0.0, "gammainc: a must be > 0");
    if x <= 0.0 {
        return 0.0;
    }
    // Series for x < a+1, otherwise Q via continued fraction.
    if x < a + 1.0 {
        let mut ap = a;
        let mut sum = 1.0 / a;
        let mut del = sum;
        for _ in 0..200 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-14 {
                break;
            }
        }
        (-x + a * x.ln() - gammaln_scalar(a)).exp() * sum
    } else {
        // Q(a,x) continued fraction; P = 1 - Q
        let mut b = x + 1.0 - a;
        let mut c = 1e30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=200 {
            let an = -i as f64 * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-14 {
                break;
            }
        }
        let q = (-x + a * x.ln() - gammaln_scalar(a)).exp() * h;
        (1.0 - q).clamp(0.0, 1.0)
    }
}

/// Shannon entropy — `scipy.stats.entropy(pk)` (natural log, axis=None).
///
/// `pk` is treated as an unnormalized probability mass (non-negative); zero
/// mass contributes 0. Matches SciPy when `qk` is None.
pub fn entropy(pk: &NdArray) -> f64 {
    let c = contig(pk);
    let s = c.as_slice().unwrap();
    let mut total = 0.0;
    for &p in s {
        assert!(p >= 0.0 && p.is_finite(), "entropy: pk must be non-negative finite");
        total += p;
    }
    assert!(total > 0.0, "entropy: pk must sum to > 0");
    let mut h = 0.0;
    for &p in s {
        if p > 0.0 {
            let q = p / total;
            h -= q * q.ln();
        }
    }
    h
}

/// `scipy.stats.zscore(a, ddof=0)` along flatten (axis=None for 1D harness).
pub fn zscore(a: &NdArray, ddof: usize) -> NdArray {
    let c = contig(a);
    let s = c.as_slice().unwrap();
    let m = mean_slice(s);
    let v = var_slice(s, ddof);
    let std = v.sqrt();
    assert!(std > 0.0, "zscore: zero standard deviation");
    let out: Vec<f64> = s.iter().map(|&x| (x - m) / std).collect();
    NdArray::from_shape_vec(a.shape(), out)
}

/// `scipy.stats.rankdata(a, method='average')`.
pub fn rankdata(a: &NdArray) -> NdArray {
    let c = contig(a);
    let s = c.as_slice().unwrap();
    let n = s.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| s[i].partial_cmp(&s[j]).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && s[idx[j]] == s[idx[i]] {
            j += 1;
        }
        // average rank of ties; ranks are 1-based
        let avg = (i + j + 1) as f64 / 2.0; // (i+1 + j)/2
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        i = j;
    }
    NdArray::from_shape_vec(a.shape(), ranks)
}

/// `scipy.stats.pearsonr(x, y)` → `(r, pvalue)`.
///
/// Two-sided p-value via Student-t approximation (matches SciPy for n≥3).
pub fn pearsonr(x: &NdArray, y: &NdArray) -> (f64, f64) {
    let xc = contig(x);
    let yc = contig(y);
    let xs = xc.as_slice().unwrap();
    let ys = yc.as_slice().unwrap();
    assert_eq!(xs.len(), ys.len(), "pearsonr: length mismatch");
    let n = xs.len();
    assert!(n >= 2, "pearsonr: need n >= 2");

    let mx = mean_slice(xs);
    let my = mean_slice(ys);
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..n {
        let dx = xs[i] - mx;
        let dy = ys[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let den = (dx2 * dy2).sqrt();
    let r = if den == 0.0 { 0.0 } else { num / den };
    let r = r.clamp(-1.0, 1.0);

    let p = if n < 3 || r.abs() >= 1.0 {
        if r.abs() >= 1.0 {
            0.0
        } else {
            f64::NAN
        }
    } else {
        let df = (n - 2) as f64;
        let t = r * (df / ((1.0 - r * r).max(0.0))).sqrt();
        2.0 * student_t_sf(t.abs(), df)
    };
    (r, p)
}

/// One-sided upper tail P(T > t) for Student-t with `df` degrees (`t >= 0`).
fn student_t_sf(t: f64, df: f64) -> f64 {
    // = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    let x = df / (df + t * t);
    0.5 * betainc_reg(0.5 * df, 0.5, x)
}

/// Regularized incomplete beta I_x(a,b).
fn betainc_reg(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use the continued-fraction side that converges faster.
    let use_complement = x > (a + 1.0) / (a + b + 2.0);
    let (aa, bb, xx, complement) = if use_complement {
        (b, a, 1.0 - x, true)
    } else {
        (a, b, x, false)
    };
    let lbeta = crate::special::gammaln_scalar(aa) + crate::special::gammaln_scalar(bb)
        - crate::special::gammaln_scalar(aa + bb);
    let front = (aa * xx.ln() + bb * (1.0 - xx).ln() - lbeta).exp() / aa;
    let cf = front * betacf(aa, bb, xx);
    if complement {
        1.0 - cf
    } else {
        cf
    }
}

fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 400;
    const EPS: f64 = 1e-14;
    const FPMIN: f64 = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_IT {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// `scipy.stats.spearmanr(x, y)` → `(correlation, pvalue)` for 1D.
pub fn spearmanr(x: &NdArray, y: &NdArray) -> (f64, f64) {
    let rx = rankdata(x);
    let ry = rankdata(y);
    pearsonr(&rx, &ry)
}

/// Result of a two-sample t-test.
#[derive(Debug, Clone, Copy)]
pub struct TtestResult {
    pub statistic: f64,
    pub pvalue: f64,
}

/// `scipy.stats.ttest_ind(a, b, equal_var=True)`.
pub fn ttest_ind(a: &NdArray, b: &NdArray) -> TtestResult {
    let ac = contig(a);
    let bc = contig(b);
    let as_ = ac.as_slice().unwrap();
    let bs = bc.as_slice().unwrap();
    let na = as_.len();
    let nb = bs.len();
    assert!(na >= 2 && nb >= 2, "ttest_ind: need n>=2 per sample");

    let ma = mean_slice(as_);
    let mb = mean_slice(bs);
    let va = var_slice(as_, 1);
    let vb = var_slice(bs, 1);

    let df = (na + nb - 2) as f64;
    let sp2 = ((na - 1) as f64 * va + (nb - 1) as f64 * vb) / df;
    let se = (sp2 * (1.0 / na as f64 + 1.0 / nb as f64)).sqrt();
    let t = if se == 0.0 { 0.0 } else { (ma - mb) / se };
    let p = 2.0 * student_t_sf(t.abs(), df);
    TtestResult {
        statistic: t,
        pvalue: p,
    }
}

/// Sample skewness — `scipy.stats.skew(a, bias=True)` (Fisher-Pearson).
pub fn skew(a: &NdArray) -> f64 {
    let c = contig(a);
    let s = c.as_slice().unwrap();
    let n = s.len() as f64;
    assert!(n >= 1.0);
    let m = mean_slice(s);
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &x in s {
        let d = x - m;
        let d2 = d * d;
        m2 += d2;
        m3 += d2 * d;
    }
    m2 /= n;
    m3 /= n;
    if m2 == 0.0 {
        return 0.0;
    }
    m3 / m2.powf(1.5)
}

/// Excess kurtosis — `scipy.stats.kurtosis(a, fisher=True, bias=True)`.
pub fn kurtosis(a: &NdArray) -> f64 {
    let c = contig(a);
    let s = c.as_slice().unwrap();
    let n = s.len() as f64;
    let m = mean_slice(s);
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &x in s {
        let d = x - m;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    if m2 == 0.0 {
        return -3.0; // excess kurtosis of degenerate
    }
    m4 / (m2 * m2) - 3.0
}

/// `scipy.stats.sem(a, ddof=1)`.
pub fn sem(a: &NdArray, ddof: usize) -> f64 {
    let c = contig(a);
    let s = c.as_slice().unwrap();
    let n = s.len();
    (var_slice(s, ddof) / n as f64).sqrt()
}

fn map_contig(a: &NdArray, f: impl Fn(f64) -> f64) -> NdArray {
    let c = a.to_contiguous();
    let out: Vec<f64> = c.as_slice().unwrap().iter().copied().map(f).collect();
    NdArray::from_shape_vec(a.shape(), out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    #[test]
    fn norm_pdf_at_zero() {
        assert_close(norm_pdf_scalar(0.0), 0.3989422804014327, 1e-12);
    }

    #[test]
    fn norm_cdf_ppf_roundtrip() {
        for x in [-1.5, 0.0, 0.7, 2.0] {
            let p = norm_cdf_scalar(x);
            assert_close(norm_ppf_scalar(p), x, 5e-6);
        }
    }

    #[test]
    fn entropy_uniform() {
        let pk = NdArray::from_vec(vec![1.0, 1.0, 1.0, 1.0]);
        assert_close(entropy(&pk), (4.0f64).ln(), 1e-12);
    }

    #[test]
    fn zscore_mean_zero() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let z = zscore(&a, 0);
        assert_close(z.sum(), 0.0, 1e-12);
    }

    #[test]
    fn rankdata_ties() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 2.0, 4.0]);
        let r = rankdata(&a);
        assert_close(r[0], 1.0, 1e-12);
        assert_close(r[1], 2.5, 1e-12);
        assert_close(r[2], 2.5, 1e-12);
        assert_close(r[3], 4.0, 1e-12);
    }

    #[test]
    fn pearsonr_perfect() {
        let x = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = NdArray::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        let (r, p) = pearsonr(&x, &y);
        assert_close(r, 1.0, 1e-12);
        assert_close(p, 0.0, 1e-12);
    }

    #[test]
    fn ttest_identical() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let r = ttest_ind(&a, &b);
        assert_close(r.statistic, 0.0, 1e-12);
        assert_close(r.pvalue, 1.0, 1e-10);
    }

    #[test]
    fn skew_symmetric() {
        let a = NdArray::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_close(skew(&a), 0.0, 1e-12);
    }

    #[test]
    fn uniform_cdf_ppf_roundtrip() {
        for p in [0.0, 0.25, 0.5, 0.9, 1.0] {
            assert_close(uniform_ppf_scalar(uniform_cdf_scalar(p)), p, 1e-12);
        }
    }

    #[test]
    fn expon_cdf_ppf_roundtrip() {
        for x in [0.0, 0.5, 1.0, 2.5] {
            let p = expon_cdf_scalar(x);
            assert_close(expon_ppf_scalar(p), x, 1e-12);
        }
    }

    #[test]
    fn laplace_pdf_symmetric() {
        assert_close(laplace_pdf_scalar(1.5), laplace_pdf_scalar(-1.5), 1e-15);
        assert_close(laplace_pdf_scalar(0.0), 0.5, 1e-15);
    }

    #[test]
    fn logistic_cdf_ppf_roundtrip() {
        for x in [-2.0, -0.5, 0.0, 1.0, 3.0] {
            let p = logistic_cdf_scalar(x);
            assert_close(logistic_ppf_scalar(p), x, 1e-12);
        }
    }

    #[test]
    fn t_chi2_gamma_beta_smoke() {
        let df = 5.0;
        assert!(t_pdf_scalar(0.0, df) > 0.0);
        assert_close(t_cdf_scalar(0.0, df), 0.5, 1e-12);
        assert_close(t_ppf_scalar(0.5, df), 0.0, 1e-6);

        assert!(chi2_pdf_scalar(2.0, df) > 0.0);
        let p = chi2_cdf_scalar(3.0, df);
        assert_close(chi2_ppf_scalar(p, df), 3.0, 1e-4);

        let a = 2.0;
        let p = gamma_cdf_scalar(1.5, a);
        assert_close(gamma_ppf_scalar(p, a), 1.5, 1e-4);

        let p = beta_cdf_scalar(0.4, 2.0, 5.0);
        assert_close(beta_ppf_scalar(p, 2.0, 5.0), 0.4, 1e-5);
    }

    #[test]
    fn poisson_binom_smoke() {
        assert_close(poisson_pmf_scalar(0.0, 0.0), 1.0, 1e-12);
        let s: f64 = (0..=20).map(|k| poisson_pmf_scalar(k as f64, 3.0)).sum();
        assert_close(s, 1.0, 1e-6);
        assert!(poisson_cdf_scalar(2.0, 3.0) < poisson_cdf_scalar(5.0, 3.0));

        let bs: f64 = (0..=10).map(|k| binom_pmf_scalar(k as f64, 10.0, 0.3)).sum();
        assert_close(bs, 1.0, 1e-6);
        assert_close(binom_cdf_scalar(10.0, 10.0, 0.3), 1.0, 1e-12);
    }
}
