//! Statistics — mirrors common `scipy.stats` entry points used in AI/ML.
//!
//! Local implementations only. Continuous distributions reuse `special::ndtr` /
//! `ndtri` where applicable.

use crate::special::{ndtr_scalar, ndtri_scalar};
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
}
