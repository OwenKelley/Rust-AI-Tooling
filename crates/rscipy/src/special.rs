//! Special functions — mirrors `scipy.special` starters.
//!
//! Elementwise APIs take/return `rnumpy::NdArray`. Scalar helpers are public
//! for tests and for composing array maps.

use rnumpy::NdArray;

/// Abramowitz & Stegun 7.1.26 rational approximation for `erf`.
pub fn erf_scalar(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    // Coefficients from A&S 7.1.26 (max error ~1.5e-7).
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + P * ax);
    let y = 1.0
        - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-ax * ax).exp();
    sign * y
}

/// `scipy.special.erf` (elementwise).
pub fn erf(a: &NdArray) -> NdArray {
    map_contig(a, erf_scalar)
}

/// `scipy.special.erfc` = `1 - erf`.
pub fn erfc(a: &NdArray) -> NdArray {
    map_contig(a, |x| 1.0 - erf_scalar(x))
}

/// Lanczos approximation constants (g = 7, N = 9) for gamma.
const LANCZOS_G: f64 = 7.0;
const LANCZOS_P: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_13,
    -176.615_029_162_140_59,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_654_078_991e-6,
    1.505_632_735_149_311_6e-7,
];

fn lanczos_ag(z: f64) -> f64 {
    let mut a = LANCZOS_P[0];
    for (i, &p) in LANCZOS_P.iter().enumerate().skip(1) {
        a += p / (z + i as f64);
    }
    a
}

/// Scalar gamma via Lanczos (reflection for z < 0.5).
pub fn gamma_scalar(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z < 0.5 {
        let sin_pi_z = (std::f64::consts::PI * z).sin();
        if sin_pi_z == 0.0 {
            return f64::NAN;
        }
        return std::f64::consts::PI / (sin_pi_z * gamma_scalar(1.0 - z));
    }
    let z = z - 1.0;
    let x = lanczos_ag(z);
    let t = z + LANCZOS_G + 0.5;
    (2.0 * std::f64::consts::PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// Scalar `log|Γ(z)|` via Lanczos.
pub fn gammaln_scalar(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z < 0.5 {
        let sin_pi_z = (std::f64::consts::PI * z).sin().abs();
        if sin_pi_z == 0.0 {
            return f64::INFINITY;
        }
        return std::f64::consts::PI.ln() - sin_pi_z.ln() - gammaln_scalar(1.0 - z);
    }
    let z = z - 1.0;
    let x = lanczos_ag(z);
    let t = z + LANCZOS_G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// `scipy.special.gamma` (elementwise).
pub fn gamma(a: &NdArray) -> NdArray {
    map_contig(a, gamma_scalar)
}

/// `scipy.special.gammaln` (elementwise).
pub fn gammaln(a: &NdArray) -> NdArray {
    map_contig(a, gammaln_scalar)
}

/// Logistic sigmoid — `scipy.special.expit`.
pub fn expit_scalar(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

/// Inverse of expit — `scipy.special.logit`.
pub fn logit_scalar(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    (p / (1.0 - p)).ln()
}

/// `scipy.special.expit` (elementwise).
pub fn expit(a: &NdArray) -> NdArray {
    map_contig(a, expit_scalar)
}

/// `scipy.special.logit` (elementwise).
pub fn logit(a: &NdArray) -> NdArray {
    map_contig(a, logit_scalar)
}

/// Stable `log(sum(exp(a)))` over all elements (SciPy `axis=None`).
pub fn logsumexp(a: &NdArray) -> f64 {
    let c = a.to_contiguous();
    let s = c.as_slice().unwrap();
    if s.is_empty() {
        return f64::NEG_INFINITY;
    }
    let mut max_v = f64::NEG_INFINITY;
    for &v in s {
        if v > max_v {
            max_v = v;
        }
    }
    if !max_v.is_finite() {
        return max_v;
    }
    let mut sum = 0.0;
    for &v in s {
        sum += (v - max_v).exp();
    }
    max_v + sum.ln()
}

/// Softmax over all elements (SciPy `axis=None` flatten semantics for harness).
pub fn softmax(a: &NdArray) -> NdArray {
    let c = a.to_contiguous();
    let s = c.as_slice().unwrap();
    let lse = logsumexp(a);
    let out: Vec<f64> = s.iter().map(|&x| (x - lse).exp()).collect();
    NdArray::from_shape_vec(a.shape(), out)
}

/// Modified Bessel function of the first kind, order 0 — `scipy.special.i0`.
///
/// Polynomial / asymptotic forms from Cephes / A&S.
pub fn i0_scalar(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2
                        + y * (0.265_973_2
                            + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        (ax.exp() / ax.sqrt())
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

/// `scipy.special.i0` (elementwise).
pub fn i0(a: &NdArray) -> NdArray {
    map_contig(a, i0_scalar)
}

/// Standard normal CDF — `scipy.special.ndtr`.
pub fn ndtr_scalar(x: f64) -> f64 {
    0.5 * (1.0 + erf_scalar(x / std::f64::consts::SQRT_2))
}

/// `scipy.special.ndtr` (elementwise).
pub fn ndtr(a: &NdArray) -> NdArray {
    map_contig(a, ndtr_scalar)
}

/// Inverse standard normal CDF — `scipy.special.ndtri`.
///
/// Rational approximation (Beasley–Springer / Moro style).
pub fn ndtri_scalar(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Coefficients from Acklam's inverse normal CDF approximation.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_690e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let plow = 0.024_25;
    let phigh = 1.0 - plow;

    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

/// `scipy.special.ndtri` (elementwise).
pub fn ndtri(a: &NdArray) -> NdArray {
    map_contig(a, ndtri_scalar)
}

fn map_contig(a: &NdArray, f: impl Fn(f64) -> f64) -> NdArray {
    let c = a.to_contiguous();
    let s = c.as_slice().unwrap();
    let mut out = Vec::with_capacity(s.len());
    // Contiguous write avoids per-element collect intermediate overhead.
    out.extend(s.iter().map(|&x| f(x)));
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
    fn erf_known() {
        assert_close(erf_scalar(0.0), 0.0, 1e-12);
        assert_close(erf_scalar(1.0), 0.8427007929497149, 2e-7);
        assert_close(erf_scalar(-1.0), -0.8427007929497149, 2e-7);
    }

    #[test]
    fn gamma_known() {
        assert_close(gamma_scalar(1.0), 1.0, 1e-12);
        assert_close(gamma_scalar(2.0), 1.0, 1e-12);
        assert_close(gamma_scalar(5.0), 24.0, 1e-10);
        assert_close(gamma_scalar(0.5), std::f64::consts::PI.sqrt(), 1e-10);
    }

    #[test]
    fn gammaln_matches_log_gamma() {
        for z in [0.5, 1.0, 2.5, 10.0, 20.0] {
            assert_close(gammaln_scalar(z), gamma_scalar(z).ln(), 1e-10);
        }
    }

    #[test]
    fn expit_logit_roundtrip() {
        for x in [-2.0, -0.5, 0.0, 0.5, 2.0] {
            let p = expit_scalar(x);
            assert_close(logit_scalar(p), x, 1e-12);
        }
    }

    #[test]
    fn logsumexp_softmax() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0]);
        let lse = logsumexp(&a);
        assert_close(lse, (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln(), 1e-12);
        let s = softmax(&a);
        assert_close(s.sum(), 1.0, 1e-12);
    }

    #[test]
    fn i0_known() {
        assert_close(i0_scalar(0.0), 1.0, 1e-12);
        assert_close(i0_scalar(1.0), 1.2660658777520082, 1e-6);
    }

    #[test]
    fn ndtr_ndtri_roundtrip() {
        for x in [-2.0, -0.5, 0.0, 0.5, 2.0] {
            let p = ndtr_scalar(x);
            assert_close(ndtri_scalar(p), x, 5e-6);
        }
    }
}
