//! Fast f32 elementwise kernels (scalar + AVX2/FMA). Local/`std` only.

/// Apply `exp` into `out` (same length as `x`).
pub fn exp_f32(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                exp_avx2(x, out);
            }
            return;
        }
    }
    for i in 0..x.len() {
        out[i] = exp_scalar(x[i]);
    }
}

/// Apply natural `log` into `out`.
pub fn log_f32(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                log_avx2(x, out);
            }
            return;
        }
    }
    for i in 0..x.len() {
        out[i] = log_scalar(x[i]);
    }
}

/// Elementwise `a.powf(b)` for equal-length slices.
pub fn pow_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                pow_avx2(a, b, out);
            }
            return;
        }
    }
    for i in 0..a.len() {
        out[i] = pow_scalar(a[i], b[i]);
    }
}

const LOG2E: f32 = 1.4426950408889634;
const LN2_HI: f32 = 0.693145751953125;
const LN2_LO: f32 = 1.428606765330187e-6;

#[inline]
pub fn exp_scalar(x: f32) -> f32 {
    let x = x.clamp(-88.0, 88.0);
    let nf = (x * LOG2E).round();
    let n = nf as i32;
    let t = (x - nf * LN2_HI) - nf * LN2_LO;
    // exp(t) ≈ 1 + t*(1 + t*(1/2 + t*(1/6 + t*(1/24 + t/120))))
    let p = {
        let c5 = 0.00833333333f32; // 1/120
        let c4 = 0.04166666666f32; // 1/24
        let c3 = 0.16666666666f32; // 1/6
        let c2 = 0.5f32;
        let mut p = c5.mul_add(t, c4);
        p = p.mul_add(t, c3);
        p = p.mul_add(t, c2);
        p = p.mul_add(t, 1.0);
        p.mul_add(t, 1.0)
    };
    let s = f32::from_bits(((n + 127) as u32) << 23);
    p * s
}

#[inline]
pub fn log_scalar(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NEG_INFINITY;
    }
    let bits = x.to_bits();
    let mut e = ((bits >> 23) as i32) - 127;
    let mut m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000); // [1, 2)
    // Center around 1: if m > √2, fold into [√2/2, √2]
    if m > 1.41421356237 {
        m *= 0.5;
        e += 1;
    }
    let f = m - 1.0;
    let f2 = f * f;
    // log1p series through f^7
    let mut p: f32 = 1.0 / 7.0;
    p = p.mul_add(f, -1.0 / 6.0);
    p = p.mul_add(f, 1.0 / 5.0);
    p = p.mul_add(f, -1.0 / 4.0);
    p = p.mul_add(f, 1.0 / 3.0);
    p = p.mul_add(f, -0.5);
    p = p.mul_add(f2, f);
    p + (e as f32) * std::f32::consts::LN_2
}

#[inline]
fn pow_scalar(a: f32, b: f32) -> f32 {
    if a > 0.0 {
        exp_scalar(b * log_scalar(a))
    } else {
        a.powf(b)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_avx2(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vmax = _mm256_set1_ps(88.0);
    let vmin = _mm256_set1_ps(-88.0);
    let vlog2e = _mm256_set1_ps(LOG2E);
    let vhalf = _mm256_set1_ps(0.5);
    let vln2_hi = _mm256_set1_ps(LN2_HI);
    let vln2_lo = _mm256_set1_ps(LN2_LO);
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(0.16666666666);
    let c4 = _mm256_set1_ps(0.04166666666);
    let c5 = _mm256_set1_ps(0.00833333333);
    let v127 = _mm256_set1_epi32(127);

    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let mut vx = _mm256_loadu_ps(x.as_ptr().add(i));
        vx = _mm256_min_ps(_mm256_max_ps(vx, vmin), vmax);
        // round(x * log2e)
        let mut vn = _mm256_fmadd_ps(vx, vlog2e, vhalf);
        vn = _mm256_floor_ps(vn);
        let ni = _mm256_cvtps_epi32(vn);
        let vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(ni, v127), 23));
        let vt = _mm256_sub_ps(vx, _mm256_mul_ps(vn, vln2_hi));
        let vt = _mm256_sub_ps(vt, _mm256_mul_ps(vn, vln2_lo));
        let mut vp = _mm256_fmadd_ps(c5, vt, c4);
        vp = _mm256_fmadd_ps(vp, vt, c3);
        vp = _mm256_fmadd_ps(vp, vt, c2);
        vp = _mm256_fmadd_ps(vp, vt, c1);
        vp = _mm256_fmadd_ps(vp, vt, c1);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(vp, vs));
        i += 8;
    }
    while i < n {
        out[i] = exp_scalar(x[i]);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn log_avx2(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vsqrt2 = _mm256_set1_ps(1.41421356237);
    let vone = _mm256_set1_ps(1.0);
    let vln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let a2 = _mm256_set1_ps(-0.5);
    let a3 = _mm256_set1_ps(1.0 / 3.0);
    let a4 = _mm256_set1_ps(-0.25);
    let a5 = _mm256_set1_ps(0.2);
    let a6 = _mm256_set1_ps(-1.0 / 6.0);
    let a7 = _mm256_set1_ps(1.0 / 7.0);
    let mant_mask = _mm256_set1_epi32(0x007F_FFFF);
    let exp_bias = _mm256_set1_epi32(127);
    let one_bits = _mm256_set1_epi32(0x3F80_0000);

    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        let bits = _mm256_castps_si256(vx);
        let e_i = _mm256_sub_epi32(_mm256_srli_epi32(bits, 23), exp_bias);
        let mut m = _mm256_castsi256_ps(_mm256_or_si256(
            _mm256_and_si256(bits, mant_mask),
            one_bits,
        ));
        let mut e = _mm256_cvtepi32_ps(e_i);
        let mask = _mm256_cmp_ps(m, vsqrt2, _CMP_GT_OS);
        m = _mm256_blendv_ps(m, _mm256_mul_ps(m, _mm256_set1_ps(0.5)), mask);
        e = _mm256_add_ps(e, _mm256_and_ps(mask, _mm256_set1_ps(1.0)));
        let f = _mm256_sub_ps(m, vone);
        let mut vp = _mm256_fmadd_ps(a7, f, a6);
        vp = _mm256_fmadd_ps(vp, f, a5);
        vp = _mm256_fmadd_ps(vp, f, a4);
        vp = _mm256_fmadd_ps(vp, f, a3);
        vp = _mm256_fmadd_ps(vp, f, a2);
        let f2 = _mm256_mul_ps(f, f);
        vp = _mm256_fmadd_ps(vp, f2, f);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_fmadd_ps(e, vln2, vp));
        i += 8;
    }
    while i < n {
        out[i] = log_scalar(x[i]);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn pow_avx2(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let mut i = 0;
    let mut tmp_log = [0.0f32; 8];
    let mut tmp_exp_in = [0.0f32; 8];
    while i + 8 <= n {
        log_avx2(&a[i..i + 8], &mut tmp_log);
        for k in 0..8 {
            tmp_exp_in[k] = b[i + k] * tmp_log[k];
        }
        exp_avx2(&tmp_exp_in, &mut out[i..i + 8]);
        i += 8;
    }
    while i < n {
        out[i] = pow_scalar(a[i], b[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_log_roundtrip() {
        for &x in &[-2.0f32, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0] {
            let e = exp_scalar(x);
            let lib = x.exp();
            assert!((e - lib).abs() < 1e-4 * lib.max(1.0), "exp {x}: {e} vs {lib}");
            if x > 0.1 {
                let l = log_scalar(x);
                let libl = x.ln();
                assert!((l - libl).abs() < 2e-4, "log {x}: {l} vs {libl}");
            }
        }
        let mut xin = [0.5f32, 1.0, 1.5, 2.0, -1.0, -0.5, 0.0, 3.0];
        let mut out = [0.0f32; 8];
        exp_f32(&xin, &mut out);
        for i in 0..8 {
            let lib = xin[i].exp();
            assert!((out[i] - lib).abs() < 1e-3 * lib.max(1.0), "vec exp {}", xin[i]);
        }
        xin = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0];
        log_f32(&xin, &mut out);
        for i in 0..8 {
            let lib = xin[i].ln();
            assert!((out[i] - lib).abs() < 2e-4, "vec log {}", xin[i]);
        }
    }
}
