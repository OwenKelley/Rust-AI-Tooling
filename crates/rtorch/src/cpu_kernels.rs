//! Portable CPU kernels (scalar + AVX2/FMA when available).

/// `out[i, :] += bias` for row-major `[batch, out_f]`.
pub fn bias_add_rows(out: &mut [f32], bias: &[f32], batch: usize, out_f: usize) {
    debug_assert_eq!(out.len(), batch * out_f);
    debug_assert_eq!(bias.len(), out_f);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if out_f >= 8 && is_x86_feature_detected!("avx2") {
            unsafe {
                bias_add_rows_avx2(out, bias, batch, out_f);
            }
            return;
        }
    }
    for i in 0..batch {
        let row = &mut out[i * out_f..(i + 1) * out_f];
        for j in 0..out_f {
            row[j] += bias[j];
        }
    }
}

/// Fused bias-add + ReLU; optional mask for autograd (`true` = kept).
pub fn bias_relu_rows(
    out: &mut [f32],
    bias: Option<&[f32]>,
    mut mask: Option<&mut [bool]>,
    batch: usize,
    out_f: usize,
) {
    debug_assert_eq!(out.len(), batch * out_f);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_f);
    }
    if let Some(m) = mask.as_ref() {
        debug_assert_eq!(m.len(), batch * out_f);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if out_f >= 8 && is_x86_feature_detected!("avx2") {
            unsafe {
                bias_relu_rows_avx2(out, bias, mask, batch, out_f);
            }
            return;
        }
    }
    for i in 0..batch {
        for j in 0..out_f {
            let idx = i * out_f + j;
            let mut v = out[idx];
            if let Some(b) = bias {
                v += b[j];
            }
            let pos = v > 0.0;
            if let Some(m) = mask.as_mut() {
                m[idx] = pos;
            }
            out[idx] = if pos { v } else { 0.0 };
        }
    }
}

/// Softmax in-place into `probs` from logits row-major `[n, c]`; returns mean NLL.
pub fn cross_entropy_mean(
    logits: &[f32],
    target: &[usize],
    probs: &mut [f32],
    n: usize,
    c: usize,
) -> f32 {
    debug_assert_eq!(logits.len(), n * c);
    debug_assert_eq!(probs.len(), n * c);
    debug_assert_eq!(target.len(), n);
    let mut loss = 0.0f32;
    for i in 0..n {
        let row = &logits[i * c..(i + 1) * c];
        let pref = &mut probs[i * c..(i + 1) * c];
        let mut m = row[0];
        for &v in row.iter().skip(1) {
            if v > m {
                m = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..c {
            let e = (row[j] - m).exp();
            pref[j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..c {
            pref[j] *= inv;
        }
        let p = pref[target[i]].max(1e-12);
        loss -= p.ln();
    }
    loss / n as f32
}

/// `db[j] += sum_i gy[i, j]`.
pub fn reduce_bias_grad(gy: &[f32], db: &mut [f32], n: usize, out_f: usize) {
    debug_assert_eq!(gy.len(), n * out_f);
    debug_assert_eq!(db.len(), out_f);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if out_f >= 8 && is_x86_feature_detected!("avx2") {
            unsafe {
                reduce_bias_grad_avx2(gy, db, n, out_f);
            }
            return;
        }
    }
    for i in 0..n {
        let row = &gy[i * out_f..(i + 1) * out_f];
        for j in 0..out_f {
            db[j] += row[j];
        }
    }
}

/// `gout = (probs - one_hot(target)) * (gy0 / n)`.
pub fn cross_entropy_input_grad(
    probs: &[f32],
    target: &[usize],
    gy0: f32,
    gout: &mut [f32],
    n: usize,
    c: usize,
) {
    let inv_n = gy0 / n as f32;
    for i in 0..n {
        for j in 0..c {
            let mut v = probs[i * c + j];
            if j == target[i] {
                v -= 1.0;
            }
            gout[i * c + j] = v * inv_n;
        }
    }
}

/// Apply ReLU mask: `out[i] = mask[i] ? gy[i] : 0`.
pub fn apply_relu_mask(gy: &[f32], mask: &[bool], out: &mut [f32]) {
    debug_assert_eq!(gy.len(), mask.len());
    debug_assert_eq!(gy.len(), out.len());
    let n = gy.len();
    let mut i = 0usize;
    while i + 8 <= n {
        for t in 0..8 {
            let j = i + t;
            out[j] = if mask[j] { gy[j] } else { 0.0 };
        }
        i += 8;
    }
    while i < n {
        out[i] = if mask[i] { gy[i] } else { 0.0 };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bias_add_rows_avx2(out: &mut [f32], bias: &[f32], batch: usize, out_f: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut j = 0usize;
    while j + 8 <= out_f {
        let bv = _mm256_loadu_ps(bias.as_ptr().add(j));
        for i in 0..batch {
            let p = out.as_mut_ptr().add(i * out_f + j);
            let ov = _mm256_loadu_ps(p);
            _mm256_storeu_ps(p, _mm256_add_ps(ov, bv));
        }
        j += 8;
    }
    while j < out_f {
        let b = *bias.get_unchecked(j);
        for i in 0..batch {
            *out.get_unchecked_mut(i * out_f + j) += b;
        }
        j += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bias_relu_rows_avx2(
    out: &mut [f32],
    bias: Option<&[f32]>,
    mut mask: Option<&mut [bool]>,
    batch: usize,
    out_f: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let mut j = 0usize;
    while j + 8 <= out_f {
        let bv = match bias {
            Some(b) => _mm256_loadu_ps(b.as_ptr().add(j)),
            None => zero,
        };
        for i in 0..batch {
            let idx = i * out_f + j;
            let p = out.as_mut_ptr().add(idx);
            let mut v = _mm256_loadu_ps(p);
            if bias.is_some() {
                v = _mm256_add_ps(v, bv);
            }
            let cmp = _mm256_cmp_ps(v, zero, _CMP_GT_OQ);
            let kept = _mm256_and_ps(v, cmp);
            _mm256_storeu_ps(p, kept);
            if let Some(m) = mask.as_mut() {
                let bits = _mm256_movemask_ps(cmp) as u32;
                for t in 0..8 {
                    m[idx + t] = (bits & (1 << t)) != 0;
                }
            }
        }
        j += 8;
    }
    while j < out_f {
        let b = bias.map(|bb| *bb.get_unchecked(j)).unwrap_or(0.0);
        for i in 0..batch {
            let idx = i * out_f + j;
            let mut v = *out.get_unchecked(idx) + b;
            let pos = v > 0.0;
            if let Some(m) = mask.as_mut() {
                m[idx] = pos;
            }
            if !pos {
                v = 0.0;
            }
            *out.get_unchecked_mut(idx) = v;
        }
        j += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn reduce_bias_grad_avx2(gy: &[f32], db: &mut [f32], n: usize, out_f: usize) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut j = 0usize;
    while j + 8 <= out_f {
        let mut acc = _mm256_loadu_ps(db.as_ptr().add(j));
        for i in 0..n {
            let gv = _mm256_loadu_ps(gy.as_ptr().add(i * out_f + j));
            acc = _mm256_add_ps(acc, gv);
        }
        _mm256_storeu_ps(db.as_mut_ptr().add(j), acc);
        j += 8;
    }
    while j < out_f {
        let mut s = *db.get_unchecked(j);
        for i in 0..n {
            s += *gy.get_unchecked(i * out_f + j);
        }
        *db.get_unchecked_mut(j) = s;
        j += 1;
    }
}
