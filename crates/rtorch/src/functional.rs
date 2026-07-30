//! `torch.nn.functional` entry points.

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::ops::{matmul_raw, mean, mul, sub, transpose_data};
use crate::tensor::{Tensor, TensorInner};

fn wrap(data: Vec<f32>, shape: &[usize], requires_grad: bool, grad_fn: Option<GradFn>) -> Tensor {
    let n = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    Tensor::from_inner(TensorInner {
        data,
        shape: shape.to_vec(),
        requires_grad,
        grad: if requires_grad {
            Some(vec![0.0; n])
        } else {
            None
        },
        grad_fn,
    })
}

/// `F.relu`
pub fn relu(x: &Tensor) -> Tensor {
    let rg = is_grad_enabled() && x.requires_grad();
    let (data, mask, shape) = {
        let xi = x.inner.borrow();
        let n = xi.data.len();
        let mut data = vec![0.0f32; n];
        let mut mask = if rg { vec![false; n] } else { Vec::new() };
        relu_kernel(xi.data.as_slice(), data.as_mut_slice(), if rg { Some(&mut mask) } else { None });
        (data, mask, xi.shape.clone())
    };
    let gf = if rg {
        Some(GradFn::Relu {
            input: x.clone(),
            mask,
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// `F.leaky_relu(x, negative_slope)`
pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Tensor {
    let xi = x.inner.borrow();
    let rg = is_grad_enabled() && x.requires_grad();
    let data: Vec<f32> = xi
        .data
        .iter()
        .map(|&v| if v >= 0.0 { v } else { v * negative_slope })
        .collect();
    let shape = xi.shape.clone();
    drop(xi);
    let gf = if rg {
        Some(GradFn::LeakyRelu {
            input: x.clone(),
            negative_slope,
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

fn relu_kernel(x: &[f32], out: &mut [f32], mask: Option<&mut [bool]>) {
    debug_assert_eq!(x.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if mask.is_none() && is_x86_feature_detected!("avx2") {
            unsafe {
                relu_avx2(x, out);
            }
            return;
        }
    }
    match mask {
        Some(m) => {
            for i in 0..x.len() {
                let pos = x[i] > 0.0;
                m[i] = pos;
                out[i] = if pos { x[i] } else { 0.0 };
            }
        }
        None => {
            for i in 0..x.len() {
                out[i] = if x[i] > 0.0 { x[i] } else { 0.0 };
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    while i < n {
        out[i] = if x[i] > 0.0 { x[i] } else { 0.0 };
        i += 1;
    }
}

/// `F.sigmoid`
pub fn sigmoid(x: &Tensor) -> Tensor {
    let rg = is_grad_enabled() && x.requires_grad();
    let (data, shape) = {
        let xi = x.inner.borrow();
        let n = xi.data.len();
        let mut data = vec![0.0f32; n];
        sigmoid_kernel(xi.data.as_slice(), data.as_mut_slice());
        (data, xi.shape.clone())
    };
    let gf = if rg {
        Some(GradFn::Sigmoid {
            input: x.clone(),
            fwd: data.clone(),
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// Numerically stable `1 / (1 + exp(-x))` with AVX2/FMA polynomial exp when available.
fn sigmoid_kernel(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                sigmoid_avx2(x, out);
            }
            return;
        }
    }
    for i in 0..x.len() {
        out[i] = sigmoid_scalar(x[i]);
    }
}

/// Cody–Waite + degree-5 exp on `-|x|`, then mirror for `x >= 0` (XNNPACK-style).
#[inline]
fn sigmoid_scalar(x: f32) -> f32 {
    let z = -x.abs();
    const DENORM_CUTOFF: f32 = f32::from_bits(0xC2AEAC4F); // -0x1.5D589Ep+6
    if z < DENORM_CUTOFF {
        return if x.is_sign_negative() { 0.0 } else { 1.0 };
    }
    const MAGIC_BIAS: f32 = f32::from_bits(0x4B40007F); // 0x1.8000FEp23
    const LOG2E: f32 = f32::from_bits(0x3FB8AA3B); // 0x1.715476p0
    const MINUS_LN2_HI: f32 = f32::from_bits(0xBF317218); // -0x1.62E43p-1
    const MINUS_LN2_LO: f32 = f32::from_bits(0x3102E308); // 0x1.05C61p-29
    const C1: f32 = f32::from_bits(0x3F7FFFFB); // 0x1.FFFFF6p-1
    const C2: f32 = f32::from_bits(0x3EFFFEE3); // 0x1.FFFDC6p-2
    const C3: f32 = f32::from_bits(0x3E2AAD40); // 0x1.555A80p-3
    const C4: f32 = f32::from_bits(0x3D2B9D0D); // 0x1.573A1Ap-5
    const C5: f32 = f32::from_bits(0x3C07CFCE); // 0x1.0F9F9Cp-7

    let mut n = z.mul_add(LOG2E, MAGIC_BIAS);
    // s = 2^n via exponent field (valid while magic-bias rounding holds).
    let s = f32::from_bits(n.to_bits() << 23);
    n -= MAGIC_BIAS;
    let t = n.mul_add(MINUS_LN2_HI, z);
    let t = n.mul_add(MINUS_LN2_LO, t);
    let p = C5.mul_add(t, C4).mul_add(t, C3).mul_add(t, C2).mul_add(t, C1);
    let e = (t * s).mul_add(p, s);
    let f = e / (e + 1.0);
    if x.is_sign_negative() {
        f
    } else {
        1.0 - f
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vsign_mask = _mm256_set1_ps(-0.0);
    let vmagic_bias = _mm256_set1_ps(f32::from_bits(0x4B40007F));
    let vlog2e = _mm256_set1_ps(f32::from_bits(0x3FB8AA3B));
    let vminus_ln2_hi = _mm256_set1_ps(f32::from_bits(0xBF317218));
    let vminus_ln2_lo = _mm256_set1_ps(f32::from_bits(0x3102E308));
    let vc5 = _mm256_set1_ps(f32::from_bits(0x3C07CFCE));
    let vc4 = _mm256_set1_ps(f32::from_bits(0x3D2B9D0D));
    let vc3 = _mm256_set1_ps(f32::from_bits(0x3E2AAD40));
    let vc2 = _mm256_set1_ps(f32::from_bits(0x3EFFFEE3));
    let vc1 = _mm256_set1_ps(f32::from_bits(0x3F7FFFFB));
    let vone = _mm256_set1_ps(1.0);
    let vdenorm_cutoff = _mm256_set1_ps(f32::from_bits(0xC2AEAC4F));

    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(i));
        // z = -abs(x)
        let vz = _mm256_or_ps(vx, vsign_mask);

        let mut vn = _mm256_fmadd_ps(vz, vlog2e, vmagic_bias);
        let vs = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(vn), 23));
        vn = _mm256_sub_ps(vn, vmagic_bias);

        let mut vt = _mm256_fmadd_ps(vn, vminus_ln2_hi, vz);
        vt = _mm256_fmadd_ps(vn, vminus_ln2_lo, vt);

        let mut vp = _mm256_fmadd_ps(vc5, vt, vc4);
        vp = _mm256_fmadd_ps(vp, vt, vc3);
        vp = _mm256_fmadd_ps(vp, vt, vc2);
        vp = _mm256_fmadd_ps(vp, vt, vc1);

        vt = _mm256_mul_ps(vt, vs);
        let ve = _mm256_fmadd_ps(vt, vp, vs);
        let vd = _mm256_add_ps(ve, vone);
        let mut vf = _mm256_div_ps(ve, vd);

        // z < denorm_cutoff → 0
        vf = _mm256_andnot_ps(_mm256_cmp_ps(vz, vdenorm_cutoff, _CMP_LT_OS), vf);
        // x < 0 ? f : 1 - f  (blendv uses sign bit of vx)
        vf = _mm256_blendv_ps(_mm256_sub_ps(vone, vf), vf, vx);

        _mm256_storeu_ps(out.as_mut_ptr().add(i), vf);
        i += 8;
    }
    while i < n {
        out[i] = sigmoid_scalar(x[i]);
        i += 1;
    }
}

#[cfg(test)]
mod sigmoid_tests {
    use super::sigmoid_scalar;

    #[test]
    fn sigmoid_scalar_matches_libm() {
        for &x in &[-8.0f32, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0] {
            let got = sigmoid_scalar(x);
            let exp = 1.0 / (1.0 + (-x).exp());
            assert!((got - exp).abs() < 2e-6, "x={x}: {got} vs {exp}");
        }
    }
}

/// `F.linear(input, weight, bias)` — input (N,In), weight (Out,In), bias (Out,).
pub fn linear(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let wt = transpose_data(weight);
    let y = matmul_raw(input, &wt);
    if let Some(b) = bias {
        let bi = b.inner.borrow();
        assert_eq!(bi.shape.len(), 1);
        let mut yi = y.inner.borrow_mut();
        assert_eq!(bi.shape[0], yi.shape[1]);
        let n = yi.shape[0];
        let out_f = yi.shape[1];
        let bd = bi.data.as_slice();
        let yd = yi.data.as_mut_slice();
        for i in 0..n {
            let row = &mut yd[i * out_f..(i + 1) * out_f];
            for j in 0..out_f {
                row[j] += bd[j];
            }
        }
    }
    let rg = is_grad_enabled()
        && (input.requires_grad()
            || weight.requires_grad()
            || bias.map(|b| b.requires_grad()).unwrap_or(false));
    if rg {
        let mut yi = y.inner.borrow_mut();
        yi.requires_grad = true;
        yi.grad = Some(vec![0.0; yi.numel()]);
        yi.grad_fn = Some(GradFn::Linear {
            input: input.clone(),
            weight: weight.clone(),
            bias: bias.cloned(),
        });
    }
    y
}

/// `F.mse_loss(input, target, reduction='mean')`
pub fn mse_loss(input: &Tensor, target: &Tensor) -> Tensor {
    let diff = sub(input, target);
    let sq = mul(&diff, &diff);
    mean(&sq)
}

/// Softmax along the last dimension for 2D `(N, C)`.
pub fn softmax(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 2, "softmax: 2D (N,C) only");
    let xi = x.inner.borrow();
    let n = xi.shape[0];
    let c = xi.shape[1];
    let mut data = vec![0.0f32; n * c];
    let mut row_exp = vec![0.0f32; c];
    for i in 0..n {
        let row = &xi.data[i * c..(i + 1) * c];
        let mut m = row[0];
        for &v in &row[1..] {
            if v > m {
                m = v;
            }
        }
        for j in 0..c {
            row_exp[j] = row[j] - m;
        }
        crate::math_kernels::exp_f32(&row_exp, &mut data[i * c..(i + 1) * c]);
        let mut sum = 0.0f32;
        for j in 0..c {
            sum += data[i * c + j];
        }
        let inv = 1.0 / sum;
        let out_row = &mut data[i * c..(i + 1) * c];
        for v in out_row.iter_mut() {
            *v *= inv;
        }
    }
    let shape = xi.shape.clone();
    let rg = is_grad_enabled() && x.requires_grad();
    drop(xi);
    let gf = if rg {
        Some(GradFn::Softmax {
            input: x.clone(),
            fwd: data.clone(),
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// Log-softmax along last dim for 2D `(N, C)`.
pub fn log_softmax(x: &Tensor) -> Tensor {
    assert_eq!(x.ndim(), 2, "log_softmax: 2D (N,C) only");
    let xi = x.inner.borrow();
    let n = xi.shape[0];
    let c = xi.shape[1];
    let mut data = vec![0.0f32; n * c];
    let mut shifted = vec![0.0f32; c];
    let mut tmp = vec![0.0f32; c];
    for i in 0..n {
        let row = &xi.data[i * c..(i + 1) * c];
        let mut m = row[0];
        for &v in &row[1..] {
            if v > m {
                m = v;
            }
        }
        for j in 0..c {
            shifted[j] = row[j] - m;
        }
        crate::math_kernels::exp_f32(&shifted, &mut tmp);
        let mut sum = 0.0f32;
        for &e in &tmp {
            sum += e;
        }
        let log_sum = crate::math_kernels::log_scalar(sum);
        for j in 0..c {
            data[i * c + j] = row[j] - m - log_sum;
        }
    }
    let shape = xi.shape.clone();
    let rg = is_grad_enabled() && x.requires_grad();
    drop(xi);
    let gf = if rg {
        Some(GradFn::LogSoftmax {
            input: x.clone(),
            fwd: data.clone(),
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// `F.cross_entropy(logits, target)` — mean reduction; `target` is class indices.
pub fn cross_entropy(logits: &Tensor, target: &[usize]) -> Tensor {
    assert_eq!(logits.ndim(), 2, "cross_entropy: logits (N,C)");
    let xi = logits.inner.borrow();
    let n = xi.shape[0];
    let c = xi.shape[1];
    assert_eq!(target.len(), n, "cross_entropy: target length");
    for &t in target {
        assert!(t < c, "cross_entropy: class {t} >= {c}");
    }
    let mut probs = vec![0.0f32; n * c];
    let mut loss = 0.0f32;
    for i in 0..n {
        let row = &xi.data[i * c..(i + 1) * c];
        let mut m = row[0];
        for &v in row.iter().skip(1) {
            if v > m {
                m = v;
            }
        }
        let mut sum = 0.0f32;
        for j in 0..c {
            let e = (row[j] - m).exp();
            probs[i * c + j] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for j in 0..c {
            probs[i * c + j] *= inv;
        }
        let p = probs[i * c + target[i]].max(1e-12);
        loss -= p.ln();
    }
    loss /= n as f32;
    let rg = is_grad_enabled() && logits.requires_grad();
    drop(xi);
    let gf = if rg {
        Some(GradFn::CrossEntropy {
            logits: logits.clone(),
            probs,
            target: target.to_vec(),
            n,
            c,
        })
    } else {
        None
    };
    wrap(vec![loss], &[], rg, gf)
}

/// `F.dropout(x, p, train)` with seeded Bernoulli when `train`.
pub fn dropout(x: &Tensor, p: f32, train: bool, seed: u64) -> Tensor {
    assert!((0.0..1.0).contains(&p), "dropout: p must be in [0,1)");
    if !train || p == 0.0 {
        return x.clone();
    }
    let scale = 1.0 / (1.0 - p);
    let xi = x.inner.borrow();
    let n = xi.data.len();
    let rg = is_grad_enabled() && x.requires_grad();
    let mut data = vec![0.0f32; n];
    let mut state = seed;
    let xd = xi.data.as_slice();
    // Fast path: no grad → write output only.
    if !rg {
        for i in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((state >> 8) & 0xFF_FFFF) as f32 * (1.0 / ((1u64 << 24) as f32));
            data[i] = if u >= p { xd[i] * scale } else { 0.0 };
        }
        let shape = xi.shape.clone();
        drop(xi);
        return wrap(data, &shape, false, None);
    }
    let mut mask = vec![0.0f32; n];
    for i in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = ((state >> 8) & 0xFF_FFFF) as f32 * (1.0 / ((1u64 << 24) as f32));
        let m = if u >= p { scale } else { 0.0 };
        data[i] = xd[i] * m;
        mask[i] = m;
    }
    let shape = xi.shape.clone();
    drop(xi);
    let gf = Some(GradFn::Dropout {
        input: x.clone(),
        mask,
    });
    wrap(data, &shape, true, gf)
}

/// `F.tanh`
pub fn tanh(x: &Tensor) -> Tensor {
    let xi = x.inner.borrow();
    let rg = is_grad_enabled() && x.requires_grad();
    let mut data = vec![0.0f32; xi.data.len()];
    tanh_kernel(xi.data.as_slice(), data.as_mut_slice());
    let fwd = if rg { data.clone() } else { Vec::new() };
    let shape = xi.shape.clone();
    drop(xi);
    let gf = if rg {
        Some(GradFn::Tanh {
            fwd,
            input: x.clone(),
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// `tanh(x) = 2*sigmoid(2x) - 1` via the fast sigmoid kernel (chunked).
fn tanh_kernel(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    const CHUNK: usize = 512;
    let mut scaled = [0.0f32; CHUNK];
    let mut sig = [0.0f32; CHUNK];
    let mut off = 0;
    while off < x.len() {
        let n = (x.len() - off).min(CHUNK);
        for i in 0..n {
            scaled[i] = 2.0 * x[off + i];
        }
        sigmoid_kernel(&scaled[..n], &mut sig[..n]);
        for i in 0..n {
            out[off + i] = 2.0 * sig[i] - 1.0;
        }
        off += n;
    }
}

/// `F.gelu(x, approximate='tanh')`
pub fn gelu(x: &Tensor) -> Tensor {
    let xi = x.inner.borrow();
    let rg = is_grad_enabled() && x.requires_grad();
    let mut data = vec![0.0f32; xi.data.len()];
    gelu_kernel(xi.data.as_slice(), data.as_mut_slice());
    let shape = xi.shape.clone();
    drop(xi);
    let gf = if rg {
        Some(GradFn::Gelu { input: x.clone() })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

fn gelu_kernel(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    const CHUNK: usize = 512;
    let k = (2.0 / std::f32::consts::PI).sqrt();
    let c = 0.044_715f32;
    let mut u = [0.0f32; CHUNK];
    let mut th = [0.0f32; CHUNK];
    let mut off = 0;
    while off < x.len() {
        let n = (x.len() - off).min(CHUNK);
        for i in 0..n {
            let v = x[off + i];
            u[i] = k * (v + c * v * v * v);
        }
        tanh_kernel(&u[..n], &mut th[..n]);
        for i in 0..n {
            out[off + i] = 0.5 * x[off + i] * (1.0 + th[i]);
        }
        off += n;
    }
}

/// `F.silu` / Swish: `x * sigmoid(x)`.
pub fn silu(x: &Tensor) -> Tensor {
    let xi = x.inner.borrow();
    let rg = is_grad_enabled() && x.requires_grad();
    let mut data = vec![0.0f32; xi.data.len()];
    silu_kernel(xi.data.as_slice(), data.as_mut_slice());
    let fwd = if rg { data.clone() } else { Vec::new() };
    let shape = xi.shape.clone();
    drop(xi);
    let gf = if rg {
        Some(GradFn::Silu {
            input: x.clone(),
            fwd,
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

fn silu_kernel(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    sigmoid_kernel(x, out);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                mul_assign_avx2(out, x);
            }
            return;
        }
    }
    for i in 0..x.len() {
        out[i] *= x[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mul_assign_avx2(a: &mut [f32], b: &[f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        _mm256_storeu_ps(a.as_mut_ptr().add(i), _mm256_mul_ps(va, vb));
        i += 8;
    }
    while i < n {
        a[i] *= b[i];
        i += 1;
    }
}

/// `F.scaled_dot_product_attention(q,k,v)` without mask/dropout.
/// Shapes: `(B, L, D)`, `(B, S, D)`, `(B, S, D)` → `(B, L, D)`.
pub fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    scaled_dot_product_attention_masked(q, k, v, None)
}

/// `F.scaled_dot_product_attention(q,k,v,attn_mask=...)` — float additive mask.
///
/// `attn_mask` is optional and broadcastable to `(B, L, S)` (e.g. `(L, S)` causal).
pub fn scaled_dot_product_attention_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
) -> Tensor {
    use crate::ops::{add, bmm, div, full, permute, reshape};
    assert_eq!(q.ndim(), 3);
    assert_eq!(k.ndim(), 3);
    assert_eq!(v.ndim(), 3);
    let (b, l, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let s = k.shape()[1];
    assert_eq!(k.shape()[0], b);
    assert_eq!(k.shape()[2], d);
    assert_eq!(v.shape(), vec![b, s, d]);
    let scale = (d as f32).sqrt();
    let kt = permute(k, &[0, 2, 1]); // (B, D, S)
    let scores = bmm(q, &kt); // (B, L, S)
    let scores = div(&scores, &full(&[1], scale, false));
    let scores = match attn_mask {
        Some(m) => {
            assert!(
                m.ndim() == 2 || m.ndim() == 3,
                "attn_mask: 2D (L,S) or 3D (B,L,S)"
            );
            if m.ndim() == 2 {
                assert_eq!(m.shape(), vec![l, s], "attn_mask 2D shape");
            } else {
                assert_eq!(m.shape(), vec![b, l, s], "attn_mask 3D shape");
            }
            add(&scores, m)
        }
        None => scores,
    };
    let flat = reshape(&scores, &[b * l, s]);
    let attn = softmax(&flat);
    let attn = reshape(&attn, &[b, l, s]);
    bmm(&attn, v)
}
