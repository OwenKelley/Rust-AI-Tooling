//! Tensor creation and forward ops (with autograd hooks).

use std::rc::Rc;

use crate::autograd::GradFn;
use crate::broadcast::{broadcast_shapes, expand_to};
use crate::context::is_grad_enabled;
use crate::gemm::gemm_f32;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::tensor::{row_major_strides, Tensor, TensorInner, TensorStorage};

fn shape_len(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

fn wants_grad(ts: &[&Tensor]) -> bool {
    is_grad_enabled() && ts.iter().any(|t| t.requires_grad())
}

fn wrap(
    data: Vec<f32>,
    shape: &[usize],
    requires_grad: bool,
    grad_fn: Option<GradFn>,
) -> Tensor {
    Tensor::from_inner(TensorInner::new_contiguous(
        data,
        shape.to_vec(),
        Device::Cpu,
        Dtype::Float32,
        requires_grad,
        if requires_grad {
            Some(vec![0.0; shape_len(shape)])
        } else {
            None
        },
        grad_fn,
    ))
}

#[derive(Clone, Copy)]
enum BinKind {
    Add,
    Sub,
    Mul,
    Div,
}

fn bin_apply(a: &[f32], b: &[f32], out: &mut [f32], kind: BinKind) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                bin_avx2(a, b, out, kind);
            }
            return;
        }
    }
    match kind {
        BinKind::Add => {
            for i in 0..a.len() {
                out[i] = a[i] + b[i];
            }
        }
        BinKind::Sub => {
            for i in 0..a.len() {
                out[i] = a[i] - b[i];
            }
        }
        BinKind::Mul => {
            for i in 0..a.len() {
                out[i] = a[i] * b[i];
            }
        }
        BinKind::Div => {
            for i in 0..a.len() {
                out[i] = a[i] / b[i];
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bin_avx2(a: &[f32], b: &[f32], out: &mut [f32], kind: BinKind) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = match kind {
            BinKind::Add => _mm256_add_ps(va, vb),
            BinKind::Sub => _mm256_sub_ps(va, vb),
            BinKind::Mul => _mm256_mul_ps(va, vb),
            BinKind::Div => _mm256_div_ps(va, vb),
        };
        _mm256_storeu_ps(out.as_mut_ptr().add(i), vr);
        i += 8;
    }
    while i < n {
        out[i] = match kind {
            BinKind::Add => a[i] + b[i],
            BinKind::Sub => a[i] - b[i],
            BinKind::Mul => a[i] * b[i],
            BinKind::Div => a[i] / b[i],
        };
        i += 1;
    }
}

/// In-place `a[i] = a[i] op b[i]` (equal lengths).
fn bin_apply_inplace(a: &mut [f32], b: &[f32], kind: BinKind) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                bin_inplace_avx2(a, b, kind);
            }
            return;
        }
    }
    match kind {
        BinKind::Add => {
            for i in 0..a.len() {
                a[i] += b[i];
            }
        }
        BinKind::Sub => {
            for i in 0..a.len() {
                a[i] -= b[i];
            }
        }
        BinKind::Mul => {
            for i in 0..a.len() {
                a[i] *= b[i];
            }
        }
        BinKind::Div => {
            for i in 0..a.len() {
                a[i] /= b[i];
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bin_inplace_avx2(a: &mut [f32], b: &[f32], kind: BinKind) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let vr = match kind {
            BinKind::Add => _mm256_add_ps(va, vb),
            BinKind::Sub => _mm256_sub_ps(va, vb),
            BinKind::Mul => _mm256_mul_ps(va, vb),
            BinKind::Div => _mm256_div_ps(va, vb),
        };
        _mm256_storeu_ps(a.as_mut_ptr().add(i), vr);
        i += 8;
    }
    while i < n {
        match kind {
            BinKind::Add => a[i] += b[i],
            BinKind::Sub => a[i] -= b[i],
            BinKind::Mul => a[i] *= b[i],
            BinKind::Div => a[i] /= b[i],
        }
        i += 1;
    }
}

fn zip_bin(a: &Tensor, b: &Tensor, kind: BinKind, make_gf: impl FnOnce() -> GradFn) -> Tensor {
    let a = a.as_contiguous();
    let b = b.as_contiguous();
    let ai = a.inner.borrow();
    let bi = b.inner.borrow();
    let out_shape = broadcast_shapes(&ai.shape, &bi.shape);
    let mut data = vec![0.0f32; shape_len(&out_shape)];
    if ai.shape == out_shape && bi.shape == out_shape {
        bin_apply(&ai.dense_data(), &bi.dense_data(), data.as_mut_slice(), kind);
    } else if ai.shape == out_shape {
        // Only expand the smaller operand.
        if bi.shape.len() == 1
            && out_shape.len() == 2
            && bi.shape[0] == out_shape[1]
        {
            // (M, N) ⊕ (N,) — fuse without materializing broadcast of b.
            let m = out_shape[0];
            let n = out_shape[1];
            let ad = &ai.dense_data();
            let bd = &bi.dense_data();
            for i in 0..m {
                let off = i * n;
                bin_apply(&ad[off..off + n], bd, &mut data[off..off + n], kind);
            }
        } else {
            let bd = expand_to(&bi.dense_data(), &bi.shape, &out_shape);
            bin_apply(&ai.dense_data(), &bd, data.as_mut_slice(), kind);
        }
    } else if bi.shape == out_shape {
        if ai.shape.len() == 1
            && out_shape.len() == 2
            && ai.shape[0] == out_shape[1]
        {
            let m = out_shape[0];
            let n = out_shape[1];
            let ad = &ai.dense_data();
            let bd = &bi.dense_data();
            for i in 0..m {
                let off = i * n;
                bin_apply(ad, &bd[off..off + n], &mut data[off..off + n], kind);
            }
        } else {
            let ad = expand_to(&ai.dense_data(), &ai.shape, &out_shape);
            bin_apply(&ad, &bi.dense_data(), data.as_mut_slice(), kind);
        }
    } else {
        let ad = expand_to(&ai.dense_data(), &ai.shape, &out_shape);
        let bd = expand_to(&bi.dense_data(), &bi.shape, &out_shape);
        bin_apply(&ad, &bd, data.as_mut_slice(), kind);
    }
    let rg_flag = wants_grad(&[&a, &b]);
    drop((ai, bi));
    let gf = if rg_flag { Some(make_gf()) } else { None };
    wrap(data, &out_shape, rg_flag, gf)
}

/// `torch.zeros(shape)`
pub fn zeros(shape: &[usize], requires_grad: bool) -> Tensor {
    Tensor::from_vec(vec![0.0; shape_len(shape)], shape, requires_grad)
}

/// `torch.ones(shape)`
pub fn ones(shape: &[usize], requires_grad: bool) -> Tensor {
    Tensor::from_vec(vec![1.0; shape_len(shape)], shape, requires_grad)
}

/// `torch.full(shape, value)`
pub fn full(shape: &[usize], value: f32, requires_grad: bool) -> Tensor {
    Tensor::from_vec(vec![value; shape_len(shape)], shape, requires_grad)
}

/// Seeded uniform in [low, high) — shared LCG with NumPy harness (f32).
pub fn seeded_uniform(shape: &[usize], seed: u64, low: f32, high: f32) -> Tensor {
    let mut state = seed;
    let n = shape_len(shape);
    let mut data = Vec::with_capacity(n);
    let span = high - low;
    for _ in 0..n {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = ((state >> 8) & 0xFF_FFFF) as f32 / ((1u64 << 24) as f32);
        data.push(low + span * u);
    }
    Tensor::from_vec(data, shape, false)
}

/// Seeded standard normal via Box–Muller on the same LCG.
pub fn randn(shape: &[usize], seed: u64, requires_grad: bool) -> Tensor {
    let mut state = seed;
    let n = shape_len(shape);
    let mut data = Vec::with_capacity(n);
    let mut next_u = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) & 0xFF_FFFF) as f32 / ((1u64 << 24) as f32)
    };
    let mut i = 0;
    while i < n {
        let u1 = next_u().max(1e-7);
        let u2 = next_u();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        data.push(r * theta.cos());
        i += 1;
        if i < n {
            data.push(r * theta.sin());
            i += 1;
        }
    }
    Tensor::from_vec(data, shape, requires_grad)
}

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    zip_bin(a, b, BinKind::Add, || GradFn::Add(Rc::new((a.clone(), b.clone()))))
}

pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    zip_bin(a, b, BinKind::Sub, || GradFn::Sub(Rc::new((a.clone(), b.clone()))))
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    zip_bin(a, b, BinKind::Mul, || GradFn::Mul(Rc::new((a.clone(), b.clone()))))
}

pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    zip_bin(a, b, BinKind::Div, || GradFn::Div(Rc::new((a.clone(), b.clone()))))
}

pub fn neg(a: &Tensor) -> Tensor {
    let ac = a.as_contiguous();
    let ai = ac.inner.borrow();
    let ad = ai.dense_data();
    let mut data = vec![0.0f32; ai.numel()];
    for i in 0..ai.numel() {
        data[i] = -ad[i];
    }
    let shape = ai.shape.clone();
    let rg = wants_grad(&[a]);
    drop(ai);
    let gf = if rg {
        Some(GradFn::Neg { input: a.clone() })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

pub fn abs(a: &Tensor) -> Tensor {
    let ac = a.as_contiguous();
    let ai = ac.inner.borrow();
    let ad = ai.dense_data();
    let mut data = vec![0.0f32; ai.numel()];
    for i in 0..ai.numel() {
        data[i] = ad[i].abs();
    }
    let shape = ai.shape.clone();
    let rg = wants_grad(&[a]);
    drop(ai);
    let gf = if rg {
        Some(GradFn::Abs { input: a.clone() })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

pub fn exp(a: &Tensor) -> Tensor {
    let ai = a.inner.borrow();
    let mut data = vec![0.0f32; ai.numel()];
    crate::math_kernels::exp_f32(&ai.dense_data(), data.as_mut_slice());
    let shape = ai.shape.clone();
    let rg = wants_grad(&[a]);
    drop(ai);
    let gf = if rg {
        Some(GradFn::Exp {
            input: a.clone(),
            fwd: data.clone(),
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

pub fn log(a: &Tensor) -> Tensor {
    let ai = a.inner.borrow();
    let mut data = vec![0.0f32; ai.numel()];
    crate::math_kernels::log_f32(&ai.dense_data(), data.as_mut_slice());
    let shape = ai.shape.clone();
    let rg = wants_grad(&[a]);
    drop(ai);
    let gf = if rg {
        Some(GradFn::Log { input: a.clone() })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// Elementwise `a.pow(b)` with broadcasting.
pub fn pow(a: &Tensor, b: &Tensor) -> Tensor {
    let ai = a.inner.borrow();
    let bi = b.inner.borrow();
    let out_shape = broadcast_shapes(&ai.shape, &bi.shape);
    let mut data = vec![0.0f32; shape_len(&out_shape)];
    if ai.shape == out_shape && bi.shape == out_shape {
        crate::math_kernels::pow_f32(&ai.dense_data(), &bi.dense_data(), data.as_mut_slice());
    } else {
        let ad = expand_to(&ai.dense_data(), &ai.shape, &out_shape);
        let bd = expand_to(&bi.dense_data(), &bi.shape, &out_shape);
        crate::math_kernels::pow_f32(&ad, &bd, data.as_mut_slice());
    }
    let rg = wants_grad(&[a, b]);
    drop((ai, bi));
    let gf = if rg {
        Some(GradFn::Pow(Rc::new((a.clone(), b.clone()))))
    } else {
        None
    };
    wrap(data, &out_shape, rg, gf)
}

pub fn clamp(a: &Tensor, min: f32, max: f32) -> Tensor {
    assert!(min <= max, "clamp: min > max");
    let ai = a.inner.borrow();
    let data: Vec<f32> = ai.dense_data().iter().map(|&v| v.clamp(min, max)).collect();
    let shape = ai.shape.clone();
    let rg = wants_grad(&[a]);
    drop(ai);
    let gf = if rg {
        Some(GradFn::Clamp {
            input: a.clone(),
            min,
            max,
        })
    } else {
        None
    };
    wrap(data, &shape, rg, gf)
}

/// 2D matmul.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let rg = wants_grad(&[a, b]);
    let out = matmul_raw(a, b);
    if rg {
        let mut t = out.inner.borrow_mut();
        t.requires_grad = true;
        t.grad = Some(vec![0.0; t.numel()]);
        t.grad_fn = Some(GradFn::Matmul(Rc::new((a.clone(), b.clone()))));
    }
    out
}

/// Raw matmul without attaching grad_fn (used in backward).
pub fn matmul_raw(a: &Tensor, b: &Tensor) -> Tensor {
    let ai = a.inner.borrow();
    let bi = b.inner.borrow();
    assert_eq!(ai.shape.len(), 2, "matmul: 2D only");
    assert_eq!(bi.shape.len(), 2, "matmul: 2D only");
    assert_eq!(ai.shape[1], bi.shape[0], "matmul: inner dim");
    let m = ai.shape[0];
    let k = ai.shape[1];
    let n = bi.shape[1];
    let out = gemm_f32(&ai.dense_data(), &bi.dense_data(), m, k, n);
    drop((ai, bi));
    Tensor::from_vec(out, &[m, n], false)
}

/// Transpose last two dims for 2D.
pub fn transpose(a: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "transpose: 2D only in v1");
    let rg = wants_grad(&[a]);
    let out = transpose_data(a);
    if rg {
        let mut t = out.inner.borrow_mut();
        t.requires_grad = true;
        t.grad = Some(vec![0.0; t.numel()]);
        t.grad_fn = Some(GradFn::Transpose2d {
            input: a.clone(),
        });
    }
    out
}

pub fn transpose_data(a: &Tensor) -> Tensor {
    let ai = a.inner.borrow();
    assert_eq!(ai.shape.len(), 2);
    let (m, n) = (ai.shape[0], ai.shape[1]);
    // Non-F32: always materialize a contiguous typed copy (no zero-copy views).
    if !ai.storage.is_f32() {
        let out = match &ai.storage {
            TensorStorage::I64(_) => {
                let src = ai.gather_i64();
                let mut dst = vec![0i64; m * n];
                for i in 0..m {
                    for j in 0..n {
                        dst[j * m + i] = src[i * n + j];
                    }
                }
                drop(ai);
                Tensor::from_i64(dst, &[n, m])
            }
            TensorStorage::Bool(_) => {
                let src = ai.gather_bool_bytes();
                let mut dst = vec![0u8; m * n];
                for i in 0..m {
                    for j in 0..n {
                        dst[j * m + i] = src[i * n + j];
                    }
                }
                let data: Vec<bool> = dst.into_iter().map(|x| x != 0).collect();
                drop(ai);
                Tensor::from_bool(data, &[n, m])
            }
            TensorStorage::F32(_) => unreachable!(),
        };
        return out;
    }
    // F32 zero-copy view: swap shape and strides.
    let out = TensorInner {
        storage: ai.storage.clone(),
        shape: vec![n, m],
        strides: vec![ai.strides[1], ai.strides[0]],
        offset: ai.offset,
        device: ai.device,
        dtype: ai.dtype,
        requires_grad: false,
        grad: None,
        grad_fn: None,
    };
    drop(ai);
    Tensor::from_inner(out)
}

pub fn reshape(a: &Tensor, shape: &[usize]) -> Tensor {
    assert_eq!(shape_len(shape), a.numel(), "reshape: numel mismatch");
    let rg = wants_grad(&[a]);
    let base = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()
    };
    let bi = base.inner.borrow();
    let out = if !bi.storage.is_f32() {
        // Non-F32: materialize contiguous typed buffer with new shape.
        match &bi.storage {
            TensorStorage::I64(_) => {
                let data = bi.gather_i64();
                drop(bi);
                Tensor::from_i64(data, shape)
            }
            TensorStorage::Bool(_) => {
                let data: Vec<bool> = bi
                    .gather_bool_bytes()
                    .into_iter()
                    .map(|x| x != 0)
                    .collect();
                drop(bi);
                Tensor::from_bool(data, shape)
            }
            TensorStorage::F32(_) => unreachable!(),
        }
    } else {
        let out = TensorInner {
            storage: bi.storage.clone(),
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: bi.offset,
            device: bi.device,
            dtype: bi.dtype,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        };
        drop(bi);
        Tensor::from_inner(out)
    };
    if rg {
        let mut t = out.inner.borrow_mut();
        t.requires_grad = true;
        t.grad = Some(vec![0.0; t.numel()]);
        t.grad_fn = Some(GradFn::Reshape {
            input: a.clone(),
        });
    }
    out
}

pub fn sum(a: &Tensor) -> Tensor {
    let (s, rg, numel) = {
        let ai = a.inner.borrow();
        let s: f32 = ai.dense_data().iter().sum();
        (s, wants_grad(&[a]), ai.numel())
    };
    let gf = if rg {
        Some(GradFn::Sum {
            input: a.clone(),
            numel,
        })
    } else {
        None
    };
    wrap(vec![s], &[], rg, gf)
}

pub fn mean(a: &Tensor) -> Tensor {
    let (s, rg, n) = {
        let ai = a.inner.borrow();
        let n = ai.numel();
        let s = ai.dense_data().iter().sum::<f32>() / n as f32;
        (s, wants_grad(&[a]), n)
    };
    let gf = if rg {
        Some(GradFn::Mean {
            input: a.clone(),
            numel: n,
        })
    } else {
        None
    };
    wrap(vec![s], &[], rg, gf)
}

/// `torch.cat(tensors, dim)`
pub fn cat(tensors: &[&Tensor], dim: usize) -> Tensor {
    assert!(!tensors.is_empty(), "cat: empty");
    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape()).collect();
    let ndim = shapes[0].len();
    assert!(dim < ndim, "cat: dim out of range");
    for s in &shapes {
        assert_eq!(s.len(), ndim, "cat: ndim mismatch");
        for (i, (&a, &b)) in shapes[0].iter().zip(s.iter()).enumerate() {
            if i != dim {
                assert_eq!(a, b, "cat: shape mismatch on non-cat dim");
            }
        }
    }
    let mut out_shape = shapes[0].clone();
    out_shape[dim] = shapes.iter().map(|s| s[dim]).sum();

    let outer: usize = out_shape[..dim].iter().product();
    let inner: usize = out_shape[dim + 1..].iter().product();
    let out_n = shape_len(&out_shape);
    let mut data = vec![0.0f32; out_n];

    for o in 0..outer {
        let mut col = 0usize;
        for t in tensors {
            let ti = t.inner.borrow();
            let dlen = ti.shape[dim];
            let chunk = dlen * inner;
            let src_off = o * chunk;
            let dst_off = (o * out_shape[dim] + col) * inner;
            data[dst_off..dst_off + chunk]
                .copy_from_slice(&ti.dense_data()[src_off..src_off + chunk]);
            col += dlen;
        }
    }

    let rg = wants_grad(tensors);
    let sizes: Vec<usize> = shapes.iter().map(|s| s[dim]).collect();
    let inputs: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let gf = if rg {
        Some(GradFn::Cat {
            inputs,
            dim,
            sizes,
        })
    } else {
        None
    };
    wrap(data, &out_shape, rg, gf)
}

/// `torch.stack(tensors, dim)` — inserts a new axis of length `len(tensors)`.
pub fn stack(tensors: &[&Tensor], dim: usize) -> Tensor {
    assert!(!tensors.is_empty(), "stack: empty");
    let base = tensors[0].shape();
    for t in tensors {
        assert_eq!(t.shape(), base, "stack: shape mismatch");
    }
    assert!(dim <= base.len(), "stack: dim out of range");
    let nstack = tensors.len();
    let mut out_shape = Vec::with_capacity(base.len() + 1);
    out_shape.extend_from_slice(&base[..dim]);
    out_shape.push(nstack);
    out_shape.extend_from_slice(&base[dim..]);

    let outer: usize = if dim == 0 {
        1
    } else {
        base[..dim].iter().product()
    };
    let inner: usize = if dim == base.len() {
        1
    } else {
        base[dim..].iter().product()
    };
    let out_n = shape_len(&out_shape);
    let mut data = Vec::with_capacity(out_n);
    unsafe {
        data.set_len(out_n);
    }
    for o in 0..outer {
        for (s, t) in tensors.iter().enumerate() {
            let src = t.inner.borrow();
            let src_off = o * inner;
            let dst_off = (o * nstack + s) * inner;
            data[dst_off..dst_off + inner].copy_from_slice(&src.dense_data()[src_off..src_off + inner]);
        }
    }
    let rg = wants_grad(tensors);
    let inputs: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
    let gf = if rg {
        Some(GradFn::Stack { inputs, dim })
    } else {
        None
    };
    wrap(data, &out_shape, rg, gf)
}

/// `torch.index_select(input, dim, index)` with `usize` indices.
pub fn index_select(input: &Tensor, dim: usize, indices: &[usize]) -> Tensor {
    let shape = input.shape();
    assert!(dim < shape.len(), "index_select: dim");
    let dim_size = shape[dim];
    for &i in indices {
        assert!(i < dim_size, "index_select: index {i} >= {dim_size}");
    }
    let mut out_shape = shape.clone();
    out_shape[dim] = indices.len();
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let mut data = vec![0.0f32; shape_len(&out_shape)];
    let src = input.inner.borrow();
    let nidx = indices.len();
    for o in 0..outer {
        for (new_k, &old_k) in indices.iter().enumerate() {
            let s = (o * dim_size + old_k) * inner;
            let d = (o * nidx + new_k) * inner;
            data[d..d + inner].copy_from_slice(&src.dense_data()[s..s + inner]);
        }
    }
    drop(src);
    let rg = wants_grad(&[input]);
    let gf = if rg {
        Some(GradFn::IndexSelect {
            input: input.clone(),
            dim,
            indices: indices.to_vec(),
            input_dim_size: dim_size,
        })
    } else {
        None
    };
    wrap(data, &out_shape, rg, gf)
}

/// `torch.chunk(input, chunks, dim)` — equal-sized chunks along `dim`.
pub fn chunk(input: &Tensor, chunks: usize, dim: usize) -> Vec<Tensor> {
    assert!(chunks > 0);
    let shape = input.shape();
    assert!(dim < shape.len());
    assert_eq!(
        shape[dim] % chunks,
        0,
        "chunk: dim size must divide evenly"
    );
    let length = shape[dim] / chunks;
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let src = input.inner.borrow();
    let rg = wants_grad(&[input]);
    let mut outs = Vec::with_capacity(chunks);
    for c in 0..chunks {
        let start = c * length;
        let mut out_shape = shape.clone();
        out_shape[dim] = length;
        let mut data = vec![0.0f32; shape_len(&out_shape)];
        for o in 0..outer {
            for k in 0..length {
                for j in 0..inner {
                    let s = (o * shape[dim] + start + k) * inner + j;
                    let d = (o * length + k) * inner + j;
                    data[d] = src.dense_data()[s];
                }
            }
        }
        let gf = if rg {
            Some(GradFn::Chunk {
                input: input.clone(),
                dim,
                start,
                length,
            })
        } else {
            None
        };
        outs.push(wrap(data, &out_shape, rg, gf));
    }
    outs
}

/// Batched matmul: `(B,M,K) @ (B,K,N) -> (B,M,N)`.
pub fn bmm(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 3, "bmm: 3D a");
    assert_eq!(b.ndim(), 3, "bmm: 3D b");
    let ash = a.shape();
    let bsh = b.shape();
    assert_eq!(ash[0], bsh[0], "bmm: batch");
    assert_eq!(ash[2], bsh[1], "bmm: inner");
    let (batch, m, k) = (ash[0], ash[1], ash[2]);
    let n = bsh[2];
    let ai = a.inner.borrow();
    let bi = b.inner.borrow();
    let mut data = vec![0.0f32; batch * m * n];
    for bi_i in 0..batch {
        let a_off = bi_i * m * k;
        let b_off = bi_i * k * n;
        let o_off = bi_i * m * n;
        let block = gemm_f32(
            &ai.dense_data()[a_off..a_off + m * k],
            &bi.dense_data()[b_off..b_off + k * n],
            m,
            k,
            n,
        );
        data[o_off..o_off + m * n].copy_from_slice(&block);
    }
    drop((ai, bi));
    let rg = wants_grad(&[a, b]);
    let gf = if rg {
        Some(GradFn::Bmm(Rc::new((a.clone(), b.clone()))))
    } else {
        None
    };
    wrap(data, &[batch, m, n], rg, gf)
}

/// `torch.permute(input, dims)`.
pub fn permute(input: &Tensor, dims: &[usize]) -> Tensor {
    let shape = input.shape();
    assert_eq!(dims.len(), shape.len());
    let mut seen = vec![false; dims.len()];
    for &d in dims {
        assert!(d < shape.len());
        assert!(!seen[d], "permute: duplicate dim");
        seen[d] = true;
    }
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let src = input.inner.borrow();
    let data = crate::autograd::permute_data(&src.dense_data(), &shape, dims);
    drop(src);
    let rg = wants_grad(&[input]);
    let gf = if rg {
        Some(GradFn::Permute {
            input: input.clone(),
            dims: dims.to_vec(),
        })
    } else {
        None
    };
    wrap(data, &out_shape, rg, gf)
}

fn assert_inplace_leaf(t: &Tensor) {
    assert!(
        t.inner.borrow().grad_fn.is_none(),
        "in-place ops require a leaf tensor (no grad_fn)"
    );
}

/// `tensor.add_(other)` — same-shape only.
pub fn add_(a: &Tensor, b: &Tensor) {
    assert_inplace_leaf(a);
    if Rc::ptr_eq(&a.inner, &b.inner) {
        let mut ai = a.inner.borrow_mut();
        let n = ai.numel();
        for i in 0..n {
            ai.data_mut_dense()[i] *= 2.0;
        }
        return;
    }
    let bi = b.inner.borrow();
    let mut ai = a.inner.borrow_mut();
    assert_eq!(ai.shape, bi.shape, "add_: same shape only");
    bin_apply_inplace(&mut *ai.data_mut_dense(), &bi.dense_data(), BinKind::Add);
}

/// `tensor.sub_(other)` — same-shape only.
pub fn sub_(a: &Tensor, b: &Tensor) {
    assert_inplace_leaf(a);
    assert!(!Rc::ptr_eq(&a.inner, &b.inner), "sub_: self alias not supported");
    let bi = b.inner.borrow();
    let mut ai = a.inner.borrow_mut();
    assert_eq!(ai.shape, bi.shape, "sub_: same shape only");
    bin_apply_inplace(&mut *ai.data_mut_dense(), &bi.dense_data(), BinKind::Sub);
}

/// `tensor.mul_(other)` — same-shape only.
pub fn mul_(a: &Tensor, b: &Tensor) {
    assert_inplace_leaf(a);
    if Rc::ptr_eq(&a.inner, &b.inner) {
        let mut ai = a.inner.borrow_mut();
        for v in ai.data_mut_dense().iter_mut() {
            *v *= *v;
        }
        return;
    }
    let bi = b.inner.borrow();
    let mut ai = a.inner.borrow_mut();
    assert_eq!(ai.shape, bi.shape, "mul_: same shape only");
    bin_apply_inplace(&mut *ai.data_mut_dense(), &bi.dense_data(), BinKind::Mul);
}

/// `tensor.relu_()`
pub fn relu_(a: &Tensor) {
    assert_inplace_leaf(a);
    let mut ai = a.inner.borrow_mut();
    relu_inplace_kernel(&mut *ai.data_mut_dense());
}

fn relu_inplace_kernel(x: &mut [f32]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                relu_inplace_avx2(x);
            }
            return;
        }
    }
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_inplace_avx2(x: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let n = x.len();
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_max_ps(v, zero));
        i += 8;
    }
    while i < n {
        if x[i] < 0.0 {
            x[i] = 0.0;
        }
        i += 1;
    }
}

/// `tensor.zero_()`
pub fn zero_(a: &Tensor) {
    fill_(a, 0.0);
}

/// `tensor.fill_(value)`
pub fn fill_(a: &Tensor, value: f32) {
    assert_inplace_leaf(a);
    let mut ai = a.inner.borrow_mut();
    for v in ai.data_mut_dense().iter_mut() {
        *v = value;
    }
}

pub fn narrow(input: &Tensor, dim: usize, start: usize, length: usize) -> Tensor {
    let shape = input.shape();
    assert!(dim < shape.len(), "narrow: dim out of range");
    assert!(start + length <= shape[dim], "narrow: range out of bounds");
    let ii = input.inner.borrow();
    let mut out_shape = shape.clone();
    out_shape[dim] = length;
    let new_offset = (ii.offset as isize + start as isize * ii.strides[dim]) as usize;
    let out = if !ii.storage.is_f32() {
        // Non-F32: gather the narrowed view into a contiguous typed copy.
        let view = TensorInner {
            storage: ii.storage.clone(),
            shape: out_shape.clone(),
            strides: ii.strides.clone(),
            offset: new_offset,
            device: ii.device,
            dtype: ii.dtype,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        };
        let out = match &view.storage {
            TensorStorage::I64(_) => Tensor::from_i64(view.gather_i64(), &out_shape),
            TensorStorage::Bool(_) => {
                let data: Vec<bool> = view
                    .gather_bool_bytes()
                    .into_iter()
                    .map(|x| x != 0)
                    .collect();
                Tensor::from_bool(data, &out_shape)
            }
            TensorStorage::F32(_) => unreachable!(),
        };
        drop(ii);
        out
    } else {
        let out = TensorInner {
            storage: ii.storage.clone(),
            shape: out_shape,
            strides: ii.strides.clone(),
            offset: new_offset,
            device: ii.device,
            dtype: ii.dtype,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        };
        drop(ii);
        Tensor::from_inner(out)
    };
    let rg = wants_grad(&[input]);
    if rg {
        let mut t = out.inner.borrow_mut();
        t.requires_grad = true;
        t.grad = Some(vec![0.0; t.numel()]);
        t.grad_fn = Some(GradFn::Chunk {
            input: input.clone(),
            dim,
            start,
            length,
        });
    }
    out
}

/// `torch.select(input, dim, index)` — owned copy with `dim` removed.
pub fn select(input: &Tensor, dim: usize, index: usize) -> Tensor {
    let t = narrow(input, dim, index, 1);
    let mut shape = t.shape();
    shape.remove(dim);
    match t.dtype() {
        Dtype::Int64 => Tensor::from_i64(t.i64_data(), &shape),
        Dtype::Bool => Tensor::from_bool(t.bool_data(), &shape),
        d => Tensor::from_vec_dtype(t.data(), &shape, false, d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::no_grad;
    use crate::nn::{Linear, Module, ReLU, MSELoss};
    use crate::optim::SGD;

    #[test]
    fn inplace_add_relu() {
        let a = seeded_uniform(&[2, 3], 1, -1.0, 1.0);
        let b = seeded_uniform(&[2, 3], 2, -0.5, 0.5);
        let expected = add(&a, &b);
        let t = Tensor::from_vec(a.data(), &a.shape(), false);
        add_(&t, &b);
        assert!((t.checksum() - expected.checksum()).abs() < 1e-5);
        relu_(&t);
        for &v in &t.data() {
            assert!(v >= 0.0);
        }
    }

    #[test]
    fn narrow_select_shape() {
        let a = seeded_uniform(&[4, 5, 3], 3, -1.0, 1.0);
        let n = narrow(&a, 1, 1, 2);
        assert_eq!(n.shape(), vec![4, 2, 3]);
        let s = select(&a, 0, 2);
        assert_eq!(s.shape(), vec![5, 3]);
    }

    #[test]
    fn train_mlp_loss_decreases() {
        let x = seeded_uniform(&[32, 4], 1, -1.0, 1.0);
        let y = seeded_uniform(&[32, 1], 2, -1.0, 1.0);
        let l1 = Linear::new(4, 8, true, 10);
        let l2 = Linear::new(8, 1, true, 20);
        let relu = ReLU;
        let loss_fn = MSELoss;
        let mut params = l1.parameters();
        params.extend(l2.parameters());
        let opt = SGD::new(params, 0.05);
        let mut last = f32::INFINITY;
        for _ in 0..20 {
            opt.zero_grad();
            let h = relu.forward(&l1.forward(&x));
            let pred = l2.forward(&h);
            let loss = loss_fn.forward(&pred, &y);
            loss.backward();
            opt.step();
            let v = loss.item();
            assert!(v.is_finite());
            last = v;
        }
        assert!(last < 2.0, "loss should drop, got {last}");
    }

    #[test]
    fn broadcast_add_works() {
        let a = seeded_uniform(&[4, 3], 1, -1.0, 1.0);
        let b = seeded_uniform(&[3], 2, -1.0, 1.0);
        let c = add(&a, &b);
        assert_eq!(c.shape(), vec![4, 3]);
    }

    #[test]
    fn no_grad_skips_graph() {
        let a = seeded_uniform(&[2, 2], 1, -1.0, 1.0);
        a.set_requires_grad(true);
        let out = no_grad(|| add(&a, &a));
        assert!(!out.requires_grad());
        assert!(out.inner.borrow().grad_fn.is_none());
    }

    #[test]
    fn cat_stack_index() {
        let a = seeded_uniform(&[2, 3], 1, -1.0, 1.0);
        let b = seeded_uniform(&[2, 3], 2, -1.0, 1.0);
        let c = cat(&[&a, &b], 0);
        assert_eq!(c.shape(), vec![4, 3]);
        let s = stack(&[&a, &b], 0);
        assert_eq!(s.shape(), vec![2, 2, 3]);
        let ix = index_select(&a, 1, &[0, 2]);
        assert_eq!(ix.shape(), vec![2, 2]);
    }

    #[test]
    fn cross_entropy_adam_trains() {
        use crate::nn::CrossEntropyLoss;
        use crate::optim::Adam;
        let x = seeded_uniform(&[16, 4], 1, -1.0, 1.0);
        let target: Vec<usize> = (0..16).map(|i| i % 3).collect();
        let l1 = Linear::new(4, 8, true, 10);
        let l2 = Linear::new(8, 3, true, 20);
        let relu = ReLU;
        let mut params = l1.parameters();
        params.extend(l2.parameters());
        let mut opt = Adam::new(params, 0.05);
        let loss_fn = CrossEntropyLoss;
        let mut last = f32::INFINITY;
        for _ in 0..30 {
            opt.zero_grad();
            let h = relu.forward(&l1.forward(&x));
            let logits = l2.forward(&h);
            let loss = loss_fn.forward(&logits, &target);
            loss.backward();
            opt.step();
            last = loss.item();
        }
        assert!(last < 1.2, "CE should drop, got {last}");
    }

    #[test]
    fn embedding_layernorm_conv_adamw() {
        use crate::nn::{Conv2d, Embedding, LayerNorm};
        use crate::optim::{AdamW, StepLR};
        let emb = Embedding::from_params(seeded_uniform(&[10, 4], 1, -0.5, 0.5));
        let y = emb.forward_indices(&[0, 3, 9, 3]);
        assert_eq!(y.shape(), vec![4, 4]);
        let ln = LayerNorm::from_params(
            seeded_uniform(&[4], 2, 0.5, 1.5),
            seeded_uniform(&[4], 3, -0.1, 0.1),
            1e-5,
        );
        let z = ln.forward(&y);
        assert_eq!(z.shape(), vec![4, 4]);
        let conv = Conv2d::from_params(
            seeded_uniform(&[2, 1, 3, 3], 4, -0.2, 0.2),
            Some(seeded_uniform(&[2], 5, -0.1, 0.1)),
        );
        let x = seeded_uniform(&[1, 1, 5, 5], 6, -1.0, 1.0);
        let out = conv.forward(&x);
        assert_eq!(out.shape(), vec![1, 2, 3, 3]);
        let loss = mean(&out);
        loss.backward();
        let mut opt = AdamW::new(conv.parameters(), 0.01, 0.01);
        opt.step();
        let mut lr = 0.1f32;
        let mut sched = StepLR::new(&mut lr, 2, 0.5);
        for _ in 0..5 {
            sched.step();
        }
        assert!((lr - 0.025).abs() < 1e-6);
    }
}
