//! Tensor creation and forward ops (with autograd hooks).

use std::rc::Rc;

use crate::autograd::GradFn;
use crate::broadcast::{broadcast_shapes, expand_to};
use crate::context::is_grad_enabled;
use crate::gemm::gemm_f32;
use crate::tensor::{Tensor, TensorInner};

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
    Tensor::from_inner(TensorInner {
        data,
        shape: shape.to_vec(),
        requires_grad,
        grad: if requires_grad {
            Some(vec![0.0; shape_len(shape)])
        } else {
            None
        },
        grad_fn,
    })
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

fn zip_bin(a: &Tensor, b: &Tensor, kind: BinKind, make_gf: impl FnOnce() -> GradFn) -> Tensor {
    let ai = a.inner.borrow();
    let bi = b.inner.borrow();
    let out_shape = broadcast_shapes(&ai.shape, &bi.shape);
    let mut data = vec![0.0f32; shape_len(&out_shape)];
    if ai.shape == out_shape && bi.shape == out_shape {
        bin_apply(ai.data.as_slice(), bi.data.as_slice(), data.as_mut_slice(), kind);
    } else {
        let ad = expand_to(&ai.data, &ai.shape, &out_shape);
        let bd = expand_to(&bi.data, &bi.shape, &out_shape);
        bin_apply(&ad, &bd, data.as_mut_slice(), kind);
    }
    let rg_flag = wants_grad(&[a, b]);
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
    let ai = a.inner.borrow();
    let mut data = vec![0.0f32; ai.data.len()];
    for i in 0..ai.data.len() {
        data[i] = -ai.data[i];
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
    let ai = a.inner.borrow();
    let mut data = vec![0.0f32; ai.data.len()];
    for i in 0..ai.data.len() {
        data[i] = ai.data[i].abs();
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
    let mut data = vec![0.0f32; ai.data.len()];
    crate::math_kernels::exp_f32(ai.data.as_slice(), data.as_mut_slice());
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
    let mut data = vec![0.0f32; ai.data.len()];
    crate::math_kernels::log_f32(ai.data.as_slice(), data.as_mut_slice());
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
        crate::math_kernels::pow_f32(ai.data.as_slice(), bi.data.as_slice(), data.as_mut_slice());
    } else {
        let ad = expand_to(&ai.data, &ai.shape, &out_shape);
        let bd = expand_to(&bi.data, &bi.shape, &out_shape);
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
    let data: Vec<f32> = ai.data.iter().map(|&v| v.clamp(min, max)).collect();
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
    let out = gemm_f32(&ai.data, &bi.data, m, k, n);
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
    let ad = ai.data.as_slice();
    let mut out = vec![0.0f32; m * n];
    const TS: usize = 32;
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + TS).min(m);
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + TS).min(n);
            for i in i0..i1 {
                for j in j0..j1 {
                    out[j * m + i] = ad[i * n + j];
                }
            }
            j0 = j1;
        }
        i0 = i1;
    }
    drop(ai);
    Tensor::from_vec(out, &[n, m], false)
}

pub fn reshape(a: &Tensor, shape: &[usize]) -> Tensor {
    assert_eq!(shape_len(shape), a.numel(), "reshape: numel mismatch");
    let rg = wants_grad(&[a]);
    let data = a.inner.borrow().data.clone();
    let gf = if rg {
        Some(GradFn::Reshape {
            input: a.clone(),
        })
    } else {
        None
    };
    wrap(data, shape, rg, gf)
}

pub fn sum(a: &Tensor) -> Tensor {
    let (s, rg, numel) = {
        let ai = a.inner.borrow();
        let s: f32 = ai.data.iter().sum();
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
        let s = ai.data.iter().sum::<f32>() / n as f32;
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
            for k in 0..dlen {
                for j in 0..inner {
                    let src = (o * dlen + k) * inner + j;
                    let dst = (o * out_shape[dim] + col + k) * inner + j;
                    data[dst] = ti.data[src];
                }
            }
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
    let mut data = vec![0.0f32; shape_len(&out_shape)];
    for o in 0..outer {
        for (s, t) in tensors.iter().enumerate() {
            let src = t.inner.borrow();
            let src_off = o * inner;
            let dst_off = (o * nstack + s) * inner;
            data[dst_off..dst_off + inner].copy_from_slice(&src.data[src_off..src_off + inner]);
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
    for o in 0..outer {
        for (new_k, &old_k) in indices.iter().enumerate() {
            for j in 0..inner {
                let s = (o * dim_size + old_k) * inner + j;
                let d = (o * indices.len() + new_k) * inner + j;
                data[d] = src.data[s];
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::no_grad;
    use crate::nn::{Linear, Module, ReLU, MSELoss};
    use crate::optim::SGD;

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
