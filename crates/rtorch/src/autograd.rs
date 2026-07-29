//! Reverse-mode autograd tape nodes.

use std::rc::Rc;

use crate::broadcast::{broadcast_shapes, expand_to, reduce_sum_to};
use crate::ops::{matmul_raw, transpose_data};
use crate::tensor::Tensor;

/// Backward function attached to a tensor produced by an op.
#[derive(Clone)]
pub enum GradFn {
    Add(Rc<(Tensor, Tensor)>),
    Sub(Rc<(Tensor, Tensor)>),
    Mul(Rc<(Tensor, Tensor)>),
    Div(Rc<(Tensor, Tensor)>),
    Matmul(Rc<(Tensor, Tensor)>),
    Sum {
        input: Tensor,
        numel: usize,
    },
    Mean {
        input: Tensor,
        numel: usize,
    },
    Relu {
        input: Tensor,
        mask: Vec<bool>,
    },
    Sigmoid {
        input: Tensor,
        fwd: Vec<f32>,
    },
    Exp {
        input: Tensor,
        fwd: Vec<f32>,
    },
    Log {
        input: Tensor,
    },
    Pow(Rc<(Tensor, Tensor)>),
    Neg {
        input: Tensor,
    },
    Abs {
        input: Tensor,
    },
    Clamp {
        input: Tensor,
        min: f32,
        max: f32,
    },
    Reshape {
        input: Tensor,
    },
    Transpose2d {
        input: Tensor,
    },
    Linear {
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Cat {
        inputs: Vec<Tensor>,
        dim: usize,
        sizes: Vec<usize>,
    },
    IndexSelect {
        input: Tensor,
        dim: usize,
        indices: Vec<usize>,
        input_dim_size: usize,
    },
    Softmax {
        input: Tensor,
        fwd: Vec<f32>,
    },
    LogSoftmax {
        input: Tensor,
        fwd: Vec<f32>,
    },
    CrossEntropy {
        logits: Tensor,
        probs: Vec<f32>,
        target: Vec<usize>,
        n: usize,
        c: usize,
    },
    Dropout {
        input: Tensor,
        mask: Vec<f32>,
    },
    Stack {
        inputs: Vec<Tensor>,
        dim: usize,
    },
    Embedding {
        weight: Tensor,
        indices: Vec<usize>,
    },
    LayerNorm {
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        mean: Vec<f32>,
        rstd: Vec<f32>,
        eps: f32,
    },
    Conv2d {
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
    },
    Tanh {
        fwd: Vec<f32>,
        input: Tensor,
    },
    Gelu {
        input: Tensor,
    },
    BatchNorm1d {
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        mean: Vec<f32>,
        rstd: Vec<f32>,
    },
    MaxPool2d {
        input: Tensor,
        indices: Vec<usize>,
        kernel_size: usize,
        stride: usize,
    },
}

impl std::fmt::Debug for GradFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            GradFn::Add(_) => "Add",
            GradFn::Sub(_) => "Sub",
            GradFn::Mul(_) => "Mul",
            GradFn::Div(_) => "Div",
            GradFn::Matmul(_) => "Matmul",
            GradFn::Sum { .. } => "Sum",
            GradFn::Mean { .. } => "Mean",
            GradFn::Relu { .. } => "Relu",
            GradFn::Sigmoid { .. } => "Sigmoid",
            GradFn::Exp { .. } => "Exp",
            GradFn::Log { .. } => "Log",
            GradFn::Pow(_) => "Pow",
            GradFn::Neg { .. } => "Neg",
            GradFn::Abs { .. } => "Abs",
            GradFn::Clamp { .. } => "Clamp",
            GradFn::Reshape { .. } => "Reshape",
            GradFn::Transpose2d { .. } => "Transpose2d",
            GradFn::Linear { .. } => "Linear",
            GradFn::Cat { .. } => "Cat",
            GradFn::IndexSelect { .. } => "IndexSelect",
            GradFn::Softmax { .. } => "Softmax",
            GradFn::LogSoftmax { .. } => "LogSoftmax",
            GradFn::CrossEntropy { .. } => "CrossEntropy",
            GradFn::Dropout { .. } => "Dropout",
            GradFn::Stack { .. } => "Stack",
            GradFn::Embedding { .. } => "Embedding",
            GradFn::LayerNorm { .. } => "LayerNorm",
            GradFn::Conv2d { .. } => "Conv2d",
            GradFn::Tanh { .. } => "Tanh",
            GradFn::Gelu { .. } => "Gelu",
            GradFn::BatchNorm1d { .. } => "BatchNorm1d",
            GradFn::MaxPool2d { .. } => "MaxPool2d",
        };
        write!(f, "GradFn::{name}")
    }
}

/// Build topological order of tensors reachable from `root` via grad_fn.
pub fn topological_sort(root: &Tensor) -> Vec<Tensor> {
    let mut visited = std::collections::HashSet::new();
    let mut order = Vec::new();
    visit(root, &mut visited, &mut order);
    order
}

fn visit(
    t: &Tensor,
    visited: &mut std::collections::HashSet<usize>,
    order: &mut Vec<Tensor>,
) {
    let ptr = std::rc::Rc::as_ptr(&t.inner) as usize;
    if !visited.insert(ptr) {
        return;
    }
    let grad_fn = t.inner.borrow().grad_fn.clone();
    if let Some(gf) = grad_fn {
        match &gf {
            GradFn::Add(ab)
            | GradFn::Sub(ab)
            | GradFn::Mul(ab)
            | GradFn::Div(ab)
            | GradFn::Matmul(ab)
            | GradFn::Pow(ab) => {
                visit(&ab.0, visited, order);
                visit(&ab.1, visited, order);
            }
            GradFn::Sum { input, .. }
            | GradFn::Mean { input, .. }
            | GradFn::Relu { input, .. }
            | GradFn::Reshape { input }
            | GradFn::Transpose2d { input }
            | GradFn::Sigmoid { input, .. }
            | GradFn::Exp { input, .. }
            | GradFn::Log { input }
            | GradFn::Neg { input }
            | GradFn::Abs { input }
            | GradFn::Clamp { input, .. }
            | GradFn::IndexSelect { input, .. }
            | GradFn::Softmax { input, .. }
            | GradFn::LogSoftmax { input, .. }
            | GradFn::Dropout { input, .. } => {
                visit(input, visited, order);
            }
            GradFn::CrossEntropy { logits, .. } => {
                visit(logits, visited, order);
            }
            GradFn::Linear {
                input,
                weight,
                bias,
            } => {
                visit(input, visited, order);
                visit(weight, visited, order);
                if let Some(b) = bias {
                    visit(b, visited, order);
                }
            }
            GradFn::Cat { inputs, .. } | GradFn::Stack { inputs, .. } => {
                for inp in inputs {
                    visit(inp, visited, order);
                }
            }
            GradFn::Embedding { weight, .. } => {
                visit(weight, visited, order);
            }
            GradFn::LayerNorm {
                input,
                weight,
                bias,
                ..
            } => {
                visit(input, visited, order);
                visit(weight, visited, order);
                visit(bias, visited, order);
            }
            GradFn::Conv2d {
                input,
                weight,
                bias,
            } => {
                visit(input, visited, order);
                visit(weight, visited, order);
                if let Some(b) = bias {
                    visit(b, visited, order);
                }
            }
            GradFn::Tanh { input, .. } | GradFn::Gelu { input } | GradFn::MaxPool2d { input, .. } => {
                visit(input, visited, order);
            }
            GradFn::BatchNorm1d {
                input,
                weight,
                bias,
                ..
            } => {
                visit(input, visited, order);
                visit(weight, visited, order);
                visit(bias, visited, order);
            }
        }
    }
    order.push(t.clone());
}

fn matmul_out_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(a[1], b[0], "matmul: inner dims");
    vec![a[0], b[1]]
}

fn accumulate_reduced(target: &Tensor, gy: &[f32], out_shape: &[usize]) {
    let tshape = target.shape();
    let g = if tshape.as_slice() == out_shape {
        gy.to_vec()
    } else {
        reduce_sum_to(gy, out_shape, &tshape)
    };
    target.inner.borrow_mut().accumulate_grad(&g);
}

/// Apply local backward for one node given upstream gradient `gy`.
pub fn apply_backward(gf: &GradFn, gy: &[f32]) {
    match gf {
        GradFn::Add(ab) => {
            let out_shape = broadcast_shapes(&ab.0.shape(), &ab.1.shape());
            accumulate_reduced(&ab.0, gy, &out_shape);
            accumulate_reduced(&ab.1, gy, &out_shape);
        }
        GradFn::Sub(ab) => {
            let out_shape = broadcast_shapes(&ab.0.shape(), &ab.1.shape());
            accumulate_reduced(&ab.0, gy, &out_shape);
            let neg: Vec<f32> = gy.iter().map(|v| -v).collect();
            accumulate_reduced(&ab.1, &neg, &out_shape);
        }
        GradFn::Mul(ab) => {
            let a_shape = ab.0.shape();
            let b_shape = ab.1.shape();
            let out_shape = broadcast_shapes(&a_shape, &b_shape);
            let (ga, gb) = {
                let a = ab.0.inner.borrow();
                let b = ab.1.inner.borrow();
                let ae = expand_to(&a.data, &a_shape, &out_shape);
                let be = expand_to(&b.data, &b_shape, &out_shape);
                let ga: Vec<f32> = gy.iter().zip(be.iter()).map(|(g, bv)| g * bv).collect();
                let gb: Vec<f32> = gy.iter().zip(ae.iter()).map(|(g, av)| g * av).collect();
                (ga, gb)
            };
            accumulate_reduced(&ab.0, &ga, &out_shape);
            accumulate_reduced(&ab.1, &gb, &out_shape);
        }
        GradFn::Div(ab) => {
            let a_shape = ab.0.shape();
            let b_shape = ab.1.shape();
            let out_shape = broadcast_shapes(&a_shape, &b_shape);
            let (ga, gb) = {
                let a = ab.0.inner.borrow();
                let b = ab.1.inner.borrow();
                let ae = expand_to(&a.data, &a_shape, &out_shape);
                let be = expand_to(&b.data, &b_shape, &out_shape);
                let ga: Vec<f32> = gy.iter().zip(be.iter()).map(|(g, bv)| g / bv).collect();
                let gb: Vec<f32> = gy
                    .iter()
                    .zip(ae.iter())
                    .zip(be.iter())
                    .map(|((g, av), bv)| -g * av / (bv * bv))
                    .collect();
                (ga, gb)
            };
            accumulate_reduced(&ab.0, &ga, &out_shape);
            accumulate_reduced(&ab.1, &gb, &out_shape);
        }
        GradFn::Pow(ab) => {
            // y = a^b; da = b * a^(b-1) * gy; db = a^b * ln(a) * gy
            let a_shape = ab.0.shape();
            let b_shape = ab.1.shape();
            let out_shape = broadcast_shapes(&a_shape, &b_shape);
            let (ga, gb) = {
                let a = ab.0.inner.borrow();
                let b = ab.1.inner.borrow();
                let ae = expand_to(&a.data, &a_shape, &out_shape);
                let be = expand_to(&b.data, &b_shape, &out_shape);
                let mut ga = vec![0.0f32; gy.len()];
                let mut gb = vec![0.0f32; gy.len()];
                for i in 0..gy.len() {
                    let av = ae[i];
                    let bv = be[i];
                    ga[i] = gy[i] * bv * av.powf(bv - 1.0);
                    gb[i] = gy[i] * av.powf(bv) * av.ln();
                }
                (ga, gb)
            };
            accumulate_reduced(&ab.0, &ga, &out_shape);
            accumulate_reduced(&ab.1, &gb, &out_shape);
        }
        GradFn::Matmul(ab) => {
            let a = &ab.0;
            let b = &ab.1;
            let c_shape = matmul_out_shape(&a.shape(), &b.shape());
            let gy_t = Tensor::from_vec(gy.to_vec(), &c_shape, false);
            let bt = transpose_data(b);
            let at = transpose_data(a);
            let ga = matmul_raw(&gy_t, &bt);
            let gb = matmul_raw(&at, &gy_t);
            a.accumulate_from(&ga);
            b.accumulate_from(&gb);
        }
        GradFn::Sum { input, numel } => {
            let g = vec![gy[0]; *numel];
            input.inner.borrow_mut().accumulate_grad(&g);
        }
        GradFn::Mean { input, numel } => {
            let scale = 1.0 / (*numel as f32);
            let g = vec![gy[0] * scale; *numel];
            input.inner.borrow_mut().accumulate_grad(&g);
        }
        GradFn::Relu { input, mask } => {
            let gin: Vec<f32> = gy
                .iter()
                .zip(mask.iter())
                .map(|(g, &m)| if m { *g } else { 0.0 })
                .collect();
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Sigmoid { input, fwd } => {
            let gin: Vec<f32> = gy
                .iter()
                .zip(fwd.iter())
                .map(|(g, &s)| g * s * (1.0 - s))
                .collect();
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Exp { input, fwd } => {
            let gin: Vec<f32> = gy.iter().zip(fwd.iter()).map(|(g, &e)| g * e).collect();
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Log { input } => {
            let gin: Vec<f32> = {
                let x = input.inner.borrow();
                gy.iter().zip(x.data.iter()).map(|(g, &v)| g / v).collect()
            };
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Neg { input } => {
            let gin: Vec<f32> = gy.iter().map(|v| -v).collect();
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Abs { input } => {
            let gin: Vec<f32> = {
                let x = input.inner.borrow();
                gy.iter()
                    .zip(x.data.iter())
                    .map(|(g, &v)| {
                        if v > 0.0 {
                            *g
                        } else if v < 0.0 {
                            -*g
                        } else {
                            0.0
                        }
                    })
                    .collect()
            };
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Clamp { input, min, max } => {
            let gin: Vec<f32> = {
                let x = input.inner.borrow();
                gy.iter()
                    .zip(x.data.iter())
                    .map(|(g, &v)| {
                        if v >= *min && v <= *max {
                            *g
                        } else {
                            0.0
                        }
                    })
                    .collect()
            };
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Reshape { input } => {
            input.inner.borrow_mut().accumulate_grad(gy);
        }
        GradFn::Transpose2d { input } => {
            let shape = input.shape();
            let gy_t = Tensor::from_vec(gy.to_vec(), &[shape[1], shape[0]], false);
            let gin = transpose_data(&gy_t);
            input.accumulate_from(&gin);
        }
        GradFn::Linear {
            input,
            weight,
            bias,
        } => {
            let x = input;
            let w = weight;
            let n = x.shape()[0];
            let out_f = w.shape()[0];
            let y_shape = vec![n, out_f];
            let gy_t = Tensor::from_vec(gy.to_vec(), &y_shape, false);
            let gx = matmul_raw(&gy_t, w);
            let gy_tt = transpose_data(&gy_t);
            let gw = matmul_raw(&gy_tt, x);
            x.accumulate_from(&gx);
            w.accumulate_from(&gw);
            if let Some(b) = bias {
                let mut db = vec![0.0f32; out_f];
                for i in 0..n {
                    for j in 0..out_f {
                        db[j] += gy[i * out_f + j];
                    }
                }
                b.inner.borrow_mut().accumulate_grad(&db);
            }
        }
        GradFn::Cat {
            inputs,
            dim,
            sizes,
        } => {
            let out_shape = {
                let mut s = inputs[0].shape();
                s[*dim] = sizes.iter().sum();
                s
            };
            let outer: usize = out_shape[..*dim].iter().product();
            let inner: usize = out_shape[*dim + 1..].iter().product();
            let out_dim = out_shape[*dim];
            let mut col = 0usize;
            for (inp, &sz) in inputs.iter().zip(sizes.iter()) {
                let mut g = vec![0.0f32; inp.numel()];
                let idim = inp.shape()[*dim];
                for o in 0..outer {
                    for k in 0..sz {
                        for j in 0..inner {
                            let src = (o * out_dim + col + k) * inner + j;
                            let dst = (o * idim + k) * inner + j;
                            g[dst] = gy[src];
                        }
                    }
                }
                inp.inner.borrow_mut().accumulate_grad(&g);
                col += sz;
            }
        }
        GradFn::IndexSelect {
            input,
            dim,
            indices,
            input_dim_size,
        } => {
            let shape = input.shape();
            let outer: usize = shape[..*dim].iter().product();
            let inner: usize = shape[*dim + 1..].iter().product();
            let mut g = vec![0.0f32; input.numel()];
            for o in 0..outer {
                for (new_k, &old_k) in indices.iter().enumerate() {
                    for j in 0..inner {
                        let s = (o * indices.len() + new_k) * inner + j;
                        let d = (o * *input_dim_size + old_k) * inner + j;
                        g[d] += gy[s];
                    }
                }
            }
            input.inner.borrow_mut().accumulate_grad(&g);
        }
        GradFn::Softmax { input, fwd } => {
            let shape = input.shape();
            assert_eq!(shape.len(), 2);
            let n = shape[0];
            let c = shape[1];
            let mut gin = vec![0.0f32; n * c];
            for i in 0..n {
                let s = &fwd[i * c..(i + 1) * c];
                let g = &gy[i * c..(i + 1) * c];
                let mut dot = 0.0f32;
                for j in 0..c {
                    dot += s[j] * g[j];
                }
                for j in 0..c {
                    gin[i * c + j] = s[j] * (g[j] - dot);
                }
            }
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::LogSoftmax { input, fwd } => {
            // y = log_softmax(x); softmax = exp(y)
            let shape = input.shape();
            assert_eq!(shape.len(), 2);
            let n = shape[0];
            let c = shape[1];
            let mut gin = vec![0.0f32; n * c];
            for i in 0..n {
                let y = &fwd[i * c..(i + 1) * c];
                let g = &gy[i * c..(i + 1) * c];
                let mut sum_g = 0.0f32;
                for j in 0..c {
                    sum_g += g[j];
                }
                for j in 0..c {
                    gin[i * c + j] = g[j] - y[j].exp() * sum_g;
                }
            }
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::CrossEntropy {
            logits,
            probs,
            target,
            n,
            c,
        } => {
            let inv_n = 1.0 / (*n as f32);
            let mut gin = vec![0.0f32; n * c];
            for i in 0..*n {
                for j in 0..*c {
                    let mut v = probs[i * *c + j];
                    if j == target[i] {
                        v -= 1.0;
                    }
                    gin[i * *c + j] = v * inv_n * gy[0];
                }
            }
            logits.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Dropout { input, mask } => {
            let gin: Vec<f32> = gy.iter().zip(mask.iter()).map(|(g, &m)| g * m).collect();
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Stack { inputs, dim } => {
            let nstack = inputs.len();
            let base = inputs[0].shape();
            let outer: usize = if *dim == 0 {
                1
            } else {
                base[..*dim].iter().product()
            };
            let inner: usize = if *dim == base.len() {
                1
            } else {
                base[*dim..].iter().product()
            };
            for (s, inp) in inputs.iter().enumerate() {
                let mut g = vec![0.0f32; inp.numel()];
                for o in 0..outer {
                    let src_off = (o * nstack + s) * inner;
                    let dst_off = o * inner;
                    g[dst_off..dst_off + inner].copy_from_slice(&gy[src_off..src_off + inner]);
                }
                inp.inner.borrow_mut().accumulate_grad(&g);
            }
        }
        GradFn::Embedding { weight, indices } => {
            let dim = weight.shape()[1];
            let mut gw = vec![0.0f32; weight.numel()];
            for (i, &idx) in indices.iter().enumerate() {
                let src = &gy[i * dim..(i + 1) * dim];
                let dst = &mut gw[idx * dim..(idx + 1) * dim];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s;
                }
            }
            weight.inner.borrow_mut().accumulate_grad(&gw);
        }
        GradFn::LayerNorm {
            input,
            weight,
            bias,
            mean,
            rstd,
            ..
        } => {
            let shape = input.shape();
            let n = shape[0];
            let c = shape[1];
            let xi = input.inner.borrow();
            let w = weight.inner.borrow();
            let mut gin = vec![0.0f32; n * c];
            let mut gw = vec![0.0f32; c];
            let mut gb = vec![0.0f32; c];
            for i in 0..n {
                let m = mean[i];
                let rs = rstd[i];
                let mut dhat = vec![0.0f32; c];
                for j in 0..c {
                    let xhat = (xi.data[i * c + j] - m) * rs;
                    gb[j] += gy[i * c + j];
                    gw[j] += gy[i * c + j] * xhat;
                    dhat[j] = gy[i * c + j] * w.data[j];
                }
                // dx = (1/c) * rstd * (c*dhat - sum(dhat) - xhat * sum(dhat * xhat))
                let mut sum_dhat = 0.0f32;
                let mut sum_dhat_xhat = 0.0f32;
                for j in 0..c {
                    let xhat = (xi.data[i * c + j] - m) * rs;
                    sum_dhat += dhat[j];
                    sum_dhat_xhat += dhat[j] * xhat;
                }
                let inv_c = 1.0 / c as f32;
                for j in 0..c {
                    let xhat = (xi.data[i * c + j] - m) * rs;
                    gin[i * c + j] = rs * inv_c * (c as f32 * dhat[j] - sum_dhat - xhat * sum_dhat_xhat);
                }
            }
            drop((xi, w));
            input.inner.borrow_mut().accumulate_grad(&gin);
            weight.inner.borrow_mut().accumulate_grad(&gw);
            bias.inner.borrow_mut().accumulate_grad(&gb);
        }
        GradFn::Conv2d {
            input,
            weight,
            bias,
        } => {
            let xi = input.inner.borrow();
            let wi = weight.inner.borrow();
            let (n, cin, h, w) = (xi.shape[0], xi.shape[1], xi.shape[2], xi.shape[3]);
            let (cout, _, kh, kw) = (wi.shape[0], wi.shape[1], wi.shape[2], wi.shape[3]);
            let oh = h - kh + 1;
            let ow = w - kw + 1;
            let mut gx = vec![0.0f32; n * cin * h * w];
            let mut gw = vec![0.0f32; cout * cin * kh * kw];
            let mut gb = vec![0.0f32; cout];
            for ni in 0..n {
                for oc in 0..cout {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let g = gy[((ni * cout + oc) * oh + oy) * ow + ox];
                            gb[oc] += g;
                            for ic in 0..cin {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let ix = ((ni * cin + ic) * h + (oy + ky)) * w + (ox + kx);
                                        let iw = ((oc * cin + ic) * kh + ky) * kw + kx;
                                        gw[iw] += xi.data[ix] * g;
                                        gx[ix] += wi.data[iw] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            drop((xi, wi));
            input.inner.borrow_mut().accumulate_grad(&gx);
            weight.inner.borrow_mut().accumulate_grad(&gw);
            if let Some(b) = bias {
                b.inner.borrow_mut().accumulate_grad(&gb);
            }
        }
        GradFn::Tanh { fwd, input } => {
            let gin: Vec<f32> = gy
                .iter()
                .zip(fwd.iter())
                .map(|(&g, &y)| g * (1.0 - y * y))
                .collect();
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Gelu { input } => {
            // d/dx of tanh-approx gelu
            let xi = input.inner.borrow();
            let k = (2.0 / std::f32::consts::PI).sqrt();
            let c = 0.044_715f32;
            let mut gin = vec![0.0f32; gy.len()];
            for i in 0..gy.len() {
                let x = xi.data[i];
                let u = k * (x + c * x * x * x);
                let t = u.tanh();
                let sech2 = 1.0 - t * t;
                let du = k * (1.0 + 3.0 * c * x * x);
                gin[i] = gy[i] * (0.5 * (1.0 + t) + 0.5 * x * sech2 * du);
            }
            drop(xi);
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::BatchNorm1d {
            input,
            weight,
            bias,
            mean,
            rstd,
        } => {
            let shape = input.shape();
            let n = shape[0];
            let c = shape[1];
            let xi = input.inner.borrow();
            let w = weight.inner.borrow();
            let mut gin = vec![0.0f32; n * c];
            let mut gw = vec![0.0f32; c];
            let mut gb = vec![0.0f32; c];
            let inv_n = 1.0 / n as f32;
            for j in 0..c {
                let mut sum_dy = 0.0f32;
                let mut sum_dy_xhat = 0.0f32;
                for i in 0..n {
                    let xhat = (xi.data[i * c + j] - mean[j]) * rstd[j];
                    let dy = gy[i * c + j] * w.data[j];
                    gb[j] += gy[i * c + j];
                    gw[j] += gy[i * c + j] * xhat;
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                }
                for i in 0..n {
                    let xhat = (xi.data[i * c + j] - mean[j]) * rstd[j];
                    let dy = gy[i * c + j] * w.data[j];
                    gin[i * c + j] = rstd[j] * inv_n * (n as f32 * dy - sum_dy - xhat * sum_dy_xhat);
                }
            }
            drop((xi, w));
            input.inner.borrow_mut().accumulate_grad(&gin);
            weight.inner.borrow_mut().accumulate_grad(&gw);
            bias.inner.borrow_mut().accumulate_grad(&gb);
        }
        GradFn::MaxPool2d {
            input,
            indices,
            ..
        } => {
            let mut gin = vec![0.0f32; input.numel()];
            for (out_i, &in_i) in indices.iter().enumerate() {
                gin[in_i] += gy[out_i];
            }
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
    }
}

impl Tensor {
    /// `tensor.backward()` for a scalar tensor.
    pub fn backward(&self) {
        assert_eq!(
            self.numel(),
            1,
            "backward: only scalar outputs supported in v1"
        );
        {
            let mut t = self.inner.borrow_mut();
            t.grad = Some(vec![1.0]);
        }
        let order = topological_sort(self);
        for node in order.iter().rev() {
            let (grad, gf) = {
                let t = node.inner.borrow();
                (t.grad.clone(), t.grad_fn.clone())
            };
            if let (Some(g), Some(gf)) = (grad, gf) {
                apply_backward(&gf, &g);
            }
        }
    }
}
