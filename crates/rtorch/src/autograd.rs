//! Reverse-mode autograd tape nodes.

use std::rc::Rc;

use crate::broadcast::{broadcast_shapes, expand_to, reduce_sum_to};
use crate::ops::{matmul_raw, transpose_data};
use crate::tensor::Tensor;

/// Saved tensors for a custom [`apply_function`] node (`ctx` in PyTorch).
#[derive(Clone, Debug, Default)]
pub struct FunctionCtx {
    saved: Vec<Tensor>,
}

impl FunctionCtx {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn save_for_backward(&mut self, tensors: &[Tensor]) {
        self.saved.extend_from_slice(tensors);
    }

    pub fn saved_tensors(&self) -> &[Tensor] {
        &self.saved
    }
}

/// Backward closure for [`GradFn::Custom`].
///
/// Returns one optional gradient buffer per input (same order as `inputs`).
pub type CustomBackward = Rc<dyn Fn(&FunctionCtx, &[f32]) -> Vec<Option<Vec<f32>>>>;

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
    LeakyRelu {
        input: Tensor,
        negative_slope: f32,
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
    /// `relu(linear(...))` with a single tape node (saves the ReLU mask).
    FusedLinearRelu {
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
        mask: Vec<bool>,
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
    /// `cross_entropy(linear(x, W, b), target)` as one tape node.
    FusedLinearCrossEntropy {
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
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
    BatchNorm2d {
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
    AvgPool2d {
        input: Tensor,
        kernel_size: usize,
        stride: usize,
    },
    AdaptiveAvgPool2d {
        input: Tensor,
        out_h: usize,
        out_w: usize,
    },
    Chunk {
        input: Tensor,
        dim: usize,
        start: usize,
        length: usize,
    },
    Bmm(Rc<(Tensor, Tensor)>),
    Permute {
        input: Tensor,
        dims: Vec<usize>,
    },
    Silu {
        input: Tensor,
        fwd: Vec<f32>,
    },
    /// User-defined VJP via [`apply_function`].
    Custom {
        inputs: Vec<Tensor>,
        ctx: Rc<FunctionCtx>,
        backward: CustomBackward,
    },
}

/// `torch.autograd.Function.apply`-style entry: run `forward` under `no_grad`, attach custom VJP.
pub fn apply_function(
    inputs: &[Tensor],
    forward: impl FnOnce(&mut FunctionCtx, &[Tensor]) -> Tensor,
    backward: impl Fn(&FunctionCtx, &[f32]) -> Vec<Option<Vec<f32>>> + 'static,
) -> Tensor {
    use crate::context::{is_grad_enabled, NoGradGuard};
    use crate::device::Device;
    use crate::dtype::Dtype;

    let mut ctx = FunctionCtx::new();
    let out = {
        let _guard = NoGradGuard::new();
        forward(&mut ctx, inputs)
    };
    let rg = is_grad_enabled() && inputs.iter().any(|t| t.requires_grad());
    let data = out.data();
    let shape = out.shape();
    let n = data.len();
    let gf = if rg {
        Some(GradFn::Custom {
            inputs: inputs.to_vec(),
            ctx: Rc::new(ctx),
            backward: Rc::new(backward),
        })
    } else {
        None
    };
    Tensor::from_inner(crate::tensor::TensorInner::new_contiguous(
        data,
        shape,
        Device::Cpu,
        Dtype::Float32,
        rg,
        if rg { Some(vec![0.0; n]) } else { None },
        gf,
    ))
}

/// Example custom Function: `y = x * x` (parity / docs).
pub fn square_function(x: &Tensor) -> Tensor {
    apply_function(
        &[x.clone()],
        |ctx, inputs| {
            ctx.save_for_backward(inputs);
            let xi = &inputs[0];
            let data: Vec<f32> = xi.data().iter().map(|&v| v * v).collect();
            Tensor::from_vec(data, &xi.shape(), false)
        },
        |ctx, gy| {
            let x = &ctx.saved_tensors()[0];
            let xd = x.data();
            assert_eq!(gy.len(), xd.len());
            let gx: Vec<f32> = gy
                .iter()
                .zip(xd.iter())
                .map(|(&g, &v)| g * 2.0 * v)
                .collect();
            vec![Some(gx)]
        },
    )
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
            GradFn::LeakyRelu { .. } => "LeakyRelu",
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
            GradFn::FusedLinearRelu { .. } => "FusedLinearRelu",
            GradFn::Cat { .. } => "Cat",
            GradFn::IndexSelect { .. } => "IndexSelect",
            GradFn::Softmax { .. } => "Softmax",
            GradFn::LogSoftmax { .. } => "LogSoftmax",
            GradFn::CrossEntropy { .. } => "CrossEntropy",
            GradFn::FusedLinearCrossEntropy { .. } => "FusedLinearCrossEntropy",
            GradFn::Dropout { .. } => "Dropout",
            GradFn::Stack { .. } => "Stack",
            GradFn::Embedding { .. } => "Embedding",
            GradFn::LayerNorm { .. } => "LayerNorm",
            GradFn::Conv2d { .. } => "Conv2d",
            GradFn::Tanh { .. } => "Tanh",
            GradFn::Gelu { .. } => "Gelu",
            GradFn::BatchNorm1d { .. } => "BatchNorm1d",
            GradFn::BatchNorm2d { .. } => "BatchNorm2d",
            GradFn::MaxPool2d { .. } => "MaxPool2d",
            GradFn::AvgPool2d { .. } => "AvgPool2d",
            GradFn::AdaptiveAvgPool2d { .. } => "AdaptiveAvgPool2d",
            GradFn::Chunk { .. } => "Chunk",
            GradFn::Bmm(_) => "Bmm",
            GradFn::Permute { .. } => "Permute",
            GradFn::Silu { .. } => "Silu",
            GradFn::Custom { .. } => "Custom",
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
    let grad_fn = {
        let inner = t.inner.borrow();
        inner.grad_fn.clone()
    };
    if let Some(gf) = grad_fn {
        match &*gf {
            GradFn::Add(ab)
            | GradFn::Sub(ab)
            | GradFn::Mul(ab)
            | GradFn::Div(ab)
            | GradFn::Matmul(ab)
            | GradFn::Pow(ab)
            | GradFn::Bmm(ab) => {
                visit(&ab.0, visited, order);
                visit(&ab.1, visited, order);
            }
            GradFn::Sum { input, .. }
            | GradFn::Mean { input, .. }
            | GradFn::Relu { input, .. }
            | GradFn::LeakyRelu { input, .. }
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
            GradFn::FusedLinearCrossEntropy {
                input,
                weight,
                bias,
                ..
            } => {
                visit(input, visited, order);
                visit(weight, visited, order);
                if let Some(b) = bias {
                    visit(b, visited, order);
                }
            }
            GradFn::Linear {
                input,
                weight,
                bias,
            }
            | GradFn::FusedLinearRelu {
                input,
                weight,
                bias,
                ..
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
            GradFn::Tanh { input, .. }
            | GradFn::Gelu { input }
            | GradFn::MaxPool2d { input, .. }
            | GradFn::AvgPool2d { input, .. }
            | GradFn::AdaptiveAvgPool2d { input, .. }
            | GradFn::Chunk { input, .. }
            | GradFn::Permute { input, .. }
            | GradFn::Silu { input, .. } => {
                visit(input, visited, order);
            }
            GradFn::BatchNorm1d {
                input,
                weight,
                bias,
                ..
            }
            | GradFn::BatchNorm2d {
                input,
                weight,
                bias,
                ..
            } => {
                visit(input, visited, order);
                visit(weight, visited, order);
                visit(bias, visited, order);
            }
            GradFn::Custom { inputs, .. } => {
                for inp in inputs {
                    visit(inp, visited, order);
                }
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
                let ae = expand_to(&a.dense_data(), &a_shape, &out_shape);
                let be = expand_to(&b.dense_data(), &b_shape, &out_shape);
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
                let ae = expand_to(&a.dense_data(), &a_shape, &out_shape);
                let be = expand_to(&b.dense_data(), &b_shape, &out_shape);
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
                let ae = expand_to(&a.dense_data(), &a_shape, &out_shape);
                let be = expand_to(&b.dense_data(), &b_shape, &out_shape);
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
            let mut t = input.inner.borrow_mut();
            if !t.requires_grad {
                return;
            }
            match &mut t.grad {
                Some(existing) => {
                    for (e, (g, &m)) in existing.iter_mut().zip(gy.iter().zip(mask.iter())) {
                        if m {
                            *e += *g;
                        }
                    }
                }
                None => {
                    let gin: Vec<f32> = gy
                        .iter()
                        .zip(mask.iter())
                        .map(|(g, &m)| if m { *g } else { 0.0 })
                        .collect();
                    t.grad = Some(gin);
                }
            }
        }
        GradFn::LeakyRelu {
            input,
            negative_slope,
        } => {
            let xi = input.inner.borrow();
            let gin: Vec<f32> = gy
                .iter()
                .zip(xi.dense_data().iter())
                .map(|(g, &v)| if v >= 0.0 { *g } else { *g * negative_slope })
                .collect();
            drop(xi);
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
                let x_d = x.dense_data();
                gy.iter().zip(x_d.iter()).map(|(g, &v)| g / v).collect()
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
                let x_d = x.dense_data();
                gy.iter()
                    .zip(x_d.iter())
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
                let x_d = x.dense_data();
                gy.iter()
                    .zip(x_d.iter())
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
            apply_linear_vjp(gy, input, weight, bias.as_ref());
        }
        GradFn::FusedLinearRelu {
            input,
            weight,
            bias,
            mask,
        } => {
            let mut gpre = crate::bufpool::take_f32(gy.len());
            crate::cpu_kernels::apply_relu_mask(gy, mask, &mut gpre);
            apply_linear_vjp(&gpre, input, weight, bias.as_ref());
            crate::bufpool::recycle_f32(gpre);
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
            let mut gin = crate::bufpool::take_f32(n * c);
            crate::cpu_kernels::cross_entropy_input_grad(probs, target, gy[0], &mut gin, *n, *c);
            logits.inner.borrow_mut().accumulate_grad(&gin);
            crate::bufpool::recycle_f32(gin);
        }
        GradFn::FusedLinearCrossEntropy {
            input,
            weight,
            bias,
            probs,
            target,
            n,
            c,
        } => {
            let mut gin = crate::bufpool::take_f32(n * c);
            crate::cpu_kernels::cross_entropy_input_grad(probs, target, gy[0], &mut gin, *n, *c);
            apply_linear_vjp(&gin, input, weight, bias.as_ref());
            crate::bufpool::recycle_f32(gin);
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
            let c = *shape.last().unwrap();
            let rows = input.numel() / c;
            let xi = input.inner.borrow();
            let w = weight.inner.borrow();
            let mut gin = vec![0.0f32; rows * c];
            let mut gw = vec![0.0f32; c];
            let mut gb = vec![0.0f32; c];
            for i in 0..rows {
                let m = mean[i];
                let rs = rstd[i];
                let mut dhat = vec![0.0f32; c];
                for j in 0..c {
                    let xhat = (xi.dense_data()[i * c + j] - m) * rs;
                    gb[j] += gy[i * c + j];
                    gw[j] += gy[i * c + j] * xhat;
                    dhat[j] = gy[i * c + j] * w.dense_data()[j];
                }
                let mut sum_dhat = 0.0f32;
                let mut sum_dhat_xhat = 0.0f32;
                for j in 0..c {
                    let xhat = (xi.dense_data()[i * c + j] - m) * rs;
                    sum_dhat += dhat[j];
                    sum_dhat_xhat += dhat[j] * xhat;
                }
                let inv_c = 1.0 / c as f32;
                for j in 0..c {
                    let xhat = (xi.dense_data()[i * c + j] - m) * rs;
                    gin[i * c + j] =
                        rs * inv_c * (c as f32 * dhat[j] - sum_dhat - xhat * sum_dhat_xhat);
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
                                        gw[iw] += xi.dense_data()[ix] * g;
                                        gx[ix] += wi.dense_data()[iw] * g;
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
                let x = xi.dense_data()[i];
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
                    let xhat = (xi.dense_data()[i * c + j] - mean[j]) * rstd[j];
                    let dy = gy[i * c + j] * w.dense_data()[j];
                    gb[j] += gy[i * c + j];
                    gw[j] += gy[i * c + j] * xhat;
                    sum_dy += dy;
                    sum_dy_xhat += dy * xhat;
                }
                for i in 0..n {
                    let xhat = (xi.dense_data()[i * c + j] - mean[j]) * rstd[j];
                    let dy = gy[i * c + j] * w.dense_data()[j];
                    gin[i * c + j] = rstd[j] * inv_n * (n as f32 * dy - sum_dy - xhat * sum_dy_xhat);
                }
            }
            drop((xi, w));
            input.inner.borrow_mut().accumulate_grad(&gin);
            weight.inner.borrow_mut().accumulate_grad(&gw);
            bias.inner.borrow_mut().accumulate_grad(&gb);
        }
        GradFn::BatchNorm2d {
            input,
            weight,
            bias,
            mean,
            rstd,
        } => {
            let shape = input.shape();
            let (n, c, h, w_s) = (shape[0], shape[1], shape[2], shape[3]);
            let m = n * h * w_s;
            let xi = input.inner.borrow();
            let wt = weight.inner.borrow();
            let mut gin = vec![0.0f32; n * c * h * w_s];
            let mut gw = vec![0.0f32; c];
            let mut gb = vec![0.0f32; c];
            let inv_m = 1.0 / m as f32;
            for j in 0..c {
                let mut sum_dy = 0.0f32;
                let mut sum_dy_xhat = 0.0f32;
                for ni in 0..n {
                    for y in 0..h {
                        for x in 0..w_s {
                            let ii = ((ni * c + j) * h + y) * w_s + x;
                            let xhat = (xi.dense_data()[ii] - mean[j]) * rstd[j];
                            let dy = gy[ii] * wt.dense_data()[j];
                            gb[j] += gy[ii];
                            gw[j] += gy[ii] * xhat;
                            sum_dy += dy;
                            sum_dy_xhat += dy * xhat;
                        }
                    }
                }
                for ni in 0..n {
                    for y in 0..h {
                        for x in 0..w_s {
                            let ii = ((ni * c + j) * h + y) * w_s + x;
                            let xhat = (xi.dense_data()[ii] - mean[j]) * rstd[j];
                            let dy = gy[ii] * wt.dense_data()[j];
                            gin[ii] =
                                rstd[j] * inv_m * (m as f32 * dy - sum_dy - xhat * sum_dy_xhat);
                        }
                    }
                }
            }
            drop((xi, wt));
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
        GradFn::AvgPool2d {
            input,
            kernel_size,
            stride,
        } => {
            let shape = input.shape();
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let k = *kernel_size;
            let s = *stride;
            let oh = (h - k) / s + 1;
            let ow = (w - k) / s + 1;
            let scale = 1.0 / (k * k) as f32;
            let mut gin = vec![0.0f32; n * c * h * w];
            for ni in 0..n {
                for ci in 0..c {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let g = gy[((ni * c + ci) * oh + oy) * ow + ox] * scale;
                            let y0 = oy * s;
                            let x0 = ox * s;
                            for ky in 0..k {
                                for kx in 0..k {
                                    let ii = ((ni * c + ci) * h + (y0 + ky)) * w + (x0 + kx);
                                    gin[ii] += g;
                                }
                            }
                        }
                    }
                }
            }
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::AdaptiveAvgPool2d {
            input,
            out_h,
            out_w,
        } => {
            let shape = input.shape();
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let oh = *out_h;
            let ow = *out_w;
            let mut gin = vec![0.0f32; n * c * h * w];
            for ni in 0..n {
                for ci in 0..c {
                    for oy in 0..oh {
                        for ox in 0..ow {
                            let y0 = oy * h / oh;
                            let y1 = ((oy + 1) * h + oh - 1) / oh;
                            let x0 = ox * w / ow;
                            let x1 = ((ox + 1) * w + ow - 1) / ow;
                            let area = ((y1 - y0) * (x1 - x0)) as f32;
                            let g = gy[((ni * c + ci) * oh + oy) * ow + ox] / area.max(1.0);
                            for y in y0..y1 {
                                for x in x0..x1 {
                                    gin[((ni * c + ci) * h + y) * w + x] += g;
                                }
                            }
                        }
                    }
                }
            }
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Chunk {
            input,
            dim,
            start,
            length,
        } => {
            let shape = input.shape();
            let mut gin = vec![0.0f32; input.numel()];
            let outer: usize = shape[..*dim].iter().product();
            let inner: usize = shape[*dim + 1..].iter().product();
            let dim_size = shape[*dim];
            for o in 0..outer {
                for k in 0..*length {
                    for j in 0..inner {
                        let src = (o * length + k) * inner + j;
                        let dst = (o * dim_size + start + k) * inner + j;
                        gin[dst] += gy[src];
                    }
                }
            }
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Bmm(ab) => {
            // a (B,M,K), b (B,K,N), gy (B,M,N)
            let a_shape = ab.0.shape();
            let b_shape = ab.1.shape();
            let (batch, m, k) = (a_shape[0], a_shape[1], a_shape[2]);
            let n = b_shape[2];
            let a = ab.0.inner.borrow();
            let b = ab.1.inner.borrow();
            let mut ga = vec![0.0f32; batch * m * k];
            let mut gb = vec![0.0f32; batch * k * n];
            for bi in 0..batch {
                let a_off = bi * m * k;
                let b_off = bi * k * n;
                let g_off = bi * m * n;
                // ga = gy @ b^T
                for i in 0..m {
                    for kk in 0..k {
                        let mut s = 0.0f32;
                        for j in 0..n {
                            s += gy[g_off + i * n + j] * b.dense_data()[b_off + kk * n + j];
                        }
                        ga[a_off + i * k + kk] = s;
                    }
                }
                // gb = a^T @ gy
                for kk in 0..k {
                    for j in 0..n {
                        let mut s = 0.0f32;
                        for i in 0..m {
                            s += a.dense_data()[a_off + i * k + kk] * gy[g_off + i * n + j];
                        }
                        gb[b_off + kk * n + j] = s;
                    }
                }
            }
            drop((a, b));
            ab.0.inner.borrow_mut().accumulate_grad(&ga);
            ab.1.inner.borrow_mut().accumulate_grad(&gb);
        }
        GradFn::Permute { input, dims } => {
            let inv = invert_permute(dims);
            let y_shape: Vec<usize> = dims.iter().map(|&d| input.shape()[d]).collect();
            let gx = permute_data(gy, &y_shape, &inv);
            input.inner.borrow_mut().accumulate_grad(&gx);
        }
        GradFn::Silu { input, .. } => {
            let xi = input.inner.borrow();
            let mut gin = vec![0.0f32; gy.len()];
            for i in 0..gy.len() {
                let x = xi.dense_data()[i];
                let s = 1.0 / (1.0 + (-x).exp());
                gin[i] = gy[i] * (s * (1.0 + x * (1.0 - s)));
            }
            drop(xi);
            input.inner.borrow_mut().accumulate_grad(&gin);
        }
        GradFn::Custom {
            inputs,
            ctx,
            backward,
        } => {
            let grads = backward(ctx, gy);
            assert_eq!(
                grads.len(),
                inputs.len(),
                "custom Function backward must return one Option per input"
            );
            for (inp, gopt) in inputs.iter().zip(grads.into_iter()) {
                if let Some(g) = gopt {
                    inp.inner.borrow_mut().accumulate_grad(&g);
                }
            }
        }
    }
}

fn invert_permute(dims: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; dims.len()];
    for (i, &d) in dims.iter().enumerate() {
        inv[d] = i;
    }
    inv
}

fn apply_linear_vjp(gy: &[f32], input: &Tensor, weight: &Tensor, bias: Option<&Tensor>) {
    use crate::cpu_kernels::reduce_bias_grad;
    use crate::gemm::{gemm_nn_into, gemm_tn_into};
    use crate::tensor::TensorStorage;

    let need_gx = input.requires_grad();
    let need_gw = weight.requires_grad();
    let need_gb = bias.map(|b| b.requires_grad()).unwrap_or(false);
    if !need_gx && !need_gw && !need_gb {
        return;
    }

    let x = input.as_contiguous();
    let w = weight.as_contiguous();
    let (n, in_f, out_f, x_ptr, w_ptr) = {
        let xi = x.inner.borrow();
        let wi = w.inner.borrow();
        assert!(xi.is_contiguous() && wi.is_contiguous());
        let n = xi.shape[0];
        let in_f = xi.shape[1];
        let out_f = wi.shape[0];
        assert_eq!(wi.shape[1], in_f);
        assert_eq!(gy.len(), n * out_f);
        let x_ptr = match &xi.storage {
            TensorStorage::F32(s) => s.borrow().as_ptr().wrapping_add(xi.offset),
            _ => panic!("linear VJP: F32 only"),
        };
        let w_ptr = match &wi.storage {
            TensorStorage::F32(s) => s.borrow().as_ptr().wrapping_add(wi.offset),
            _ => panic!("linear VJP: F32 only"),
        };
        // Keep storage Rcs alive via x/w tensors below.
        (n, in_f, out_f, x_ptr, w_ptr)
    };
    // Storage Rcs are owned by x/w; pointers remain valid while those live.
    let _keep = (&x, &w);

    if need_gx {
        let mut ii = input.inner.borrow_mut();
        let numel = n * in_f;
        if ii.grad.is_none() {
            ii.grad = Some(vec![0.0; numel]);
        }
        let g = ii.grad.as_mut().unwrap();
        debug_assert_eq!(g.len(), numel);
        unsafe {
            let a = std::slice::from_raw_parts(gy.as_ptr(), n * out_f);
            let b = std::slice::from_raw_parts(w_ptr, out_f * in_f);
            gemm_nn_into(a, b, g, n, out_f, in_f, 1.0);
        }
    }
    if need_gw {
        let mut wi = weight.inner.borrow_mut();
        let numel = out_f * in_f;
        if wi.grad.is_none() {
            wi.grad = Some(vec![0.0; numel]);
        }
        let g = wi.grad.as_mut().unwrap();
        debug_assert_eq!(g.len(), numel);
        unsafe {
            let a = std::slice::from_raw_parts(gy.as_ptr(), n * out_f);
            let b = std::slice::from_raw_parts(x_ptr, n * in_f);
            gemm_tn_into(a, b, g, out_f, n, in_f, 1.0);
        }
    }
    if need_gb {
        if let Some(b) = bias {
            let mut bi = b.inner.borrow_mut();
            if bi.grad.is_none() {
                bi.grad = Some(vec![0.0; out_f]);
            }
            let g = bi.grad.as_mut().unwrap();
            reduce_bias_grad(gy, g, n, out_f);
        }
    }
}

pub(crate) fn permute_data(data: &[f32], shape: &[usize], dims: &[usize]) -> Vec<f32> {
    let ndim = shape.len();
    assert_eq!(dims.len(), ndim);
    let out_shape: Vec<usize> = dims.iter().map(|&d| shape[d]).collect();
    let numel: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; numel];
    let mut stride = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    let mut out_stride = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_stride[i] = out_stride[i + 1] * out_shape[i + 1];
    }
    for out_idx in 0..numel {
        let mut rem = out_idx;
        let mut in_idx = 0usize;
        for i in 0..ndim {
            let coord = rem / out_stride[i];
            rem %= out_stride[i];
            in_idx += coord * stride[dims[i]];
        }
        out[out_idx] = data[in_idx];
    }
    out
}

impl Tensor {
    /// `tensor.backward()` for a scalar tensor.
    pub fn backward(&self) {
        self.backward_with(false);
    }

    /// `tensor.backward(create_graph=...)` for a scalar tensor.
    ///
    /// When `create_graph` is true, leaf `.grad` buffers are filled via differentiable
    /// VJPs (see [`grad`]); use [`grad`] directly when you need the returned tensors.
    pub fn backward_with(&self, create_graph: bool) {
        assert_eq!(
            self.numel(),
            1,
            "backward: only scalar outputs supported in v1"
        );
        if create_graph {
            let order = topological_sort(self);
            let inputs: Vec<Tensor> = order
                .iter()
                .filter(|t| t.requires_grad() && t.inner.borrow().grad_fn.is_none())
                .cloned()
                .collect();
            let refs: Vec<&Tensor> = inputs.iter().collect();
            let gvec = grad(self, &refs, true);
            for (t, g) in inputs.iter().zip(gvec.iter()) {
                t.inner.borrow_mut().grad = Some(g.data());
            }
            return;
        }
        {
            let mut t = self.inner.borrow_mut();
            t.grad = Some(vec![1.0]);
        }
        let order = topological_sort(self);
        for node in order.iter().rev() {
            let gf = {
                let t = node.inner.borrow();
                t.grad_fn.clone()
            };
            let Some(gf) = gf else {
                continue;
            };
            let g = node.inner.borrow_mut().grad.take();
            if let Some(g) = g {
                apply_backward(&gf, &g);
            }
        }
    }
}

fn tensor_ptr(t: &Tensor) -> usize {
    std::rc::Rc::as_ptr(&t.inner) as usize
}

fn accumulate_grad_map(
    grads: &mut std::collections::HashMap<usize, Tensor>,
    target: &Tensor,
    g: Tensor,
) {
    let p = tensor_ptr(target);
    let next = match grads.remove(&p) {
        Some(old) => crate::ops::add(&old, &g),
        None => g,
    };
    grads.insert(p, next);
}

/// Conv2d VJP buffers (stride=1, pad=0). Pass empty `x`/`w` when that buffer is unused.
fn conv2d_vjp_bufs(
    gy: &[f32],
    x: &[f32],
    w: &[f32],
    n: usize,
    cin: usize,
    h: usize,
    ww: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    oh: usize,
    ow: usize,
    need_gw: bool,
    need_gx: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut gx = vec![0.0f32; n * cin * h * ww];
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
                                let ix = ((ni * cin + ic) * h + (oy + ky)) * ww + (ox + kx);
                                let iw = ((oc * cin + ic) * kh + ky) * kw + kx;
                                if need_gw {
                                    gw[iw] += x[ix] * g;
                                }
                                if need_gx {
                                    gx[ix] += w[iw] * g;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (gx, gw, gb)
}

fn apply_vjp_tensor(
    gf: &GradFn,
    gy: &Tensor,
    grads: &mut std::collections::HashMap<usize, Tensor>,
) {
    use crate::ops::{mul, neg, ones};
    match gf {
        GradFn::Add(ab) => {
            assert_eq!(
                ab.0.shape(),
                ab.1.shape(),
                "create_graph Add: same shape only"
            );
            accumulate_grad_map(grads, &ab.0, gy.clone());
            accumulate_grad_map(grads, &ab.1, gy.clone());
        }
        GradFn::Sub(ab) => {
            assert_eq!(
                ab.0.shape(),
                ab.1.shape(),
                "create_graph Sub: same shape only"
            );
            accumulate_grad_map(grads, &ab.0, gy.clone());
            accumulate_grad_map(grads, &ab.1, neg(gy));
        }
        GradFn::Mul(ab) => {
            assert_eq!(
                ab.0.shape(),
                ab.1.shape(),
                "create_graph Mul: same shape only"
            );
            accumulate_grad_map(grads, &ab.0, mul(gy, &ab.1));
            accumulate_grad_map(grads, &ab.1, mul(gy, &ab.0));
        }
        GradFn::Sum { input, .. } => {
            let scale = ones(&input.shape(), false);
            accumulate_grad_map(grads, input, mul(gy, &scale));
        }
        GradFn::Mean { input, numel } => {
            let scale = crate::ops::full(&input.shape(), 1.0 / (*numel as f32), false);
            accumulate_grad_map(grads, input, mul(gy, &scale));
        }
        GradFn::Neg { input } => {
            accumulate_grad_map(grads, input, neg(gy));
        }
        GradFn::Relu { input, mask } => {
            let mdata: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            let m = Tensor::from_vec(mdata, &input.shape(), false);
            accumulate_grad_map(grads, input, mul(gy, &m));
        }
        GradFn::Pow(ab) => {
            use crate::ops::{log, pow, sub};
            assert_eq!(
                ab.0.shape(),
                ab.1.shape(),
                "create_graph Pow: same shape only"
            );
            let a = &ab.0;
            let b = &ab.1;
            let one = crate::ops::ones(&a.shape(), false);
            let bm1 = sub(b, &one);
            let a_pow_bm1 = pow(a, &bm1);
            let ga = mul(gy, &mul(b, &a_pow_bm1));
            let a_pow_b = pow(a, b);
            let gb = mul(gy, &mul(&a_pow_b, &log(a)));
            accumulate_grad_map(grads, a, ga);
            accumulate_grad_map(grads, b, gb);
        }
        GradFn::Matmul(ab) => {
            use crate::ops::{matmul, transpose};
            let a = &ab.0;
            let b = &ab.1;
            assert_eq!(a.ndim(), 2);
            assert_eq!(b.ndim(), 2);
            let bt = transpose(b);
            let at = transpose(a);
            let ga = matmul(gy, &bt);
            let gb = matmul(&at, gy);
            accumulate_grad_map(grads, a, ga);
            accumulate_grad_map(grads, b, gb);
        }
        GradFn::Div(ab) => {
            use crate::ops::div;
            assert_eq!(
                ab.0.shape(),
                ab.1.shape(),
                "create_graph Div: same shape only"
            );
            let a = &ab.0;
            let b = &ab.1;
            let ga = div(gy, b);
            // gb = -gy * a / b^2
            let gb = neg(&mul(gy, &div(a, &mul(b, b))));
            accumulate_grad_map(grads, a, ga);
            accumulate_grad_map(grads, b, gb);
        }
        GradFn::Exp { input, .. } => {
            use crate::ops::exp;
            accumulate_grad_map(grads, input, mul(gy, &exp(input)));
        }
        GradFn::Log { input } => {
            use crate::ops::div;
            accumulate_grad_map(grads, input, div(gy, input));
        }
        GradFn::Sigmoid { input, .. } => {
            use crate::ops::sub;
            use crate::functional::sigmoid;
            let s = sigmoid(input);
            let one = ones(&input.shape(), false);
            let ds = mul(&s, &sub(&one, &s));
            accumulate_grad_map(grads, input, mul(gy, &ds));
        }
        GradFn::Tanh { input, .. } => {
            use crate::ops::sub;
            use crate::functional::tanh;
            let t = tanh(input);
            let one = ones(&input.shape(), false);
            let dt = sub(&one, &mul(&t, &t));
            accumulate_grad_map(grads, input, mul(gy, &dt));
        }
        GradFn::Abs { input } => {
            // sign(x) treated as constant for higher-order (abs'' = 0 a.e.)
            let xd = input.data();
            let sign: Vec<f32> = xd
                .iter()
                .map(|&v| {
                    if v > 0.0 {
                        1.0
                    } else if v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let s = Tensor::from_vec(sign, &input.shape(), false);
            accumulate_grad_map(grads, input, mul(gy, &s));
        }
        GradFn::Softmax { input, .. } => {
            use crate::functional::softmax;
            use crate::ops::{matmul, sub};
            assert_eq!(input.ndim(), 2, "create_graph Softmax: 2D only");
            let shape = input.shape();
            let c = shape[1];
            let s = softmax(input);
            let sg = mul(&s, gy);
            let ones_col = ones(&[c, 1], false);
            let dots = matmul(&sg, &ones_col); // [n, 1]
            let ones_row = ones(&[1, c], false);
            let dots_bc = matmul(&dots, &ones_row); // [n, c]
            accumulate_grad_map(grads, input, mul(&s, &sub(gy, &dots_bc)));
        }
        GradFn::LogSoftmax { input, .. } => {
            use crate::functional::softmax;
            use crate::ops::{matmul, sub};
            assert_eq!(input.ndim(), 2, "create_graph LogSoftmax: 2D only");
            let shape = input.shape();
            let c = shape[1];
            let s = softmax(input);
            let ones_col = ones(&[c, 1], false);
            let sum_g = matmul(gy, &ones_col);
            let ones_row = ones(&[1, c], false);
            let sum_bc = matmul(&sum_g, &ones_row);
            accumulate_grad_map(grads, input, sub(gy, &mul(&s, &sum_bc)));
        }
        GradFn::CrossEntropy {
            logits,
            probs,
            target,
            n,
            c,
        } => {
            let probs = probs.clone();
            let target = target.clone();
            let n = *n;
            let c = *c;
            let inv_n = 1.0 / n as f32;
            let shape = logits.shape();
            let gin = apply_function(
                &[gy.clone()],
                {
                    let probs = probs.clone();
                    let target = target.clone();
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let g0 = inputs[0].item();
                        let mut gin = vec![0.0f32; n * c];
                        for i in 0..n {
                            for j in 0..c {
                                let mut v = probs[i * c + j];
                                if j == target[i] {
                                    v -= 1.0;
                                }
                                gin[i * c + j] = v * inv_n * g0;
                            }
                        }
                        Tensor::from_vec(gin, &shape, false)
                    }
                },
                {
                    let probs = probs.clone();
                    let target = target.clone();
                    move |_ctx, ggin| {
                        let mut acc = 0.0f32;
                        for i in 0..n {
                            for j in 0..c {
                                let mut v = probs[i * c + j];
                                if j == target[i] {
                                    v -= 1.0;
                                }
                                acc += ggin[i * c + j] * v * inv_n;
                            }
                        }
                        vec![Some(vec![acc])]
                    }
                },
            );
            accumulate_grad_map(grads, logits, gin);
        }
        GradFn::FusedLinearCrossEntropy {
            input,
            weight,
            bias,
            probs,
            target,
            n,
            c,
        } => {
            use crate::ops::{matmul, reshape, transpose};
            let mut gin_data = vec![0.0f32; n * c];
            crate::cpu_kernels::cross_entropy_input_grad(
                probs,
                target,
                gy.item(),
                &mut gin_data,
                *n,
                *c,
            );
            let gpre = Tensor::from_vec(gin_data, &[*n, *c], false);
            assert_eq!(input.ndim(), 2);
            assert_eq!(weight.ndim(), 2);
            let batch = input.shape()[0];
            let out_f = weight.shape()[0];
            let gx = matmul(&gpre, weight);
            let gw = matmul(&transpose(&gpre), input);
            accumulate_grad_map(grads, input, gx);
            accumulate_grad_map(grads, weight, gw);
            if let Some(b) = bias {
                let ones_row = ones(&[1, batch], false);
                let gb = matmul(&ones_row, &gpre);
                accumulate_grad_map(grads, b, reshape(&gb, &[out_f]));
            }
        }
        GradFn::Silu { input, .. } => {
            use crate::functional::sigmoid;
            use crate::ops::{add, sub};
            let s = sigmoid(input);
            let one = ones(&input.shape(), false);
            // σ(x) * (1 + x * (1 - σ(x)))
            let ds = mul(&s, &add(&one, &mul(input, &sub(&one, &s))));
            accumulate_grad_map(grads, input, mul(gy, &ds));
        }
        GradFn::Gelu { input } => {
            use crate::functional::tanh;
            use crate::ops::{add, full, sub};
            let k = (2.0f32 / std::f32::consts::PI).sqrt();
            let c = 0.044_715f32;
            let sh = input.shape();
            let x2 = mul(input, input);
            let x3 = mul(&x2, input);
            let u = mul(
                &full(&sh, k, false),
                &add(input, &mul(&full(&sh, c, false), &x3)),
            );
            let t = tanh(&u);
            let one = ones(&sh, false);
            let half = full(&sh, 0.5, false);
            let sech2 = sub(&one, &mul(&t, &t));
            let du = mul(
                &full(&sh, k, false),
                &add(&one, &mul(&full(&sh, 3.0 * c, false), &x2)),
            );
            let term1 = mul(&half, &add(&one, &t));
            let term2 = mul(&half, &mul(input, &mul(&sech2, &du)));
            let dgelu = add(&term1, &term2);
            accumulate_grad_map(grads, input, mul(gy, &dgelu));
        }
        GradFn::Clamp { input, min, max } => {
            let xd = input.data();
            let mask: Vec<f32> = xd
                .iter()
                .map(|&v| {
                    if v >= *min && v <= *max {
                        1.0
                    } else {
                        0.0
                    }
                })
                .collect();
            let m = Tensor::from_vec(mask, &input.shape(), false);
            accumulate_grad_map(grads, input, mul(gy, &m));
        }
        GradFn::Reshape { input } => {
            use crate::ops::reshape;
            accumulate_grad_map(grads, input, reshape(gy, &input.shape()));
        }
        GradFn::Transpose2d { input } => {
            use crate::ops::transpose;
            assert_eq!(gy.ndim(), 2, "create_graph Transpose2d: 2D only");
            accumulate_grad_map(grads, input, transpose(gy));
        }
        GradFn::Linear {
            input,
            weight,
            bias,
        } => {
            use crate::ops::{matmul, reshape, transpose};
            assert_eq!(input.ndim(), 2);
            assert_eq!(weight.ndim(), 2);
            assert_eq!(gy.ndim(), 2);
            let n = input.shape()[0];
            let out_f = weight.shape()[0];
            let gx = matmul(gy, weight);
            let gw = matmul(&transpose(gy), input);
            accumulate_grad_map(grads, input, gx);
            accumulate_grad_map(grads, weight, gw);
            if let Some(b) = bias {
                let ones_row = ones(&[1, n], false);
                let gb = matmul(&ones_row, gy); // [1, out]
                accumulate_grad_map(grads, b, reshape(&gb, &[out_f]));
            }
        }
        GradFn::FusedLinearRelu {
            input,
            weight,
            bias,
            mask,
        } => {
            use crate::ops::{matmul, mul, reshape, transpose};
            assert_eq!(input.ndim(), 2);
            assert_eq!(weight.ndim(), 2);
            assert_eq!(gy.ndim(), 2);
            let n = input.shape()[0];
            let out_f = weight.shape()[0];
            let mdata: Vec<f32> = mask.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
            let m = Tensor::from_vec(mdata, &gy.shape(), false);
            let gpre = mul(gy, &m);
            let gx = matmul(&gpre, weight);
            let gw = matmul(&transpose(&gpre), input);
            accumulate_grad_map(grads, input, gx);
            accumulate_grad_map(grads, weight, gw);
            if let Some(b) = bias {
                let ones_row = ones(&[1, n], false);
                let gb = matmul(&ones_row, &gpre);
                accumulate_grad_map(grads, b, reshape(&gb, &[out_f]));
            }
        }
        GradFn::Cat {
            inputs,
            dim,
            sizes,
        } => {
            use crate::ops::narrow;
            let mut col = 0usize;
            for (inp, &sz) in inputs.iter().zip(sizes.iter()) {
                let gin = narrow(gy, *dim, col, sz);
                accumulate_grad_map(grads, inp, gin);
                col += sz;
            }
        }
        GradFn::Stack { inputs, dim } => {
            use crate::ops::{narrow, reshape};
            for (s, inp) in inputs.iter().enumerate() {
                let piece = narrow(gy, *dim, s, 1);
                accumulate_grad_map(grads, inp, reshape(&piece, &inp.shape()));
            }
        }
        GradFn::LeakyRelu {
            input,
            negative_slope,
        } => {
            let xd = input.data();
            let mask: Vec<f32> = xd
                .iter()
                .map(|&v| {
                    if v >= 0.0 {
                        1.0
                    } else {
                        *negative_slope
                    }
                })
                .collect();
            let m = Tensor::from_vec(mask, &input.shape(), false);
            accumulate_grad_map(grads, input, mul(gy, &m));
        }
        GradFn::Permute { input, dims } => {
            use crate::ops::permute;
            let inv = invert_permute(dims);
            accumulate_grad_map(grads, input, permute(gy, &inv));
        }
        GradFn::Bmm(ab) => {
            use crate::ops::{bmm, permute};
            assert_eq!(ab.0.ndim(), 3);
            assert_eq!(ab.1.ndim(), 3);
            assert_eq!(gy.ndim(), 3);
            // ga = gy @ b^T ; gb = a^T @ gy
            let bt = permute(&ab.1, &[0, 2, 1]);
            let at = permute(&ab.0, &[0, 2, 1]);
            accumulate_grad_map(grads, &ab.0, bmm(gy, &bt));
            accumulate_grad_map(grads, &ab.1, bmm(&at, gy));
        }
        GradFn::LayerNorm {
            input,
            weight,
            bias,
            mean,
            rstd,
            ..
        } => {
            use crate::ops::{full, matmul, reshape, sub};
            let shape = input.shape();
            let c = *shape.last().unwrap();
            let rows = input.numel() / c;
            let gy2 = reshape(gy, &[rows, c]);
            let x2 = reshape(input, &[rows, c]);
            let ones_r = ones(&[1, rows], false);
            let ones_c = ones(&[c, 1], false);
            let ones_row = ones(&[1, c], false);

            // bias grad: sum over rows
            let gb = reshape(&matmul(&ones_r, &gy2), &[c]);

            // xhat = (x - mean) * rstd  (mean/rstd treated as saved constants)
            let mean_t = Tensor::from_vec(mean.clone(), &[rows, 1], false);
            let rstd_t = Tensor::from_vec(rstd.clone(), &[rows, 1], false);
            let mean_bc = matmul(&mean_t, &ones_row);
            let rstd_bc = matmul(&rstd_t, &ones_row);
            let xhat = mul(&sub(&x2, &mean_bc), &rstd_bc);

            // weight grad: sum_i gy * xhat
            let gw = reshape(&matmul(&ones_r, &mul(&gy2, &xhat)), &[c]);

            // dhat = gy * weight
            let dhat = mul(&gy2, weight);
            let sum_dhat = matmul(&dhat, &ones_c); // [rows, 1]
            let sum_dhat_xhat = matmul(&mul(&dhat, &xhat), &ones_c);
            let sum_dhat_bc = matmul(&sum_dhat, &ones_row);
            let sum_dhat_xhat_bc = matmul(&sum_dhat_xhat, &ones_row);
            let inv_c = full(&[rows, c], 1.0 / c as f32, false);
            let c_t = full(&[rows, c], c as f32, false);
            // gin = rstd/c * (c*dhat - sum_dhat - xhat*sum_dhat_xhat)
            let inner = sub(
                &sub(&mul(&c_t, &dhat), &sum_dhat_bc),
                &mul(&xhat, &sum_dhat_xhat_bc),
            );
            let gin = reshape(&mul(&rstd_bc, &mul(&inv_c, &inner)), &shape);

            accumulate_grad_map(grads, input, gin);
            accumulate_grad_map(grads, weight, gw);
            accumulate_grad_map(grads, bias, gb);
        }
        GradFn::Conv2d {
            input,
            weight,
            bias,
        } => {
            let xshape = input.shape();
            let wshape = weight.shape();
            let (n, cin, h, ww) = (xshape[0], xshape[1], xshape[2], xshape[3]);
            let (cout, _, kh, kw) = (wshape[0], wshape[1], wshape[2], wshape[3]);
            let oh = h - kh + 1;
            let ow = ww - kw + 1;
            let x_c = Tensor::from_vec(input.data(), &xshape, false);
            let w_c = Tensor::from_vec(weight.data(), &wshape, false);

            let gx = apply_function(
                &[gy.clone(), w_c.clone()],
                {
                    let xshape = xshape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let wd = inputs[1].data();
                        let (gx, _, _) = conv2d_vjp_bufs(
                            &gyd, &[], &wd, n, cin, h, ww, cout, kh, kw, oh, ow, false, true,
                        );
                        Tensor::from_vec(gx, &xshape, false)
                    }
                },
                {
                    move |ctx, ggx| {
                        let gyd = ctx.saved_tensors()[0].data();
                        let wd = ctx.saved_tensors()[1].data();
                        let mut ggy = vec![0.0f32; n * cout * oh * ow];
                        let mut gw = vec![0.0f32; cout * cin * kh * kw];
                        for ni in 0..n {
                            for oc in 0..cout {
                                for oy in 0..oh {
                                    for ox in 0..ow {
                                        let mut acc = 0.0f32;
                                        for ic in 0..cin {
                                            for ky in 0..kh {
                                                for kx in 0..kw {
                                                    let ix = ((ni * cin + ic) * h + (oy + ky)) * ww
                                                        + (ox + kx);
                                                    let iw =
                                                        ((oc * cin + ic) * kh + ky) * kw + kx;
                                                    acc += wd[iw] * ggx[ix];
                                                    gw[iw] += gyd
                                                        [((ni * cout + oc) * oh + oy) * ow + ox]
                                                        * ggx[ix];
                                                }
                                            }
                                        }
                                        ggy[((ni * cout + oc) * oh + oy) * ow + ox] = acc;
                                    }
                                }
                            }
                        }
                        vec![Some(ggy), Some(gw)]
                    }
                },
            );

            let gw = apply_function(
                &[gy.clone(), x_c.clone()],
                {
                    let wshape = wshape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let xd = inputs[1].data();
                        let (_, gw, _) = conv2d_vjp_bufs(
                            &gyd, &xd, &[], n, cin, h, ww, cout, kh, kw, oh, ow, true, false,
                        );
                        Tensor::from_vec(gw, &wshape, false)
                    }
                },
                {
                    move |ctx, ggw| {
                        let gyd = ctx.saved_tensors()[0].data();
                        let xd = ctx.saved_tensors()[1].data();
                        let mut ggy = vec![0.0f32; n * cout * oh * ow];
                        let mut gx = vec![0.0f32; n * cin * h * ww];
                        for ni in 0..n {
                            for oc in 0..cout {
                                for oy in 0..oh {
                                    for ox in 0..ow {
                                        let g_out = ((ni * cout + oc) * oh + oy) * ow + ox;
                                        let mut acc = 0.0f32;
                                        for ic in 0..cin {
                                            for ky in 0..kh {
                                                for kx in 0..kw {
                                                    let ix = ((ni * cin + ic) * h + (oy + ky)) * ww
                                                        + (ox + kx);
                                                    let iw =
                                                        ((oc * cin + ic) * kh + ky) * kw + kx;
                                                    acc += xd[ix] * ggw[iw];
                                                    gx[ix] += gyd[g_out] * ggw[iw];
                                                }
                                            }
                                        }
                                        ggy[g_out] = acc;
                                    }
                                }
                            }
                        }
                        vec![Some(ggy), Some(gx)]
                    }
                },
            );

            accumulate_grad_map(grads, input, gx);
            accumulate_grad_map(grads, weight, gw);
            if let Some(b) = bias {
                let gb = apply_function(
                    &[gy.clone()],
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut gb = vec![0.0f32; cout];
                        for ni in 0..n {
                            for oc in 0..cout {
                                for oy in 0..oh {
                                    for ox in 0..ow {
                                        gb[oc] +=
                                            gyd[((ni * cout + oc) * oh + oy) * ow + ox];
                                    }
                                }
                            }
                        }
                        Tensor::from_vec(gb, &[cout], false)
                    },
                    move |_ctx, ggb| {
                        let mut ggy = vec![0.0f32; n * cout * oh * ow];
                        for ni in 0..n {
                            for oc in 0..cout {
                                for oy in 0..oh {
                                    for ox in 0..ow {
                                        ggy[((ni * cout + oc) * oh + oy) * ow + ox] = ggb[oc];
                                    }
                                }
                            }
                        }
                        vec![Some(ggy)]
                    },
                );
                accumulate_grad_map(grads, b, gb);
            }
        }
        GradFn::BatchNorm1d {
            input,
            weight,
            bias,
            mean,
            rstd,
        } => {
            use crate::ops::{full, matmul, reshape, sub};
            let shape = input.shape();
            let n = shape[0];
            let c = shape[1];
            let ones_n = ones(&[n, 1], false);
            let ones_r = ones(&[1, n], false);
            let mean_t = Tensor::from_vec(mean.clone(), &[1, c], false);
            let rstd_t = Tensor::from_vec(rstd.clone(), &[1, c], false);
            let mean_bc = matmul(&ones_n, &mean_t);
            let rstd_bc = matmul(&ones_n, &rstd_t);
            let xhat = mul(&sub(input, &mean_bc), &rstd_bc);

            let gb = reshape(&matmul(&ones_r, gy), &[c]);
            let gw = reshape(&matmul(&ones_r, &mul(gy, &xhat)), &[c]);

            let w_bc = matmul(&ones_n, &reshape(weight, &[1, c]));
            let dhat = mul(gy, &w_bc);
            let sum_dhat = matmul(&ones_r, &dhat); // [1, c]
            let sum_dhat_xhat = matmul(&ones_r, &mul(&dhat, &xhat));
            let sum_dhat_bc = matmul(&ones_n, &sum_dhat);
            let sum_dhat_xhat_bc = matmul(&ones_n, &sum_dhat_xhat);
            let inv_n = full(&[n, c], 1.0 / n as f32, false);
            let n_t = full(&[n, c], n as f32, false);
            let inner = sub(
                &sub(&mul(&n_t, &dhat), &sum_dhat_bc),
                &mul(&xhat, &sum_dhat_xhat_bc),
            );
            let gin = mul(&rstd_bc, &mul(&inv_n, &inner));

            accumulate_grad_map(grads, input, gin);
            accumulate_grad_map(grads, weight, gw);
            accumulate_grad_map(grads, bias, gb);
        }
        GradFn::BatchNorm2d {
            input,
            weight,
            bias,
            mean,
            rstd,
        } => {
            let shape = input.shape();
            let (n, c, h, w_s) = (shape[0], shape[1], shape[2], shape[3]);
            let m = n * h * w_s;
            let mean = mean.clone();
            let rstd = rstd.clone();
            let x_c = Tensor::from_vec(input.data(), &shape, false);
            let w_c = Tensor::from_vec(weight.data(), &weight.shape(), false);

            let gin = apply_function(
                &[gy.clone(), x_c.clone(), w_c.clone()],
                {
                    let mean = mean.clone();
                    let rstd = rstd.clone();
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let xd = inputs[1].data();
                        let wd = inputs[2].data();
                        let inv_m = 1.0 / m as f32;
                        let mut gin = vec![0.0f32; n * c * h * w_s];
                        for j in 0..c {
                            let mut sum_dy = 0.0f32;
                            let mut sum_dy_xhat = 0.0f32;
                            for ni in 0..n {
                                for y in 0..h {
                                    for x in 0..w_s {
                                        let ii = ((ni * c + j) * h + y) * w_s + x;
                                        let xhat = (xd[ii] - mean[j]) * rstd[j];
                                        let dy = gyd[ii] * wd[j];
                                        sum_dy += dy;
                                        sum_dy_xhat += dy * xhat;
                                    }
                                }
                            }
                            for ni in 0..n {
                                for y in 0..h {
                                    for x in 0..w_s {
                                        let ii = ((ni * c + j) * h + y) * w_s + x;
                                        let xhat = (xd[ii] - mean[j]) * rstd[j];
                                        let dy = gyd[ii] * wd[j];
                                        gin[ii] = rstd[j]
                                            * inv_m
                                            * (m as f32 * dy - sum_dy - xhat * sum_dy_xhat);
                                    }
                                }
                            }
                        }
                        Tensor::from_vec(gin, &shape, false)
                    }
                },
                {
                    let mean = mean.clone();
                    let rstd = rstd.clone();
                    move |ctx, ggin| {
                        let xd = ctx.saved_tensors()[1].data();
                        let wd = ctx.saved_tensors()[2].data();
                        let inv_m = 1.0 / m as f32;
                        let mut ggy = vec![0.0f32; n * c * h * w_s];
                        for j in 0..c {
                            let mut sum_g = 0.0f32;
                            let mut sum_g_xhat = 0.0f32;
                            for ni in 0..n {
                                for y in 0..h {
                                    for x in 0..w_s {
                                        let ii = ((ni * c + j) * h + y) * w_s + x;
                                        let xhat = (xd[ii] - mean[j]) * rstd[j];
                                        sum_g += ggin[ii];
                                        sum_g_xhat += ggin[ii] * xhat;
                                    }
                                }
                            }
                            let scale = rstd[j] * inv_m * wd[j];
                            for ni in 0..n {
                                for y in 0..h {
                                    for x in 0..w_s {
                                        let ii = ((ni * c + j) * h + y) * w_s + x;
                                        let xhat = (xd[ii] - mean[j]) * rstd[j];
                                        ggy[ii] = scale
                                            * (m as f32 * ggin[ii] - sum_g - xhat * sum_g_xhat);
                                    }
                                }
                            }
                        }
                        vec![Some(ggy), None, None]
                    }
                },
            );

            let gw = apply_function(
                &[gy.clone(), x_c.clone()],
                {
                    let mean = mean.clone();
                    let rstd = rstd.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let xd = inputs[1].data();
                        let mut gw = vec![0.0f32; c];
                        for j in 0..c {
                            for ni in 0..n {
                                for y in 0..h {
                                    for x in 0..w_s {
                                        let ii = ((ni * c + j) * h + y) * w_s + x;
                                        let xhat = (xd[ii] - mean[j]) * rstd[j];
                                        gw[j] += gyd[ii] * xhat;
                                    }
                                }
                            }
                        }
                        Tensor::from_vec(gw, &[c], false)
                    }
                },
                {
                    let mean = mean.clone();
                    let rstd = rstd.clone();
                    move |ctx, ggw| {
                        let xd = ctx.saved_tensors()[1].data();
                        let mut ggy = vec![0.0f32; n * c * h * w_s];
                        for j in 0..c {
                            for ni in 0..n {
                                for y in 0..h {
                                    for x in 0..w_s {
                                        let ii = ((ni * c + j) * h + y) * w_s + x;
                                        let xhat = (xd[ii] - mean[j]) * rstd[j];
                                        ggy[ii] = ggw[j] * xhat;
                                    }
                                }
                            }
                        }
                        vec![Some(ggy), None]
                    }
                },
            );

            let gb = apply_function(
                &[gy.clone()],
                move |ctx, inputs| {
                    ctx.save_for_backward(inputs);
                    let gyd = inputs[0].data();
                    let mut gb = vec![0.0f32; c];
                    for j in 0..c {
                        for ni in 0..n {
                            for y in 0..h {
                                for x in 0..w_s {
                                    gb[j] += gyd[((ni * c + j) * h + y) * w_s + x];
                                }
                            }
                        }
                    }
                    Tensor::from_vec(gb, &[c], false)
                },
                move |_ctx, ggb| {
                    let mut ggy = vec![0.0f32; n * c * h * w_s];
                    for j in 0..c {
                        for ni in 0..n {
                            for y in 0..h {
                                for x in 0..w_s {
                                    ggy[((ni * c + j) * h + y) * w_s + x] = ggb[j];
                                }
                            }
                        }
                    }
                    vec![Some(ggy)]
                },
            );

            accumulate_grad_map(grads, input, gin);
            accumulate_grad_map(grads, weight, gw);
            accumulate_grad_map(grads, bias, gb);
        }
        GradFn::MaxPool2d {
            input,
            indices,
            ..
        } => {
            let indices = indices.clone();
            let shape = input.shape();
            let in_numel = input.numel();
            let n_out = indices.len();
            let gin = apply_function(
                &[gy.clone()],
                {
                    let indices = indices.clone();
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut g = vec![0.0f32; in_numel];
                        for (out_i, &in_i) in indices.iter().enumerate() {
                            g[in_i] += gyd[out_i];
                        }
                        Tensor::from_vec(g, &shape, false)
                    }
                },
                {
                    let indices = indices.clone();
                    move |_ctx, ggin| {
                        let mut ggy = vec![0.0f32; n_out];
                        for (out_i, &in_i) in indices.iter().enumerate() {
                            ggy[out_i] += ggin[in_i];
                        }
                        vec![Some(ggy)]
                    }
                },
            );
            accumulate_grad_map(grads, input, gin);
        }
        GradFn::AvgPool2d {
            input,
            kernel_size,
            stride,
        } => {
            let shape = input.shape();
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let k = *kernel_size;
            let s = *stride;
            let oh = (h - k) / s + 1;
            let ow = (w - k) / s + 1;
            let scale = 1.0 / (k * k) as f32;
            let gin = apply_function(
                &[gy.clone()],
                {
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut gin = vec![0.0f32; n * c * h * w];
                        for ni in 0..n {
                            for ci in 0..c {
                                for oy in 0..oh {
                                    for ox in 0..ow {
                                        let g =
                                            gyd[((ni * c + ci) * oh + oy) * ow + ox] * scale;
                                        let y0 = oy * s;
                                        let x0 = ox * s;
                                        for ky in 0..k {
                                            for kx in 0..k {
                                                let ii = ((ni * c + ci) * h + (y0 + ky)) * w
                                                    + (x0 + kx);
                                                gin[ii] += g;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Tensor::from_vec(gin, &shape, false)
                    }
                },
                move |_ctx, ggin| {
                    let mut ggy = vec![0.0f32; n * c * oh * ow];
                    for ni in 0..n {
                        for ci in 0..c {
                            for oy in 0..oh {
                                for ox in 0..ow {
                                    let y0 = oy * s;
                                    let x0 = ox * s;
                                    let mut acc = 0.0f32;
                                    for ky in 0..k {
                                        for kx in 0..k {
                                            let ii = ((ni * c + ci) * h + (y0 + ky)) * w
                                                + (x0 + kx);
                                            acc += ggin[ii];
                                        }
                                    }
                                    ggy[((ni * c + ci) * oh + oy) * ow + ox] = acc * scale;
                                }
                            }
                        }
                    }
                    vec![Some(ggy)]
                },
            );
            accumulate_grad_map(grads, input, gin);
        }
        GradFn::AdaptiveAvgPool2d {
            input,
            out_h,
            out_w,
        } => {
            let shape = input.shape();
            let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
            let oh = *out_h;
            let ow = *out_w;
            let gin = apply_function(
                &[gy.clone()],
                {
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut gin = vec![0.0f32; n * c * h * w];
                        for ni in 0..n {
                            for ci in 0..c {
                                for oy in 0..oh {
                                    for ox in 0..ow {
                                        let y0 = oy * h / oh;
                                        let y1 = ((oy + 1) * h + oh - 1) / oh;
                                        let x0 = ox * w / ow;
                                        let x1 = ((ox + 1) * w + ow - 1) / ow;
                                        let area = ((y1 - y0) * (x1 - x0)) as f32;
                                        let g = gyd[((ni * c + ci) * oh + oy) * ow + ox]
                                            / area.max(1.0);
                                        for y in y0..y1 {
                                            for x in x0..x1 {
                                                gin[((ni * c + ci) * h + y) * w + x] += g;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Tensor::from_vec(gin, &shape, false)
                    }
                },
                move |_ctx, ggin| {
                    let mut ggy = vec![0.0f32; n * c * oh * ow];
                    for ni in 0..n {
                        for ci in 0..c {
                            for oy in 0..oh {
                                for ox in 0..ow {
                                    let y0 = oy * h / oh;
                                    let y1 = ((oy + 1) * h + oh - 1) / oh;
                                    let x0 = ox * w / ow;
                                    let x1 = ((ox + 1) * w + ow - 1) / ow;
                                    let area = ((y1 - y0) * (x1 - x0)) as f32;
                                    let inv = 1.0 / area.max(1.0);
                                    let mut acc = 0.0f32;
                                    for y in y0..y1 {
                                        for x in x0..x1 {
                                            acc += ggin[((ni * c + ci) * h + y) * w + x];
                                        }
                                    }
                                    ggy[((ni * c + ci) * oh + oy) * ow + ox] = acc * inv;
                                }
                            }
                        }
                    }
                    vec![Some(ggy)]
                },
            );
            accumulate_grad_map(grads, input, gin);
        }
        GradFn::Chunk {
            input,
            dim,
            start,
            length,
        } => {
            let dim = *dim;
            let start = *start;
            let length = *length;
            let shape = input.shape();
            let in_numel = input.numel();
            let outer: usize = shape[..dim].iter().product();
            let inner: usize = shape[dim + 1..].iter().product();
            let dim_size = shape[dim];
            let gin = apply_function(
                &[gy.clone()],
                {
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut gin = vec![0.0f32; in_numel];
                        for o in 0..outer {
                            for k in 0..length {
                                for j in 0..inner {
                                    let src = (o * length + k) * inner + j;
                                    let dst = (o * dim_size + start + k) * inner + j;
                                    gin[dst] += gyd[src];
                                }
                            }
                        }
                        Tensor::from_vec(gin, &shape, false)
                    }
                },
                move |_ctx, ggin| {
                    let mut ggy = vec![0.0f32; outer * length * inner];
                    for o in 0..outer {
                        for k in 0..length {
                            for j in 0..inner {
                                let src = (o * length + k) * inner + j;
                                let dst = (o * dim_size + start + k) * inner + j;
                                ggy[src] += ggin[dst];
                            }
                        }
                    }
                    vec![Some(ggy)]
                },
            );
            accumulate_grad_map(grads, input, gin);
        }
        GradFn::Dropout { input, mask } => {
            let m = Tensor::from_vec(mask.clone(), &input.shape(), false);
            accumulate_grad_map(grads, input, mul(gy, &m));
        }
        GradFn::IndexSelect {
            input,
            dim,
            indices,
            input_dim_size,
        } => {
            let shape = input.shape();
            let dim = *dim;
            let indices = indices.clone();
            let input_dim_size = *input_dim_size;
            let outer: usize = shape[..dim].iter().product();
            let inner: usize = shape[dim + 1..].iter().product();
            let nidx = indices.len();
            let in_numel = input.numel();
            let gin = apply_function(
                &[gy.clone()],
                {
                    let indices = indices.clone();
                    let shape = shape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut g = vec![0.0f32; in_numel];
                        for o in 0..outer {
                            for (new_k, &old_k) in indices.iter().enumerate() {
                                for j in 0..inner {
                                    let s = (o * nidx + new_k) * inner + j;
                                    let d = (o * input_dim_size + old_k) * inner + j;
                                    g[d] += gyd[s];
                                }
                            }
                        }
                        Tensor::from_vec(g, &shape, false)
                    }
                },
                {
                    let indices = indices.clone();
                    move |_ctx, ggin| {
                        let mut ggy = vec![0.0f32; outer * nidx * inner];
                        for o in 0..outer {
                            for (new_k, &old_k) in indices.iter().enumerate() {
                                for j in 0..inner {
                                    let s = (o * nidx + new_k) * inner + j;
                                    let d = (o * input_dim_size + old_k) * inner + j;
                                    ggy[s] += ggin[d];
                                }
                            }
                        }
                        vec![Some(ggy)]
                    }
                },
            );
            accumulate_grad_map(grads, input, gin);
        }
        GradFn::Embedding { weight, indices } => {
            let dim = weight.shape()[1];
            let n = indices.len();
            let indices = indices.clone();
            let wshape = weight.shape();
            let w_numel = weight.numel();
            let gw = apply_function(
                &[gy.clone()],
                {
                    let indices = indices.clone();
                    let wshape = wshape.clone();
                    move |ctx, inputs| {
                        ctx.save_for_backward(inputs);
                        let gyd = inputs[0].data();
                        let mut g = vec![0.0f32; w_numel];
                        for (i, &idx) in indices.iter().enumerate() {
                            let src = &gyd[i * dim..(i + 1) * dim];
                            let dst = &mut g[idx * dim..(idx + 1) * dim];
                            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                                *d += s;
                            }
                        }
                        Tensor::from_vec(g, &wshape, false)
                    }
                },
                {
                    let indices = indices.clone();
                    move |_ctx, ggin| {
                        let mut ggy = vec![0.0f32; n * dim];
                        for (i, &idx) in indices.iter().enumerate() {
                            let src = &ggin[idx * dim..(idx + 1) * dim];
                            let dst = &mut ggy[i * dim..(i + 1) * dim];
                            dst.copy_from_slice(src);
                        }
                        vec![Some(ggy)]
                    }
                },
            );
            accumulate_grad_map(grads, weight, gw);
        }
        other => {
            // Non-differentiable-through fallback: classic VJP into `.grad`, then lift.
            let data = gy.data();
            apply_backward(other, &data);
            // Pull any newly written leaf grads into the map (no graph).
            for t in other_inputs(other) {
                if let Some(g) = t.grad() {
                    let p = tensor_ptr(&t);
                    if !grads.contains_key(&p) {
                        grads.insert(p, Tensor::from_vec(g, &t.shape(), false));
                    }
                }
            }
        }
    }
}

fn other_inputs(gf: &GradFn) -> Vec<Tensor> {
    let mut v = Vec::new();
    match gf {
        GradFn::Add(ab)
        | GradFn::Sub(ab)
        | GradFn::Mul(ab)
        | GradFn::Div(ab)
        | GradFn::Matmul(ab)
        | GradFn::Pow(ab)
        | GradFn::Bmm(ab) => {
            v.push(ab.0.clone());
            v.push(ab.1.clone());
        }
        GradFn::Sum { input, .. }
        | GradFn::Mean { input, .. }
        | GradFn::Relu { input, .. }
        | GradFn::LeakyRelu { input, .. }
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
        | GradFn::Dropout { input, .. }
        | GradFn::Tanh { input, .. }
        | GradFn::Gelu { input }
        | GradFn::MaxPool2d { input, .. }
        | GradFn::AvgPool2d { input, .. }
        | GradFn::AdaptiveAvgPool2d { input, .. }
        | GradFn::Chunk { input, .. }
        | GradFn::Permute { input, .. }
        | GradFn::Silu { input, .. } => v.push(input.clone()),
        GradFn::CrossEntropy { logits, .. } => v.push(logits.clone()),
        GradFn::FusedLinearCrossEntropy {
            input,
            weight,
            bias,
            ..
        } => {
            v.push(input.clone());
            v.push(weight.clone());
            if let Some(b) = bias {
                v.push(b.clone());
            }
        },
        GradFn::Linear {
            input,
            weight,
            bias,
        }
        | GradFn::FusedLinearRelu {
            input,
            weight,
            bias,
            ..
        } => {
            v.push(input.clone());
            v.push(weight.clone());
            if let Some(b) = bias {
                v.push(b.clone());
            }
        }
        GradFn::Cat { inputs, .. } | GradFn::Stack { inputs, .. } => v.extend(inputs.clone()),
        GradFn::Embedding { weight, .. } => v.push(weight.clone()),
        GradFn::LayerNorm {
            input,
            weight,
            bias,
            ..
        } => {
            v.push(input.clone());
            v.push(weight.clone());
            v.push(bias.clone());
        }
        GradFn::Conv2d {
            input,
            weight,
            bias,
        } => {
            v.push(input.clone());
            v.push(weight.clone());
            if let Some(b) = bias {
                v.push(b.clone());
            }
        }
        GradFn::BatchNorm1d {
            input,
            weight,
            bias,
            ..
        }
        | GradFn::BatchNorm2d {
            input,
            weight,
            bias,
            ..
        } => {
            v.push(input.clone());
            v.push(weight.clone());
            v.push(bias.clone());
        }
        GradFn::Custom { inputs, .. } => v.extend(inputs.clone()),
    }
    v
}

/// `torch.autograd.grad(outputs, inputs, create_graph=...)`.
///
/// `create_graph=true` builds a differentiable graph for
/// Add/Mul/Div/Pow/Matmul/Sum/Mean/Sub/Neg/Relu/Exp/Log/Sigmoid/Tanh/Abs/
/// Softmax/LogSoftmax/Silu/Gelu/Clamp/Reshape/Transpose2d/Linear/Cat/Stack/LeakyRelu/Permute/Bmm/
/// Dropout/IndexSelect/Embedding/LayerNorm/Conv2d/BatchNorm1d/BatchNorm2d/
/// MaxPool2d/AvgPool2d/AdaptiveAvgPool2d/CrossEntropy/Chunk
/// (same-shape elementwise binaries; 2D matmul / 2D softmax / 2D linear / 3D bmm). Other ops fall back to first-order-only VJPs.
pub fn grad(output: &Tensor, inputs: &[&Tensor], create_graph: bool) -> Vec<Tensor> {
    assert_eq!(output.numel(), 1, "grad: scalar output only in v1");
    if !create_graph {
        for inp in inputs {
            inp.zero_grad();
        }
        output.backward();
        return inputs
            .iter()
            .map(|inp| {
                let g = inp
                    .grad()
                    .unwrap_or_else(|| vec![0.0; inp.numel()]);
                Tensor::from_vec(g, &inp.shape(), false)
            })
            .collect();
    }

    let mut grads: std::collections::HashMap<usize, Tensor> = std::collections::HashMap::new();
    let ones = crate::ops::ones(&[], true);
    ones.set_requires_grad(true);
    grads.insert(tensor_ptr(output), ones);

    let order = topological_sort(output);
    for node in order.iter().rev() {
        let gf = node.inner.borrow().grad_fn.clone();
        let Some(gy) = grads.get(&tensor_ptr(node)).cloned() else {
            continue;
        };
        let Some(gf) = gf else {
            continue;
        };
        apply_vjp_tensor(&gf, &gy, &mut grads);
    }

    inputs
        .iter()
        .map(|inp| {
            grads
                .get(&tensor_ptr(inp))
                .cloned()
                .unwrap_or_else(|| crate::ops::zeros(&inp.shape(), false))
        })
        .collect()
}

/// Central-difference check of analytic `∂f/∂x` for a scalar `f(x)`.
///
/// Returns the max relative error over elements:
/// `|g_num - g_an| / (1 + max(|g_num|, |g_an|))`.
pub fn gradcheck_max_error(
    mut f: impl FnMut(&Tensor) -> Tensor,
    x: &Tensor,
    eps: f32,
) -> f32 {
    assert!(eps > 0.0);
    x.set_requires_grad(true);
    x.zero_grad();
    let y = f(x);
    assert_eq!(y.numel(), 1, "gradcheck: f must return a scalar");
    let g_an = grad(&y, &[x], false)[0].data();
    let shape = x.shape();
    let mut data = x.data();
    let n = data.len();
    let mut max_err = 0.0f32;
    for i in 0..n {
        let orig = data[i];
        data[i] = orig + eps;
        let xp = Tensor::from_vec(data.clone(), &shape, false);
        let yp = f(&xp).item();
        data[i] = orig - eps;
        let xm = Tensor::from_vec(data.clone(), &shape, false);
        let ym = f(&xm).item();
        data[i] = orig;
        let g_num = (yp - ym) / (2.0 * eps);
        let ga = g_an[i];
        let denom = 1.0 + g_num.abs().max(ga.abs());
        let err = (g_num - ga).abs() / denom;
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

#[cfg(test)]
mod grad_tests {
    use super::*;
    use crate::ops::{matmul, mul, pow, seeded_uniform, sum};

    #[test]
    fn second_derivative_of_square() {
        let x = seeded_uniform(&[4], 7, -1.0, 1.0);
        x.set_requires_grad(true);
        let y = sum(&mul(&x, &x));
        let g = grad(&y, &[&x], true);
        assert_eq!(g.len(), 1);
        let gsum = sum(&g[0]);
        let g2 = grad(&gsum, &[&x], false);
        for &v in &g2[0].data() {
            assert!((v - 2.0).abs() < 1e-4, "got {v}");
        }
    }

    #[test]
    fn create_graph_pow_cubic() {
        // y = sum(x^3); dy/dx = 3 x^2; d2y/dx2 = 6 x
        let x = seeded_uniform(&[3], 11, 0.5, 1.5);
        x.set_requires_grad(true);
        let three = crate::ops::full(&x.shape(), 3.0, false);
        let y = sum(&pow(&x, &three));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        let xd = x.data();
        let g2d = g2[0].data();
        for i in 0..xd.len() {
            assert!((g2d[i] - 6.0 * xd[i]).abs() < 1e-3, "{} vs {}", g2d[i], 6.0 * xd[i]);
        }
    }

    #[test]
    fn gradcheck_square_small_error() {
        let x = seeded_uniform(&[5], 3, -0.8, 0.8);
        let err = gradcheck_max_error(|t| sum(&mul(t, t)), &x, 1e-3);
        assert!(err < 1e-2, "gradcheck err {err}");
    }

    #[test]
    fn create_graph_matmul() {
        let a = seeded_uniform(&[2, 3], 1, -1.0, 1.0);
        let b = seeded_uniform(&[3, 2], 2, -1.0, 1.0);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let y = sum(&matmul(&a, &b));
        let gs = grad(&y, &[&a, &b], true);
        let g2a = grad(&sum(&gs[0]), &[&a], false);
        // ∂²/∂a² of sum(a@b) is 0
        for &v in &g2a[0].data() {
            assert!(v.abs() < 1e-5, "got {v}");
        }
    }

    #[test]
    fn custom_square_function_grad() {
        let x = seeded_uniform(&[4], 9, -1.0, 1.0);
        x.set_requires_grad(true);
        let y = sum(&square_function(&x));
        y.backward();
        let g = x.grad().unwrap();
        let xd = x.data();
        for i in 0..xd.len() {
            assert!((g[i] - 2.0 * xd[i]).abs() < 1e-5, "{} vs {}", g[i], 2.0 * xd[i]);
        }
    }

    #[test]
    fn create_graph_exp() {
        use crate::ops::exp;
        let x = seeded_uniform(&[4], 13, -0.5, 0.5);
        x.set_requires_grad(true);
        let y = sum(&exp(&x));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        let expected = exp(&x).data();
        let g2d = g2[0].data();
        for i in 0..expected.len() {
            assert!((g2d[i] - expected[i]).abs() < 1e-3, "{} vs {}", g2d[i], expected[i]);
        }
    }

    #[test]
    fn create_graph_sigmoid() {
        use crate::functional::sigmoid;
        use crate::ops::sub;
        let x = seeded_uniform(&[4], 17, -1.0, 1.0);
        x.set_requires_grad(true);
        let y = sum(&sigmoid(&x));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        let s = sigmoid(&x);
        let one = crate::ops::ones(&x.shape(), false);
        let ds = mul(&s, &sub(&one, &s));
        // d²σ/dx² = σ(1-σ)(1-2σ)
        let two_s = mul(&s, &crate::ops::full(&x.shape(), 2.0, false));
        let expected = mul(&ds, &sub(&one, &two_s)).data();
        let g2d = g2[0].data();
        for i in 0..expected.len() {
            assert!((g2d[i] - expected[i]).abs() < 1e-3, "{} vs {}", g2d[i], expected[i]);
        }
    }

    #[test]
    fn create_graph_silu() {
        use crate::functional::{sigmoid, silu};
        use crate::ops::{add, sub};
        let x = seeded_uniform(&[4], 19, -1.0, 1.0);
        x.set_requires_grad(true);
        let y = sum(&silu(&x));
        let g = grad(&y, &[&x], true);
        let gd = g[0].data();
        let s = sigmoid(&x);
        let one = crate::ops::ones(&x.shape(), false);
        let expected = mul(&s, &add(&one, &mul(&x, &sub(&one, &s)))).data();
        for i in 0..expected.len() {
            assert!((gd[i] - expected[i]).abs() < 1e-3, "{} vs {}", gd[i], expected[i]);
        }
        // second derivative finite
        let g2 = grad(&sum(&g[0]), &[&x], false);
        assert!(g2[0].data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn create_graph_softmax_sq() {
        use crate::functional::softmax;
        let x = seeded_uniform(&[2, 3], 23, -1.0, 1.0);
        x.set_requires_grad(true);
        let s = softmax(&x);
        let y = sum(&mul(&s, &s));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        assert!(g[0].data().iter().all(|v| v.is_finite()));
        assert!(g2[0].data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn create_graph_clamp_sq() {
        use crate::ops::clamp;
        let x = seeded_uniform(&[5], 29, -1.5, 1.5);
        x.set_requires_grad(true);
        let y = sum(&mul(&clamp(&x, -0.5, 0.5), &clamp(&x, -0.5, 0.5)));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        let xd = x.data();
        let g2d = g2[0].data();
        for i in 0..xd.len() {
            let expected = if xd[i] >= -0.5 && xd[i] <= 0.5 { 2.0 } else { 0.0 };
            assert!((g2d[i] - expected).abs() < 1e-3, "{} vs {}", g2d[i], expected);
        }
    }

    #[test]
    fn create_graph_linear_sq() {
        use crate::functional::linear;
        use crate::ops::add;
        let x = seeded_uniform(&[3, 4], 31, -1.0, 1.0);
        let w = seeded_uniform(&[2, 4], 32, -0.5, 0.5);
        let b = seeded_uniform(&[2], 33, -0.1, 0.1);
        x.set_requires_grad(true);
        w.set_requires_grad(true);
        b.set_requires_grad(true);
        let out = linear(&x, &w, Some(&b));
        let y = sum(&mul(&out, &out));
        let gs = grad(&y, &[&x, &w, &b], true);
        assert_eq!(gs.len(), 3);
        let gsum = add(&add(&sum(&gs[0]), &sum(&gs[1])), &sum(&gs[2]));
        let g2 = grad(&gsum, &[&x, &w], false);
        assert!(g2[0].data().iter().all(|v| v.is_finite()));
        assert!(g2[1].data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn create_graph_cat_sq() {
        use crate::ops::{add, cat};
        let a = seeded_uniform(&[2, 3], 41, -1.0, 1.0);
        let b = seeded_uniform(&[2, 3], 42, -1.0, 1.0);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let c = cat(&[&a, &b], 0);
        let y = sum(&mul(&c, &c));
        let gs = grad(&y, &[&a, &b], true);
        let gsum = add(&sum(&gs[0]), &sum(&gs[1]));
        let g2 = grad(&gsum, &[&a, &b], false);
        for i in 0..a.numel() {
            assert!((g2[0].data()[i] - 2.0).abs() < 1e-3);
        }
        for i in 0..b.numel() {
            assert!((g2[1].data()[i] - 2.0).abs() < 1e-3);
        }
    }

    #[test]
    fn create_graph_stack_sq() {
        use crate::ops::{add, stack};
        let a = seeded_uniform(&[2, 2], 43, -1.0, 1.0);
        let b = seeded_uniform(&[2, 2], 44, -1.0, 1.0);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let c = stack(&[&a, &b], 0);
        let y = sum(&mul(&c, &c));
        let gs = grad(&y, &[&a, &b], true);
        let gsum = add(&sum(&gs[0]), &sum(&gs[1]));
        let g2 = grad(&gsum, &[&a, &b], false);
        assert!(g2[0].data().iter().all(|&v| (v - 2.0).abs() < 1e-3));
        assert!(g2[1].data().iter().all(|&v| (v - 2.0).abs() < 1e-3));
    }

    #[test]
    fn create_graph_bmm_sq() {
        use crate::ops::{add, bmm};
        let a = seeded_uniform(&[2, 3, 4], 51, -1.0, 1.0);
        let b = seeded_uniform(&[2, 4, 3], 52, -1.0, 1.0);
        a.set_requires_grad(true);
        b.set_requires_grad(true);
        let c = bmm(&a, &b);
        let y = sum(&mul(&c, &c));
        let gs = grad(&y, &[&a, &b], true);
        let gsum = add(&sum(&gs[0]), &sum(&gs[1]));
        let g2 = grad(&gsum, &[&a, &b], false);
        assert!(g2[0].data().iter().all(|v| v.is_finite()));
        assert!(g2[1].data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn create_graph_permute_sq() {
        use crate::ops::permute;
        let x = seeded_uniform(&[2, 3, 4], 53, -1.0, 1.0);
        x.set_requires_grad(true);
        let y = sum(&mul(&permute(&x, &[2, 0, 1]), &permute(&x, &[2, 0, 1])));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        assert!(g2[0].data().iter().all(|&v| (v - 2.0).abs() < 1e-3));
    }

    #[test]
    fn create_graph_dropout_sq() {
        use crate::functional::dropout;
        let x = seeded_uniform(&[4], 62, -1.0, 1.0);
        x.set_requires_grad(true);
        let d = dropout(&x, 0.25, true, 77);
        let y = sum(&mul(&d, &d));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        let p = 0.25f32;
        let scale = 1.0 / (1.0 - p);
        let mut state = 77u64;
        let g2d = g2[0].data();
        for i in 0..4 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let u = ((state >> 8) & 0xFF_FFFF) as f32 * (1.0 / ((1u64 << 24) as f32));
            let m = if u >= p { scale } else { 0.0 };
            let expected = 2.0 * m * m;
            assert!((g2d[i] - expected).abs() < 1e-3, "{} vs {}", g2d[i], expected);
        }
    }

    #[test]
    fn create_graph_index_select_sq() {
        use crate::ops::index_select;
        let x = seeded_uniform(&[5, 3], 64, -1.0, 1.0);
        x.set_requires_grad(true);
        let s = index_select(&x, 0, &[0, 2, 4]);
        let y = sum(&mul(&s, &s));
        let g = grad(&y, &[&x], true);
        let g2 = grad(&sum(&g[0]), &[&x], false);
        let g2d = g2[0].data();
        for row in 0..5 {
            let expected = if matches!(row, 0 | 2 | 4) { 2.0 } else { 0.0 };
            for j in 0..3 {
                assert!((g2d[row * 3 + j] - expected).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn create_graph_layernorm_sq() {
        use crate::nn::{LayerNorm, Module};
        use crate::ops::add;
        let x = seeded_uniform(&[2, 4], 71, -1.0, 1.0);
        let ln = LayerNorm::new(4, 1e-5);
        x.set_requires_grad(true);
        let out = ln.forward(&x);
        let y = sum(&mul(&out, &out));
        let w = &ln.weight;
        let b = &ln.bias;
        let gs = grad(&y, &[&x, w, b], true);
        let classic = grad(&y, &[&x, w, b], false);
        for i in 0..3 {
            let a = gs[i].data();
            let b = classic[i].data();
            for j in 0..a.len() {
                assert!((a[j] - b[j]).abs() < 1e-4, "{} vs {}", a[j], b[j]);
            }
        }
        // Graph is live enough for a second backward to stay finite.
        let gsum = add(&add(&sum(&gs[0]), &sum(&gs[1])), &sum(&gs[2]));
        let g2 = grad(&gsum, &[&x, w], false);
        assert!(g2[0].data().iter().all(|v| v.is_finite()));
        assert!(g2[1].data().iter().all(|v| v.is_finite()));
    }
}
