//! `nn.LayerNorm`

use crate::context::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::nn::Module;
use crate::ops::{add, full, matmul, mul, ones, pow, reshape, sub, zeros};
use crate::tensor::{Tensor, TensorInner};

/// `torch.nn.LayerNorm(normalized_shape)` — normalize over the last dimension.
pub struct LayerNorm {
    pub normalized_shape: usize,
    pub eps: f32,
    pub weight: Tensor, // (C,)
    pub bias: Tensor,   // (C,)
}

impl LayerNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Self {
        Self {
            normalized_shape,
            eps,
            weight: ones(&[normalized_shape], true),
            bias: zeros(&[normalized_shape], true),
        }
    }

    pub fn from_params(weight: Tensor, bias: Tensor, eps: f32) -> Self {
        let c = weight.numel();
        assert_eq!(bias.numel(), c);
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);
        Self {
            normalized_shape: c,
            eps,
            weight,
            bias,
        }
    }
}

/// Fast contiguous LayerNorm when no autograd is needed.
fn layernorm_forward_nograd(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32, c: usize) -> Tensor {
    let shape = input.shape();
    let rows = input.numel() / c;
    let xd = input.inner.borrow().dense_data();
    let wd = weight.inner.borrow().dense_data();
    let bd = bias.inner.borrow().dense_data();
    let mut data = vec![0.0f32; rows * c];
    for i in 0..rows {
        let row = &xd[i * c..(i + 1) * c];
        let mut m = 0.0f32;
        for &v in row {
            m += v;
        }
        m /= c as f32;
        let mut var = 0.0f32;
        for &v in row {
            let d = v - m;
            var += d * d;
        }
        var /= c as f32;
        let rs = 1.0 / (var + eps).sqrt();
        for j in 0..c {
            let xhat = (row[j] - m) * rs;
            data[i * c + j] = xhat * wd[j] + bd[j];
        }
    }
    Tensor::from_inner(TensorInner::new_contiguous(
        data,
        shape,
        Device::Cpu,
        Dtype::Float32,
        false,
        None,
        None,
    ))
}

/// Differentiable LayerNorm via tensor ops (supports full `create_graph` / Hessian).
fn layernorm_forward_diff(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f32, c: usize) -> Tensor {
    let shape = input.shape();
    let rows = input.numel() / c;
    let x2 = reshape(input, &[rows, c]);
    let ones_c = ones(&[c, 1], false);
    let ones_row = ones(&[1, c], false);
    let ones_n = ones(&[rows, 1], false);
    let inv_c = full(&[rows, 1], 1.0 / c as f32, false);

    // mean: [rows, 1]
    let mean = mul(&matmul(&x2, &ones_c), &inv_c);
    let mean_bc = matmul(&mean, &ones_row);
    let centered = sub(&x2, &mean_bc);
    // var: [rows, 1]
    let var = mul(&matmul(&mul(&centered, &centered), &ones_c), &inv_c);
    let eps_t = full(&[rows, 1], eps, false);
    let neg_half = full(&[rows, 1], -0.5, false);
    let rstd = pow(&add(&var, &eps_t), &neg_half);
    let rstd_bc = matmul(&rstd, &ones_row);
    let xhat = mul(&centered, &rstd_bc);
    let w_bc = matmul(&ones_n, &reshape(weight, &[1, c]));
    let b_bc = matmul(&ones_n, &reshape(bias, &[1, c]));
    let out2 = add(&mul(&xhat, &w_bc), &b_bc);
    reshape(&out2, &shape)
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        let c = self.normalized_shape;
        assert_eq!(
            *shape.last().expect("LayerNorm: empty shape"),
            c,
            "LayerNorm: last dim"
        );
        let rg = is_grad_enabled()
            && (input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad());
        if !rg {
            return layernorm_forward_nograd(input, &self.weight, &self.bias, self.eps, c);
        }
        layernorm_forward_diff(input, &self.weight, &self.bias, self.eps, c)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
