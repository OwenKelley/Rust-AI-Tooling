//! `nn.LayerNorm`

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::nn::Module;
use crate::ops::{ones, zeros};
use crate::tensor::{Tensor, TensorInner};

/// `torch.nn.LayerNorm(normalized_shape)` — last-dim normalize for 2D `(N, C)`.
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

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 2, "LayerNorm: 2D (N,C) only");
        let c = self.normalized_shape;
        assert_eq!(input.shape()[1], c, "LayerNorm: feature dim");
        let xi = input.inner.borrow();
        let n = xi.shape[0];
        let w = self.weight.inner.borrow();
        let b = self.bias.inner.borrow();
        let mut data = vec![0.0f32; n * c];
        let mut mean = vec![0.0f32; n];
        let mut rstd = vec![0.0f32; n];
        for i in 0..n {
            let row = &xi.data[i * c..(i + 1) * c];
            let mut m = 0.0f32;
            for &v in row {
                m += v;
            }
            m /= c as f32;
            mean[i] = m;
            let mut var = 0.0f32;
            for &v in row {
                let d = v - m;
                var += d * d;
            }
            var /= c as f32;
            let rs = 1.0 / (var + self.eps).sqrt();
            rstd[i] = rs;
            for j in 0..c {
                let xhat = (row[j] - m) * rs;
                data[i * c + j] = xhat * w.data[j] + b.data[j];
            }
        }
        drop((xi, w, b));
        let rg = is_grad_enabled()
            && (input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad());
        let gf = if rg {
            Some(GradFn::LayerNorm {
                input: input.clone(),
                weight: self.weight.clone(),
                bias: self.bias.clone(),
                mean,
                rstd,
                eps: self.eps,
            })
        } else {
            None
        };
        Tensor::from_inner(TensorInner {
            data,
            shape: vec![n, c],
            requires_grad: rg,
            grad: if rg {
                Some(vec![0.0; n * c])
            } else {
                None
            },
            grad_fn: gf,
        })
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
