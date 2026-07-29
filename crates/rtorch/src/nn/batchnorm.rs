//! `nn.BatchNorm1d`

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::nn::Module;
use crate::ops::{ones, zeros};
use crate::tensor::{Tensor, TensorInner};
use std::cell::RefCell;

/// `torch.nn.BatchNorm1d(num_features)` — 2D `(N, C)` training/eval.
pub struct BatchNorm1d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: RefCell<Vec<f32>>,
    pub running_var: RefCell<Vec<f32>>,
    pub training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize, eps: f32, momentum: f32) -> Self {
        Self {
            num_features,
            eps,
            momentum,
            weight: ones(&[num_features], true),
            bias: zeros(&[num_features], true),
            running_mean: RefCell::new(vec![0.0; num_features]),
            running_var: RefCell::new(vec![1.0; num_features]),
            training: true,
        }
    }

    pub fn from_params(weight: Tensor, bias: Tensor, eps: f32, momentum: f32) -> Self {
        let c = weight.numel();
        assert_eq!(bias.numel(), c);
        weight.set_requires_grad(true);
        bias.set_requires_grad(true);
        Self {
            num_features: c,
            eps,
            momentum,
            weight,
            bias,
            running_mean: RefCell::new(vec![0.0; c]),
            running_var: RefCell::new(vec![1.0; c]),
            training: true,
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 2, "BatchNorm1d: 2D (N,C) only");
        let c = self.num_features;
        assert_eq!(input.shape()[1], c);
        let xi = input.inner.borrow();
        let n = xi.shape[0];
        let w = self.weight.inner.borrow();
        let b = self.bias.inner.borrow();
        let mut mean = vec![0.0f32; c];
        let mut rstd = vec![0.0f32; c];
        if self.training {
            for j in 0..c {
                let mut m = 0.0f32;
                for i in 0..n {
                    m += xi.data[i * c + j];
                }
                m /= n as f32;
                mean[j] = m;
                let mut var = 0.0f32;
                for i in 0..n {
                    let d = xi.data[i * c + j] - m;
                    var += d * d;
                }
                var /= n as f32;
                rstd[j] = 1.0 / (var + self.eps).sqrt();
            }
            // Update running stats (PyTorch: unbiased var for running_var).
            let mut rm = self.running_mean.borrow_mut();
            let mut rv = self.running_var.borrow_mut();
            let mom = self.momentum;
            let unbias = if n > 1 {
                (n as f32) / ((n - 1) as f32)
            } else {
                1.0
            };
            for j in 0..c {
                let mut var = 0.0f32;
                for i in 0..n {
                    let d = xi.data[i * c + j] - mean[j];
                    var += d * d;
                }
                var = var / n as f32 * unbias;
                rm[j] = (1.0 - mom) * rm[j] + mom * mean[j];
                rv[j] = (1.0 - mom) * rv[j] + mom * var;
            }
        } else {
            let rm = self.running_mean.borrow();
            let rv = self.running_var.borrow();
            for j in 0..c {
                mean[j] = rm[j];
                rstd[j] = 1.0 / (rv[j] + self.eps).sqrt();
            }
        }
        let mut data = vec![0.0f32; n * c];
        for i in 0..n {
            for j in 0..c {
                let xhat = (xi.data[i * c + j] - mean[j]) * rstd[j];
                data[i * c + j] = xhat * w.data[j] + b.data[j];
            }
        }
        drop((xi, w, b));
        let rg = is_grad_enabled()
            && (input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad());
        let gf = if rg {
            Some(GradFn::BatchNorm1d {
                input: input.clone(),
                weight: self.weight.clone(),
                bias: self.bias.clone(),
                mean,
                rstd,
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
