//! `nn.BatchNorm1d`

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Dtype;
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
        let xd = xi.dense_data();
        let n = xi.shape[0];
        let w = self.weight.inner.borrow();
        let b = self.bias.inner.borrow();
        let wd = w.dense_data();
        let bd = b.dense_data();
        let mut mean = vec![0.0f32; c];
        let mut rstd = vec![0.0f32; c];
        if self.training {
            for j in 0..c {
                let mut m = 0.0f32;
                for i in 0..n {
                    m += xd[i * c + j];
                }
                m /= n as f32;
                mean[j] = m;
                let mut var = 0.0f32;
                for i in 0..n {
                    let d = xd[i * c + j] - m;
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
                    let d = xd[i * c + j] - mean[j];
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
                let xhat = (xd[i * c + j] - mean[j]) * rstd[j];
                data[i * c + j] = xhat * wd[j] + bd[j];
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
        Tensor::from_inner(TensorInner::new_contiguous(
        data,
        vec![n, c],
        Device::Cpu,
        Dtype::Float32,
        rg,
        if rg {
                Some(vec![0.0; n * c])
            } else {
                None
            },
        gf,
    ))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// `torch.nn.BatchNorm2d(num_features)` — 4D `(N, C, H, W)`.
pub struct BatchNorm2d {
    pub num_features: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: RefCell<Vec<f32>>,
    pub running_var: RefCell<Vec<f32>>,
    pub training: bool,
}

impl BatchNorm2d {
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

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        assert_eq!(input.ndim(), 4, "BatchNorm2d: NCHW only");
        let c = self.num_features;
        let xi = input.inner.borrow();
        let (n, cin, h, w) = (xi.shape[0], xi.shape[1], xi.shape[2], xi.shape[3]);
        assert_eq!(cin, c);
        let m = n * h * w;
        let wt = self.weight.inner.borrow();
        let b = self.bias.inner.borrow();
        let mut mean = vec![0.0f32; c];
        let mut rstd = vec![0.0f32; c];
        if self.training {
            for j in 0..c {
                let mut s = 0.0f32;
                for ni in 0..n {
                    for y in 0..h {
                        for x in 0..w {
                            s += xi.dense_data()[((ni * c + j) * h + y) * w + x];
                        }
                    }
                }
                mean[j] = s / m as f32;
                let mut var = 0.0f32;
                for ni in 0..n {
                    for y in 0..h {
                        for x in 0..w {
                            let v = xi.dense_data()[((ni * c + j) * h + y) * w + x] - mean[j];
                            var += v * v;
                        }
                    }
                }
                var /= m as f32;
                rstd[j] = 1.0 / (var + self.eps).sqrt();
            }
            let mut rm = self.running_mean.borrow_mut();
            let mut rv = self.running_var.borrow_mut();
            let mom = self.momentum;
            let unbias = if m > 1 {
                (m as f32) / ((m - 1) as f32)
            } else {
                1.0
            };
            for j in 0..c {
                let mut var = 0.0f32;
                for ni in 0..n {
                    for y in 0..h {
                        for x in 0..w {
                            let v = xi.dense_data()[((ni * c + j) * h + y) * w + x] - mean[j];
                            var += v * v;
                        }
                    }
                }
                var = var / m as f32 * unbias;
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
        let mut data = vec![0.0f32; n * c * h * w];
        for ni in 0..n {
            for j in 0..c {
                for y in 0..h {
                    for x in 0..w {
                        let ii = ((ni * c + j) * h + y) * w + x;
                        let xhat = (xi.dense_data()[ii] - mean[j]) * rstd[j];
                        data[ii] = xhat * wt.dense_data()[j] + b.dense_data()[j];
                    }
                }
            }
        }
        drop((xi, wt, b));
        let rg = is_grad_enabled()
            && (input.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad());
        let gf = if rg {
            Some(GradFn::BatchNorm2d {
                input: input.clone(),
                weight: self.weight.clone(),
                bias: self.bias.clone(),
                mean,
                rstd,
            })
        } else {
            None
        };
        let numel = n * c * h * w;
        Tensor::from_inner(TensorInner::new_contiguous(
        data,
        vec![n, c, h, w],
        Device::Cpu,
        Dtype::Float32,
        rg,
        if rg { Some(vec![0.0; numel]) } else { None },
        gf,
    ))
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
