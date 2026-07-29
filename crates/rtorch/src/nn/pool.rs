//! Pooling modules.

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::nn::Module;
use crate::ops::reshape;
use crate::tensor::{Tensor, TensorInner};

/// `F.max_pool2d` / `nn.MaxPool2d` — NCHW, padding=0.
pub fn max_pool2d(input: &Tensor, kernel_size: usize, stride: usize) -> Tensor {
    let xi = input.inner.borrow();
    assert_eq!(xi.shape.len(), 4, "max_pool2d: NCHW");
    let (n, c, h, w) = (xi.shape[0], xi.shape[1], xi.shape[2], xi.shape[3]);
    assert!(h >= kernel_size && w >= kernel_size);
    let oh = (h - kernel_size) / stride + 1;
    let ow = (w - kernel_size) / stride + 1;
    let out_n = n * c * oh * ow;
    let mut data = vec![0.0f32; out_n];
    let mut indices = vec![0usize; out_n];
    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..oh {
                for ox in 0..ow {
                    let y0 = oy * stride;
                    let x0 = ox * stride;
                    let mut best = f32::NEG_INFINITY;
                    let mut best_i = 0usize;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let iy = y0 + ky;
                            let ix = x0 + kx;
                            let ii = ((ni * c + ci) * h + iy) * w + ix;
                            let v = xi.data[ii];
                            if v > best {
                                best = v;
                                best_i = ii;
                            }
                        }
                    }
                    let oi = ((ni * c + ci) * oh + oy) * ow + ox;
                    data[oi] = best;
                    indices[oi] = best_i;
                }
            }
        }
    }
    let shape = vec![n, c, oh, ow];
    drop(xi);
    let rg = is_grad_enabled() && input.requires_grad();
    let gf = if rg {
        Some(GradFn::MaxPool2d {
            input: input.clone(),
            indices,
            kernel_size,
            stride,
        })
    } else {
        None
    };
    Tensor::from_inner(TensorInner {
        data,
        shape,
        requires_grad: rg,
        grad: if rg { Some(vec![0.0; out_n]) } else { None },
        grad_fn: gf,
    })
}

pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        max_pool2d(input, self.kernel_size, self.stride)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

/// `nn.Flatten(start_dim=1)` for contiguous tensors.
pub struct Flatten {
    pub start_dim: usize,
}

impl Flatten {
    pub fn new(start_dim: usize) -> Self {
        Self { start_dim }
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self { start_dim: 1 }
    }
}

impl Module for Flatten {
    fn forward(&self, input: &Tensor) -> Tensor {
        let shape = input.shape();
        assert!(self.start_dim < shape.len());
        let mut out = Vec::with_capacity(self.start_dim + 1);
        out.extend_from_slice(&shape[..self.start_dim]);
        let flat: usize = shape[self.start_dim..].iter().product();
        out.push(flat);
        reshape(input, &out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
