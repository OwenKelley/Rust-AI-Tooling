//! `nn.Conv2d` — NCHW, stride=1, padding=0, dilation=1, groups=1.

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Dtype;
use crate::nn::Module;
use crate::ops::randn;
use crate::tensor::{Tensor, TensorInner};

/// `torch.nn.Conv2d(in_channels, out_channels, kernel_size)` (square kernel).
pub struct Conv2d {
    pub weight: Tensor, // (out_c, in_c, kH, kW)
    pub bias: Option<Tensor>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
        seed: u64,
    ) -> Self {
        let fan_in = in_channels * kernel_size * kernel_size;
        let scale = (1.0 / fan_in as f32).sqrt();
        let w = randn(&[out_channels, in_channels, kernel_size, kernel_size], seed, true);
        {
            let mut inner = w.inner.borrow_mut();
            for v in inner.data_mut_dense().iter_mut() {
                *v *= scale;
            }
        }
        let b = if bias {
            let bb = randn(&[out_channels], seed + 1, true);
            {
                let mut inner = bb.inner.borrow_mut();
                for v in inner.data_mut_dense().iter_mut() {
                    *v *= scale;
                }
            }
            Some(bb)
        } else {
            None
        };
        Self {
            weight: w,
            bias: b,
            in_channels,
            out_channels,
            kernel_size,
        }
    }

    pub fn from_params(weight: Tensor, bias: Option<Tensor>) -> Self {
        weight.set_requires_grad(true);
        if let Some(ref b) = bias {
            b.set_requires_grad(true);
        }
        let shape = weight.shape();
        assert_eq!(shape.len(), 4, "Conv2d weight must be 4D");
        Self {
            in_channels: shape[1],
            out_channels: shape[0],
            kernel_size: shape[2],
            weight,
            bias,
        }
    }
}

/// Forward conv2d NCHW, stride 1, pad 0.
pub fn conv2d_forward(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> Tensor {
    let xi = input.inner.borrow();
    let wi = weight.inner.borrow();
    let xd = xi.dense_data();
    let wd = wi.dense_data();
    assert_eq!(xi.shape.len(), 4, "conv2d input NCHW");
    assert_eq!(wi.shape.len(), 4, "conv2d weight OIHW");
    let (n, cin, h, w) = (xi.shape[0], xi.shape[1], xi.shape[2], xi.shape[3]);
    let (cout, cin_w, kh, kw) = (wi.shape[0], wi.shape[1], wi.shape[2], wi.shape[3]);
    assert_eq!(cin, cin_w);
    assert!(h >= kh && w >= kw, "conv2d: spatial too small");
    let oh = h - kh + 1;
    let ow = w - kw + 1;
    let mut data = vec![0.0f32; n * cout * oh * ow];
    let bias_d = bias.map(|b| b.data());
    for ni in 0..n {
        for oc in 0..cout {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = 0.0f32;
                    for ic in 0..cin {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iv = xd[((ni * cin + ic) * h + (oy + ky)) * w + (ox + kx)];
                                let wv = wd[((oc * cin + ic) * kh + ky) * kw + kx];
                                acc += iv * wv;
                            }
                        }
                    }
                    if let Some(ref bd) = bias_d {
                        acc += bd[oc];
                    }
                    data[((ni * cout + oc) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    drop((xi, wi));
    let rg = is_grad_enabled()
        && (input.requires_grad()
            || weight.requires_grad()
            || bias.map(|b| b.requires_grad()).unwrap_or(false));
    let gf = if rg {
        Some(GradFn::Conv2d {
            input: input.clone(),
            weight: weight.clone(),
            bias: bias.cloned(),
        })
    } else {
        None
    };
    let shape = vec![n, cout, oh, ow];
    let numel = n * cout * oh * ow;
    Tensor::from_inner(TensorInner::new_contiguous(
        data,
        shape,
        Device::Cpu,
        Dtype::Float32,
        rg,
        if rg { Some(vec![0.0; numel]) } else { None },
        gf,
    ))
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Tensor {
        conv2d_forward(input, &self.weight, self.bias.as_ref())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            ps.push(b.clone());
        }
        ps
    }
}
