//! `nn.Linear`

use crate::functional;
use crate::nn::Module;
use crate::ops::randn;
use crate::tensor::Tensor;

/// `torch.nn.Linear(in_features, out_features, bias=True)`
pub struct Linear {
    pub weight: Tensor, // (out, in)
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        // Kaiming-ish uniform scale ~ 1/sqrt(in)
        let scale = (1.0 / in_features as f32).sqrt();
        let w = randn(&[out_features, in_features], seed, true);
        {
            let mut inner = w.inner.borrow_mut();
            for v in inner.data_mut_dense().iter_mut() {
                *v *= scale;
            }
        }
        let b = if bias {
            let bb = randn(&[out_features], seed + 1, true);
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
        Self { weight: w, bias: b }
    }

    /// Build from existing parameter tensors (for parity harnesses).
    pub fn from_params(weight: Tensor, bias: Option<Tensor>) -> Self {
        weight.set_requires_grad(true);
        if let Some(ref b) = bias {
            b.set_requires_grad(true);
        }
        Self { weight, bias }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::linear(input, &self.weight, self.bias.as_ref())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            ps.push(b.clone());
        }
        ps
    }
}
