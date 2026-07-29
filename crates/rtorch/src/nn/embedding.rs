//! `nn.Embedding`

use crate::autograd::GradFn;
use crate::context::is_grad_enabled;
use crate::nn::Module;
use crate::ops::randn;
use crate::tensor::{Tensor, TensorInner};

/// `torch.nn.Embedding(num_embeddings, embedding_dim)`
pub struct Embedding {
    pub weight: Tensor, // (num_embeddings, embedding_dim)
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize, seed: u64) -> Self {
        let w = randn(&[num_embeddings, embedding_dim], seed, true);
        {
            let mut inner = w.inner.borrow_mut();
            let scale = (1.0 / embedding_dim as f32).sqrt();
            for v in inner.data.iter_mut() {
                *v *= scale;
            }
        }
        Self { weight: w }
    }

    pub fn from_params(weight: Tensor) -> Self {
        weight.set_requires_grad(true);
        Self { weight }
    }

    /// Lookup rows: `indices` length N → output `(N, embedding_dim)`.
    pub fn forward_indices(&self, indices: &[usize]) -> Tensor {
        let w = self.weight.inner.borrow();
        let num_emb = w.shape[0];
        let dim = w.shape[1];
        let n = indices.len();
        let mut data = vec![0.0f32; n * dim];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < num_emb, "Embedding: index {idx} >= {num_emb}");
            let src = &w.data[idx * dim..(idx + 1) * dim];
            data[i * dim..(i + 1) * dim].copy_from_slice(src);
        }
        drop(w);
        let rg = is_grad_enabled() && self.weight.requires_grad();
        let gf = if rg {
            Some(GradFn::Embedding {
                weight: self.weight.clone(),
                indices: indices.to_vec(),
            })
        } else {
            None
        };
        let shape = vec![n, dim];
        let numel = n * dim;
        Tensor::from_inner(TensorInner {
            data,
            shape,
            requires_grad: rg,
            grad: if rg { Some(vec![0.0; numel]) } else { None },
            grad_fn: gf,
        })
    }
}

impl Module for Embedding {
    fn forward(&self, _input: &Tensor) -> Tensor {
        panic!("Embedding::forward: use forward_indices(&[usize])");
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
