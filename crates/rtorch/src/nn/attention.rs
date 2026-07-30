//! `nn.MultiheadAttention` — batch_first, no bias_kv / add_zero_attn.

use std::rc::Rc;

use crate::functional::{linear, scaled_dot_product_attention_masked};
use crate::nn::{Linear, Module};
use crate::ops::{chunk, permute, reshape, seeded_uniform};
use crate::tensor::Tensor;

/// `torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)`.
pub struct MultiheadAttention {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub in_proj_weight: Tensor, // (3E, E)
    pub in_proj_bias: Tensor,   // (3E,)
    pub out_proj: Linear,
}

fn linear_last(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    let shape = x.shape();
    if shape.len() == 2 {
        return linear(x, weight, bias);
    }
    assert_eq!(shape.len(), 3, "linear_last: 2D or 3D");
    let (n, l, e) = (shape[0], shape[1], shape[2]);
    assert_eq!(e, weight.shape()[1]);
    let out_f = weight.shape()[0];
    let flat = reshape(x, &[n * l, e]);
    let y = linear(&flat, weight, bias);
    reshape(&y, &[n, l, out_f])
}

impl MultiheadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, seed: u64) -> Self {
        assert!(embed_dim % num_heads == 0);
        let head_dim = embed_dim / num_heads;
        let scale = (1.0 / embed_dim as f32).sqrt();
        let in_proj_weight = seeded_uniform(&[3 * embed_dim, embed_dim], seed, -scale, scale);
        let in_proj_bias = seeded_uniform(&[3 * embed_dim], seed + 1, -scale, scale);
        in_proj_weight.set_requires_grad(true);
        in_proj_bias.set_requires_grad(true);
        let out_proj = Linear::from_params(
            seeded_uniform(&[embed_dim, embed_dim], seed + 2, -scale, scale),
            Some(seeded_uniform(&[embed_dim], seed + 3, -scale, scale)),
        );
        Self {
            embed_dim,
            num_heads,
            head_dim,
            in_proj_weight,
            in_proj_bias,
            out_proj,
        }
    }

    pub fn from_params(
        in_proj_weight: Tensor,
        in_proj_bias: Tensor,
        out_proj_weight: Tensor,
        out_proj_bias: Tensor,
        num_heads: usize,
    ) -> Self {
        let embed_dim = in_proj_weight.shape()[1];
        assert_eq!(in_proj_weight.shape()[0], 3 * embed_dim);
        assert_eq!(in_proj_bias.numel(), 3 * embed_dim);
        assert!(embed_dim % num_heads == 0);
        in_proj_weight.set_requires_grad(true);
        in_proj_bias.set_requires_grad(true);
        let out_proj = Linear::from_params(out_proj_weight, Some(out_proj_bias));
        Self {
            embed_dim,
            num_heads,
            head_dim: embed_dim / num_heads,
            in_proj_weight,
            in_proj_bias,
            out_proj,
        }
    }

    fn project_qkv(&self, x: &Tensor) -> (Tensor, Tensor, Tensor) {
        let proj = linear_last(x, &self.in_proj_weight, Some(&self.in_proj_bias));
        let parts = chunk(&proj, 3, 2);
        (parts[0].clone(), parts[1].clone(), parts[2].clone())
    }

    fn to_heads(&self, x: &Tensor, n: usize, l: usize) -> Tensor {
        let t = reshape(x, &[n, l, self.num_heads, self.head_dim]);
        permute(&t, &[0, 2, 1, 3])
    }

    fn from_heads(&self, x: &Tensor, n: usize, l: usize) -> Tensor {
        let t = permute(x, &[0, 2, 1, 3]);
        reshape(&t, &[n, l, self.embed_dim])
    }

    /// Self- or cross-attention. Returns `(output, None)` (attn weights omitted).
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
    ) -> (Tensor, Option<Tensor>) {
        self.forward_qkv_masked(query, key, value, None)
    }

    /// Like `forward_qkv` with optional float additive `attn_mask` of shape `(Lq, Lk)`.
    pub fn forward_qkv_masked(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> (Tensor, Option<Tensor>) {
        assert_eq!(query.ndim(), 3);
        assert_eq!(key.ndim(), 3);
        assert_eq!(value.ndim(), 3);
        let (n, lq, e) = (query.shape()[0], query.shape()[1], query.shape()[2]);
        let lk = key.shape()[1];
        assert_eq!(e, self.embed_dim);
        assert_eq!(key.shape()[2], self.embed_dim);
        assert_eq!(value.shape()[2], self.embed_dim);
        assert_eq!(key.shape()[0], n);
        assert_eq!(value.shape()[0], n);
        assert_eq!(value.shape()[1], lk);

        let (q, k, v) = if Rc::ptr_eq(&query.inner, &key.inner)
            && Rc::ptr_eq(&key.inner, &value.inner)
        {
            self.project_qkv(query)
        } else {
            let w = chunk(&self.in_proj_weight, 3, 0);
            let b = chunk(&self.in_proj_bias, 3, 0);
            let q = linear_last(query, &w[0], Some(&b[0]));
            let k = linear_last(key, &w[1], Some(&b[1]));
            let v = linear_last(value, &w[2], Some(&b[2]));
            (q, k, v)
        };

        let qh = self.to_heads(&q, n, lq);
        let kh = self.to_heads(&k, n, lk);
        let vh = self.to_heads(&v, n, lk);
        let q2 = reshape(&qh, &[n * self.num_heads, lq, self.head_dim]);
        let k2 = reshape(&kh, &[n * self.num_heads, lk, self.head_dim]);
        let v2 = reshape(&vh, &[n * self.num_heads, lk, self.head_dim]);
        let ctx = scaled_dot_product_attention_masked(&q2, &k2, &v2, attn_mask);
        let ctx = reshape(&ctx, &[n, self.num_heads, lq, self.head_dim]);
        let flat = self.from_heads(&ctx, n, lq);
        let out = linear_last(&flat, &self.out_proj.weight, self.out_proj.bias.as_ref());
        (out, None)
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_qkv(input, input, input).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = vec![self.in_proj_weight.clone(), self.in_proj_bias.clone()];
        ps.extend(self.out_proj.parameters());
        ps
    }
}
