//! `nn.TransformerEncoder*` / `nn.TransformerDecoder*` — batch_first, dropout=0 path.

use crate::functional::{gelu, relu};
use crate::nn::{LayerNorm, Linear, Module, MultiheadAttention};
use crate::ops::{add, reshape, seeded_uniform};
use crate::tensor::Tensor;

fn linear_last(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Tensor {
    use crate::functional::linear;
    let shape = x.shape();
    if shape.len() == 2 {
        return linear(x, weight, bias);
    }
    assert_eq!(shape.len(), 3);
    let (n, l, e) = (shape[0], shape[1], shape[2]);
    let out_f = weight.shape()[0];
    let flat = reshape(x, &[n * l, e]);
    let y = linear(&flat, weight, bias);
    reshape(&y, &[n, l, out_f])
}

/// `nn.Transformer.generate_square_subsequent_mask(sz)` — float additive causal mask.
/// Allowed positions are `0`; future positions are a large negative (`-1e9`).
pub fn generate_square_subsequent_mask(sz: usize) -> Tensor {
    let mut data = vec![0.0f32; sz * sz];
    for i in 0..sz {
        for j in (i + 1)..sz {
            data[i * sz + j] = -1e9;
        }
    }
    Tensor::from_vec(data, &[sz, sz], false)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformerActivation {
    Relu,
    Gelu,
}

/// `torch.nn.TransformerEncoderLayer(..., batch_first=True, norm_first=False)`.
pub struct TransformerEncoderLayer {
    pub self_attn: MultiheadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub activation: TransformerActivation,
    pub d_model: usize,
    pub dim_feedforward: usize,
}

impl TransformerEncoderLayer {
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        activation: TransformerActivation,
        seed: u64,
    ) -> Self {
        let self_attn = MultiheadAttention::new(d_model, nhead, seed);
        let scale = (1.0 / d_model as f32).sqrt();
        let linear1 = Linear::from_params(
            seeded_uniform(&[dim_feedforward, d_model], seed + 10, -scale, scale),
            Some(seeded_uniform(&[dim_feedforward], seed + 11, -scale, scale)),
        );
        let linear2 = Linear::from_params(
            seeded_uniform(&[d_model, dim_feedforward], seed + 12, -scale, scale),
            Some(seeded_uniform(&[d_model], seed + 13, -scale, scale)),
        );
        Self {
            self_attn,
            linear1,
            linear2,
            norm1: LayerNorm::new(d_model, 1e-5),
            norm2: LayerNorm::new(d_model, 1e-5),
            activation,
            d_model,
            dim_feedforward,
        }
    }

    pub fn from_parts(
        self_attn: MultiheadAttention,
        linear1: Linear,
        linear2: Linear,
        norm1: LayerNorm,
        norm2: LayerNorm,
        activation: TransformerActivation,
    ) -> Self {
        let d_model = self_attn.embed_dim;
        let dim_feedforward = linear1.weight.shape()[0];
        Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            activation,
            d_model,
            dim_feedforward,
        }
    }

    fn ff(&self, x: &Tensor) -> Tensor {
        let h = linear_last(x, &self.linear1.weight, self.linear1.bias.as_ref());
        let h = match self.activation {
            TransformerActivation::Relu => relu(&h),
            TransformerActivation::Gelu => gelu(&h),
        };
        linear_last(&h, &self.linear2.weight, self.linear2.bias.as_ref())
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, src: &Tensor) -> Tensor {
        // Post-norm (norm_first=False), dropout=0:
        // x = norm1(x + self_attn(x)); x = norm2(x + ff(x))
        let attn = self.self_attn.forward(src);
        let x = self.norm1.forward(&add(src, &attn));
        let ff = self.ff(&x);
        self.norm2.forward(&add(&x, &ff))
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.self_attn.parameters();
        ps.extend(self.linear1.parameters());
        ps.extend(self.linear2.parameters());
        ps.extend(self.norm1.parameters());
        ps.extend(self.norm2.parameters());
        ps
    }
}

/// `torch.nn.TransformerEncoder(layer, num_layers)` without final norm.
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
}

impl TransformerEncoder {
    pub fn new(layer: TransformerEncoderLayer, num_layers: usize) -> Self {
        assert!(num_layers >= 1);
        let mut layers = Vec::with_capacity(num_layers);
        layers.push(layer);
        let d_model = layers[0].d_model;
        let nhead = layers[0].self_attn.num_heads;
        let dim_ff = layers[0].dim_feedforward;
        let act = layers[0].activation;
        for i in 1..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                nhead,
                dim_ff,
                act,
                1000 + i as u64 * 17,
            ));
        }
        Self { layers }
    }

    pub fn from_layers(layers: Vec<TransformerEncoderLayer>) -> Self {
        assert!(!layers.is_empty());
        Self { layers }
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, src: &Tensor) -> Tensor {
        let mut x = src.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = Vec::new();
        for layer in &self.layers {
            ps.extend(layer.parameters());
        }
        ps
    }
}

/// `torch.nn.TransformerDecoderLayer(..., batch_first=True, norm_first=False)`.
pub struct TransformerDecoderLayer {
    pub self_attn: MultiheadAttention,
    pub multihead_attn: MultiheadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub norm3: LayerNorm,
    pub activation: TransformerActivation,
    pub d_model: usize,
    pub dim_feedforward: usize,
}

impl TransformerDecoderLayer {
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        activation: TransformerActivation,
        seed: u64,
    ) -> Self {
        let self_attn = MultiheadAttention::new(d_model, nhead, seed);
        let multihead_attn = MultiheadAttention::new(d_model, nhead, seed + 50);
        let scale = (1.0 / d_model as f32).sqrt();
        let linear1 = Linear::from_params(
            seeded_uniform(&[dim_feedforward, d_model], seed + 10, -scale, scale),
            Some(seeded_uniform(&[dim_feedforward], seed + 11, -scale, scale)),
        );
        let linear2 = Linear::from_params(
            seeded_uniform(&[d_model, dim_feedforward], seed + 12, -scale, scale),
            Some(seeded_uniform(&[d_model], seed + 13, -scale, scale)),
        );
        Self {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1: LayerNorm::new(d_model, 1e-5),
            norm2: LayerNorm::new(d_model, 1e-5),
            norm3: LayerNorm::new(d_model, 1e-5),
            activation,
            d_model,
            dim_feedforward,
        }
    }

    pub fn from_parts(
        self_attn: MultiheadAttention,
        multihead_attn: MultiheadAttention,
        linear1: Linear,
        linear2: Linear,
        norm1: LayerNorm,
        norm2: LayerNorm,
        norm3: LayerNorm,
        activation: TransformerActivation,
    ) -> Self {
        let d_model = self_attn.embed_dim;
        let dim_feedforward = linear1.weight.shape()[0];
        Self {
            self_attn,
            multihead_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            activation,
            d_model,
            dim_feedforward,
        }
    }

    fn ff(&self, x: &Tensor) -> Tensor {
        let h = linear_last(x, &self.linear1.weight, self.linear1.bias.as_ref());
        let h = match self.activation {
            TransformerActivation::Relu => relu(&h),
            TransformerActivation::Gelu => gelu(&h),
        };
        linear_last(&h, &self.linear2.weight, self.linear2.bias.as_ref())
    }

    /// Post-norm decoder layer. `tgt_mask` is optional float additive `(Lt, Lt)`.
    pub fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
    ) -> Tensor {
        let sa = self
            .self_attn
            .forward_qkv_masked(tgt, tgt, tgt, tgt_mask)
            .0;
        let x = self.norm1.forward(&add(tgt, &sa));
        let ca = self
            .multihead_attn
            .forward_qkv_masked(&x, memory, memory, None)
            .0;
        let x = self.norm2.forward(&add(&x, &ca));
        let ff = self.ff(&x);
        self.norm3.forward(&add(&x, &ff))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.self_attn.parameters();
        ps.extend(self.multihead_attn.parameters());
        ps.extend(self.linear1.parameters());
        ps.extend(self.linear2.parameters());
        ps.extend(self.norm1.parameters());
        ps.extend(self.norm2.parameters());
        ps.extend(self.norm3.parameters());
        ps
    }
}

/// `torch.nn.TransformerDecoder(layer, num_layers)` without final norm.
pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderLayer>,
}

impl TransformerDecoder {
    pub fn from_layers(layers: Vec<TransformerDecoderLayer>) -> Self {
        assert!(!layers.is_empty());
        Self { layers }
    }

    pub fn forward(
        &self,
        tgt: &Tensor,
        memory: &Tensor,
        tgt_mask: Option<&Tensor>,
    ) -> Tensor {
        let mut x = tgt.clone();
        for layer in &self.layers {
            x = layer.forward(&x, memory, tgt_mask);
        }
        x
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut ps = Vec::new();
        for layer in &self.layers {
            ps.extend(layer.parameters());
        }
        ps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::seeded_uniform;

    #[test]
    fn encoder_layer_preserves_shape() {
        let layer = TransformerEncoderLayer::new(8, 2, 16, TransformerActivation::Relu, 7);
        let x = seeded_uniform(&[2, 4, 8], 1, -1.0, 1.0);
        let y = layer.forward(&x);
        assert_eq!(y.shape(), &[2, 4, 8]);
        assert!(y.checksum().is_finite());
    }

    #[test]
    fn encoder_stack_preserves_shape() {
        let enc = TransformerEncoder::from_layers(vec![
            TransformerEncoderLayer::new(8, 2, 16, TransformerActivation::Gelu, 3),
            TransformerEncoderLayer::new(8, 2, 16, TransformerActivation::Gelu, 9),
        ]);
        let x = seeded_uniform(&[1, 3, 8], 2, -1.0, 1.0);
        let y = enc.forward(&x);
        assert_eq!(y.shape(), &[1, 3, 8]);
    }

    #[test]
    fn causal_mask_upper_blocked() {
        let m = generate_square_subsequent_mask(3);
        let d = m.inner.borrow().data.clone();
        assert_eq!(d[0], 0.0);
        assert!(d[1] < -1e8);
        assert!(d[2] < -1e8);
        assert_eq!(d[3], 0.0); // (1,0)
        assert_eq!(d[4], 0.0); // (1,1)
        assert!(d[5] < -1e8); // (1,2)
    }

    #[test]
    fn decoder_layer_preserves_shape() {
        let layer = TransformerDecoderLayer::new(8, 2, 16, TransformerActivation::Relu, 5);
        let tgt = seeded_uniform(&[2, 3, 8], 1, -1.0, 1.0);
        let mem = seeded_uniform(&[2, 4, 8], 2, -1.0, 1.0);
        let mask = generate_square_subsequent_mask(3);
        let y = layer.forward(&tgt, &mem, Some(&mask));
        assert_eq!(y.shape(), &[2, 3, 8]);
        assert!(y.checksum().is_finite());
    }
}
