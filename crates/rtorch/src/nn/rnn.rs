//! `nn.GRU` — single-layer, unidirectional, `batch_first=true`.

use crate::functional::{linear, sigmoid, tanh};
use crate::nn::Module;
use crate::ops::{add, chunk, index_select, mul, reshape, seeded_uniform, stack, sub, zeros};
use crate::tensor::Tensor;

/// `torch.nn.GRU(input_size, hidden_size, batch_first=True)`
pub struct GRU {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor, // (3H, I)
    pub weight_hh: Tensor, // (3H, H)
    pub bias_ih: Tensor,   // (3H,)
    pub bias_hh: Tensor,   // (3H,)
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let scale = (1.0 / hidden_size as f32).sqrt();
        let weight_ih = seeded_uniform(&[3 * hidden_size, input_size], seed, -scale, scale);
        let weight_hh = seeded_uniform(&[3 * hidden_size, hidden_size], seed + 1, -scale, scale);
        let bias_ih = seeded_uniform(&[3 * hidden_size], seed + 2, -scale, scale);
        let bias_hh = seeded_uniform(&[3 * hidden_size], seed + 3, -scale, scale);
        weight_ih.set_requires_grad(true);
        weight_hh.set_requires_grad(true);
        bias_ih.set_requires_grad(true);
        bias_hh.set_requires_grad(true);
        Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    pub fn from_params(
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Self {
        let hidden_size = bias_ih.numel() / 3;
        let input_size = weight_ih.shape()[1];
        assert_eq!(weight_ih.shape()[0], 3 * hidden_size);
        assert_eq!(weight_hh.shape(), vec![3 * hidden_size, hidden_size]);
        assert_eq!(bias_hh.numel(), 3 * hidden_size);
        weight_ih.set_requires_grad(true);
        weight_hh.set_requires_grad(true);
        bias_ih.set_requires_grad(true);
        bias_hh.set_requires_grad(true);
        Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    fn cell(&self, x_t: &Tensor, h_prev: &Tensor) -> Tensor {
        let gi = linear(x_t, &self.weight_ih, Some(&self.bias_ih));
        let gh = linear(h_prev, &self.weight_hh, Some(&self.bias_hh));
        let gi_c = chunk(&gi, 3, 1);
        let gh_c = chunk(&gh, 3, 1);
        let r = sigmoid(&add(&gi_c[0], &gh_c[0]));
        let z = sigmoid(&add(&gi_c[1], &gh_c[1]));
        let n = tanh(&add(&gi_c[2], &mul(&r, &gh_c[2])));
        let one = crate::ops::ones(&z.shape(), false);
        let h_t = add(&mul(&sub(&one, &z), &n), &mul(&z, h_prev));
        h_t
    }

    /// `input`: `(N, T, I)` → `(output (N,T,H), h_n (1,N,H))`.
    pub fn forward_seq(&self, input: &Tensor, h0: Option<&Tensor>) -> (Tensor, Tensor) {
        assert_eq!(input.ndim(), 3, "GRU: batch_first (N,T,I)");
        let shape = input.shape();
        let (n, t_len, i) = (shape[0], shape[1], shape[2]);
        assert_eq!(i, self.input_size);
        let mut h = match h0 {
            Some(h) => {
                assert_eq!(h.shape(), vec![n, self.hidden_size]);
                h.clone()
            }
            None => zeros(&[n, self.hidden_size], false),
        };
        let mut outs = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let xt = index_select(input, 1, &[t]);
            let xt = reshape(&xt, &[n, self.input_size]);
            h = self.cell(&xt, &h);
            outs.push(h.clone());
        }
        let out_refs: Vec<&Tensor> = outs.iter().collect();
        let output = stack(&out_refs, 1);
        let h_n = reshape(&h, &[1, n, self.hidden_size]);
        (output, h_n)
    }

    pub fn named_parameters(&self) -> Vec<(&str, Tensor)> {
        vec![
            ("weight_ih_l0", self.weight_ih.clone()),
            ("weight_hh_l0", self.weight_hh.clone()),
            ("bias_ih_l0", self.bias_ih.clone()),
            ("bias_hh_l0", self.bias_hh.clone()),
        ]
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_seq(input, None).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }
}

/// `torch.nn.LSTM(input_size, hidden_size, batch_first=True)` — 1-layer, unidirectional.
pub struct LSTM {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weight_ih: Tensor, // (4H, I)
    pub weight_hh: Tensor, // (4H, H)
    pub bias_ih: Tensor,   // (4H,)
    pub bias_hh: Tensor,   // (4H,)
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, seed: u64) -> Self {
        let scale = (1.0 / hidden_size as f32).sqrt();
        let weight_ih = seeded_uniform(&[4 * hidden_size, input_size], seed, -scale, scale);
        let weight_hh = seeded_uniform(&[4 * hidden_size, hidden_size], seed + 1, -scale, scale);
        let bias_ih = seeded_uniform(&[4 * hidden_size], seed + 2, -scale, scale);
        let bias_hh = seeded_uniform(&[4 * hidden_size], seed + 3, -scale, scale);
        weight_ih.set_requires_grad(true);
        weight_hh.set_requires_grad(true);
        bias_ih.set_requires_grad(true);
        bias_hh.set_requires_grad(true);
        Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    pub fn from_params(
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
    ) -> Self {
        let hidden_size = bias_ih.numel() / 4;
        let input_size = weight_ih.shape()[1];
        assert_eq!(weight_ih.shape()[0], 4 * hidden_size);
        assert_eq!(weight_hh.shape(), vec![4 * hidden_size, hidden_size]);
        assert_eq!(bias_hh.numel(), 4 * hidden_size);
        weight_ih.set_requires_grad(true);
        weight_hh.set_requires_grad(true);
        bias_ih.set_requires_grad(true);
        bias_hh.set_requires_grad(true);
        Self {
            input_size,
            hidden_size,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
        }
    }

    fn cell(&self, x_t: &Tensor, h_prev: &Tensor, c_prev: &Tensor) -> (Tensor, Tensor) {
        let gi = linear(x_t, &self.weight_ih, Some(&self.bias_ih));
        let gh = linear(h_prev, &self.weight_hh, Some(&self.bias_hh));
        let gates = add(&gi, &gh);
        let parts = chunk(&gates, 4, 1);
        let i = sigmoid(&parts[0]);
        let f = sigmoid(&parts[1]);
        let g = tanh(&parts[2]);
        let o = sigmoid(&parts[3]);
        let c_t = add(&mul(&f, c_prev), &mul(&i, &g));
        let h_t = mul(&o, &tanh(&c_t));
        (h_t, c_t)
    }

    /// `input`: `(N, T, I)` → `(output (N,T,H), (h_n (1,N,H), c_n (1,N,H)))`.
    pub fn forward_seq(
        &self,
        input: &Tensor,
        hx: Option<(&Tensor, &Tensor)>,
    ) -> (Tensor, (Tensor, Tensor)) {
        assert_eq!(input.ndim(), 3, "LSTM: batch_first (N,T,I)");
        let shape = input.shape();
        let (n, t_len, i) = (shape[0], shape[1], shape[2]);
        assert_eq!(i, self.input_size);
        let (mut h, mut c) = match hx {
            Some((h0, c0)) => {
                assert_eq!(h0.shape(), vec![n, self.hidden_size]);
                assert_eq!(c0.shape(), vec![n, self.hidden_size]);
                (h0.clone(), c0.clone())
            }
            None => (
                zeros(&[n, self.hidden_size], false),
                zeros(&[n, self.hidden_size], false),
            ),
        };
        let mut outs = Vec::with_capacity(t_len);
        for t in 0..t_len {
            let xt = index_select(input, 1, &[t]);
            let xt = reshape(&xt, &[n, self.input_size]);
            let (hn, cn) = self.cell(&xt, &h, &c);
            h = hn;
            c = cn;
            outs.push(h.clone());
        }
        let out_refs: Vec<&Tensor> = outs.iter().collect();
        let output = stack(&out_refs, 1);
        let h_n = reshape(&h, &[1, n, self.hidden_size]);
        let c_n = reshape(&c, &[1, n, self.hidden_size]);
        (output, (h_n, c_n))
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward_seq(input, None).0
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.weight_ih.clone(),
            self.weight_hh.clone(),
            self.bias_ih.clone(),
            self.bias_hh.clone(),
        ]
    }
}
