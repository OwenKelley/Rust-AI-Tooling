//! Nested / jagged tensors (`torch.nested`).

use crate::ops::full;
use crate::tensor::Tensor;

/// Jagged batch of same-rank tensors (1D list), mirroring a common `nested_tensor` use.
#[derive(Clone, Debug)]
pub struct NestedTensor {
    pub tensors: Vec<Tensor>,
}

/// `torch.nested.nested_tensor(tensors)` — all tensors must share rank and dtype.
pub fn nested_tensor(tensors: Vec<Tensor>) -> NestedTensor {
    assert!(!tensors.is_empty(), "nested_tensor: empty list");
    let rank = tensors[0].ndim();
    let dtype = tensors[0].dtype();
    for (i, t) in tensors.iter().enumerate() {
        assert_eq!(t.ndim(), rank, "nested_tensor: rank mismatch at {i}");
        assert_eq!(t.dtype(), dtype, "nested_tensor: dtype mismatch at {i}");
    }
    NestedTensor { tensors }
}

impl NestedTensor {
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// `nested.unbind()` — constituent tensors.
    pub fn unbind(&self) -> Vec<Tensor> {
        self.tensors.clone()
    }

    /// Parity checksum: sum of constituent checksums + length.
    pub fn checksum(&self) -> f64 {
        self.len() as f64 + self.tensors.iter().map(|t| t.checksum()).sum::<f64>()
    }

    /// `torch.nested.to_padded_tensor(nt, padding)` — pad along dim 0 of each 1D component,
    /// or along the first dim for higher rank, producing a dense batch tensor.
    ///
    /// For 1D constituents of lengths `L_i`, result shape is `[N, max_L]`.
    /// For ND constituents with shape `[L_i, D...]`, result is `[N, max_L, D...]`.
    pub fn to_padded_tensor(&self, padding_value: f32) -> Tensor {
        assert!(!self.tensors.is_empty());
        let n = self.tensors.len();
        let t0 = &self.tensors[0];
        let rank = t0.ndim();
        assert!(rank >= 1, "to_padded_tensor: need rank >= 1");

        let max_l = self
            .tensors
            .iter()
            .map(|t| t.shape()[0])
            .max()
            .unwrap();

        let mut out_shape = vec![n, max_l];
        if rank > 1 {
            out_shape.extend_from_slice(&t0.shape()[1..]);
        }
        for t in &self.tensors {
            assert_eq!(
                &t.shape()[1..],
                &out_shape[2..],
                "to_padded_tensor: trailing shape mismatch"
            );
        }

        let out = full(&out_shape, padding_value, false);
        let trail: usize = if rank == 1 {
            1
        } else {
            out_shape[2..].iter().product()
        };

        {
            let mut oi = out.inner.borrow_mut();
            oi.make_contiguous_unique();
            let mut dst = oi.data_mut_dense();
            for (i, t) in self.tensors.iter().enumerate() {
                let li = t.shape()[0];
                let src = t.inner.borrow().dense_data();
                for r in 0..li {
                    let so = r * trail;
                    let doff = (i * max_l + r) * trail;
                    dst[doff..doff + trail].copy_from_slice(&src[so..so + trail]);
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn nested_pad_1d() {
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2], false);
        let b = Tensor::from_vec(vec![3.0], &[1], false);
        let nt = nested_tensor(vec![a, b]);
        let p = nt.to_padded_tensor(0.0);
        assert_eq!(p.shape(), vec![2, 2]);
        assert_eq!(p.data(), vec![1.0, 2.0, 3.0, 0.0]);
    }
}
