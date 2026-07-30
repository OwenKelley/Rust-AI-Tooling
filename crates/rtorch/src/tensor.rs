//! Contiguous f32 tensor with optional autograd metadata.

use std::cell::RefCell;
use std::rc::Rc;

use crate::autograd::GradFn;

pub type TensorRef = Rc<RefCell<TensorInner>>;

/// Shared mutable tensor storage.
#[derive(Debug)]
pub struct TensorInner {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
    pub grad_fn: Option<GradFn>,
}

impl TensorInner {
    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            let n = self.numel();
            self.grad = Some(vec![0.0; n]);
        }
    }

    pub fn accumulate_grad(&mut self, g: &[f32]) {
        assert_eq!(g.len(), self.numel(), "grad size mismatch");
        if !self.requires_grad {
            return;
        }
        match &mut self.grad {
            Some(existing) => {
                for (e, &v) in existing.iter_mut().zip(g.iter()) {
                    *e += v;
                }
            }
            None => {
                self.grad = Some(g.to_vec());
            }
        }
    }
}

/// `torch.Tensor` analogue (CPU f32).
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) inner: TensorRef,
}

impl Tensor {
    pub(crate) fn from_inner(inner: TensorInner) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Self {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(data.len(), expected, "data length != shape product");
        Self::from_inner(TensorInner {
            data,
            shape: shape.to_vec(),
            requires_grad,
            grad: if requires_grad {
                Some(vec![0.0; expected])
            } else {
                None
            },
            grad_fn: None,
        })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    pub fn ndim(&self) -> usize {
        self.inner.borrow().shape.len()
    }

    pub fn numel(&self) -> usize {
        self.inner.borrow().numel()
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Borrow data without cloning (preferred in hot paths).
    pub fn with_data<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        f(&self.inner.borrow().data)
    }

    /// Accumulate gradient from another tensor's data without an extra clone.
    pub(crate) fn accumulate_from(&self, src: &Tensor) {
        let g = src.inner.borrow();
        self.inner.borrow_mut().accumulate_grad(&g.data);
    }

    pub fn set_requires_grad(&self, flag: bool) {
        let mut t = self.inner.borrow_mut();
        t.requires_grad = flag;
        if flag && t.grad.is_none() {
            let n = t.numel();
            t.grad = Some(vec![0.0; n]);
        }
        if !flag {
            t.grad = None;
            t.grad_fn = None;
        }
    }

    /// Clones storage; prefer [`Self::with_data`] in hot paths.
    pub fn data(&self) -> Vec<f32> {
        self.inner.borrow().data.clone()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.borrow().grad.clone()
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().zero_grad();
    }

    /// Scalar checksum for parity (sum of finite values).
    pub fn checksum(&self) -> f64 {
        self.inner
            .borrow()
            .data
            .iter()
            .map(|&x| if x.is_finite() { x as f64 } else { 0.0 })
            .sum()
    }

    /// Grad checksum (0 if no grad).
    pub fn grad_checksum(&self) -> f64 {
        match &self.inner.borrow().grad {
            Some(g) => g
                .iter()
                .map(|&x| if x.is_finite() { x as f64 } else { 0.0 })
                .sum(),
            None => 0.0,
        }
    }

    pub fn item(&self) -> f32 {
        let t = self.inner.borrow();
        assert_eq!(t.numel(), 1, "item: tensor must have one element");
        t.data[0]
    }

    /// Detach: new tensor sharing data values without grad history.
    pub fn detach(&self) -> Tensor {
        let t = self.inner.borrow();
        Self::from_vec(t.data.clone(), &t.shape, false)
    }

    /// Overwrite storage from a contiguous `src` (same numel). Leaf only.
    pub fn copy_from_slice(&self, src: &[f32]) {
        let mut t = self.inner.borrow_mut();
        assert!(t.grad_fn.is_none(), "copy_from_slice: leaf only");
        assert_eq!(src.len(), t.data.len());
        t.data.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_shape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert_eq!(t.shape(), vec![2, 2]);
        assert!((t.checksum() - 10.0).abs() < 1e-6);
    }
}
