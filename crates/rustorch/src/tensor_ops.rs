//! Operator overloads and PyTorch-like methods on [`Tensor`].
//!
//! Thin wrappers over [`crate::ops`] / free functions so autograd wiring is unchanged.
//! Low-risk set: arithmetic `+ - * /` and `-`, in-place `+= -= *=`, plus common methods.
//!
//! Scalar–tensor ops, indexing, and `*` as matmul are intentionally not included.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::ops::{
    abs, add, add_, bmm, div, exp, log, matmul, mean, mul, mul_, neg, pow, reshape, sub, sub_, sum,
    transpose,
};
use crate::tensor::Tensor;

// --- binary arithmetic: all ownership combos forward to free functions -----------

macro_rules! impl_bin_op {
    ($trait:ident, $method:ident, $fn:path) => {
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: Tensor) -> Tensor {
                $fn(&self, &rhs)
            }
        }
        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: &Tensor) -> Tensor {
                $fn(&self, rhs)
            }
        }
        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: Tensor) -> Tensor {
                $fn(self, &rhs)
            }
        }
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            #[inline]
            fn $method(self, rhs: &Tensor) -> Tensor {
                $fn(self, rhs)
            }
        }
    };
}

impl_bin_op!(Add, add, add);
impl_bin_op!(Sub, sub, sub);
impl_bin_op!(Mul, mul, mul);
impl_bin_op!(Div, div, div);

impl Neg for Tensor {
    type Output = Tensor;
    #[inline]
    fn neg(self) -> Tensor {
        neg(&self)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;
    #[inline]
    fn neg(self) -> Tensor {
        neg(self)
    }
}

// --- in-place (same semantics as add_ / sub_ / mul_) ----------------------------

impl AddAssign<Tensor> for Tensor {
    #[inline]
    fn add_assign(&mut self, rhs: Tensor) {
        add_(self, &rhs);
    }
}

impl AddAssign<&Tensor> for Tensor {
    #[inline]
    fn add_assign(&mut self, rhs: &Tensor) {
        add_(self, rhs);
    }
}

impl SubAssign<Tensor> for Tensor {
    #[inline]
    fn sub_assign(&mut self, rhs: Tensor) {
        sub_(self, &rhs);
    }
}

impl SubAssign<&Tensor> for Tensor {
    #[inline]
    fn sub_assign(&mut self, rhs: &Tensor) {
        sub_(self, rhs);
    }
}

impl MulAssign<Tensor> for Tensor {
    #[inline]
    fn mul_assign(&mut self, rhs: Tensor) {
        mul_(self, &rhs);
    }
}

impl MulAssign<&Tensor> for Tensor {
    #[inline]
    fn mul_assign(&mut self, rhs: &Tensor) {
        mul_(self, rhs);
    }
}

// --- convenience methods --------------------------------------------------------

impl Tensor {
    /// `torch.matmul` / `@` (use this; `*` is elementwise).
    #[inline]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        matmul(self, other)
    }

    /// `torch.bmm`
    #[inline]
    pub fn bmm(&self, other: &Tensor) -> Tensor {
        bmm(self, other)
    }

    /// 2D transpose (`torch.t` / `transpose(0, 1)`).
    #[inline]
    pub fn t(&self) -> Tensor {
        transpose(self)
    }

    /// Alias of [`Self::t`].
    #[inline]
    pub fn transpose(&self) -> Tensor {
        transpose(self)
    }

    /// `tensor.reshape(...)`
    #[inline]
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        reshape(self, shape)
    }

    /// `tensor.view(...)` — same as [`Self::reshape`] in this CPU build.
    #[inline]
    pub fn view(&self, shape: &[usize]) -> Tensor {
        reshape(self, shape)
    }

    /// `tensor.sum()`
    #[inline]
    pub fn sum(&self) -> Tensor {
        sum(self)
    }

    /// `tensor.mean()`
    #[inline]
    pub fn mean(&self) -> Tensor {
        mean(self)
    }

    /// `tensor.pow(exponent)`
    #[inline]
    pub fn pow(&self, exponent: &Tensor) -> Tensor {
        pow(self, exponent)
    }

    /// `tensor.abs()`
    #[inline]
    pub fn abs(&self) -> Tensor {
        abs(self)
    }

    /// `tensor.exp()`
    #[inline]
    pub fn exp(&self) -> Tensor {
        exp(self)
    }

    /// `tensor.log()`
    #[inline]
    pub fn log(&self) -> Tensor {
        log(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{full, seeded_uniform};

    #[test]
    fn arithmetic_ops_match_free_fns() {
        let a = seeded_uniform(&[2, 3], 1, -1.0, 1.0);
        let b = seeded_uniform(&[2, 3], 2, -1.0, 1.0);
        assert!(((&a + &b).checksum() - add(&a, &b).checksum()).abs() < 1e-5);
        assert!(((&a - &b).checksum() - sub(&a, &b).checksum()).abs() < 1e-5);
        assert!(((&a * &b).checksum() - mul(&a, &b).checksum()).abs() < 1e-5);
        assert!(((&a / &b).checksum() - div(&a, &b).checksum()).abs() < 1e-5);
        assert!(((-&a).checksum() - neg(&a).checksum()).abs() < 1e-5);
        let _ = a.clone() + b.clone();
        let _ = a.clone() + &b;
        let _ = &a + b.clone();
    }

    #[test]
    fn assign_ops_match_inplace_fns() {
        let b = seeded_uniform(&[2, 2], 3, -0.5, 0.5);
        let mut a1 = seeded_uniform(&[2, 2], 4, -0.5, 0.5);
        let a2 = a1.clone();
        a1 += &b;
        let a2b = a2;
        add_(&a2b, &b);
        // add_ takes &Tensor and mutates via RefCell — same storage path
        assert!((a1.checksum() - a2b.checksum()).abs() < 1e-5);

        let mut m = full(&[2, 2], 2.0, false);
        m *= &full(&[2, 2], 3.0, false);
        assert!((m.checksum() - 24.0).abs() < 1e-5);
        m -= &full(&[2, 2], 1.0, false);
        assert!((m.checksum() - 20.0).abs() < 1e-5);
    }

    #[test]
    fn methods_match_free_fns() {
        let a = seeded_uniform(&[2, 3], 5, -1.0, 1.0);
        let b = seeded_uniform(&[3, 2], 6, -1.0, 1.0);
        assert!((a.matmul(&b).checksum() - matmul(&a, &b).checksum()).abs() < 1e-4);
        assert!((a.t().checksum() - transpose(&a).checksum()).abs() < 1e-5);
        assert!((a.sum().checksum() - sum(&a).checksum()).abs() < 1e-5);
        assert!((a.mean().checksum() - mean(&a).checksum()).abs() < 1e-5);
        let e = full(&a.shape(), 2.0, false);
        assert!((a.pow(&e).checksum() - pow(&a, &e).checksum()).abs() < 1e-4);
        assert!((a.abs().checksum() - abs(&a).checksum()).abs() < 1e-5);
        assert!((a.reshape(&[3, 2]).checksum() - a.view(&[6]).reshape(&[3, 2]).checksum()).abs() < 1e-5);
    }
}
