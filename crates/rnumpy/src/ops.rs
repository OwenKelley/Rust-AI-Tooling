//! Element-wise / ufunc-style ops — mirrors `numpy` arithmetic & math ufuncs.

use crate::NdArray;
use ndarray::Zip;

/// Prefer Zip into a pre-sized buffer (one alloc, tight loop).
fn zip2(a: &NdArray, b: &NdArray, f: impl Fn(f64, f64) -> f64) -> NdArray {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    let mut out = NdArray::zeros(a.raw_dim());
    Zip::from(&mut out)
        .and(a)
        .and(b)
        .for_each(|o, &x, &y| *o = f(x, y));
    out
}

fn map1(a: &NdArray, f: impl Fn(f64) -> f64) -> NdArray {
    let mut out = NdArray::zeros(a.raw_dim());
    Zip::from(&mut out).and(a).for_each(|o, &x| *o = f(x));
    out
}

/// `np.add(a, b)`
pub fn add(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x + y)
}

/// `np.subtract(a, b)`
pub fn subtract(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x - y)
}

/// `np.multiply(a, b)`
pub fn multiply(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x * y)
}

/// `np.divide(a, b)`
pub fn divide(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x / y)
}

/// `np.power(a, b)` element-wise
pub fn power(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, f64::powf)
}

/// `np.sqrt(a)`
pub fn sqrt(a: &NdArray) -> NdArray {
    map1(a, f64::sqrt)
}

/// `np.exp(a)`
pub fn exp(a: &NdArray) -> NdArray {
    map1(a, f64::exp)
}

/// `np.log(a)` natural log
pub fn log(a: &NdArray) -> NdArray {
    map1(a, f64::ln)
}

/// `np.negative(a)`
pub fn negative(a: &NdArray) -> NdArray {
    map1(a, |x| -x)
}

/// `np.abs(a)`
pub fn abs(a: &NdArray) -> NdArray {
    map1(a, f64::abs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{ones, seeded_uniform};
    use approx::assert_abs_diff_eq;

    #[test]
    fn add_ones() {
        let a = ones(&[2, 2]);
        let b = ones(&[2, 2]);
        let c = add(&a, &b);
        assert!(c.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn multiply_scale() {
        let a = seeded_uniform(&[4], 1, 0.0, 1.0);
        let b = ones(&[4]);
        let c = multiply(&a, &b);
        for (x, y) in a.iter().zip(c.iter()) {
            assert_abs_diff_eq!(*x, *y, epsilon = 1e-12);
        }
    }
}
