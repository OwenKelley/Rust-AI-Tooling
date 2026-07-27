//! Linear algebra — mirrors common `numpy` / `np.linalg` entry points.

use crate::NdArray;
use ndarray::{ArrayView1, ArrayView2, Ix1, Ix2};

fn view2(a: &NdArray) -> ArrayView2<'_, f64> {
    assert_eq!(a.ndim(), 2, "expected 2D array");
    a.view().into_dimensionality::<Ix2>().expect("2D")
}

fn view1(a: &NdArray) -> ArrayView1<'_, f64> {
    assert_eq!(a.ndim(), 1, "expected 1D array");
    a.view().into_dimensionality::<Ix1>().expect("1D")
}

/// `np.transpose(a)` / `a.T` for ND.
///
/// Returns an owned row-major copy (NumPy's `transpose` is often a view;
/// materializing matches callers that need contiguous owned data).
pub fn transpose(a: &NdArray) -> NdArray {
    // Materialize; NumPy often returns a view (O(1)), so this op will lose
    // on microbenchmarks until we expose a view-based API.
    a.t().to_owned()
}

/// `np.matmul(a, b)` for 2D matrices — views only, no extra copies.
pub fn matmul(a: &NdArray, b: &NdArray) -> NdArray {
    view2(a).dot(&view2(b)).into_dyn()
}

/// `np.dot(a, b)` — vectors → scalar-as-0d; matrices → matmul.
pub fn dot(a: &NdArray, b: &NdArray) -> NdArray {
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            let av = view1(a);
            let bv = view1(b);
            assert_eq!(av.len(), bv.len(), "dot: vector lengths must match");
            let s = av.dot(&bv);
            NdArray::from_elem(ndarray::IxDyn(&[]), s)
        }
        (2, 2) => matmul(a, b),
        (2, 1) => view2(a).dot(&view1(b)).into_dyn(),
        (1, 2) => view1(a).dot(&view2(b)).into_dyn(),
        _ => panic!(
            "dot: unsupported ndims {} and {}",
            a.ndim(),
            b.ndim()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{eye, ones};
    use approx::assert_abs_diff_eq;

    #[test]
    fn matmul_eye() {
        let a = ones(&[3, 3]);
        let i = eye(3);
        let c = matmul(&a, &i);
        for x in c.iter() {
            assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn dot_vectors() {
        let a = ones(&[3]);
        let b = ones(&[3]);
        let s = dot(&a, &b);
        assert_abs_diff_eq!(s[[]], 3.0, epsilon = 1e-12);
    }
}
