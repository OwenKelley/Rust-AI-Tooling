//! Linear algebra — mirrors common `numpy` / `np.linalg` entry points.
//!
//! Heavy flops go through in-house `gemm` (std only), not ndarray/BLAS crates.

use crate::gemm::{dot_f64, gemm_rowmajor, gemv_rowmajor, gevm_rowmajor};
use crate::NdArray;
use ndarray::{ArrayD, IxDyn};

fn require_slice(a: &NdArray) -> &[f64] {
    a.as_slice_memory_order()
        .expect("linalg ops currently require contiguous row-major storage")
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

/// `np.matmul(a, b)` for 2D matrices.
pub fn matmul(a: &NdArray, b: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "matmul: expected 2D A");
    assert_eq!(b.ndim(), 2, "matmul: expected 2D B");
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    assert_eq!(k, k2, "matmul: inner dims must match");

    let data = gemm_rowmajor(require_slice(a), require_slice(b), m, k, n);
    ArrayD::from_shape_vec(IxDyn(&[m, n]), data).expect("shape matches len")
}

/// `np.dot(a, b)` — vectors → scalar-as-0d; matrices → matmul.
pub fn dot(a: &NdArray, b: &NdArray) -> NdArray {
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            let s = dot_f64(require_slice(a), require_slice(b));
            NdArray::from_elem(IxDyn(&[]), s)
        }
        (2, 2) => matmul(a, b),
        (2, 1) => {
            let m = a.shape()[0];
            let k = a.shape()[1];
            assert_eq!(b.len(), k, "dot: matrix-vector width mismatch");
            let data = gemv_rowmajor(require_slice(a), require_slice(b), m, k);
            ArrayD::from_shape_vec(IxDyn(&[m]), data).expect("shape matches len")
        }
        (1, 2) => {
            let k = a.len();
            let k2 = b.shape()[0];
            let n = b.shape()[1];
            assert_eq!(k, k2, "dot: vector-matrix width mismatch");
            let data = gevm_rowmajor(require_slice(a), require_slice(b), k, n);
            ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("shape matches len")
        }
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
    fn matmul_parallel_path() {
        // Large enough to hit the std::thread parallel GEMM path.
        let a = ones(&[128, 128]);
        let b = eye(128);
        let c = matmul(&a, &b);
        for x in c.iter() {
            assert_abs_diff_eq!(*x, 1.0, epsilon = 1e-9);
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
