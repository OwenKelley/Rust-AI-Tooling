//! `rnumpy` — NumPy-shaped numerical API for Rust.
//!
//! Function names intentionally mirror NumPy (`np.zeros`, `np.add`, …)
//! so Python/Rust parity tests can call the same conceptual surface.

pub mod array;
pub mod broadcast;
pub mod creation;
pub mod gemm;
pub mod indexing;
pub mod linalg;
pub mod manipulation;
pub mod ops;
pub mod reductions;

pub use array::{AxisSlice, NdArray, NdArrayF32};
pub use creation::*;
pub use indexing::*;
pub use linalg::*;
pub use manipulation::*;
pub use ops::*;
pub use reductions::*;

/// Shared floating-point tolerance used by parity checks.
pub const DEFAULT_RTOL: f64 = 1e-7;
pub const DEFAULT_ATOL: f64 = 1e-8;

#[cfg(test)]
pub(crate) mod test_util {
    /// Local stand-in for `approx::assert_abs_diff_eq` (no third-party dep).
    #[inline]
    pub fn assert_abs_diff_eq(a: f64, b: f64, epsilon: f64) {
        let diff = (a - b).abs();
        assert!(
            diff <= epsilon,
            "abs_diff_eq failed: |{a} - {b}| = {diff} > {epsilon}"
        );
    }
}
