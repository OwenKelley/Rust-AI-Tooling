//! `rnumpy` — NumPy-shaped numerical API for Rust.
//!
//! Function names intentionally mirror NumPy (`np.zeros`, `np.add`, …)
//! so Python/Rust parity tests can call the same conceptual surface.

pub mod creation;
pub mod linalg;
pub mod ops;
pub mod reductions;

pub use creation::*;
pub use linalg::*;
pub use ops::*;
pub use reductions::*;

use ndarray::ArrayD;

/// Owned N-dimensional `f64` array (NumPy `ndarray` analogue for this crate).
pub type NdArray = ArrayD<f64>;

/// Shared floating-point tolerance used by parity checks.
pub const DEFAULT_RTOL: f64 = 1e-7;
pub const DEFAULT_ATOL: f64 = 1e-8;
