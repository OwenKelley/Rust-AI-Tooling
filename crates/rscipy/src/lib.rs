//! `rscipy` — SciPy-shaped scientific API for Rust.
//!
//! Built on `rnumpy`. Names mirror SciPy so Python/Rust parity tests share
//! the same conceptual surface.

pub mod fft;
pub mod integrate;
pub mod linalg;
pub mod optimize;
pub mod signal;
pub mod sparse;
pub mod special;
pub mod stats;

pub use fft::*;
pub use integrate::*;
pub use linalg::*;
pub use optimize::*;
pub use signal::*;
pub use sparse::*;
pub use special::*;
pub use stats::*;
