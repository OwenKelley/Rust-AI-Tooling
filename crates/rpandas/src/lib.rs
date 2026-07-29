//! `rpandas` — Pandas-shaped tabular API for Rust.
//!
//! Built on `rnumpy`. Names mirror Pandas so Python/Rust parity tests share
//! the same conceptual surface.

pub mod frame;
pub mod groupby;
pub mod index;
pub mod io;
pub mod merge;
pub mod ops;
pub mod reshape;
pub mod rolling;
pub mod series;

pub use frame::{Column, DataFrame};
pub use groupby::{groupby_agg, Agg};
pub use index::RangeIndex;
pub use io::{read_csv, read_csv_str, to_csv, to_csv_string};
pub use merge::{merge, MergeHow};
pub use ops::{
    describe, dropna, fillna, filter, filter_gt, mean, sort_values, sum,
};
pub use reshape::{melt, pivot_table};
pub use rolling::{rolling_mean, rolling_mean_frame, rolling_sum};
pub use series::Series;
