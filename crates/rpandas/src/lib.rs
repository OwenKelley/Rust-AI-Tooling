//! `rpandas` — Pandas-shaped tabular API for Rust.
//!
//! Built on `rnumpy`. Names mirror Pandas so Python/Rust parity tests share
//! the same conceptual surface.

pub mod arrow_convert;
pub mod categorical;
pub mod datetime;
pub mod frame;
pub mod groupby;
pub mod index;
pub mod io;
pub mod ipc;
pub mod merge;
pub mod ops;
pub mod reshape;
pub mod resample;
pub mod rolling;
pub mod series;

pub use arrow_convert::{dataframe_to_record_batch, record_batch_to_dataframe};

pub use categorical::{categorical_codes, Categorical};
pub use datetime::{date_range, DatetimeIndex, Freq};
pub use frame::{Column, DataFrame};
pub use groupby::{groupby_agg, Agg};
pub use index::{Index, RangeIndex};
pub use io::{read_csv, read_csv_str, to_csv, to_csv_string};
pub use ipc::{
    read_ipc, read_ipc_bytes, read_parquet_bytes, to_ipc, to_ipc_bytes, to_parquet_bytes,
};
pub use merge::{merge, merge_on, MergeHow};
pub use ops::{
    describe, dropna, fillna, filter, filter_gt, mean, sort_values, sum,
};
pub use reshape::{melt, pivot_table};
pub use resample::{resample_mean, resample_mean_index, resample_sum, resample_sum_index};
pub use rolling::{rolling_mean, rolling_mean_frame, rolling_sum};
pub use series::{apply_f64, map_f64, Series};
