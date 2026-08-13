//! `rarrow` — PyArrow-shaped Arrow / Parquet API for Rust (`std` only).

pub mod array;
pub mod flatbuf;
pub mod ipc;
pub mod parquet;
pub mod record_batch;
pub mod rev_fbb;
pub mod schema;

pub use array::{
    Array, BooleanArray, DictionaryUtf8Array, Float64Array, Int64Array, ListFloat64Array,
    StringArray,
};
pub use ipc::{read_ipc_file, read_ipc_stream, write_ipc_file, write_ipc_stream};
pub use parquet::{read_parquet, write_parquet, write_parquet_par1, write_parquet_rpqt};
pub use record_batch::{batch_from_columns, RecordBatch};
pub use schema::{DataType, Field, Schema};

/// Convert an `rpandas`-compatible column map: names + arrays.
pub fn checksum_batch(batch: &RecordBatch) -> f64 {
    batch.checksum()
}
