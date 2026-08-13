//! Columnar IPC — Arrow-inspired binary frame format (`std` only).
//!
//! This is **not** full Apache Arrow / Parquet. It is a compact local IPC used
//! until a future `rarrow` crate can speak real Arrow/Parquet. Layout is stable
//! for rpandas ↔ Python harness roundtrips.
//!
//! ```text
//! magic: b"RPIC" (4)
//! version: u32 LE = 1
//! ncols: u32 LE
//! nrows: u64 LE
//! for each column:
//!   name_len: u32 LE, name UTF-8 bytes
//!   dtype: u8  (0=f64, 1=i64, 2=bool, 3=utf8)
//!   payload:
//!     f64: nrows × f64 LE
//!     i64: nrows × i64 LE, then nrows null bytes (0/1)
//!     bool: nrows value bytes (0/1), then nrows null bytes
//!     utf8: for each row: u32 LE length (u32::MAX = null), then bytes if present
//! ```

use std::fs;
use std::path::Path;

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};

const MAGIC: &[u8; 4] = b"RPIC";
const VERSION: u32 = 1;

const DTYPE_F64: u8 = 0;
const DTYPE_I64: u8 = 1;
const DTYPE_BOOL: u8 = 2;
const DTYPE_UTF8: u8 = 3;

/// Serialize a DataFrame to IPC bytes.
pub fn to_ipc_bytes(df: &DataFrame) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(df.ncols() as u32).to_le_bytes());
    out.extend_from_slice(&(df.nrows() as u64).to_le_bytes());

    for (name, col) in df.columns_ref() {
        let nb = name.as_bytes();
        out.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        out.extend_from_slice(nb);
        match col {
            Column::Float64(a) => {
                out.push(DTYPE_F64);
                let c = a.to_contiguous();
                for &v in c.as_slice().unwrap() {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Column::Int64 { values, nulls } => {
                out.push(DTYPE_I64);
                for &v in values {
                    out.extend_from_slice(&v.to_le_bytes());
                }
                for &n in nulls {
                    out.push(u8::from(n));
                }
            }
            Column::Bool { values, nulls } => {
                out.push(DTYPE_BOOL);
                for &v in values {
                    out.push(u8::from(v));
                }
                for &n in nulls {
                    out.push(u8::from(n));
                }
            }
            Column::Utf8 { values, nulls } => {
                out.push(DTYPE_UTF8);
                for (s, &is_null) in values.iter().zip(nulls.iter()) {
                    if is_null {
                        out.extend_from_slice(&u32::MAX.to_le_bytes());
                    } else {
                        let b = s.as_bytes();
                        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
                        out.extend_from_slice(b);
                    }
                }
            }
        }
    }
    out
}

/// Write IPC to a file path.
pub fn to_ipc(df: &DataFrame, path: impl AsRef<Path>) -> std::io::Result<()> {
    fs::write(path, to_ipc_bytes(df))
}

/// Deserialize a DataFrame from IPC bytes.
pub fn read_ipc_bytes(bytes: &[u8]) -> DataFrame {
    assert!(bytes.len() >= 4 + 4 + 4 + 8, "read_ipc: truncated header");
    assert_eq!(&bytes[0..4], MAGIC, "read_ipc: bad magic");
    let mut off = 4usize;
    let version = read_u32(bytes, &mut off);
    assert_eq!(version, VERSION, "read_ipc: unsupported version {version}");
    let ncols = read_u32(bytes, &mut off) as usize;
    let nrows = read_u64(bytes, &mut off) as usize;

    let mut cols = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        let name_len = read_u32(bytes, &mut off) as usize;
        let name = String::from_utf8(bytes[off..off + name_len].to_vec())
            .expect("read_ipc: column name utf8");
        off += name_len;
        let dtype = bytes[off];
        off += 1;
        let col = match dtype {
            DTYPE_F64 => {
                let mut data = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    data.push(read_f64(bytes, &mut off));
                }
                Column::Float64(NdArray::from_vec(data))
            }
            DTYPE_I64 => {
                let mut values = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    values.push(read_i64(bytes, &mut off));
                }
                let mut nulls = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    nulls.push(bytes[off] != 0);
                    off += 1;
                }
                Column::Int64 { values, nulls }
            }
            DTYPE_BOOL => {
                let mut values = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    values.push(bytes[off] != 0);
                    off += 1;
                }
                let mut nulls = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    nulls.push(bytes[off] != 0);
                    off += 1;
                }
                Column::Bool { values, nulls }
            }
            DTYPE_UTF8 => {
                let mut values = Vec::with_capacity(nrows);
                let mut nulls = Vec::with_capacity(nrows);
                for _ in 0..nrows {
                    let len = read_u32(bytes, &mut off);
                    if len == u32::MAX {
                        values.push(String::new());
                        nulls.push(true);
                    } else {
                        let n = len as usize;
                        values.push(
                            String::from_utf8(bytes[off..off + n].to_vec())
                                .expect("read_ipc: utf8 cell"),
                        );
                        nulls.push(false);
                        off += n;
                    }
                }
                Column::Utf8 { values, nulls }
            }
            other => panic!("read_ipc: unknown dtype {other}"),
        };
        cols.push((name, col));
    }
    DataFrame::from_columns(cols)
}

/// Read IPC from a file path.
pub fn read_ipc(path: impl AsRef<Path>) -> std::io::Result<DataFrame> {
    let bytes = fs::read(path)?;
    Ok(read_ipc_bytes(&bytes))
}

/// Parquet-shaped export alias (same v1 IPC payload; `.rparquet` companion).
///
/// Not Apache Parquet. Prefer [`to_ipc_bytes`] in new code.
pub fn to_parquet_bytes(df: &DataFrame) -> Vec<u8> {
    to_ipc_bytes(df)
}

/// Parquet-shaped import alias for [`read_ipc_bytes`].
pub fn read_parquet_bytes(bytes: &[u8]) -> DataFrame {
    read_ipc_bytes(bytes)
}

fn read_u32(buf: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes(buf[*off..*off + 4].try_into().unwrap());
    *off += 4;
    v
}

fn read_u64(buf: &[u8], off: &mut usize) -> u64 {
    let v = u64::from_le_bytes(buf[*off..*off + 8].try_into().unwrap());
    *off += 8;
    v
}

fn read_i64(buf: &[u8], off: &mut usize) -> i64 {
    let v = i64::from_le_bytes(buf[*off..*off + 8].try_into().unwrap());
    *off += 8;
    v
}

fn read_f64(buf: &[u8], off: &mut usize) -> f64 {
    let v = f64::from_le_bytes(buf[*off..*off + 8].try_into().unwrap());
    *off += 8;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ipc_roundtrip_mixed() {
        let df = DataFrame::from_columns(vec![
            (
                "f".into(),
                Column::Float64(NdArray::from_vec(vec![1.5, f64::NAN, -2.0])),
            ),
            (
                "i".into(),
                Column::Int64 {
                    values: vec![1, 0, 3],
                    nulls: vec![false, true, false],
                },
            ),
            (
                "s".into(),
                Column::Utf8 {
                    values: vec!["a".into(), String::new(), "c".into()],
                    nulls: vec![false, true, false],
                },
            ),
        ]);
        let bytes = to_ipc_bytes(&df);
        let back = read_ipc_bytes(&bytes);
        assert_eq!(back.nrows(), 3);
        assert_eq!(back.ncols(), 3);
        assert!((back.float_slice("f")[0] - 1.5).abs() < 1e-15);
        assert!(back.float_slice("f")[1].is_nan());
        assert_eq!(back.checksum(), df.checksum());
    }
}
