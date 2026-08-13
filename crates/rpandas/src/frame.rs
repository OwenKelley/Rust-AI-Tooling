//! DataFrame and column storage — mirrors `pandas.DataFrame`.

use rnumpy::NdArray;

use crate::index::Index;
use crate::series::Series;

/// Column payload.
///
/// - `Float64`: missing = NaN
/// - `Int64` / `Bool` / `Utf8`: missing via parallel null mask
#[derive(Debug, Clone)]
pub enum Column {
    Float64(NdArray),
    Int64 {
        values: Vec<i64>,
        nulls: Vec<bool>,
    },
    Bool {
        values: Vec<bool>,
        nulls: Vec<bool>,
    },
    Utf8 {
        values: Vec<String>,
        nulls: Vec<bool>,
    },
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Column::Float64(a) => a.len(),
            Column::Int64 { values, .. } => values.len(),
            Column::Bool { values, .. } => values.len(),
            Column::Utf8 { values, .. } => values.len(),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Column::Float64(_))
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Column::Float64(_) | Column::Int64 { .. })
    }

    pub fn is_utf8(&self) -> bool {
        matches!(self, Column::Utf8 { .. })
    }

    /// Null / NaN mask (`true` = missing).
    pub fn null_mask(&self) -> Vec<bool> {
        match self {
            Column::Float64(a) => {
                let c = a.to_contiguous();
                c.as_slice()
                    .unwrap()
                    .iter()
                    .map(|x| x.is_nan())
                    .collect()
            }
            Column::Int64 { nulls, .. }
            | Column::Bool { nulls, .. }
            | Column::Utf8 { nulls, .. } => nulls.clone(),
        }
    }

    pub fn take_rows(&self, indices: &[usize]) -> Column {
        match self {
            Column::Float64(a) => {
                let c = a.to_contiguous();
                let s = c.as_slice().unwrap();
                let data: Vec<f64> = indices.iter().map(|&i| s[i]).collect();
                Column::Float64(NdArray::from_vec(data))
            }
            Column::Int64 { values, nulls } => {
                let mut v = Vec::with_capacity(indices.len());
                let mut n = Vec::with_capacity(indices.len());
                for &i in indices {
                    v.push(values[i]);
                    n.push(nulls[i]);
                }
                Column::Int64 {
                    values: v,
                    nulls: n,
                }
            }
            Column::Bool { values, nulls } => {
                let mut v = Vec::with_capacity(indices.len());
                let mut n = Vec::with_capacity(indices.len());
                for &i in indices {
                    v.push(values[i]);
                    n.push(nulls[i]);
                }
                Column::Bool {
                    values: v,
                    nulls: n,
                }
            }
            Column::Utf8 { values, nulls } => {
                let mut v = Vec::with_capacity(indices.len());
                let mut n = Vec::with_capacity(indices.len());
                for &i in indices {
                    v.push(values[i].clone());
                    n.push(nulls[i]);
                }
                Column::Utf8 {
                    values: v,
                    nulls: n,
                }
            }
        }
    }

    pub fn slice_rows(&self, start: usize, end: usize) -> Column {
        match self {
            Column::Float64(a) => {
                let c = a.to_contiguous();
                let s = c.as_slice().unwrap();
                Column::Float64(NdArray::from_vec(s[start..end].to_vec()))
            }
            Column::Int64 { values, nulls } => Column::Int64 {
                values: values[start..end].to_vec(),
                nulls: nulls[start..end].to_vec(),
            },
            Column::Bool { values, nulls } => Column::Bool {
                values: values[start..end].to_vec(),
                nulls: nulls[start..end].to_vec(),
            },
            Column::Utf8 { values, nulls } => Column::Utf8 {
                values: values[start..end].to_vec(),
                nulls: nulls[start..end].to_vec(),
            },
        }
    }

    /// Cast to f64 for numeric reductions (null/NaN → NaN).
    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            Column::Float64(a) => {
                Some(a.to_contiguous().as_slice().unwrap().to_vec())
            }
            Column::Int64 { values, nulls } => Some(
                values
                    .iter()
                    .zip(nulls.iter())
                    .map(|(&v, &n)| if n { f64::NAN } else { v as f64 })
                    .collect(),
            ),
            Column::Bool { values, nulls } => Some(
                values
                    .iter()
                    .zip(nulls.iter())
                    .map(|(&v, &n)| {
                        if n {
                            f64::NAN
                        } else if v {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
            Column::Utf8 { .. } => None,
        }
    }

    /// Checksum contribution for parity harness.
    pub fn checksum(&self) -> f64 {
        match self {
            Column::Float64(a) => {
                let c = a.to_contiguous();
                c.as_slice()
                    .unwrap()
                    .iter()
                    .map(|x| if x.is_nan() { 0.0 } else { *x })
                    .sum()
            }
            Column::Int64 { values, nulls } => values
                .iter()
                .zip(nulls.iter())
                .map(|(&v, &n)| if n { 0.0 } else { v as f64 })
                .sum(),
            Column::Bool { values, nulls } => values
                .iter()
                .zip(nulls.iter())
                .map(|(&v, &n)| {
                    if n {
                        0.0
                    } else if v {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum(),
            Column::Utf8 { values, nulls } => values
                .iter()
                .zip(nulls.iter())
                .map(|(s, &n)| if n { 0.0 } else { s.len() as f64 })
                .sum(),
        }
    }

    /// Format cell for CSV.
    pub fn csv_cell(&self, i: usize) -> String {
        match self {
            Column::Float64(a) => {
                let c = a.to_contiguous();
                let x = c.as_slice().unwrap()[i];
                if x.is_nan() {
                    String::new()
                } else {
                    format!("{x}")
                }
            }
            Column::Int64 { values, nulls } => {
                if nulls[i] {
                    String::new()
                } else {
                    values[i].to_string()
                }
            }
            Column::Bool { values, nulls } => {
                if nulls[i] {
                    String::new()
                } else if values[i] {
                    "True".into()
                } else {
                    "False".into()
                }
            }
            Column::Utf8 { values, nulls } => {
                if nulls[i] {
                    String::new()
                } else {
                    values[i].clone()
                }
            }
        }
    }
}

/// `pandas.DataFrame` analogue (column-oriented).
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub index: Index,
    columns: Vec<(String, Column)>,
}

impl DataFrame {
    /// Empty frame.
    pub fn new() -> Self {
        Self {
            index: Index::range(0),
            columns: Vec::new(),
        }
    }

    /// Build from named columns (all must share the same length).
    pub fn from_columns(cols: Vec<(String, Column)>) -> Self {
        let nrows = if cols.is_empty() {
            0
        } else {
            let n = cols[0].1.len();
            for (name, c) in &cols {
                assert_eq!(
                    c.len(),
                    n,
                    "column '{name}' length {} != {n}",
                    c.len()
                );
            }
            n
        };
        Self {
            index: Index::range(nrows),
            columns: cols,
        }
    }

    /// `pd.DataFrame` from float columns only: names + row-major matrix (nrows × ncols).
    pub fn from_numeric(names: &[&str], data: &NdArray) -> Self {
        assert_eq!(data.ndim(), 2, "from_numeric: expected 2D");
        let nrows = data.shape()[0];
        let ncols = data.shape()[1];
        assert_eq!(names.len(), ncols, "from_numeric: name count");
        let c = data.to_contiguous();
        let s = c.as_slice().unwrap();
        let mut cols = Vec::with_capacity(ncols);
        for (j, &name) in names.iter().enumerate() {
            let mut col = Vec::with_capacity(nrows);
            for i in 0..nrows {
                col.push(s[i * ncols + j]);
            }
            cols.push((name.to_string(), Column::Float64(NdArray::from_vec(col))));
        }
        Self::from_columns(cols)
    }

    pub fn nrows(&self) -> usize {
        self.index.len()
    }

    pub fn ncols(&self) -> usize {
        self.columns.len()
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.columns.iter().any(|(n, _)| n == name)
    }

    fn col_index(&self, name: &str) -> usize {
        self.columns
            .iter()
            .position(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("column '{name}' not found"))
    }

    /// `df[col]` → Series.
    pub fn column(&self, name: &str) -> Series {
        let i = self.col_index(name);
        let (n, c) = &self.columns[i];
        Series {
            name: n.clone(),
            index: self.index.clone(),
            data: c.clone(),
        }
    }

    /// `df[[cols]]`.
    pub fn select(&self, names: &[&str]) -> DataFrame {
        let cols: Vec<(String, Column)> = names
            .iter()
            .map(|name| {
                let i = self.col_index(name);
                self.columns[i].clone()
            })
            .collect();
        DataFrame {
            index: self.index.clone(),
            columns: cols,
        }
    }

    /// Assign / overwrite a column (`df[name] = ...`).
    pub fn with_column(mut self, name: impl Into<String>, col: Column) -> Self {
        let name = name.into();
        assert_eq!(col.len(), self.nrows(), "with_column: length mismatch");
        if let Some(pos) = self.columns.iter().position(|(n, _)| n == &name) {
            self.columns[pos] = (name, col);
        } else {
            self.columns.push((name, col));
        }
        self
    }

    /// `df.set_index(DatetimeIndex)` — replace the row index (length must match).
    pub fn set_index(mut self, index: impl Into<Index>) -> Self {
        let index = index.into();
        assert_eq!(
            index.len(),
            self.nrows(),
            "set_index: length {} != nrows {}",
            index.len(),
            self.nrows()
        );
        self.index = index;
        self
    }

    /// `df.head(n)`.
    pub fn head(&self, n: usize) -> DataFrame {
        let end = n.min(self.nrows());
        self.slice_rows(0, end)
    }

    /// `df.tail(n)`.
    pub fn tail(&self, n: usize) -> DataFrame {
        let nrows = self.nrows();
        let start = nrows.saturating_sub(n);
        self.slice_rows(start, nrows)
    }

    pub(crate) fn slice_rows(&self, start: usize, end: usize) -> DataFrame {
        let cols: Vec<(String, Column)> = self
            .columns
            .iter()
            .map(|(n, c)| (n.clone(), c.slice_rows(start, end)))
            .collect();
        DataFrame {
            index: self.index.slice_rows(start, end),
            columns: cols,
        }
    }

    pub(crate) fn take_rows(&self, indices: &[usize]) -> DataFrame {
        let cols: Vec<(String, Column)> = self
            .columns
            .iter()
            .map(|(n, c)| (n.clone(), c.take_rows(indices)))
            .collect();
        DataFrame {
            index: self.index.take_rows(indices),
            columns: cols,
        }
    }

    pub(crate) fn columns_ref(&self) -> &[(String, Column)] {
        &self.columns
    }

    /// Float column as contiguous slice (panic if missing / not float).
    pub fn float_slice(&self, name: &str) -> Vec<f64> {
        match &self.columns[self.col_index(name)].1 {
            Column::Float64(a) => a.to_contiguous().as_slice().unwrap().to_vec(),
            other => other
                .as_f64_vec()
                .unwrap_or_else(|| panic!("column '{name}' is not numeric")),
        }
    }

    /// Parity checksum: nrows + ncols + sum of column checksums.
    pub fn checksum(&self) -> f64 {
        let mut s = self.nrows() as f64 + self.ncols() as f64;
        for (_, c) in &self.columns {
            s += c.checksum();
        }
        s
    }
}

impl Default for DataFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_numeric_roundtrip_names() {
        let data = NdArray::from_shape_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let df = DataFrame::from_numeric(&["a", "b"], &data);
        assert_eq!(df.nrows(), 2);
        assert_eq!(df.ncols(), 2);
        assert_eq!(df.float_slice("a"), vec![1.0, 3.0]);
        assert_eq!(df.float_slice("b"), vec![2.0, 4.0]);
    }

    #[test]
    fn head_tail() {
        let data = NdArray::from_shape_vec(&[5, 1], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let df = DataFrame::from_numeric(&["x"], &data);
        assert_eq!(df.head(2).float_slice("x"), vec![1.0, 2.0]);
        assert_eq!(df.tail(2).float_slice("x"), vec![4.0, 5.0]);
    }

    #[test]
    fn int_bool_columns() {
        let df = DataFrame::from_columns(vec![
            (
                "i".into(),
                Column::Int64 {
                    values: vec![1, 2],
                    nulls: vec![false, false],
                },
            ),
            (
                "b".into(),
                Column::Bool {
                    values: vec![true, false],
                    nulls: vec![false, false],
                },
            ),
        ]);
        assert_eq!(df.checksum(), 2.0 + 2.0 + 3.0 + 1.0);
    }
}
