//! Eager DataFrame — mirrors `polars.DataFrame`.

use rarrow::{batch_from_columns, Array, RecordBatch};

use crate::expr::Expr;
use crate::groupby::GroupBy;
use crate::join::JoinHow;
use crate::lazy::LazyFrame;
use crate::series::Series;

#[derive(Debug, Clone, PartialEq)]
pub struct DataFrame {
    columns: Vec<Series>,
}

impl DataFrame {
    pub fn new(columns: Vec<Series>) -> Self {
        if let Some(first) = columns.first() {
            let n = first.len();
            for s in &columns {
                assert_eq!(s.len(), n, "column length mismatch: {}", s.name());
            }
        }
        let mut names = std::collections::HashSet::new();
        for s in &columns {
            assert!(names.insert(s.name.clone()), "duplicate column {}", s.name());
        }
        Self { columns }
    }

    pub fn from_series(columns: Vec<Series>) -> Self {
        Self::new(columns)
    }

    pub fn from_record_batch(batch: &RecordBatch) -> Self {
        let cols = batch
            .schema
            .fields
            .iter()
            .zip(batch.columns.iter())
            .map(|(f, a)| Series::new(f.name.clone(), a.clone()))
            .collect();
        Self::new(cols)
    }

    pub fn to_record_batch(&self) -> RecordBatch {
        batch_from_columns(
            self.columns
                .iter()
                .map(|s| (s.name.clone(), s.data.clone()))
                .collect(),
        )
    }

    pub fn height(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn width(&self) -> usize {
        self.columns.len()
    }

    pub fn get_column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    pub fn column(&self, name: &str) -> &Series {
        self.columns
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("column '{name}' not found"))
    }

    pub fn columns(&self) -> &[Series] {
        &self.columns
    }

    pub fn checksum(&self) -> f64 {
        let mut s = self.height() as f64 + self.width() as f64;
        for c in &self.columns {
            s += c.checksum();
        }
        s
    }

    pub fn select(&self, names: &[&str]) -> DataFrame {
        let cols = names
            .iter()
            .map(|n| self.column(n).clone())
            .collect();
        DataFrame::new(cols)
    }

    pub fn drop(&self, names: &[&str]) -> DataFrame {
        let drop: std::collections::HashSet<&str> = names.iter().copied().collect();
        let cols = self
            .columns
            .iter()
            .filter(|c| !drop.contains(c.name.as_str()))
            .cloned()
            .collect();
        DataFrame::new(cols)
    }

    pub fn rename(&self, mapping: &[(&str, &str)]) -> DataFrame {
        let map: std::collections::HashMap<&str, &str> =
            mapping.iter().copied().collect();
        let cols = self
            .columns
            .iter()
            .map(|c| {
                if let Some(&new) = map.get(c.name.as_str()) {
                    c.clone().rename(new)
                } else {
                    c.clone()
                }
            })
            .collect();
        DataFrame::new(cols)
    }

    /// Append or replace columns (Polars `with_columns`).
    pub fn with_columns(&self, extras: Vec<Series>) -> DataFrame {
        let mut cols = self.columns.clone();
        for s in extras {
            if let Some(i) = cols.iter().position(|c| c.name == s.name) {
                assert_eq!(s.len(), self.height(), "with_columns length mismatch");
                cols[i] = s;
            } else {
                assert_eq!(s.len(), self.height(), "with_columns length mismatch");
                cols.push(s);
            }
        }
        DataFrame::new(cols)
    }

    pub fn filter(&self, predicate: &Expr) -> DataFrame {
        let mask = predicate.eval_bool(self);
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i) } else { None })
            .collect();
        self.take_rows(&indices)
    }

    pub fn take_rows(&self, indices: &[usize]) -> DataFrame {
        let cols = self
            .columns
            .iter()
            .map(|c| c.take(indices))
            .collect();
        DataFrame::new(cols)
    }

    pub fn head(&self, n: usize) -> DataFrame {
        self.slice(0, n)
    }

    pub fn tail(&self, n: usize) -> DataFrame {
        let h = self.height();
        let start = h.saturating_sub(n);
        self.slice(start as i64, n)
    }

    pub fn slice(&self, offset: i64, length: usize) -> DataFrame {
        let h = self.height() as i64;
        let start = if offset < 0 {
            (h + offset).max(0) as usize
        } else {
            (offset as usize).min(self.height())
        };
        let end = (start + length).min(self.height());
        let indices: Vec<usize> = (start..end).collect();
        self.take_rows(&indices)
    }

    pub fn sort(&self, by: &[&str], descending: bool) -> DataFrame {
        assert!(!by.is_empty(), "sort: need at least one column");
        let n = self.height();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            for &name in by {
                let cmp = cmp_rows(&self.column(name).data, i, j);
                let primary = if descending { cmp.reverse() } else { cmp };
                if primary != std::cmp::Ordering::Equal {
                    return primary;
                }
            }
            i.cmp(&j)
        });
        self.take_rows(&order)
    }

    pub fn groupby<'a>(&'a self, keys: &[&'a str]) -> GroupBy<'a> {
        GroupBy::new(self, keys)
    }

    pub fn join(&self, other: &DataFrame, on: &[&str], how: JoinHow) -> DataFrame {
        crate::join::join(self, other, on, how)
    }

    pub fn lazy(&self) -> LazyFrame {
        LazyFrame::new(self.clone())
    }
}

fn cmp_rows(data: &Array, i: usize, j: usize) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match data {
        Array::Float64(a) => match (a.nulls[i], a.nulls[j]) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.values[i].partial_cmp(&a.values[j]).unwrap_or(Ordering::Equal),
        },
        Array::Int64(a) => match (a.nulls[i], a.nulls[j]) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.values[i].cmp(&a.values[j]),
        },
        Array::Boolean(a) => match (a.nulls[i], a.nulls[j]) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.values[i].cmp(&a.values[j]),
        },
        Array::Utf8(a) => match (&a.values[i], &a.values[j]) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(x), Some(y)) => x.cmp(y),
        },
        Array::TimestampNs(a) => match (a.nulls[i], a.nulls[j]) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => a.values[i].cmp(&a.values[j]),
        },
        Array::ListFloat64(_) => Ordering::Equal,
        Array::DictionaryUtf8(a) => {
            let vi = if a.nulls[i] {
                None
            } else {
                Some(a.dictionary[a.indices[i] as usize].as_str())
            };
            let vj = if a.nulls[j] {
                None
            } else {
                Some(a.dictionary[a.indices[j] as usize].as_str())
            };
            match (vi, vj) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Greater,
                (Some(_), None) => Ordering::Less,
                (Some(x), Some(y)) => x.cmp(y),
            }
        }
    }
}

/// Row-wise equality for join keys (nulls match nulls).
pub(crate) fn key_eq(cols: &[&Array], i: usize, j_cols: &[&Array], j: usize) -> bool {
    cols.iter().zip(j_cols.iter()).all(|(a, b)| row_eq(a, i, b, j))
}

fn row_eq(a: &Array, i: usize, b: &Array, j: usize) -> bool {
    match (a, b) {
        (Array::Float64(x), Array::Float64(y)) => match (x.nulls[i], y.nulls[j]) {
            (true, true) => true,
            (false, false) => x.values[i].to_bits() == y.values[j].to_bits(),
            _ => false,
        },
        (Array::Int64(x), Array::Int64(y)) => match (x.nulls[i], y.nulls[j]) {
            (true, true) => true,
            (false, false) => x.values[i] == y.values[j],
            _ => false,
        },
        (Array::Boolean(x), Array::Boolean(y)) => match (x.nulls[i], y.nulls[j]) {
            (true, true) => true,
            (false, false) => x.values[i] == y.values[j],
            _ => false,
        },
        (Array::Utf8(x), Array::Utf8(y)) => x.values[i] == y.values[j],
        _ => false,
    }
}

pub(crate) fn hash_key(cols: &[&Array], i: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    for c in cols {
        match c {
            Array::Float64(a) => {
                a.nulls[i].hash(&mut h);
                if !a.nulls[i] {
                    a.values[i].to_bits().hash(&mut h);
                }
            }
            Array::Int64(a) => {
                a.nulls[i].hash(&mut h);
                if !a.nulls[i] {
                    a.values[i].hash(&mut h);
                }
            }
            Array::Boolean(a) => {
                a.nulls[i].hash(&mut h);
                if !a.nulls[i] {
                    a.values[i].hash(&mut h);
                }
            }
            Array::Utf8(a) => a.values[i].hash(&mut h),
            Array::TimestampNs(a) => {
                a.nulls[i].hash(&mut h);
                if !a.nulls[i] {
                    a.values[i].hash(&mut h);
                }
            }
            Array::ListFloat64(a) => {
                a.nulls[i].hash(&mut h);
                if !a.nulls[i] {
                    let start = a.offsets[i] as usize;
                    let end = a.offsets[i + 1] as usize;
                    for v in &a.values[start..end] {
                        v.to_bits().hash(&mut h);
                    }
                }
            }
            Array::DictionaryUtf8(a) => {
                a.nulls[i].hash(&mut h);
                if !a.nulls[i] {
                    a.dictionary[a.indices[i] as usize].hash(&mut h);
                }
            }
        }
    }
    h.finish()
}

pub(crate) fn take_array_pub(data: &Array, indices: &[usize]) -> Array {
    crate::series::take_array(data, indices)
}
