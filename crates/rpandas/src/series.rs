//! Named 1D column — mirrors `pandas.Series` (f64 / i64 / bool / string).

use rnumpy::NdArray;

use crate::frame::Column;
use crate::index::Index;

/// `pandas.Series` analogue.
#[derive(Debug, Clone)]
pub struct Series {
    pub name: String,
    pub index: Index,
    pub data: Column,
}

impl Series {
    /// `pd.Series(values, name=name)` for float data.
    pub fn from_f64(values: Vec<f64>, name: impl Into<String>) -> Self {
        let n = values.len();
        Self {
            name: name.into(),
            index: Index::range(n),
            data: Column::Float64(NdArray::from_vec(values)),
        }
    }

    /// `pd.Series(values, name=name)` for int64 (`None` → null).
    pub fn from_i64(values: Vec<Option<i64>>, name: impl Into<String>) -> Self {
        let n = values.len();
        let mut vals = Vec::with_capacity(n);
        let mut nulls = Vec::with_capacity(n);
        for v in values {
            match v {
                Some(x) => {
                    vals.push(x);
                    nulls.push(false);
                }
                None => {
                    vals.push(0);
                    nulls.push(true);
                }
            }
        }
        Self {
            name: name.into(),
            index: Index::range(n),
            data: Column::Int64 {
                values: vals,
                nulls,
            },
        }
    }

    /// `pd.Series(values, name=name)` for bool (`None` → null).
    pub fn from_bool(values: Vec<Option<bool>>, name: impl Into<String>) -> Self {
        let n = values.len();
        let mut vals = Vec::with_capacity(n);
        let mut nulls = Vec::with_capacity(n);
        for v in values {
            match v {
                Some(x) => {
                    vals.push(x);
                    nulls.push(false);
                }
                None => {
                    vals.push(false);
                    nulls.push(true);
                }
            }
        }
        Self {
            name: name.into(),
            index: Index::range(n),
            data: Column::Bool {
                values: vals,
                nulls,
            },
        }
    }

    /// `pd.Series(values, name=name)` for string data (`None` → null).
    pub fn from_str(values: Vec<Option<String>>, name: impl Into<String>) -> Self {
        let n = values.len();
        let mut vals = Vec::with_capacity(n);
        let mut nulls = Vec::with_capacity(n);
        for v in values {
            match v {
                Some(s) => {
                    vals.push(s);
                    nulls.push(false);
                }
                None => {
                    vals.push(String::new());
                    nulls.push(true);
                }
            }
        }
        Self {
            name: name.into(),
            index: Index::range(n),
            data: Column::Utf8 {
                values: vals,
                nulls,
            },
        }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sum of non-null numeric values.
    pub fn sum(&self) -> f64 {
        match self.data.as_f64_vec() {
            Some(xs) => xs.iter().filter(|x| !x.is_nan()).sum(),
            None => f64::NAN,
        }
    }

    /// Mean of non-null numeric values.
    pub fn mean(&self) -> f64 {
        match self.data.as_f64_vec() {
            Some(xs) => {
                let mut sum = 0.0;
                let mut n = 0usize;
                for &x in &xs {
                    if !x.is_nan() {
                        sum += x;
                        n += 1;
                    }
                }
                if n == 0 {
                    f64::NAN
                } else {
                    sum / n as f64
                }
            }
            None => f64::NAN,
        }
    }

    /// `Series.map(f)` / elementwise `apply` for float series.
    ///
    /// NaN inputs stay NaN (pandas `na_action=None` still passes NaN into `f` for
    /// map; we match apply-style: call `f` on every value including NaN).
    pub fn map_f64(&self, f: impl Fn(f64) -> f64) -> Series {
        let xs = self
            .data
            .as_f64_vec()
            .unwrap_or_else(|| panic!("map_f64: series '{}' is not float", self.name));
        let out: Vec<f64> = xs.into_iter().map(f).collect();
        Series {
            name: self.name.clone(),
            index: self.index.clone(),
            data: Column::Float64(NdArray::from_vec(out)),
        }
    }

    /// Elementwise `Series.apply(f)` for float series (same as [`map_f64`] in v1).
    pub fn apply_f64(&self, f: impl Fn(f64) -> f64) -> Series {
        self.map_f64(f)
    }
}

/// `df[col].map(f)` → float column Series as a one-column frame for harnesses.
pub fn map_f64(df: &crate::frame::DataFrame, col: &str, f: impl Fn(f64) -> f64) -> Series {
    df.column(col).map_f64(f)
}

/// `df[col].apply(f)`.
pub fn apply_f64(df: &crate::frame::DataFrame, col: &str, f: impl Fn(f64) -> f64) -> Series {
    df.column(col).apply_f64(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_square() {
        let s = Series::from_f64(vec![1.0, -2.0, f64::NAN], "x");
        let out = s.map_f64(|x| x * x);
        let xs = out.data.as_f64_vec().unwrap();
        assert!((xs[0] - 1.0).abs() < 1e-12);
        assert!((xs[1] - 4.0).abs() < 1e-12);
        assert!(xs[2].is_nan());
    }
}
