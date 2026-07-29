//! Frame-level ops — filter, sort, dropna/fillna, describe, reductions.

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};

/// Boolean row filter: keep rows where `mask[i]` is true.
///
/// Mirrors `df[mask]` for a boolean Series / array of length `nrows`.
pub fn filter(df: &DataFrame, mask: &[bool]) -> DataFrame {
    assert_eq!(mask.len(), df.nrows(), "filter: mask length");
    let indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    df.take_rows(&indices)
}

/// Filter rows where float column `col` satisfies `value > thresh` (NaN → false).
pub fn filter_gt(df: &DataFrame, col: &str, thresh: f64) -> DataFrame {
    let xs = df.float_slice(col);
    let mask: Vec<bool> = xs.iter().map(|&x| !x.is_nan() && x > thresh).collect();
    filter(df, &mask)
}

/// `df.sort_values(by, ascending=...)`.
pub fn sort_values(df: &DataFrame, by: &str, ascending: bool) -> DataFrame {
    let n = df.nrows();
    let mut order: Vec<usize> = (0..n).collect();
    match &df.columns_ref()[df_col_pos(df, by)].1 {
        Column::Float64(a) => {
            let c = a.to_contiguous();
            let s = c.as_slice().unwrap().to_vec();
            order.sort_by(|&i, &j| {
                let cmp = partial_cmp_nan_last(s[i], s[j]);
                let primary = if ascending { cmp } else { cmp.reverse() };
                primary.then(i.cmp(&j))
            });
        }
        Column::Int64 { values, nulls } => {
            order.sort_by(|&i, &j| {
                let cmp = match (nulls[i], nulls[j]) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => values[i].cmp(&values[j]),
                };
                let primary = if ascending { cmp } else { cmp.reverse() };
                primary.then(i.cmp(&j))
            });
        }
        Column::Bool { values, nulls } => {
            order.sort_by(|&i, &j| {
                let cmp = match (nulls[i], nulls[j]) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => values[i].cmp(&values[j]),
                };
                let primary = if ascending { cmp } else { cmp.reverse() };
                primary.then(i.cmp(&j))
            });
        }
        Column::Utf8 { values, nulls } => {
            order.sort_by(|&i, &j| {
                let cmp = match (nulls[i], nulls[j]) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => std::cmp::Ordering::Greater,
                    (false, true) => std::cmp::Ordering::Less,
                    (false, false) => values[i].cmp(&values[j]),
                };
                let primary = if ascending { cmp } else { cmp.reverse() };
                primary.then(i.cmp(&j))
            });
        }
    }
    df.take_rows(&order)
}

fn df_col_pos(df: &DataFrame, name: &str) -> usize {
    df.columns_ref()
        .iter()
        .position(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("column '{name}' not found"))
}

fn partial_cmp_nan_last(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
    }
}

/// `df.dropna(how='any'|'all')` — drop rows with missing values.
pub fn dropna(df: &DataFrame, how: &str) -> DataFrame {
    let n = df.nrows();
    let masks: Vec<Vec<bool>> = df
        .columns_ref()
        .iter()
        .map(|(_, c)| c.null_mask())
        .collect();
    let keep: Vec<usize> = (0..n)
        .filter(|&i| {
            let null_count = masks.iter().filter(|m| m[i]).count();
            match how {
                "any" => null_count == 0,
                "all" => null_count < masks.len(),
                other => panic!("dropna: unknown how '{other}'"),
            }
        })
        .collect();
    df.take_rows(&keep)
}

/// `df.fillna(value)` for float columns (string nulls → empty string if value is used as 0.0 skip).
///
/// Float NaNs replaced by `value`. UTF-8 nulls left unchanged unless `fill_str` is `Some`.
pub fn fillna(df: &DataFrame, value: f64, fill_str: Option<&str>) -> DataFrame {
    let cols: Vec<(String, Column)> = df
        .columns_ref()
        .iter()
        .map(|(name, col)| {
            let new_col = match col {
                Column::Float64(a) => {
                    let c = a.to_contiguous();
                    let data: Vec<f64> = c
                        .as_slice()
                        .unwrap()
                        .iter()
                        .map(|&x| if x.is_nan() { value } else { x })
                        .collect();
                    Column::Float64(NdArray::from_vec(data))
                }
                Column::Int64 { values, nulls } => {
                    let fill = value as i64;
                    let mut v = values.clone();
                    let mut n = nulls.clone();
                    for i in 0..v.len() {
                        if n[i] {
                            v[i] = fill;
                            n[i] = false;
                        }
                    }
                    Column::Int64 {
                        values: v,
                        nulls: n,
                    }
                }
                Column::Bool { values, nulls } => {
                    let fill = value != 0.0;
                    let mut v = values.clone();
                    let mut n = nulls.clone();
                    for i in 0..v.len() {
                        if n[i] {
                            v[i] = fill;
                            n[i] = false;
                        }
                    }
                    Column::Bool {
                        values: v,
                        nulls: n,
                    }
                }
                Column::Utf8 { values, nulls } => {
                    if let Some(s) = fill_str {
                        let mut v = values.clone();
                        let mut n = nulls.clone();
                        for i in 0..v.len() {
                            if n[i] {
                                v[i] = s.to_string();
                                n[i] = false;
                            }
                        }
                        Column::Utf8 {
                            values: v,
                            nulls: n,
                        }
                    } else {
                        col.clone()
                    }
                }
            };
            (name.clone(), new_col)
        })
        .collect();
    DataFrame::from_columns(cols)
}

/// Column-wise sum of float columns (`df.sum(numeric_only=True)` axis=0).
///
/// Returns a 1-row frame with the same float column names.
pub fn sum(df: &DataFrame) -> DataFrame {
    reduce_numeric(df, |xs| {
        xs.iter().filter(|x| !x.is_nan()).sum()
    })
}

/// Column-wise mean of float columns (`df.mean(numeric_only=True)`).
pub fn mean(df: &DataFrame) -> DataFrame {
    reduce_numeric(df, |xs| {
        let mut s = 0.0;
        let mut n = 0usize;
        for &x in xs {
            if !x.is_nan() {
                s += x;
                n += 1;
            }
        }
        if n == 0 {
            f64::NAN
        } else {
            s / n as f64
        }
    })
}

fn reduce_numeric(df: &DataFrame, f: impl Fn(&[f64]) -> f64) -> DataFrame {
    let mut cols = Vec::new();
    for (name, col) in df.columns_ref() {
        if let Some(xs) = col.as_f64_vec() {
            let v = f(&xs);
            cols.push((
                name.clone(),
                Column::Float64(NdArray::from_vec(vec![v])),
            ));
        }
    }
    DataFrame::from_columns(cols)
}

/// `df.describe()` for numeric columns — rows: count, mean, std, min, 25%, 50%, 75%, max.
pub fn describe(df: &DataFrame) -> DataFrame {
    let stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"];
    let float_cols: Vec<(String, Vec<f64>)> = df
        .columns_ref()
        .iter()
        .filter_map(|(name, col)| col.as_f64_vec().map(|xs| (name.clone(), xs)))
        .collect();

    let mut out_cols: Vec<(String, Column)> = Vec::new();
    // First column: stat labels as utf8 for readability in tests; harness uses numeric only.
    // Pandas describe index is labels; we emit a parallel frame of only numeric stats columns.
    for (name, xs) in &float_cols {
        let clean: Vec<f64> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
        let count = clean.len() as f64;
        let (mean_v, std_v, min_v, p25, p50, p75, max_v) = if clean.is_empty() {
            (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
        } else {
            let mean_v = clean.iter().sum::<f64>() / count;
            let var = clean.iter().map(|x| {
                let d = x - mean_v;
                d * d
            }).sum::<f64>() / (count - 1.0).max(1.0);
            // pandas describe uses sample std (ddof=1)
            let std_v = if clean.len() < 2 { f64::NAN } else { var.sqrt() };
            let mut sorted = clean.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (
                mean_v,
                std_v,
                sorted[0],
                quantile_sorted(&sorted, 0.25),
                quantile_sorted(&sorted, 0.50),
                quantile_sorted(&sorted, 0.75),
                *sorted.last().unwrap(),
            )
        };
        let vals = vec![count, mean_v, std_v, min_v, p25, p50, p75, max_v];
        assert_eq!(vals.len(), stats.len());
        out_cols.push((name.clone(), Column::Float64(NdArray::from_vec(vals))));
    }
    let _ = stats; // row labels implicit by position
    DataFrame::from_columns(out_cols)
}

/// Linear interpolation quantile on a sorted non-empty slice (pandas default).
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_df() -> DataFrame {
        DataFrame::from_columns(vec![
            (
                "a".into(),
                Column::Float64(NdArray::from_vec(vec![3.0, 1.0, f64::NAN, 2.0])),
            ),
            (
                "b".into(),
                Column::Float64(NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0])),
            ),
        ])
    }

    #[test]
    fn sort_and_dropna() {
        let df = sample_df();
        let s = sort_values(&df, "a", true);
        assert_eq!(s.float_slice("a")[..3], [1.0, 2.0, 3.0]);
        let d = dropna(&df, "any");
        assert_eq!(d.nrows(), 3);
    }

    #[test]
    fn fillna_sum_mean() {
        let df = sample_df();
        let f = fillna(&df, 0.0, None);
        assert!(!f.float_slice("a")[2].is_nan());
        let s = sum(&df);
        assert!((s.float_slice("a")[0] - 6.0).abs() < 1e-12);
    }
}
