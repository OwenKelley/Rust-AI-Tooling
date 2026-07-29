//! Rolling window ops — mirrors `Series.rolling(window).mean/sum`.

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};

/// `df[col].rolling(window).mean()` → float column (same length; leading NaNs).
///
/// Uses `min_periods == window` (pandas default).
pub fn rolling_mean(df: &DataFrame, col: &str, window: usize) -> Column {
    rolling_reduce(df, col, window, true)
}

/// `df[col].rolling(window).sum()`.
pub fn rolling_sum(df: &DataFrame, col: &str, window: usize) -> Column {
    rolling_reduce(df, col, window, false)
}

/// Apply rolling mean to all numeric columns; non-numeric columns copied.
pub fn rolling_mean_frame(df: &DataFrame, window: usize) -> DataFrame {
    let mut cols = Vec::new();
    for (name, col) in df.columns_ref() {
        if col.as_f64_vec().is_some() {
            cols.push((name.clone(), rolling_mean(df, name, window)));
        } else {
            cols.push((name.clone(), col.clone()));
        }
    }
    DataFrame::from_columns(cols)
}

fn rolling_reduce(df: &DataFrame, col: &str, window: usize, mean: bool) -> Column {
    assert!(window >= 1, "rolling: window must be >= 1");
    let xs = df
        .columns_ref()
        .iter()
        .find(|(n, _)| n == col)
        .and_then(|(_, c)| c.as_f64_vec())
        .unwrap_or_else(|| panic!("rolling: column '{col}' not numeric"));
    let n = xs.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        if i + 1 < window {
            continue;
        }
        let start = i + 1 - window;
        let mut s = 0.0;
        let mut c = 0usize;
        for j in start..=i {
            let v = xs[j];
            if !v.is_nan() {
                s += v;
                c += 1;
            }
        }
        // pandas: min_periods=window ⇒ need `window` non-NaN observations
        if c >= window {
            out[i] = if mean { s / c as f64 } else { s };
        }
    }
    Column::Float64(NdArray::from_vec(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_mean_known() {
        let df = DataFrame::from_columns(vec![(
            "x".into(),
            Column::Float64(NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0])),
        )]);
        let col = rolling_mean(&df, "x", 2);
        let xs = match col {
            Column::Float64(a) => a.to_contiguous().as_slice().unwrap().to_vec(),
            _ => panic!(),
        };
        assert!(xs[0].is_nan());
        assert!((xs[1] - 1.5).abs() < 1e-12);
        assert!((xs[2] - 2.5).abs() < 1e-12);
        assert!((xs[3] - 3.5).abs() < 1e-12);
    }
}
