//! Resample — mirrors `DataFrame.resample(freq).mean/sum` for fixed freqs.
//!
//! Semantics (v1):
//! - `closed='left'`, `label='left'`
//! - bins from `floor(min_ts)` through `floor(max_ts)` step `freq`
//! - empty bins: mean → NaN, sum → 0.0
//! - result includes a leading `ts` float column (bin start, epoch ns)

use rnumpy::NdArray;

use crate::datetime::{floor_bin, DatetimeIndex, Freq};
use crate::frame::{Column, DataFrame};

/// `df.set_index(dt).resample(freq).mean().reset_index()` (numeric cols only).
pub fn resample_mean(df: &DataFrame, index: &DatetimeIndex, freq: Freq) -> DataFrame {
    resample_agg(df, index, freq, true)
}

/// `df.set_index(dt).resample(freq).sum().reset_index()` (numeric cols only).
pub fn resample_sum(df: &DataFrame, index: &DatetimeIndex, freq: Freq) -> DataFrame {
    resample_agg(df, index, freq, false)
}

/// Resample using `df.index` when it is a [`DatetimeIndex`].
pub fn resample_mean_index(df: &DataFrame, freq: Freq) -> DataFrame {
    let idx = df
        .index
        .as_datetime()
        .expect("resample_mean_index: DataFrame.index must be DatetimeIndex");
    resample_mean(df, idx, freq)
}

/// Resample using `df.index` when it is a [`DatetimeIndex`].
pub fn resample_sum_index(df: &DataFrame, freq: Freq) -> DataFrame {
    let idx = df
        .index
        .as_datetime()
        .expect("resample_sum_index: DataFrame.index must be DatetimeIndex");
    resample_sum(df, idx, freq)
}

fn resample_agg(df: &DataFrame, index: &DatetimeIndex, freq: Freq, mean: bool) -> DataFrame {
    assert_eq!(
        df.nrows(),
        index.len(),
        "resample: index length must match nrows"
    );
    let period = freq.as_ns();
    let ts = index.values();
    if ts.is_empty() {
        return DataFrame::from_columns(vec![(
            "ts".into(),
            Column::Float64(NdArray::from_vec(vec![])),
        )]);
    }

    // Require non-decreasing timestamps (pandas typically assumes sorted).
    for w in ts.windows(2) {
        assert!(w[0] <= w[1], "resample: DatetimeIndex must be sorted ascending");
    }

    let first_bin = floor_bin(ts[0], period);
    let last_bin = floor_bin(*ts.last().unwrap(), period);
    let n_bins = ((last_bin - first_bin) / period) as usize + 1;

    // Numeric columns only.
    let mut names: Vec<String> = Vec::new();
    let mut series: Vec<Vec<f64>> = Vec::new();
    for (name, col) in df.columns_ref() {
        if let Some(xs) = col.as_f64_vec() {
            names.push(name.clone());
            series.push(xs);
        }
    }

    let mut sums = vec![vec![0.0f64; n_bins]; series.len()];
    let mut counts = vec![vec![0usize; n_bins]; series.len()];

    for (row, &t) in ts.iter().enumerate() {
        let bin = floor_bin(t, period);
        let bi = ((bin - first_bin) / period) as usize;
        for (c, xs) in series.iter().enumerate() {
            let v = xs[row];
            if !v.is_nan() {
                sums[c][bi] += v;
                counts[c][bi] += 1;
            }
        }
    }

    let mut ts_out = Vec::with_capacity(n_bins);
    let mut t = first_bin;
    for _ in 0..n_bins {
        ts_out.push(t as f64);
        t += period;
    }

    let mut cols = vec![("ts".into(), Column::Float64(NdArray::from_vec(ts_out)))];
    for (c, name) in names.into_iter().enumerate() {
        let mut out = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            let n = counts[c][b];
            if n == 0 {
                out.push(if mean { f64::NAN } else { 0.0 });
            } else if mean {
                out.push(sums[c][b] / n as f64);
            } else {
                out.push(sums[c][b]);
            }
        }
        cols.push((name, Column::Float64(NdArray::from_vec(out))));
    }
    DataFrame::from_columns(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datetime::date_range;

    #[test]
    fn resample_hourly_to_daily_mean() {
        // 2020-01-01 .. 48 hours → 2 daily bins
        let start = 1_577_836_800_000_000_000i64; // 2020-01-01T00:00:00
        let idx = date_range(start, 48, Freq::H);
        let vals: Vec<f64> = (0..48).map(|i| i as f64).collect();
        let df = DataFrame::from_columns(vec![(
            "c0".into(),
            Column::Float64(NdArray::from_vec(vals)),
        )]);
        let out = resample_mean(&df, &idx, Freq::D);
        assert_eq!(out.nrows(), 2);
        let c0 = out
            .columns_ref()
            .iter()
            .find(|(n, _)| n == "c0")
            .unwrap()
            .1
            .as_f64_vec()
            .unwrap();
        // hours 0..23 mean = 11.5; hours 24..47 mean = 35.5
        assert!((c0[0] - 11.5).abs() < 1e-12);
        assert!((c0[1] - 35.5).abs() < 1e-12);

        let df_idx = df.clone().set_index(idx);
        let out2 = resample_mean_index(&df_idx, Freq::D);
        assert_eq!(out2.checksum(), out.checksum());
    }
}
