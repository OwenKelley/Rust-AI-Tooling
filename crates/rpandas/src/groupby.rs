//! Group-by aggregation — mirrors `DataFrame.groupby(...).agg(...)`.

use std::collections::HashMap;

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};

/// Aggregation function name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Agg {
    Sum,
    Mean,
    Count,
    Min,
    Max,
}

impl Agg {
    pub fn parse(s: &str) -> Self {
        match s {
            "sum" => Agg::Sum,
            "mean" => Agg::Mean,
            "count" => Agg::Count,
            "min" => Agg::Min,
            "max" => Agg::Max,
            other => panic!("groupby_agg: unknown agg '{other}'"),
        }
    }
}

/// `df.groupby(key).agg({col: agg, ...})` for a single key column.
///
/// Key may be float (exact bit equality; NaN groups together) or string.
/// Output rows are ordered by first appearance of each key.
pub fn groupby_agg(df: &DataFrame, key: &str, aggs: &[(&str, Agg)]) -> DataFrame {
    let n = df.nrows();
    let key_col = df
        .columns_ref()
        .iter()
        .find(|(n, _)| n == key)
        .map(|(_, c)| c)
        .unwrap_or_else(|| panic!("groupby: key '{key}' not found"));

    // Map key-id → group index; preserve first-seen order.
    let mut order: Vec<usize> = Vec::new();
    let mut group_of: Vec<usize> = vec![0; n];

    match key_col {
        Column::Float64(a) => {
            let c = a.to_contiguous();
            let xs = c.as_slice().unwrap();
            let mut map: HashMap<u64, usize> = HashMap::new();
            for i in 0..n {
                let bits = if xs[i].is_nan() {
                    u64::MAX
                } else {
                    xs[i].to_bits()
                };
                let g = if let Some(&g) = map.get(&bits) {
                    g
                } else {
                    let g = order.len();
                    map.insert(bits, g);
                    order.push(i);
                    g
                };
                group_of[i] = g;
            }
        }
        Column::Int64 { values, nulls } => {
            let mut map: HashMap<(bool, i64), usize> = HashMap::new();
            for i in 0..n {
                let k = (nulls[i], values[i]);
                let g = if let Some(&g) = map.get(&k) {
                    g
                } else {
                    let g = order.len();
                    map.insert(k, g);
                    order.push(i);
                    g
                };
                group_of[i] = g;
            }
        }
        Column::Bool { values, nulls } => {
            let mut map: HashMap<(bool, bool), usize> = HashMap::new();
            for i in 0..n {
                let k = (nulls[i], values[i]);
                let g = if let Some(&g) = map.get(&k) {
                    g
                } else {
                    let g = order.len();
                    map.insert(k, g);
                    order.push(i);
                    g
                };
                group_of[i] = g;
            }
        }
        Column::Utf8 { values, nulls } => {
            let mut map: HashMap<(bool, String), usize> = HashMap::new();
            for i in 0..n {
                let k = (nulls[i], values[i].clone());
                let g = if let Some(&g) = map.get(&k) {
                    g
                } else {
                    let g = order.len();
                    map.insert(k, g);
                    order.push(i);
                    g
                };
                group_of[i] = g;
            }
        }
    }

    let ng = order.len();
    let mut out: Vec<(String, Column)> = Vec::new();

    // Key column: one value per group (from representative row).
    out.push((key.to_string(), key_col.take_rows(&order)));

    for &(col_name, agg) in aggs {
        if col_name == key {
            continue;
        }
        let col = df
            .columns_ref()
            .iter()
            .find(|(n, _)| n == col_name)
            .map(|(_, c)| c)
            .unwrap_or_else(|| panic!("groupby: column '{col_name}' not found"));

        let series = match col {
            Column::Float64(_) | Column::Int64 { .. } | Column::Bool { .. } => {
                let xs = col.as_f64_vec().unwrap();
                let mut acc = vec![AggState::new(agg); ng];
                for i in 0..n {
                    acc[group_of[i]].push(xs[i]);
                }
                let vals: Vec<f64> = acc.iter().map(|s| s.finish()).collect();
                Column::Float64(NdArray::from_vec(vals))
            }
            Column::Utf8 { nulls, .. } => {
                assert!(
                    matches!(agg, Agg::Count),
                    "groupby: string column only supports count"
                );
                let mut counts = vec![0.0; ng];
                for i in 0..n {
                    if !nulls[i] {
                        counts[group_of[i]] += 1.0;
                    }
                }
                Column::Float64(NdArray::from_vec(counts))
            }
        };
        let out_name = format!("{col_name}_{}", agg_name(agg));
        out.push((out_name, series));
    }

    DataFrame::from_columns(out)
}

fn agg_name(a: Agg) -> &'static str {
    match a {
        Agg::Sum => "sum",
        Agg::Mean => "mean",
        Agg::Count => "count",
        Agg::Min => "min",
        Agg::Max => "max",
    }
}

#[derive(Clone)]
struct AggState {
    kind: Agg,
    sum: f64,
    count: usize,
    min: f64,
    max: f64,
}

impl AggState {
    fn new(kind: Agg) -> Self {
        Self {
            kind,
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn push(&mut self, x: f64) {
        if x.is_nan() {
            // pandas skipna=True default for these aggs
            return;
        }
        self.sum += x;
        self.count += 1;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn finish(&self) -> f64 {
        match self.kind {
            Agg::Sum => self.sum,
            Agg::Mean => {
                if self.count == 0 {
                    f64::NAN
                } else {
                    self.sum / self.count as f64
                }
            }
            Agg::Count => self.count as f64,
            Agg::Min => {
                if self.count == 0 {
                    f64::NAN
                } else {
                    self.min
                }
            }
            Agg::Max => {
                if self.count == 0 {
                    f64::NAN
                } else {
                    self.max
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groupby_sum() {
        let df = DataFrame::from_columns(vec![
            (
                "g".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0, 1.0, 2.0])),
            ),
            (
                "v".into(),
                Column::Float64(NdArray::from_vec(vec![10.0, 20.0, 30.0, 40.0])),
            ),
        ]);
        let out = groupby_agg(&df, "g", &[("v", Agg::Sum)]);
        assert_eq!(out.nrows(), 2);
        // first-seen order: g=1 then g=2
        assert_eq!(out.float_slice("g"), vec![1.0, 2.0]);
        assert_eq!(out.float_slice("v_sum"), vec![40.0, 60.0]);
    }
}
