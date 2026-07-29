//! Reshape ops — `melt` and `pivot_table`.

use std::collections::HashMap;

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};
use crate::groupby::Agg;

/// `pd.melt(df, id_vars=..., value_vars=...)`.
///
/// Output columns: `id_vars...`, `variable`, `value` (value is Float64 when
/// melted cols are numeric; else Utf8).
pub fn melt(df: &DataFrame, id_vars: &[&str], value_vars: &[&str]) -> DataFrame {
    assert!(!value_vars.is_empty(), "melt: need value_vars");
    let n = df.nrows();
    let out_rows = n * value_vars.len();

    // id columns repeated for each value_var
    let mut out: Vec<(String, Column)> = Vec::new();
    for &id in id_vars {
        let col = df
            .columns_ref()
            .iter()
            .find(|(name, _)| name == id)
            .map(|(_, c)| c)
            .unwrap_or_else(|| panic!("melt: id_var '{id}' not found"));
        let mut indices = Vec::with_capacity(out_rows);
        for _ in 0..value_vars.len() {
            for i in 0..n {
                indices.push(i);
            }
        }
        out.push((id.to_string(), col.take_rows(&indices)));
    }

    // variable column
    let mut var_vals = Vec::with_capacity(out_rows);
    let var_nulls = vec![false; out_rows];
    for &vn in value_vars {
        for _ in 0..n {
            var_vals.push(vn.to_string());
        }
    }
    out.push((
        "variable".into(),
        Column::Utf8 {
            values: var_vals,
            nulls: var_nulls,
        },
    ));

    // value column — prefer float when all value_vars numeric
    let all_num = value_vars.iter().all(|&vn| {
        df.columns_ref()
            .iter()
            .find(|(n, _)| n == vn)
            .map(|(_, c)| c.as_f64_vec().is_some())
            .unwrap_or(false)
    });

    if all_num {
        let mut vals = Vec::with_capacity(out_rows);
        for &vn in value_vars {
            let xs = df
                .columns_ref()
                .iter()
                .find(|(n, _)| n == vn)
                .unwrap()
                .1
                .as_f64_vec()
                .unwrap();
            vals.extend(xs);
        }
        out.push(("value".into(), Column::Float64(NdArray::from_vec(vals))));
    } else {
        let mut vals = Vec::with_capacity(out_rows);
        let mut nulls = Vec::with_capacity(out_rows);
        for &vn in value_vars {
            let col = &df
                .columns_ref()
                .iter()
                .find(|(n, _)| n == vn)
                .unwrap()
                .1;
            for i in 0..n {
                let cell = col.csv_cell(i);
                if cell.is_empty() && col.null_mask()[i] {
                    vals.push(String::new());
                    nulls.push(true);
                } else {
                    vals.push(cell);
                    nulls.push(false);
                }
            }
        }
        out.push((
            "value".into(),
            Column::Utf8 {
                values: vals,
                nulls,
            },
        ));
    }

    DataFrame::from_columns(out)
}

/// `pd.pivot_table(df, index, columns, values, aggfunc)`.
///
/// Single index / columns / values. Column order is first-seen unique column
/// keys. Missing combinations are NaN.
pub fn pivot_table(
    df: &DataFrame,
    index: &str,
    columns: &str,
    values: &str,
    agg: Agg,
) -> DataFrame {
    let n = df.nrows();
    let idx_col = find(df, index);
    let col_col = find(df, columns);
    let val_col = find(df, values);
    let val_f = val_col
        .as_f64_vec()
        .unwrap_or_else(|| panic!("pivot_table: values must be numeric"));

    // Unique index keys (first seen) and column keys (first seen)
    let mut index_keys: Vec<Key> = Vec::new();
    let mut index_map: HashMap<Key, usize> = HashMap::new();
    let mut col_keys: Vec<Key> = Vec::new();
    let mut col_map: HashMap<Key, usize> = HashMap::new();

    for i in 0..n {
        let ik = key_at(idx_col, i);
        if !index_map.contains_key(&ik) {
            index_map.insert(ik.clone(), index_keys.len());
            index_keys.push(ik);
        }
        let ck = key_at(col_col, i);
        if !col_map.contains_key(&ck) {
            col_map.insert(ck.clone(), col_keys.len());
            col_keys.push(ck);
        }
    }

    let nr = index_keys.len();
    let nc = col_keys.len();
    let mut cells: Vec<AggState> = (0..nr * nc).map(|_| AggState::new(agg)).collect();

    for i in 0..n {
        let ri = index_map[&key_at(idx_col, i)];
        let ci = col_map[&key_at(col_col, i)];
        cells[ri * nc + ci].push(val_f[i]);
    }

    let mut out = Vec::new();
    out.push((index.to_string(), materialize_keys(idx_col, &index_keys)));

    for (ci, ck) in col_keys.iter().enumerate() {
        let name = key_label(ck);
        let mut vals = Vec::with_capacity(nr);
        for ri in 0..nr {
            vals.push(cells[ri * nc + ci].finish());
        }
        out.push((name, Column::Float64(NdArray::from_vec(vals))));
    }

    DataFrame::from_columns(out)
}

fn find<'a>(df: &'a DataFrame, name: &str) -> &'a Column {
    df.columns_ref()
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, c)| c)
        .unwrap_or_else(|| panic!("column '{name}' not found"))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Float(u64),
    Int(i64),
    Bool(bool),
    Str(String),
    Null,
}

fn key_at(col: &Column, i: usize) -> Key {
    match col {
        Column::Float64(a) => {
            let x = a.to_contiguous().as_slice().unwrap()[i];
            if x.is_nan() {
                Key::Null
            } else {
                Key::Float(x.to_bits())
            }
        }
        Column::Int64 { values, nulls } => {
            if nulls[i] {
                Key::Null
            } else {
                Key::Int(values[i])
            }
        }
        Column::Bool { values, nulls } => {
            if nulls[i] {
                Key::Null
            } else {
                Key::Bool(values[i])
            }
        }
        Column::Utf8 { values, nulls } => {
            if nulls[i] {
                Key::Null
            } else {
                Key::Str(values[i].clone())
            }
        }
    }
}

fn key_label(k: &Key) -> String {
    match k {
        Key::Float(bits) => format!("{}", f64::from_bits(*bits)),
        Key::Int(v) => v.to_string(),
        Key::Bool(v) => {
            if *v {
                "True".into()
            } else {
                "False".into()
            }
        }
        Key::Str(s) => s.clone(),
        Key::Null => "nan".into(),
    }
}

fn materialize_keys(template: &Column, keys: &[Key]) -> Column {
    match template {
        Column::Float64(_) => {
            let vals: Vec<f64> = keys
                .iter()
                .map(|k| match k {
                    Key::Float(b) => f64::from_bits(*b),
                    Key::Null => f64::NAN,
                    Key::Int(v) => *v as f64,
                    _ => f64::NAN,
                })
                .collect();
            Column::Float64(NdArray::from_vec(vals))
        }
        Column::Int64 { .. } => {
            let mut values = Vec::new();
            let mut nulls = Vec::new();
            for k in keys {
                match k {
                    Key::Int(v) => {
                        values.push(*v);
                        nulls.push(false);
                    }
                    Key::Null => {
                        values.push(0);
                        nulls.push(true);
                    }
                    Key::Float(b) => {
                        values.push(f64::from_bits(*b) as i64);
                        nulls.push(false);
                    }
                    _ => {
                        values.push(0);
                        nulls.push(true);
                    }
                }
            }
            Column::Int64 { values, nulls }
        }
        Column::Bool { .. } => {
            let mut values = Vec::new();
            let mut nulls = Vec::new();
            for k in keys {
                match k {
                    Key::Bool(v) => {
                        values.push(*v);
                        nulls.push(false);
                    }
                    Key::Null => {
                        values.push(false);
                        nulls.push(true);
                    }
                    _ => {
                        values.push(false);
                        nulls.push(true);
                    }
                }
            }
            Column::Bool { values, nulls }
        }
        Column::Utf8 { .. } => {
            let mut values = Vec::new();
            let mut nulls = Vec::new();
            for k in keys {
                match k {
                    Key::Str(s) => {
                        values.push(s.clone());
                        nulls.push(false);
                    }
                    Key::Null => {
                        values.push(String::new());
                        nulls.push(true);
                    }
                    other => {
                        values.push(key_label(other));
                        nulls.push(false);
                    }
                }
            }
            Column::Utf8 { values, nulls }
        }
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
        if self.count == 0 && !matches!(self.kind, Agg::Sum | Agg::Count) {
            return f64::NAN;
        }
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
    fn melt_basic() {
        let df = DataFrame::from_columns(vec![
            (
                "id".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0])),
            ),
            (
                "a".into(),
                Column::Float64(NdArray::from_vec(vec![10.0, 20.0])),
            ),
            (
                "b".into(),
                Column::Float64(NdArray::from_vec(vec![30.0, 40.0])),
            ),
        ]);
        let m = melt(&df, &["id"], &["a", "b"]);
        assert_eq!(m.nrows(), 4);
        assert_eq!(m.float_slice("value"), vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn pivot_sum() {
        let df = DataFrame::from_columns(vec![
            (
                "i".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 1.0, 2.0])),
            ),
            (
                "c".into(),
                Column::Utf8 {
                    values: vec!["x".into(), "y".into(), "x".into()],
                    nulls: vec![false, false, false],
                },
            ),
            (
                "v".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0, 3.0])),
            ),
        ]);
        let p = pivot_table(&df, "i", "c", "v", Agg::Sum);
        assert_eq!(p.nrows(), 2);
        assert!(p.has_column("x"));
        assert!(p.has_column("y"));
    }
}
