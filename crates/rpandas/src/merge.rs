//! Frame merge / join — mirrors `pandas.merge` for a single key.

use std::collections::HashMap;

use crate::frame::{Column, DataFrame};

/// Join kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeHow {
    Inner,
    Left,
}

impl MergeHow {
    pub fn parse(s: &str) -> Self {
        match s {
            "inner" => MergeHow::Inner,
            "left" => MergeHow::Left,
            other => panic!("merge: unknown how '{other}'"),
        }
    }
}

/// `pd.merge(left, right, on=key, how=...)`.
///
/// Single-key merge. Overlapping non-key column names get `_x` / `_y` suffixes
/// (pandas default when suffixes=('_x','_y')).
pub fn merge(left: &DataFrame, right: &DataFrame, on: &str, how: MergeHow) -> DataFrame {
    let n_left = left.nrows();
    let n_right = right.nrows();

    let left_key = find_col(left, on);
    let right_key = find_col(right, on);

    // Build right multimap: key → row indices
    let mut right_map: HashMap<Key, Vec<usize>> = HashMap::new();
    for i in 0..n_right {
        let k = row_key(right_key, i);
        right_map.entry(k).or_default().push(i);
    }

    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new(); // usize::MAX = unmatched (left join)

    match left_key {
        Column::Float64(_)
        | Column::Int64 { .. }
        | Column::Bool { .. }
        | Column::Utf8 { .. } => {
            for i in 0..n_left {
                let k = row_key(left_key, i);
                if let Some(rs) = right_map.get(&k) {
                    for &j in rs {
                        left_idx.push(i);
                        right_idx.push(j);
                    }
                } else if matches!(how, MergeHow::Left) {
                    left_idx.push(i);
                    right_idx.push(usize::MAX);
                }
            }
        }
    }

    // Column naming
    let left_names: Vec<String> = left.column_names().into_iter().map(str::to_string).collect();
    let right_names: Vec<String> = right.column_names().into_iter().map(str::to_string).collect();
    let left_set: std::collections::HashSet<&str> =
        left_names.iter().map(|s| s.as_str()).collect();

    let mut out_cols: Vec<(String, Column)> = Vec::new();

    // Key from left
    out_cols.push((on.to_string(), left_key.take_rows(&left_idx)));

    // Left non-key columns
    for name in &left_names {
        if name == on {
            continue;
        }
        let col = find_col(left, name);
        let out_name = if right_names.iter().any(|r| r == name) {
            format!("{name}_x")
        } else {
            name.clone()
        };
        out_cols.push((out_name, col.take_rows(&left_idx)));
    }

    // Right non-key columns
    for name in &right_names {
        if name == on {
            continue;
        }
        let col = find_col(right, name);
        let out_name = if left_set.contains(name.as_str()) {
            format!("{name}_y")
        } else {
            name.clone()
        };
        out_cols.push((out_name, take_right_with_nulls(col, &right_idx)));
    }

    DataFrame::from_columns(out_cols)
}

fn find_col<'a>(df: &'a DataFrame, name: &str) -> &'a Column {
    df.columns_ref()
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, c)| c)
        .unwrap_or_else(|| panic!("merge: column '{name}' not found"))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Key {
    Float(u64),
    Int(i64),
    Bool(bool),
    Str { null: bool, s: String },
    Null,
}

fn row_key(col: &Column, i: usize) -> Key {
    match col {
        Column::Float64(a) => {
            let c = a.to_contiguous();
            let x = c.as_slice().unwrap()[i];
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
                Key::Str {
                    null: false,
                    s: values[i].clone(),
                }
            }
        }
    }
}

fn take_right_with_nulls(col: &Column, right_idx: &[usize]) -> Column {
    match col {
        Column::Float64(a) => {
            let c = a.to_contiguous();
            let s = c.as_slice().unwrap();
            let data: Vec<f64> = right_idx
                .iter()
                .map(|&j| {
                    if j == usize::MAX {
                        f64::NAN
                    } else {
                        s[j]
                    }
                })
                .collect();
            Column::Float64(rnumpy::NdArray::from_vec(data))
        }
        Column::Int64 { values, nulls } => {
            let mut v = Vec::with_capacity(right_idx.len());
            let mut n = Vec::with_capacity(right_idx.len());
            for &j in right_idx {
                if j == usize::MAX {
                    v.push(0);
                    n.push(true);
                } else {
                    v.push(values[j]);
                    n.push(nulls[j]);
                }
            }
            Column::Int64 { values: v, nulls: n }
        }
        Column::Bool { values, nulls } => {
            let mut v = Vec::with_capacity(right_idx.len());
            let mut n = Vec::with_capacity(right_idx.len());
            for &j in right_idx {
                if j == usize::MAX {
                    v.push(false);
                    n.push(true);
                } else {
                    v.push(values[j]);
                    n.push(nulls[j]);
                }
            }
            Column::Bool { values: v, nulls: n }
        }
        Column::Utf8 { values, nulls } => {
            let mut v = Vec::with_capacity(right_idx.len());
            let mut n = Vec::with_capacity(right_idx.len());
            for &j in right_idx {
                if j == usize::MAX {
                    v.push(String::new());
                    n.push(true);
                } else {
                    v.push(values[j].clone());
                    n.push(nulls[j]);
                }
            }
            Column::Utf8 { values: v, nulls: n }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnumpy::NdArray;

    #[test]
    fn merge_inner_left() {
        let left = DataFrame::from_columns(vec![
            (
                "k".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0, 3.0])),
            ),
            (
                "v".into(),
                Column::Float64(NdArray::from_vec(vec![10.0, 20.0, 30.0])),
            ),
        ]);
        let right = DataFrame::from_columns(vec![
            (
                "k".into(),
                Column::Float64(NdArray::from_vec(vec![2.0, 3.0, 4.0])),
            ),
            (
                "w".into(),
                Column::Float64(NdArray::from_vec(vec![200.0, 300.0, 400.0])),
            ),
        ]);
        let inner = merge(&left, &right, "k", MergeHow::Inner);
        assert_eq!(inner.nrows(), 2);
        assert_eq!(inner.float_slice("k"), vec![2.0, 3.0]);

        let left_j = merge(&left, &right, "k", MergeHow::Left);
        assert_eq!(left_j.nrows(), 3);
        assert!(left_j.float_slice("w")[0].is_nan());
    }
}
