//! Frame merge / join — mirrors `pandas.merge` (single- or multi-key).

use std::collections::{HashMap, HashSet};

use crate::frame::{Column, DataFrame};

/// Join kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeHow {
    Inner,
    Left,
    Right,
    Outer,
}

impl MergeHow {
    pub fn parse(s: &str) -> Self {
        match s {
            "inner" => MergeHow::Inner,
            "left" => MergeHow::Left,
            "right" => MergeHow::Right,
            "outer" => MergeHow::Outer,
            other => panic!("merge: unknown how '{other}'"),
        }
    }
}

/// `pd.merge(left, right, on=key, how=...)` (single key).
pub fn merge(left: &DataFrame, right: &DataFrame, on: &str, how: MergeHow) -> DataFrame {
    merge_on(left, right, &[on], how)
}

/// `pd.merge(left, right, on=[…], how=…)`.
///
/// Overlapping non-key column names get `_x` / `_y` suffixes.
pub fn merge_on(left: &DataFrame, right: &DataFrame, on: &[&str], how: MergeHow) -> DataFrame {
    assert!(!on.is_empty(), "merge: need at least one key");
    for &k in on {
        assert!(left.has_column(k), "merge: left missing key '{k}'");
        assert!(right.has_column(k), "merge: right missing key '{k}'");
    }

    let n_left = left.nrows();
    let n_right = right.nrows();
    let left_keys: Vec<&Column> = on.iter().map(|&k| find_col(left, k)).collect();
    let right_keys: Vec<&Column> = on.iter().map(|&k| find_col(right, k)).collect();

    let mut right_map: HashMap<CompositeKey, Vec<usize>> = HashMap::new();
    for j in 0..n_right {
        let key = composite_key(&right_keys, j);
        right_map.entry(key).or_default().push(j);
    }

    let mut left_idx: Vec<usize> = Vec::new();
    let mut right_idx: Vec<usize> = Vec::new(); // usize::MAX = unmatched
    let mut matched_right: HashSet<usize> = HashSet::new();

    match how {
        MergeHow::Inner | MergeHow::Left | MergeHow::Outer => {
            for i in 0..n_left {
                let key = composite_key(&left_keys, i);
                if let Some(rs) = right_map.get(&key) {
                    for &j in rs {
                        left_idx.push(i);
                        right_idx.push(j);
                        matched_right.insert(j);
                    }
                } else if matches!(how, MergeHow::Left | MergeHow::Outer) {
                    left_idx.push(i);
                    right_idx.push(usize::MAX);
                }
            }
            if matches!(how, MergeHow::Outer) {
                for j in 0..n_right {
                    if !matched_right.contains(&j) {
                        left_idx.push(usize::MAX);
                        right_idx.push(j);
                    }
                }
            }
        }
        MergeHow::Right => {
            let mut left_map: HashMap<CompositeKey, Vec<usize>> = HashMap::new();
            for i in 0..n_left {
                let key = composite_key(&left_keys, i);
                left_map.entry(key).or_default().push(i);
            }
            for j in 0..n_right {
                let key = composite_key(&right_keys, j);
                if let Some(ls) = left_map.get(&key) {
                    for &i in ls {
                        left_idx.push(i);
                        right_idx.push(j);
                    }
                } else {
                    left_idx.push(usize::MAX);
                    right_idx.push(j);
                }
            }
        }
    }

    let left_names: Vec<String> = left.column_names().into_iter().map(str::to_string).collect();
    let right_names: Vec<String> = right.column_names().into_iter().map(str::to_string).collect();
    let on_set: HashSet<&str> = on.iter().copied().collect();
    let left_set: HashSet<&str> = left_names.iter().map(|s| s.as_str()).collect();

    let mut out_cols: Vec<(String, Column)> = Vec::new();

    // Keys: prefer left value, fall back to right when left unmatched (outer/right).
    for &k in on {
        let lk = find_col(left, k);
        let rk = find_col(right, k);
        out_cols.push((k.to_string(), coalesce_keys(lk, rk, &left_idx, &right_idx)));
    }

    for name in &left_names {
        if on_set.contains(name.as_str()) {
            continue;
        }
        let col = find_col(left, name);
        let out_name = if right_names.iter().any(|r| r == name) {
            format!("{name}_x")
        } else {
            name.clone()
        };
        out_cols.push((out_name, take_with_nulls(col, &left_idx)));
    }

    for name in &right_names {
        if on_set.contains(name.as_str()) {
            continue;
        }
        let col = find_col(right, name);
        let out_name = if left_set.contains(name.as_str()) {
            format!("{name}_y")
        } else {
            name.clone()
        };
        out_cols.push((out_name, take_with_nulls(col, &right_idx)));
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
enum Atom {
    Float(u64),
    Int(i64),
    Bool(bool),
    Str(String),
    Null,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CompositeKey(Vec<Atom>);

fn atom_at(col: &Column, i: usize) -> Atom {
    match col {
        Column::Float64(a) => {
            let c = a.to_contiguous();
            let x = c.as_slice().unwrap()[i];
            if x.is_nan() {
                Atom::Null
            } else {
                Atom::Float(x.to_bits())
            }
        }
        Column::Int64 { values, nulls } => {
            if nulls[i] {
                Atom::Null
            } else {
                Atom::Int(values[i])
            }
        }
        Column::Bool { values, nulls } => {
            if nulls[i] {
                Atom::Null
            } else {
                Atom::Bool(values[i])
            }
        }
        Column::Utf8 { values, nulls } => {
            if nulls[i] {
                Atom::Null
            } else {
                Atom::Str(values[i].clone())
            }
        }
    }
}

fn composite_key(cols: &[&Column], i: usize) -> CompositeKey {
    CompositeKey(cols.iter().map(|c| atom_at(c, i)).collect())
}

fn coalesce_keys(
    left: &Column,
    right: &Column,
    left_idx: &[usize],
    right_idx: &[usize],
) -> Column {
    // Always emit Float64 for float keys; otherwise follow left dtype with nulls.
    match left {
        Column::Float64(_) => {
            let l = left.as_f64_vec().unwrap();
            let r = right.as_f64_vec().unwrap_or_else(|| {
                panic!("merge: key dtype mismatch (expected float on right)")
            });
            let data: Vec<f64> = left_idx
                .iter()
                .zip(right_idx.iter())
                .map(|(&i, &j)| {
                    if i != usize::MAX {
                        l[i]
                    } else if j != usize::MAX {
                        r[j]
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            Column::Float64(rnumpy::NdArray::from_vec(data))
        }
        Column::Int64 { .. } => take_coalesce_int(left, right, left_idx, right_idx),
        Column::Bool { .. } => take_coalesce_bool(left, right, left_idx, right_idx),
        Column::Utf8 { .. } => take_coalesce_utf8(left, right, left_idx, right_idx),
    }
}

fn take_coalesce_int(
    left: &Column,
    right: &Column,
    left_idx: &[usize],
    right_idx: &[usize],
) -> Column {
    let (Column::Int64 {
        values: lv,
        nulls: ln,
    },
    Column::Int64 {
        values: rv,
        nulls: rn,
    }) = (left, right)
    else {
        panic!("merge: key dtype mismatch (int)")
    };
    let mut values = Vec::with_capacity(left_idx.len());
    let mut nulls = Vec::with_capacity(left_idx.len());
    for (&i, &j) in left_idx.iter().zip(right_idx.iter()) {
        if i != usize::MAX {
            values.push(lv[i]);
            nulls.push(ln[i]);
        } else if j != usize::MAX {
            values.push(rv[j]);
            nulls.push(rn[j]);
        } else {
            values.push(0);
            nulls.push(true);
        }
    }
    Column::Int64 { values, nulls }
}

fn take_coalesce_bool(
    left: &Column,
    right: &Column,
    left_idx: &[usize],
    right_idx: &[usize],
) -> Column {
    let (Column::Bool {
        values: lv,
        nulls: ln,
    },
    Column::Bool {
        values: rv,
        nulls: rn,
    }) = (left, right)
    else {
        panic!("merge: key dtype mismatch (bool)")
    };
    let mut values = Vec::with_capacity(left_idx.len());
    let mut nulls = Vec::with_capacity(left_idx.len());
    for (&i, &j) in left_idx.iter().zip(right_idx.iter()) {
        if i != usize::MAX {
            values.push(lv[i]);
            nulls.push(ln[i]);
        } else if j != usize::MAX {
            values.push(rv[j]);
            nulls.push(rn[j]);
        } else {
            values.push(false);
            nulls.push(true);
        }
    }
    Column::Bool { values, nulls }
}

fn take_coalesce_utf8(
    left: &Column,
    right: &Column,
    left_idx: &[usize],
    right_idx: &[usize],
) -> Column {
    let (Column::Utf8 {
        values: lv,
        nulls: ln,
    },
    Column::Utf8 {
        values: rv,
        nulls: rn,
    }) = (left, right)
    else {
        panic!("merge: key dtype mismatch (utf8)")
    };
    let mut values = Vec::with_capacity(left_idx.len());
    let mut nulls = Vec::with_capacity(left_idx.len());
    for (&i, &j) in left_idx.iter().zip(right_idx.iter()) {
        if i != usize::MAX {
            values.push(lv[i].clone());
            nulls.push(ln[i]);
        } else if j != usize::MAX {
            values.push(rv[j].clone());
            nulls.push(rn[j]);
        } else {
            values.push(String::new());
            nulls.push(true);
        }
    }
    Column::Utf8 { values, nulls }
}

fn take_with_nulls(col: &Column, idx: &[usize]) -> Column {
    match col {
        Column::Float64(a) => {
            let c = a.to_contiguous();
            let s = c.as_slice().unwrap();
            let data: Vec<f64> = idx
                .iter()
                .map(|&j| if j == usize::MAX { f64::NAN } else { s[j] })
                .collect();
            Column::Float64(rnumpy::NdArray::from_vec(data))
        }
        Column::Int64 { values, nulls } => {
            let mut v = Vec::with_capacity(idx.len());
            let mut n = Vec::with_capacity(idx.len());
            for &j in idx {
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
            let mut v = Vec::with_capacity(idx.len());
            let mut n = Vec::with_capacity(idx.len());
            for &j in idx {
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
            let mut v = Vec::with_capacity(idx.len());
            let mut n = Vec::with_capacity(idx.len());
            for &j in idx {
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

    #[test]
    fn merge_multi_key_and_outer() {
        let left = DataFrame::from_columns(vec![
            (
                "a".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 1.0, 2.0])),
            ),
            (
                "b".into(),
                Column::Float64(NdArray::from_vec(vec![10.0, 20.0, 10.0])),
            ),
            (
                "v".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0, 3.0])),
            ),
        ]);
        let right = DataFrame::from_columns(vec![
            (
                "a".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0, 3.0])),
            ),
            (
                "b".into(),
                Column::Float64(NdArray::from_vec(vec![10.0, 10.0, 10.0])),
            ),
            (
                "w".into(),
                Column::Float64(NdArray::from_vec(vec![100.0, 200.0, 300.0])),
            ),
        ]);
        let inner = merge_on(&left, &right, &["a", "b"], MergeHow::Inner);
        assert_eq!(inner.nrows(), 2); // (1,10) and (2,10)
        let outer = merge_on(&left, &right, &["a", "b"], MergeHow::Outer);
        assert_eq!(outer.nrows(), 4); // 2 matches + left (1,20) + right (3,10)
    }
}
