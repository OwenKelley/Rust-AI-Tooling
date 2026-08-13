//! Joins — mirrors Polars `DataFrame.join` (inner / left).

use std::collections::{HashMap, HashSet};

use rarrow::Array;

use crate::frame::{hash_key, key_eq, DataFrame};
use crate::series::Series;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinHow {
    Inner,
    Left,
}

impl JoinHow {
    pub fn parse(s: &str) -> Self {
        match s {
            "inner" => JoinHow::Inner,
            "left" => JoinHow::Left,
            other => panic!("join: unknown how '{other}'"),
        }
    }
}

pub fn join(left: &DataFrame, right: &DataFrame, on: &[&str], how: JoinHow) -> DataFrame {
    assert!(!on.is_empty(), "join: need keys");
    for &k in on {
        let _ = left.column(k);
        let _ = right.column(k);
    }

    let left_keys: Vec<&Array> = on.iter().map(|k| &left.column(k).data).collect();
    let right_keys: Vec<&Array> = on.iter().map(|k| &right.column(k).data).collect();

    let mut right_map: HashMap<u64, Vec<usize>> = HashMap::new();
    for j in 0..right.height() {
        let h = hash_key(&right_keys, j);
        right_map.entry(h).or_default().push(j);
    }

    let mut li = Vec::new();
    let mut ri = Vec::new();

    for i in 0..left.height() {
        let h = hash_key(&left_keys, i);
        let mut matched = false;
        if let Some(cands) = right_map.get(&h) {
            for &j in cands {
                if key_eq(&left_keys, i, &right_keys, j) {
                    li.push(i);
                    ri.push(j);
                    matched = true;
                }
            }
        }
        if !matched && how == JoinHow::Left {
            li.push(i);
            ri.push(usize::MAX);
        }
    }

    let on_set: HashSet<&str> = on.iter().copied().collect();
    let mut out = Vec::new();

    // keys from left
    for &k in on {
        out.push(left.column(k).take(&li));
    }

    // left non-key columns
    for s in left.columns() {
        if on_set.contains(s.name.as_str()) {
            continue;
        }
        out.push(s.take(&li));
    }

    // right non-key columns (suffix _right on collision)
    let left_names: HashSet<String> = left
        .columns()
        .iter()
        .map(|c| c.name.clone())
        .collect();
    for s in right.columns() {
        if on_set.contains(s.name.as_str()) {
            continue;
        }
        let name = if left_names.contains(&s.name) {
            format!("{}_right", s.name)
        } else {
            s.name.clone()
        };
        let data = take_right(&s.data, &ri);
        out.push(Series::new(name, data));
    }

    DataFrame::new(out)
}

fn take_right(data: &Array, indices: &[usize]) -> Array {
    // usize::MAX → null row
    match data {
        Array::Float64(a) => {
            let mut values = Vec::with_capacity(indices.len());
            let mut nulls = Vec::with_capacity(indices.len());
            for &i in indices {
                if i == usize::MAX {
                    values.push(0.0);
                    nulls.push(true);
                } else {
                    values.push(a.values[i]);
                    nulls.push(a.nulls[i]);
                }
            }
            Array::Float64(rarrow::Float64Array { values, nulls })
        }
        Array::Int64(a) => {
            let mut values = Vec::with_capacity(indices.len());
            let mut nulls = Vec::with_capacity(indices.len());
            for &i in indices {
                if i == usize::MAX {
                    values.push(0);
                    nulls.push(true);
                } else {
                    values.push(a.values[i]);
                    nulls.push(a.nulls[i]);
                }
            }
            Array::Int64(rarrow::Int64Array { values, nulls })
        }
        Array::Boolean(a) => {
            let mut values = Vec::with_capacity(indices.len());
            let mut nulls = Vec::with_capacity(indices.len());
            for &i in indices {
                if i == usize::MAX {
                    values.push(false);
                    nulls.push(true);
                } else {
                    values.push(a.values[i]);
                    nulls.push(a.nulls[i]);
                }
            }
            Array::Boolean(rarrow::BooleanArray { values, nulls })
        }
        Array::Utf8(a) => {
            let mut values = Vec::with_capacity(indices.len());
            for &i in indices {
                if i == usize::MAX {
                    values.push(None);
                } else {
                    values.push(a.values[i].clone());
                }
            }
            Array::Utf8(rarrow::StringArray { values })
        }
        Array::TimestampNs(a) => {
            let mut values = Vec::with_capacity(indices.len());
            let mut nulls = Vec::with_capacity(indices.len());
            for &i in indices {
                if i == usize::MAX {
                    values.push(0);
                    nulls.push(true);
                } else {
                    values.push(a.values[i]);
                    nulls.push(a.nulls[i]);
                }
            }
            Array::TimestampNs(rarrow::Int64Array { values, nulls })
        }
        Array::ListFloat64(_) => panic!("join: ListFloat64 not supported"),
        Array::DictionaryUtf8(_) => panic!("join: DictionaryUtf8 not supported"),
    }
}
