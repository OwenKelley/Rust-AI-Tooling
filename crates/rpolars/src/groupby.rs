//! Group-by aggregations — mirrors `DataFrame.group_by(...).agg(...)`.

use std::collections::HashMap;

use rarrow::{Array, Float64Array, Int64Array};

use crate::frame::{hash_key, key_eq, DataFrame};
use crate::series::Series;

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
            other => panic!("agg: unknown '{other}'"),
        }
    }
}

pub struct GroupBy<'a> {
    df: &'a DataFrame,
    keys: Vec<&'a str>,
}

impl<'a> GroupBy<'a> {
    pub fn new(df: &'a DataFrame, keys: &[&'a str]) -> Self {
        assert!(!keys.is_empty(), "groupby: need keys");
        for &k in keys {
            let _ = df.column(k);
        }
        Self {
            df,
            keys: keys.to_vec(),
        }
    }

    /// `aggs` is `(column, agg)` — output column named `{col}_{agg}` except count → `{col}_count`.
    pub fn agg(&self, aggs: &[(&str, Agg)]) -> DataFrame {
        let n = self.df.height();
        let key_arrs: Vec<&Array> = self.keys.iter().map(|k| &self.df.column(k).data).collect();

        let mut first_row: Vec<usize> = Vec::new();
        let mut group_of = vec![0usize; n];
        let mut map: HashMap<u64, Vec<usize>> = HashMap::new();

        for i in 0..n {
            let h = hash_key(&key_arrs, i);
            let entry = map.entry(h).or_default();
            let mut found = None;
            for &g in entry.iter() {
                let j = first_row[g];
                if key_eq(&key_arrs, i, &key_arrs, j) {
                    found = Some(g);
                    break;
                }
            }
            let g = if let Some(g) = found {
                g
            } else {
                let g = first_row.len();
                first_row.push(i);
                entry.push(g);
                g
            };
            group_of[i] = g;
        }

        let ng = first_row.len();
        let mut out_cols: Vec<Series> = Vec::new();

        for &k in &self.keys {
            let src = &self.df.column(k).data;
            let indices = &first_row;
            out_cols.push(Series::new(k.to_string(), crate::frame::take_array_pub(src, indices)));
        }

        for &(col, agg) in aggs {
            let src = &self.df.column(col).data;
            let name = match agg {
                Agg::Sum => format!("{col}_sum"),
                Agg::Mean => format!("{col}_mean"),
                Agg::Count => format!("{col}_count"),
                Agg::Min => format!("{col}_min"),
                Agg::Max => format!("{col}_max"),
            };
            out_cols.push(Series::new(name, aggregate(src, &group_of, ng, agg)));
        }

        DataFrame::new(out_cols)
    }
}

fn aggregate(src: &Array, group_of: &[usize], ng: usize, agg: Agg) -> Array {
    match src {
        Array::Float64(a) => {
            let mut sums = vec![0.0; ng];
            let mut mins = vec![f64::INFINITY; ng];
            let mut maxs = vec![f64::NEG_INFINITY; ng];
            let mut counts = vec![0i64; ng];
            for (i, &g) in group_of.iter().enumerate() {
                if a.nulls[i] {
                    continue;
                }
                let v = a.values[i];
                sums[g] += v;
                counts[g] += 1;
                mins[g] = mins[g].min(v);
                maxs[g] = maxs[g].max(v);
            }
            match agg {
                Agg::Sum => Array::Float64(Float64Array {
                    values: sums,
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                Agg::Mean => Array::Float64(Float64Array {
                    values: sums
                        .iter()
                        .zip(counts.iter())
                        .map(|(&s, &c)| if c == 0 { 0.0 } else { s / c as f64 })
                        .collect(),
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                Agg::Count => Array::Int64(Int64Array {
                    values: counts,
                    nulls: vec![false; ng],
                }),
                Agg::Min => Array::Float64(Float64Array {
                    values: mins
                        .iter()
                        .enumerate()
                        .map(|(g, &v)| if counts[g] == 0 { 0.0 } else { v })
                        .collect(),
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                Agg::Max => Array::Float64(Float64Array {
                    values: maxs
                        .iter()
                        .enumerate()
                        .map(|(g, &v)| if counts[g] == 0 { 0.0 } else { v })
                        .collect(),
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
            }
        }
        Array::Int64(a) => {
            let mut sums = vec![0i64; ng];
            let mut mins = vec![i64::MAX; ng];
            let mut maxs = vec![i64::MIN; ng];
            let mut counts = vec![0i64; ng];
            for (i, &g) in group_of.iter().enumerate() {
                if a.nulls[i] {
                    continue;
                }
                let v = a.values[i];
                sums[g] = sums[g].saturating_add(v);
                counts[g] += 1;
                mins[g] = mins[g].min(v);
                maxs[g] = maxs[g].max(v);
            }
            match agg {
                Agg::Sum => Array::Int64(Int64Array {
                    values: sums,
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                Agg::Mean => Array::Float64(Float64Array {
                    values: sums
                        .iter()
                        .zip(counts.iter())
                        .map(|(&s, &c)| if c == 0 { 0.0 } else { s as f64 / c as f64 })
                        .collect(),
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                Agg::Count => Array::Int64(Int64Array {
                    values: counts,
                    nulls: vec![false; ng],
                }),
                Agg::Min => Array::Int64(Int64Array {
                    values: mins
                        .iter()
                        .enumerate()
                        .map(|(g, &v)| if counts[g] == 0 { 0 } else { v })
                        .collect(),
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                Agg::Max => Array::Int64(Int64Array {
                    values: maxs
                        .iter()
                        .enumerate()
                        .map(|(g, &v)| if counts[g] == 0 { 0 } else { v })
                        .collect(),
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
            }
        }
        Array::Boolean(a) => {
            // count only (or sum as 0/1)
            let mut sums = vec![0i64; ng];
            let mut counts = vec![0i64; ng];
            for (i, &g) in group_of.iter().enumerate() {
                if a.nulls[i] {
                    continue;
                }
                counts[g] += 1;
                if a.values[i] {
                    sums[g] += 1;
                }
            }
            match agg {
                Agg::Count => Array::Int64(Int64Array {
                    values: counts,
                    nulls: vec![false; ng],
                }),
                Agg::Sum => Array::Int64(Int64Array {
                    values: sums,
                    nulls: counts.iter().map(|&c| c == 0).collect(),
                }),
                _ => panic!("bool column only supports sum/count aggs"),
            }
        }
        Array::Utf8(_) => match agg {
            Agg::Count => {
                let mut counts = vec![0i64; ng];
                for (i, &g) in group_of.iter().enumerate() {
                    if let Array::Utf8(a) = src {
                        if a.values[i].is_some() {
                            counts[g] += 1;
                        }
                    }
                }
                Array::Int64(Int64Array {
                    values: counts,
                    nulls: vec![false; ng],
                })
            }
            _ => panic!("utf8 column only supports count agg"),
        },
        Array::TimestampNs(_) => match agg {
            Agg::Count => {
                let mut counts = vec![0i64; ng];
                for (i, &g) in group_of.iter().enumerate() {
                    if let Array::TimestampNs(a) = src {
                        if !a.nulls[i] {
                            counts[g] += 1;
                        }
                    }
                }
                Array::Int64(Int64Array {
                    values: counts,
                    nulls: vec![false; ng],
                })
            }
            _ => panic!("timestamp column only supports count agg"),
        },
        Array::ListFloat64(_) => panic!("list column aggs not supported"),
        Array::DictionaryUtf8(_) => match agg {
            Agg::Count => {
                let mut counts = vec![0i64; ng];
                for (i, &g) in group_of.iter().enumerate() {
                    if let Array::DictionaryUtf8(a) = src {
                        if !a.nulls[i] {
                            counts[g] += 1;
                        }
                    }
                }
                Array::Int64(Int64Array {
                    values: counts,
                    nulls: vec![false; ng],
                })
            }
            _ => panic!("dictionary column only supports count agg"),
        },
    }
}
