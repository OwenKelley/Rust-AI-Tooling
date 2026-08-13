//! Categoricals — mirrors a minimal `pandas.Categorical` / `.astype('category')`.
//!
//! v1: ordered-by-appearance categories, integer codes (`-1` = null), no ordered flag.

use rnumpy::NdArray;

use crate::frame::{Column, DataFrame};

/// Compact categorical encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Categorical {
    /// Category codes; `-1` means missing.
    codes: Vec<i64>,
    categories: Vec<String>,
}

impl Categorical {
    pub fn codes(&self) -> &[i64] {
        &self.codes
    }

    pub fn categories(&self) -> &[String] {
        &self.categories
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }

    pub fn n_categories(&self) -> usize {
        self.categories.len()
    }

    /// Build from optional strings; categories sorted lexicographically (pandas default).
    pub fn from_strings(values: &[Option<String>]) -> Self {
        let mut set = std::collections::BTreeSet::new();
        for v in values.iter().flatten() {
            set.insert(v.clone());
        }
        let categories: Vec<String> = set.into_iter().collect();
        let index: std::collections::HashMap<&str, i64> = categories
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as i64))
            .collect();
        let mut codes = Vec::with_capacity(values.len());
        for v in values {
            match v {
                None => codes.push(-1),
                Some(s) => codes.push(*index.get(s.as_str()).expect("category")),
            }
        }
        Self { codes, categories }
    }

    /// Decode codes back to optional strings.
    pub fn to_strings(&self) -> Vec<Option<String>> {
        self.codes
            .iter()
            .map(|&c| {
                if c < 0 {
                    None
                } else {
                    Some(self.categories[c as usize].clone())
                }
            })
            .collect()
    }
}

/// `Series.astype('category')` style helper from a UTF-8 column → codes frame.
///
/// Result columns: `codes` (f64), `n_categories` (repeated scalar as f64 for checksuming).
pub fn categorical_codes(df: &DataFrame, col: &str) -> DataFrame {
    let series = df.column(col);
    let values = match &series.data {
        Column::Utf8 { values, nulls } => values
            .iter()
            .zip(nulls.iter())
            .map(|(s, &n)| if n { None } else { Some(s.clone()) })
            .collect::<Vec<_>>(),
        _ => panic!("categorical_codes: column '{col}' must be utf8"),
    };
    let cat = Categorical::from_strings(&values);
    let codes_f: Vec<f64> = cat.codes.iter().map(|&c| c as f64).collect();
    let n = cat.len();
    let n_cats = cat.n_categories() as f64;
    DataFrame::from_columns(vec![
        ("codes".into(), Column::Float64(NdArray::from_vec(codes_f))),
        (
            "n_categories".into(),
            Column::Float64(NdArray::from_vec(vec![n_cats; n])),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_strings_codes() {
        let vals = vec![
            Some("a".into()),
            Some("b".into()),
            Some("a".into()),
            None,
            Some("b".into()),
        ];
        let cat = Categorical::from_strings(&vals);
        assert_eq!(cat.codes(), &[0, 1, 0, -1, 1]);
        assert_eq!(cat.categories(), &["a".to_string(), "b".to_string()]);
        assert_eq!(cat.to_strings()[3], None);
    }
}
