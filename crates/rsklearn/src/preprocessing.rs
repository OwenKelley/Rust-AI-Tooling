//! Preprocessing — `StandardScaler`, `LabelEncoder`.

use std::collections::HashMap;

use rnumpy::NdArray;

use crate::model_selection::from_shape_helper;

/// `sklearn.preprocessing.StandardScaler` (with_mean=True, with_std=True).
#[derive(Debug, Clone)]
pub struct StandardScaler {
    pub mean_: Vec<f64>,
    pub scale_: Vec<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean_: Vec::new(),
            scale_: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &NdArray) -> &mut Self {
        assert_eq!(x.shape().len(), 2);
        let (n, d) = (x.shape()[0], x.shape()[1]);
        assert!(n > 0);
        self.mean_ = vec![0.0; d];
        self.scale_ = vec![0.0; d];
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..n {
                s += x.get(&[i, j]);
            }
            self.mean_[j] = s / n as f64;
        }
        for j in 0..d {
            let mut s = 0.0;
            for i in 0..n {
                let dlt = x.get(&[i, j]) - self.mean_[j];
                s += dlt * dlt;
            }
            let var = s / n as f64; // population (ddof=0) like sklearn
            self.scale_[j] = if var > 0.0 { var.sqrt() } else { 1.0 };
        }
        self
    }

    pub fn transform(&self, x: &NdArray) -> NdArray {
        assert_eq!(x.shape().len(), 2);
        let (n, d) = (x.shape()[0], x.shape()[1]);
        assert_eq!(d, self.mean_.len());
        let mut out = Vec::with_capacity(n * d);
        for i in 0..n {
            for j in 0..d {
                out.push((x.get(&[i, j]) - self.mean_[j]) / self.scale_[j]);
            }
        }
        from_shape_helper(&[n, d], out)
    }

    pub fn fit_transform(&mut self, x: &NdArray) -> NdArray {
        self.fit(x);
        self.transform(x)
    }
}

impl Default for StandardScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// `sklearn.preprocessing.LabelEncoder`.
#[derive(Debug, Clone, Default)]
pub struct LabelEncoder {
    pub classes_: Vec<String>,
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, y: &[String]) -> &mut Self {
        let mut seen = HashMap::new();
        self.classes_.clear();
        for s in y {
            if !seen.contains_key(s) {
                seen.insert(s.clone(), self.classes_.len());
                self.classes_.push(s.clone());
            }
        }
        self.classes_.sort();
        self
    }

    pub fn transform(&self, y: &[String]) -> Vec<i64> {
        y.iter()
            .map(|s| {
                self.classes_
                    .iter()
                    .position(|c| c == s)
                    .expect("unseen label") as i64
            })
            .collect()
    }

    pub fn fit_transform(&mut self, y: &[String]) -> Vec<i64> {
        self.fit(y);
        self.transform(y)
    }

    pub fn inverse_transform(&self, y: &[i64]) -> Vec<String> {
        y.iter()
            .map(|&i| self.classes_[i as usize].clone())
            .collect()
    }
}
