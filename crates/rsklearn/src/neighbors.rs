//! Brute-force k-NN.

use rnumpy::NdArray;

#[derive(Debug, Clone)]
pub struct KNeighborsClassifier {
    pub n_neighbors: usize,
    x_train: Option<NdArray>,
    y_train: Vec<i64>,
}

impl KNeighborsClassifier {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            x_train: None,
            y_train: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &NdArray, y: &[i64]) -> &mut Self {
        assert_eq!(x.shape()[0], y.len());
        self.x_train = Some(x.clone());
        self.y_train = y.to_vec();
        self
    }

    pub fn predict(&self, x: &NdArray) -> Vec<i64> {
        let train = self.x_train.as_ref().expect("not fitted");
        let k = self.n_neighbors.min(self.y_train.len());
        let n = x.shape()[0];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut dists: Vec<(f64, i64)> = Vec::with_capacity(train.shape()[0]);
            for t in 0..train.shape()[0] {
                let mut d = 0.0;
                for j in 0..x.shape()[1] {
                    let diff = x.get(&[i, j]) - train.get(&[t, j]);
                    d += diff * diff;
                }
                dists.push((d, self.y_train[t]));
            }
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut votes: std::collections::HashMap<i64, usize> =
                std::collections::HashMap::new();
            for &(_, lab) in dists.iter().take(k) {
                *votes.entry(lab).or_default() += 1;
            }
            let best = votes
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)))
                .map(|(lab, _)| lab)
                .unwrap();
            out.push(best);
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct KNeighborsRegressor {
    pub n_neighbors: usize,
    x_train: Option<NdArray>,
    y_train: Vec<f64>,
}

impl KNeighborsRegressor {
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors,
            x_train: None,
            y_train: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &NdArray, y: &[f64]) -> &mut Self {
        assert_eq!(x.shape()[0], y.len());
        self.x_train = Some(x.clone());
        self.y_train = y.to_vec();
        self
    }

    pub fn predict(&self, x: &NdArray) -> Vec<f64> {
        let train = self.x_train.as_ref().expect("not fitted");
        let k = self.n_neighbors.min(self.y_train.len());
        let n = x.shape()[0];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut dists: Vec<(f64, f64)> = Vec::with_capacity(train.shape()[0]);
            for t in 0..train.shape()[0] {
                let mut d = 0.0;
                for j in 0..x.shape()[1] {
                    let diff = x.get(&[i, j]) - train.get(&[t, j]);
                    d += diff * diff;
                }
                dists.push((d, self.y_train[t]));
            }
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mean = dists.iter().take(k).map(|(_, v)| *v).sum::<f64>() / k as f64;
            out.push(mean);
        }
        out
    }
}
