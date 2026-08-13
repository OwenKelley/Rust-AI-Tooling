//! K-means clustering.

use rnumpy::NdArray;

use crate::model_selection::from_shape_helper;

#[derive(Debug, Clone)]
pub struct KMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub random_state: u64,
    pub cluster_centers_: Option<NdArray>,
    pub labels_: Vec<i64>,
}

impl KMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 100,
            random_state: 0,
            cluster_centers_: None,
            labels_: Vec::new(),
        }
    }

    pub fn fit(&mut self, x: &NdArray) -> &mut Self {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        assert!(self.n_clusters <= n);
        // Init: first k rows (deterministic; matches sklearn init=X[:k])
        let mut centers = Vec::with_capacity(self.n_clusters * d);
        for i in 0..self.n_clusters {
            for j in 0..d {
                centers.push(x.get(&[i, j]));
            }
        }
        let mut labels = vec![0i64; n];
        for _ in 0..self.max_iter {
            let mut changed = false;
            for i in 0..n {
                let mut best = 0usize;
                let mut best_d = f64::INFINITY;
                for c in 0..self.n_clusters {
                    let mut dist = 0.0;
                    for j in 0..d {
                        let diff = x.get(&[i, j]) - centers[c * d + j];
                        dist += diff * diff;
                    }
                    if dist < best_d {
                        best_d = dist;
                        best = c;
                    }
                }
                if labels[i] != best as i64 {
                    changed = true;
                    labels[i] = best as i64;
                }
            }
            let mut sums = vec![0.0; self.n_clusters * d];
            let mut counts = vec![0usize; self.n_clusters];
            for i in 0..n {
                let c = labels[i] as usize;
                counts[c] += 1;
                for j in 0..d {
                    sums[c * d + j] += x.get(&[i, j]);
                }
            }
            for c in 0..self.n_clusters {
                if counts[c] == 0 {
                    continue;
                }
                for j in 0..d {
                    centers[c * d + j] = sums[c * d + j] / counts[c] as f64;
                }
            }
            if !changed {
                break;
            }
        }
        self.cluster_centers_ = Some(from_shape_helper(&[self.n_clusters, d], centers));
        self.labels_ = labels;
        self
    }

    pub fn predict(&self, x: &NdArray) -> Vec<i64> {
        let centers = self.cluster_centers_.as_ref().expect("not fitted");
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut best = 0usize;
            let mut best_d = f64::INFINITY;
            for c in 0..self.n_clusters {
                let mut dist = 0.0;
                for j in 0..d {
                    let diff = x.get(&[i, j]) - centers.get(&[c, j]);
                    dist += diff * diff;
                }
                if dist < best_d {
                    best_d = dist;
                    best = c;
                }
            }
            out.push(best as i64);
        }
        out
    }
}
