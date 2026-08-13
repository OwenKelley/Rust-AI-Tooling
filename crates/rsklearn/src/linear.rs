//! Linear / logistic regression.

use rnumpy::NdArray;

/// Ordinary least squares via normal equations (with bias column).
#[derive(Debug, Clone, Default)]
pub struct LinearRegression {
    pub coef_: Vec<f64>,
    pub intercept_: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, x: &NdArray, y: &[f64]) -> &mut Self {
        assert_eq!(x.shape().len(), 2);
        let (n, d) = (x.shape()[0], x.shape()[1]);
        assert_eq!(y.len(), n);
        // Solve (X'X) β = X'y with augmented X = [1 | X]
        let p = d + 1;
        let mut xtx = vec![0.0; p * p];
        let mut xty = vec![0.0; p];
        for i in 0..n {
            let mut row = vec![1.0; p];
            for j in 0..d {
                row[j + 1] = x.get(&[i, j]);
            }
            for a in 0..p {
                xty[a] += row[a] * y[i];
                for b in 0..p {
                    xtx[a * p + b] += row[a] * row[b];
                }
            }
        }
        let beta = solve_symmetric(&xtx, &xty, p);
        self.intercept_ = beta[0];
        self.coef_ = beta[1..].to_vec();
        self
    }

    pub fn predict(&self, x: &NdArray) -> Vec<f64> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        assert_eq!(d, self.coef_.len());
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = self.intercept_;
            for j in 0..d {
                v += self.coef_[j] * x.get(&[i, j]);
            }
            out.push(v);
        }
        out
    }
}

/// Binary logistic regression with GD (labels in {0,1}).
#[derive(Debug, Clone)]
pub struct LogisticRegression {
    pub coef_: Vec<f64>,
    pub intercept_: f64,
    pub lr: f64,
    pub max_iter: usize,
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self {
            coef_: Vec::new(),
            intercept_: 0.0,
            lr: 0.1,
            max_iter: 500,
        }
    }
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fit(&mut self, x: &NdArray, y: &[f64]) -> &mut Self {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        assert_eq!(y.len(), n);
        self.coef_ = vec![0.0; d];
        self.intercept_ = 0.0;
        for _ in 0..self.max_iter {
            let mut g_w = vec![0.0; d];
            let mut g_b = 0.0;
            for i in 0..n {
                let mut z = self.intercept_;
                for j in 0..d {
                    z += self.coef_[j] * x.get(&[i, j]);
                }
                let p = 1.0 / (1.0 + (-z).exp());
                let err = p - y[i];
                g_b += err;
                for j in 0..d {
                    g_w[j] += err * x.get(&[i, j]);
                }
            }
            self.intercept_ -= self.lr * g_b / n as f64;
            for j in 0..d {
                self.coef_[j] -= self.lr * g_w[j] / n as f64;
            }
        }
        self
    }

    pub fn predict_proba(&self, x: &NdArray) -> Vec<f64> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut z = self.intercept_;
            for j in 0..d {
                z += self.coef_[j] * x.get(&[i, j]);
            }
            out.push(1.0 / (1.0 + (-z).exp()));
        }
        out
    }

    pub fn predict(&self, x: &NdArray) -> Vec<i64> {
        self.predict_proba(x)
            .into_iter()
            .map(|p| if p >= 0.5 { 1 } else { 0 })
            .collect()
    }
}

fn solve_symmetric(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    // Gaussian elimination with partial pivoting on a copy.
    let mut m = a.to_vec();
    let mut y = b.to_vec();
    for k in 0..n {
        let mut piv = k;
        for i in k + 1..n {
            if m[i * n + k].abs() > m[piv * n + k].abs() {
                piv = i;
            }
        }
        for j in 0..n {
            m.swap(k * n + j, piv * n + j);
        }
        y.swap(k, piv);
        let diag = m[k * n + k];
        assert!(diag.abs() > 1e-12, "singular system");
        for i in k + 1..n {
            let f = m[i * n + k] / diag;
            for j in k..n {
                m[i * n + j] -= f * m[k * n + j];
            }
            y[i] -= f * y[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in i + 1..n {
            s -= m[i * n + j] * x[j];
        }
        x[i] = s / m[i * n + i];
    }
    x
}
