//! `rsklearn` — scikit-learn-shaped classical ML (`std` only, on `rnumpy`).

pub mod cluster;
pub mod export;
pub mod feature_extraction;
pub mod linear;
pub mod metrics;
pub mod model_selection;
pub mod neighbors;
pub mod preprocessing;

pub use cluster::KMeans;
pub use export::ModelArtifact;
pub use feature_extraction::{CountVectorizer, HashingVectorizer};
pub use linear::{LinearRegression, LogisticRegression};
pub use metrics::{
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score,
};
pub use model_selection::train_test_split;
pub use neighbors::{KNeighborsClassifier, KNeighborsRegressor};
pub use preprocessing::{LabelEncoder, StandardScaler};

#[cfg(test)]
mod tests {
    use super::*;
    use rnumpy::NdArray;

    #[test]
    fn linear_and_metrics() {
        // y = 1 + 2*x0 + 3*x1
        let x = NdArray::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0]).reshape_view(&[4, 2]);
        let y: Vec<f64> = (0..4)
            .map(|i| 1.0 + 2.0 * x.get(&[i, 0]) + 3.0 * x.get(&[i, 1]))
            .collect();
        let mut lr = LinearRegression::new();
        lr.fit(&x, &y);
        let pred = lr.predict(&x);
        assert!(r2_score(&y, &pred) > 0.999);
        assert!(mean_squared_error(&y, &pred) < 1e-8);
    }

    #[test]
    fn knn_and_kmeans() {
        let x = NdArray::from_vec(vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).reshape_view(&[4, 2]);
        let y = vec![0, 1, 0, 1];
        let mut knn = KNeighborsClassifier::new(1);
        knn.fit(&x, &y);
        assert_eq!(knn.predict(&x), y);
        let mut km = KMeans::new(2);
        km.fit(&x);
        assert_eq!(km.labels_.len(), 4);
    }

    #[test]
    fn scaler_split() {
        let x = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape_view(&[4, 2]);
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let mut sc = StandardScaler::new();
        let xs = sc.fit_transform(&x);
        assert!((xs.get(&[0, 0]) + xs.get(&[1, 0]) + xs.get(&[2, 0]) + xs.get(&[3, 0])).abs() < 1e-9);
        let (tr, te, _, _) = train_test_split(&x, &y, 0.25, 0, true);
        assert_eq!(tr.shape()[0] + te.shape()[0], 4);
    }
}
