//! Train/test split — mirrors `sklearn.model_selection.train_test_split`.

use rnumpy::NdArray;

fn lcg_shuffle(indices: &mut [usize], seed: u64) {
    let mut state = seed | 1;
    for i in (1..indices.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

fn take_rows(x: &NdArray, indices: &[usize]) -> NdArray {
    assert_eq!(x.shape().len(), 2, "X must be 2D");
    let ncols = x.shape()[1];
    let mut data = Vec::with_capacity(indices.len() * ncols);
    for &i in indices {
        for j in 0..ncols {
            data.push(x.get(&[i, j]));
        }
    }
    from_shape_helper(&[indices.len(), ncols], data)
}

pub(crate) fn from_shape_helper(shape: &[usize], data: Vec<f64>) -> NdArray {
    assert_eq!(NdArray::shape_len(shape), data.len());
    NdArray::from_vec(data).reshape_view(shape)
}

fn take_y(y: &[f64], indices: &[usize]) -> Vec<f64> {
    indices.iter().map(|&i| y[i]).collect()
}

/// `train_test_split(X, y, test_size=…, random_state=…, shuffle=…)`.
///
/// With `shuffle=false`, matches sklearn: train = first rows, test = last
/// `ceil(test_size * n)` rows.
pub fn train_test_split(
    x: &NdArray,
    y: &[f64],
    test_size: f64,
    random_state: u64,
    shuffle: bool,
) -> (NdArray, NdArray, Vec<f64>, Vec<f64>) {
    assert_eq!(x.shape().len(), 2);
    let n = x.shape()[0];
    assert_eq!(y.len(), n);
    assert!(test_size > 0.0 && test_size < 1.0);
    // sklearn: n_test = ceil(test_size * n_samples)
    let n_test = ((n as f64) * test_size).ceil() as usize;
    let n_test = n_test.clamp(1, n - 1);
    let n_train = n - n_test;
    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        lcg_shuffle(&mut indices, random_state);
    }
    // After optional shuffle: train = first n_train, test = last n_test (sklearn).
    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];
    (
        take_rows(x, train_idx),
        take_rows(x, test_idx),
        take_y(y, train_idx),
        take_y(y, test_idx),
    )
}
