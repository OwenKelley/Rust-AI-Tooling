//! Array creation — mirrors `numpy` constructors.

use crate::NdArray;

/// `np.zeros(shape)`
pub fn zeros(shape: &[usize]) -> NdArray {
    NdArray::zeros(shape)
}

/// `np.ones(shape)`
pub fn ones(shape: &[usize]) -> NdArray {
    NdArray::ones(shape)
}

/// `np.full(shape, fill_value)`
pub fn full(shape: &[usize], fill_value: f64) -> NdArray {
    NdArray::from_elem(shape, fill_value)
}

/// `np.arange(start, stop, step)` — exclusive `stop`, like NumPy.
pub fn arange(start: f64, stop: f64, step: f64) -> NdArray {
    assert!(step != 0.0, "step must be non-zero");
    let mut values = Vec::new();
    if step > 0.0 {
        let mut x = start;
        while x < stop {
            values.push(x);
            x += step;
        }
    } else {
        let mut x = start;
        while x > stop {
            values.push(x);
            x += step;
        }
    }
    NdArray::from_vec(values)
}

/// `np.linspace(start, stop, num)` — inclusive endpoints.
pub fn linspace(start: f64, stop: f64, num: usize) -> NdArray {
    if num == 0 {
        return NdArray::zeros(&[0]);
    }
    if num == 1 {
        return NdArray::from_vec(vec![start]);
    }
    let step = (stop - start) / (num as f64 - 1.0);
    let values: Vec<f64> = (0..num).map(|i| start + step * i as f64).collect();
    NdArray::from_vec(values)
}

/// `np.eye(n)` — 2D identity.
pub fn eye(n: usize) -> NdArray {
    let mut a = zeros(&[n, n]);
    for i in 0..n {
        a[[i, i]] = 1.0;
    }
    a
}

/// Deterministic filled array from a linear congruential RNG (parity-friendly).
/// Not a NumPy API; used so Python and Rust share identical inputs without
/// depending on NumPy's RNG stream.
pub fn seeded_uniform(shape: &[usize], seed: u64, low: f64, high: f64) -> NdArray {
    let mut state = seed;
    let total: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let mut data = Vec::with_capacity(total);
    let span = high - low;
    for _ in 0..total {
        // Numerical Recipes LCG
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        // 24-bit fraction in [0, 1) — must match python/.../rng.py
        let u = ((state >> 8) & 0xFF_FFFF) as f64 / ((1u64 << 24) as f64);
        data.push(low + span * u);
    }
    NdArray::from_shape_vec(shape, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_abs_diff_eq;

    #[test]
    fn zeros_shape() {
        let a = zeros(&[2, 3]);
        assert_eq!(a.shape(), &[2, 3]);
        assert!(a.iter().all(|x| x == 0.0));
    }

    #[test]
    fn arange_matches_numpy_semantics() {
        let a = arange(0.0, 5.0, 1.0);
        assert_eq!(a.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn linspace_endpoints() {
        let a = linspace(0.0, 1.0, 5);
        assert_abs_diff_eq(a[0], 0.0, 1e-12);
        assert_abs_diff_eq(a[4], 1.0, 1e-12);
    }

    #[test]
    fn eye_diagonal() {
        let a = eye(3);
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 2]], 0.0);
    }
}
