//! Reductions — mirrors `numpy` aggregate functions.
//!
//! Contiguous slice paths use simple `<`/`>` loops so LLVM can vectorize.
//! (Iterator `min_by(total_cmp)` is correct for NaNs but much slower.)

use crate::NdArray;

/// `np.sum(a)`
pub fn sum(a: &NdArray) -> f64 {
    a.sum()
}

/// `np.mean(a)`
pub fn mean(a: &NdArray) -> f64 {
    a.mean().unwrap_or(f64::NAN)
}

/// `np.min(a)` — prefers contiguous SIMD-friendly scan.
pub fn min(a: &NdArray) -> f64 {
    if let Some(slice) = a.as_slice_memory_order() {
        min_slice(slice)
    } else {
        *a.iter()
            .min_by(|x, y| x.total_cmp(y))
            .expect("non-empty array")
    }
}

/// `np.max(a)`
pub fn max(a: &NdArray) -> f64 {
    if let Some(slice) = a.as_slice_memory_order() {
        max_slice(slice)
    } else {
        *a.iter()
            .max_by(|x, y| x.total_cmp(y))
            .expect("non-empty array")
    }
}

/// `np.var(a)` — population variance (ddof=0), NumPy default.
pub fn var(a: &NdArray) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let m = mean(a);
    let sum_sq = if let Some(slice) = a.as_slice_memory_order() {
        let mut acc = 0.0;
        for &x in slice {
            let d = x - m;
            acc += d * d;
        }
        acc
    } else {
        a.iter()
            .map(|&x| {
                let d = x - m;
                d * d
            })
            .sum::<f64>()
    };
    sum_sq / n as f64
}

/// `np.std(a)` — population std (ddof=0), NumPy default.
pub fn std(a: &NdArray) -> f64 {
    var(a).sqrt()
}

/// `np.argmin(a)` — flat index.
pub fn argmin(a: &NdArray) -> usize {
    if let Some(slice) = a.as_slice_memory_order() {
        argmin_slice(slice)
    } else {
        a.iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| x.total_cmp(y))
            .map(|(i, _)| i)
            .expect("non-empty array")
    }
}

/// `np.argmax(a)` — flat index.
pub fn argmax(a: &NdArray) -> usize {
    if let Some(slice) = a.as_slice_memory_order() {
        argmax_slice(slice)
    } else {
        a.iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.total_cmp(y))
            .map(|(i, _)| i)
            .expect("non-empty array")
    }
}

#[inline]
fn min_slice(slice: &[f64]) -> f64 {
    assert!(!slice.is_empty(), "non-empty array");
    let mut m = slice[0];
    for &x in &slice[1..] {
        if x < m {
            m = x;
        }
    }
    m
}

#[inline]
fn max_slice(slice: &[f64]) -> f64 {
    assert!(!slice.is_empty(), "non-empty array");
    let mut m = slice[0];
    for &x in &slice[1..] {
        if x > m {
            m = x;
        }
    }
    m
}

#[inline]
fn argmin_slice(slice: &[f64]) -> usize {
    assert!(!slice.is_empty(), "non-empty array");
    let mut best_i = 0;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v < best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

#[inline]
fn argmax_slice(slice: &[f64]) -> usize {
    assert!(!slice.is_empty(), "non-empty array");
    let mut best_i = 0;
    let mut best_v = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::arange;

    #[test]
    fn sum_arange() {
        let a = arange(1.0, 5.0, 1.0);
        assert_eq!(sum(&a), 10.0);
    }

    #[test]
    fn argmax_basic() {
        let a = arange(1.0, 5.0, 1.0);
        assert_eq!(argmax(&a), 3);
    }

    #[test]
    fn min_max_basic() {
        let a = arange(1.0, 5.0, 1.0);
        assert_eq!(min(&a), 1.0);
        assert_eq!(max(&a), 4.0);
    }
}
