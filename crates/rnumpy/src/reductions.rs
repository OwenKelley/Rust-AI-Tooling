//! Reductions — mirrors `numpy` aggregate functions.
//!
//! Contiguous slice paths use simple `<`/`>` loops so LLVM can vectorize.
//! Optional `axis` mirrors NumPy's `axis=` (single axis only for now).

use crate::NdArray;

/// `np.sum(a)` or `np.sum(a, axis=axis)`
pub fn sum(a: &NdArray) -> f64 {
    a.sum()
}

/// `np.sum(a, axis=axis)` — returns an array with that axis removed.
pub fn sum_axis(a: &NdArray, axis: usize) -> NdArray {
    if let Some(out) = sum_axis_fast(a, axis) {
        return out;
    }
    reduce_axis(a, axis, 0.0, |acc, x| acc + x)
}

fn sum_axis_fast(a: &NdArray, axis: usize) -> Option<NdArray> {
    if a.ndim() != 2 || !a.is_c_contiguous() {
        return None;
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let s = a.as_slice()?;
    if axis == 0 {
        let mut out = vec![0.0; n];
        for i in 0..m {
            let row = &s[i * n..(i + 1) * n];
            for j in 0..n {
                out[j] += row[j];
            }
        }
        Some(NdArray::from_vec(out))
    } else if axis == 1 {
        let mut out = vec![0.0; m];
        for i in 0..m {
            let row = &s[i * n..(i + 1) * n];
            let mut acc = 0.0;
            for &x in row {
                acc += x;
            }
            out[i] = acc;
        }
        Some(NdArray::from_vec(out))
    } else {
        None
    }
}

/// `np.mean(a)`
pub fn mean(a: &NdArray) -> f64 {
    a.mean().unwrap_or(f64::NAN)
}

/// `np.mean(a, axis=axis)`
pub fn mean_axis(a: &NdArray, axis: usize) -> NdArray {
    assert!(axis < a.ndim(), "mean_axis: axis out of bounds");
    let n = a.shape()[axis] as f64;
    let mut out = sum_axis(a, axis);
    let slice = out.as_slice_mut().unwrap();
    for x in slice.iter_mut() {
        *x /= n;
    }
    out
}

/// `np.min(a)`
pub fn min(a: &NdArray) -> f64 {
    if let Some(s) = a.as_slice() {
        return min_slice(s);
    }
    let c = a.to_contiguous();
    min_slice(c.as_slice().expect("contiguous storage"))
}

/// `np.min(a, axis=axis)`
pub fn min_axis(a: &NdArray, axis: usize) -> NdArray {
    if let Some(out) = minmax_axis_fast(a, axis, true) {
        return out;
    }
    reduce_axis(a, axis, f64::INFINITY, |acc, x| if x < acc { x } else { acc })
}

/// `np.max(a)`
pub fn max(a: &NdArray) -> f64 {
    if let Some(s) = a.as_slice() {
        return max_slice(s);
    }
    let c = a.to_contiguous();
    max_slice(c.as_slice().expect("contiguous storage"))
}

/// `np.max(a, axis=axis)`
pub fn max_axis(a: &NdArray, axis: usize) -> NdArray {
    if let Some(out) = minmax_axis_fast(a, axis, false) {
        return out;
    }
    reduce_axis(a, axis, f64::NEG_INFINITY, |acc, x| if x > acc { x } else { acc })
}

fn minmax_axis_fast(a: &NdArray, axis: usize, is_min: bool) -> Option<NdArray> {
    if a.ndim() != 2 || !a.is_c_contiguous() {
        return None;
    }
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let s = a.as_slice()?;
    if axis == 0 {
        let mut out = vec![0.0; n];
        out.copy_from_slice(&s[..n]);
        for i in 1..m {
            let row = &s[i * n..(i + 1) * n];
            for j in 0..n {
                if is_min {
                    if row[j] < out[j] {
                        out[j] = row[j];
                    }
                } else if row[j] > out[j] {
                    out[j] = row[j];
                }
            }
        }
        Some(NdArray::from_vec(out))
    } else if axis == 1 {
        let mut out = vec![0.0; m];
        for i in 0..m {
            let row = &s[i * n..(i + 1) * n];
            let mut v = row[0];
            for &x in &row[1..] {
                if is_min {
                    if x < v {
                        v = x;
                    }
                } else if x > v {
                    v = x;
                }
            }
            out[i] = v;
        }
        Some(NdArray::from_vec(out))
    } else {
        None
    }
}

/// `np.var(a)` — population variance (ddof=0), NumPy default.
pub fn var(a: &NdArray) -> f64 {
    let n = a.len();
    if n == 0 {
        return f64::NAN;
    }
    let m = mean(a);
    let c = a.to_contiguous();
    let slice = c.as_slice().expect("contiguous storage");
    let mut acc = 0.0;
    for &x in slice {
        let d = x - m;
        acc += d * d;
    }
    acc / n as f64
}

/// `np.std(a)` — population std (ddof=0), NumPy default.
pub fn std(a: &NdArray) -> f64 {
    var(a).sqrt()
}

/// `np.argmin(a)` — flat index.
pub fn argmin(a: &NdArray) -> usize {
    let c = a.to_contiguous();
    argmin_slice(c.as_slice().expect("contiguous storage"))
}

/// `np.argmax(a)` — flat index.
pub fn argmax(a: &NdArray) -> usize {
    let c = a.to_contiguous();
    argmax_slice(c.as_slice().expect("contiguous storage"))
}

/// `np.cumsum(a)` — flat cumulative sum (C-order).
pub fn cumsum(a: &NdArray) -> NdArray {
    let c = a.to_contiguous();
    let mut out = c.as_slice().unwrap().to_vec();
    let mut acc = 0.0;
    for x in &mut out {
        acc += *x;
        *x = acc;
    }
    NdArray::from_shape_vec(a.shape(), out)
}

/// `np.cumsum(a, axis=axis)`
pub fn cumsum_axis(a: &NdArray, axis: usize) -> NdArray {
    scan_axis(a, axis, 0.0, |acc, x| acc + x)
}

/// `np.cumprod(a)` — flat cumulative product (C-order).
pub fn cumprod(a: &NdArray) -> NdArray {
    let c = a.to_contiguous();
    let mut out = c.as_slice().unwrap().to_vec();
    let mut acc = 1.0;
    for x in &mut out {
        acc *= *x;
        *x = acc;
    }
    NdArray::from_shape_vec(a.shape(), out)
}

/// `np.cumprod(a, axis=axis)`
pub fn cumprod_axis(a: &NdArray, axis: usize) -> NdArray {
    scan_axis(a, axis, 1.0, |acc, x| acc * x)
}

fn scan_axis(a: &NdArray, axis: usize, init: f64, f: impl Fn(f64, f64) -> f64) -> NdArray {
    let a = a.to_contiguous();
    let ndim = a.ndim();
    assert!(axis < ndim, "axis {axis} out of bounds for ndim {ndim}");
    let shape = a.shape().to_vec();
    let data = a.as_slice().unwrap();
    let mut out = data.to_vec();
    let mut stride = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        stride[d] = stride[d + 1] * shape[d + 1];
    }
    let axis_n = shape[axis];
    let axis_stride = stride[axis];
    let n_outer = a.len() / axis_n;

    for outer in 0..n_outer {
        let base = outer % axis_stride + (outer / axis_stride) * axis_stride * axis_n;
        let mut acc = init;
        for k in 0..axis_n {
            let idx = base + k * axis_stride;
            acc = f(acc, data[idx]);
            out[idx] = acc;
        }
    }
    NdArray::from_shape_vec(&shape, out)
}

fn reduce_axis(a: &NdArray, axis: usize, init: f64, f: impl Fn(f64, f64) -> f64) -> NdArray {
    let a = a.to_contiguous();
    let ndim = a.ndim();
    assert!(axis < ndim, "axis {axis} out of bounds for ndim {ndim}");
    let shape = a.shape();
    let mut out_shape = Vec::with_capacity(ndim.saturating_sub(1));
    out_shape.extend_from_slice(&shape[..axis]);
    out_shape.extend_from_slice(&shape[axis + 1..]);

    let out_len: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut out = vec![init; out_len];
    let data = a.as_slice().unwrap();

    let mut a_stride = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        a_stride[d] = a_stride[d + 1] * shape[d + 1];
    }

    let axis_n = shape[axis];
    for out_flat in 0..out_len {
        let mut rem = out_flat;
        let mut coords = vec![0usize; ndim];
        let mut out_stride = vec![1usize; out_shape.len()];
        if !out_shape.is_empty() {
            for d in (0..out_shape.len() - 1).rev() {
                out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
            }
            let mut out_axis = 0usize;
            for d in 0..ndim {
                if d == axis {
                    continue;
                }
                coords[d] = rem / out_stride[out_axis];
                rem %= out_stride[out_axis];
                out_axis += 1;
            }
        }

        let mut acc = init;
        for k in 0..axis_n {
            coords[axis] = k;
            let mut flat = 0usize;
            for d in 0..ndim {
                flat += coords[d] * a_stride[d];
            }
            acc = f(acc, data[flat]);
        }
        out[out_flat] = acc;
    }

    NdArray::from_shape_vec(&out_shape, out)
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
    use crate::NdArray;

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

    #[test]
    fn sum_axis0_2d() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = sum_axis(&a, 0);
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn mean_axis1_2d() {
        let a = NdArray::from_shape_vec(&[2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let m = mean_axis(&a, 1);
        assert_eq!(m.shape(), &[2]);
        assert_eq!(m.as_slice().unwrap(), &[2.5, 6.5]);
    }

    #[test]
    fn cumsum_flat() {
        let a = arange(1.0, 5.0, 1.0);
        let c = cumsum(&a);
        assert_eq!(c.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn cumsum_axis0() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = cumsum_axis(&a, 0);
        assert_eq!(c.as_slice().unwrap(), &[1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
    }
}
