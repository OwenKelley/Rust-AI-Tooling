//! Indexing helpers — `take`, boolean compress, and axis slicing.

use crate::array::AxisSlice;
use crate::NdArray;

/// `np.take(a, indices, axis=axis)` — gather along one axis (or flat if `None`).
pub fn take(a: &NdArray, indices: &[usize], axis: Option<usize>) -> NdArray {
    match axis {
        None => {
            if let Some(s) = a.as_slice() {
                let mut data = Vec::with_capacity(indices.len());
                for &i in indices {
                    data.push(s[i]);
                }
                return NdArray::from_vec(data);
            }
            let mut data = Vec::with_capacity(indices.len());
            for &i in indices {
                data.push(a.get_flat(i));
            }
            NdArray::from_vec(data)
        }
        Some(0) if a.ndim() >= 1 && a.is_c_contiguous() => {
            // Fast row gather for C-contiguous arrays.
            let axis_n = a.shape()[0];
            for &i in indices {
                assert!(i < axis_n, "take index {i} out of bounds for axis size {axis_n}");
            }
            let row_stride: usize = if a.ndim() == 1 {
                1
            } else {
                a.shape()[1..].iter().product()
            };
            let mut out_shape = a.shape().to_vec();
            out_shape[0] = indices.len();
            let mut out = vec![0.0; NdArray::shape_len(&out_shape)];
            let src = a.as_slice().unwrap();
            for (dst_i, &src_i) in indices.iter().enumerate() {
                let src_off = src_i * row_stride;
                let dst_off = dst_i * row_stride;
                out[dst_off..dst_off + row_stride]
                    .copy_from_slice(&src[src_off..src_off + row_stride]);
            }
            NdArray::from_shape_vec(&out_shape, out)
        }
        Some(axis) => {
            assert!(axis < a.ndim());
            let axis_n = a.shape()[axis];
            for &i in indices {
                assert!(i < axis_n, "take index {i} out of bounds for axis size {axis_n}");
            }
            let mut out_shape = a.shape().to_vec();
            out_shape[axis] = indices.len();
            let out_len = NdArray::shape_len(&out_shape);
            let mut out = vec![0.0; out_len];
            let mut coords = vec![0usize; a.ndim()];
            for (flat, dest) in out.iter_mut().enumerate() {
                let mut rem = flat;
                for d in (0..out_shape.len()).rev() {
                    let dim = out_shape[d].max(1);
                    coords[d] = rem % dim;
                    rem /= dim;
                }
                let mut src = coords.clone();
                src[axis] = indices[coords[axis]];
                *dest = a.get(&src);
            }
            NdArray::from_shape_vec(&out_shape, out)
        }
    }
}

/// Boolean index / `np.compress` for 1D (or raveled) condition as 0/1 floats.
pub fn compress(condition: &NdArray, a: &NdArray, axis: Option<usize>) -> NdArray {
    match axis {
        None => {
            assert_eq!(condition.len(), a.len(), "compress: condition length mismatch");
            if let (Some(c), Some(s)) = (condition.as_slice(), a.as_slice()) {
                let mut data = Vec::new();
                for i in 0..c.len() {
                    if c[i] != 0.0 {
                        data.push(s[i]);
                    }
                }
                return NdArray::from_vec(data);
            }
            let mut data = Vec::new();
            for i in 0..a.len() {
                if condition.get_flat(i) != 0.0 {
                    data.push(a.get_flat(i));
                }
            }
            NdArray::from_vec(data)
        }
        Some(axis) => {
            assert!(axis < a.ndim());
            assert_eq!(
                condition.len(),
                a.shape()[axis],
                "compress: condition length must match axis"
            );
            let mut keep = Vec::new();
            if let Some(c) = condition.as_slice() {
                for (i, &v) in c.iter().enumerate() {
                    if v != 0.0 {
                        keep.push(i);
                    }
                }
            } else {
                for i in 0..condition.len() {
                    if condition.get_flat(i) != 0.0 {
                        keep.push(i);
                    }
                }
            }
            take(a, &keep, Some(axis))
        }
    }
}

/// `a[start:stop:step, ...]` convenience over [`NdArray::slice`].
pub fn slice_array(a: &NdArray, specs: &[AxisSlice]) -> NdArray {
    a.slice(specs)
}

/// Integer fancy index along axis 0 for 1D/2D: `a[indices]` style gather of rows.
pub fn take_rows(a: &NdArray, indices: &[usize]) -> NdArray {
    if a.ndim() == 1 {
        take(a, indices, None)
    } else {
        take(a, indices, Some(0))
    }
}

/// Boolean fancy index `a[mask]` — `mask` must match `a.shape`; returns 1D (C-order).
///
/// Nonzero mask entries (as floats) select elements, matching NumPy truthiness on 0/1.
pub fn boolean_index(a: &NdArray, mask: &NdArray) -> NdArray {
    assert_eq!(
        a.shape(),
        mask.shape(),
        "boolean_index: mask shape must match array"
    );
    let n = a.len();
    let mut data = Vec::new();
    if let (Some(av), Some(mv)) = (a.as_slice(), mask.as_slice()) {
        for i in 0..n {
            if mv[i] != 0.0 {
                data.push(av[i]);
            }
        }
    } else {
        for i in 0..n {
            if mask.get_flat(i) != 0.0 {
                data.push(a.get_flat(i));
            }
        }
    }
    NdArray::from_vec(data)
}

/// Advanced integer indexing for 2D: `a[rows, cols]` with equal-length index lists.
///
/// Returns a 1D array of `a[rows[i], cols[i]]` (NumPy point-indexing style).
pub fn fancy_index_2d(a: &NdArray, rows: &[usize], cols: &[usize]) -> NdArray {
    assert_eq!(a.ndim(), 2, "fancy_index_2d: expected 2D array");
    assert_eq!(
        rows.len(),
        cols.len(),
        "fancy_index_2d: rows and cols length mismatch"
    );
    let nr = a.shape()[0];
    let nc = a.shape()[1];
    let mut data = Vec::with_capacity(rows.len());
    for i in 0..rows.len() {
        let r = rows[i];
        let c = cols[i];
        assert!(r < nr, "fancy_index_2d: row {r} out of bounds");
        assert!(c < nc, "fancy_index_2d: col {c} out of bounds");
        data.push(a[[r, c]]);
    }
    NdArray::from_vec(data)
}

/// `np.take_along_axis(a, indices, axis)` for 1D index arrays broadcast along `axis`.
///
/// `indices` is treated as 1D of length `a.shape[axis]` replacements along that axis;
/// for full NumPy broadcasting, pass an `indices` NdArray matching `a` except on `axis`.
pub fn take_along_axis(a: &NdArray, indices: &NdArray, axis: usize) -> NdArray {
    assert!(axis < a.ndim(), "take_along_axis: axis out of bounds");
    assert_eq!(indices.ndim(), a.ndim(), "take_along_axis: ndim mismatch");
    let mut out_shape = a.shape().to_vec();
    out_shape[axis] = indices.shape()[axis];
    for d in 0..a.ndim() {
        if d != axis {
            assert_eq!(
                a.shape()[d],
                indices.shape()[d],
                "take_along_axis: shape mismatch on axis {d}"
            );
        }
    }
    let axis_n = a.shape()[axis];
    let out_len = NdArray::shape_len(&out_shape);
    let mut out = vec![0.0; out_len];
    let mut coords = vec![0usize; a.ndim()];
    for (flat, dest) in out.iter_mut().enumerate() {
        let mut rem = flat;
        for d in (0..out_shape.len()).rev() {
            let dim = out_shape[d].max(1);
            coords[d] = rem % dim;
            rem /= dim;
        }
        let idx_f = indices.get(&coords);
        let idx = idx_f as isize;
        let idx = if idx < 0 {
            (axis_n as isize + idx) as usize
        } else {
            idx as usize
        };
        assert!(idx < axis_n, "take_along_axis: index {idx} out of bounds");
        let mut src = coords.clone();
        src[axis] = idx;
        *dest = a.get(&src);
    }
    NdArray::from_shape_vec(&out_shape, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::arange;

    #[test]
    fn take_flat() {
        let a = arange(10.0, 15.0, 1.0);
        let t = take(&a, &[0, 2, 4], None);
        assert_eq!(t.as_slice().unwrap(), &[10.0, 12.0, 14.0]);
    }

    #[test]
    fn take_axis0() {
        let a = NdArray::from_shape_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = take(&a, &[2, 0], Some(0));
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.as_slice().unwrap(), &[5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn compress_1d() {
        let a = arange(0.0, 5.0, 1.0);
        let c = NdArray::from_vec(vec![1.0, 0.0, 1.0, 0.0, 1.0]);
        let out = compress(&c, &a, None);
        assert_eq!(out.as_slice().unwrap(), &[0.0, 2.0, 4.0]);
    }

    #[test]
    fn boolean_index_2d() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = NdArray::from_shape_vec(&[2, 3], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let out = boolean_index(&a, &m);
        assert_eq!(out.as_slice().unwrap(), &[1.0, 3.0, 5.0]);
    }

    #[test]
    fn fancy_index_2d_points() {
        let a = NdArray::from_shape_vec(&[3, 3], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let out = fancy_index_2d(&a, &[0, 2, 1], &[1, 2, 0]);
        assert_eq!(out.as_slice().unwrap(), &[1.0, 8.0, 3.0]);
    }

    #[test]
    fn take_along_axis_rows() {
        let a = NdArray::from_shape_vec(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Along axis 0, pick row indices [[1,0],[2,1],[0,0]] shaped like a
        let idx = NdArray::from_shape_vec(&[3, 2], vec![1.0, 0.0, 2.0, 1.0, 0.0, 0.0]);
        let out = take_along_axis(&a, &idx, 0);
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.as_slice().unwrap(), &[3.0, 2.0, 5.0, 4.0, 1.0, 2.0]);
    }
}
