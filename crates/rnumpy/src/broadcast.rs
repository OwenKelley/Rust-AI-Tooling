//! NumPy-style broadcasting helpers (right-aligned shape rules).

/// Compute the broadcast shape of two arrays, or panic if incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let ndim = a.len().max(b.len());
    let mut out = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let da = dim_from_right(a, ndim, i);
        let db = dim_from_right(b, ndim, i);
        assert!(
            da == db || da == 1 || db == 1,
            "operands could not be broadcast together: {a:?} vs {b:?}"
        );
        out.push(da.max(db));
    }
    out
}

#[inline]
fn dim_from_right(shape: &[usize], out_ndim: usize, axis: usize) -> usize {
    let pad = out_ndim - shape.len();
    if axis < pad {
        1
    } else {
        shape[axis - pad]
    }
}

/// Map an output multi-index into a source flat index under broadcasting.
pub fn broadcast_flat_index(shape: &[usize], out_shape: &[usize], out_multi: &[usize]) -> usize {
    debug_assert_eq!(out_shape.len(), out_multi.len());
    let ndim = out_shape.len();
    let pad = ndim - shape.len();
    let mut stride = 1usize;
    let mut offset = 0usize;
    for axis in (0..ndim).rev() {
        let src_dim = if axis < pad { 1 } else { shape[axis - pad] };
        let idx = if src_dim == 1 { 0 } else { out_multi[axis] };
        offset += idx * stride;
        stride = stride.saturating_mul(src_dim.max(1));
    }
    offset
}

/// Decode a flat row-major index into a multi-index for `shape`.
pub fn unravel_index(flat: usize, shape: &[usize], out_multi: &mut [usize]) {
    debug_assert_eq!(shape.len(), out_multi.len());
    let mut rem = flat;
    for d in (0..shape.len()).rev() {
        let dim = shape[d].max(1);
        out_multi[d] = rem % dim;
        rem /= dim;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_matrix_row() {
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[3, 4], &[4]), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[5], &[3, 1]), vec![3, 5]);
    }
}
