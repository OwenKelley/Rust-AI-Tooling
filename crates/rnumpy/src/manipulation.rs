//! Array manipulation — mirrors common `numpy` reshape / stack helpers.

use crate::NdArray;

fn reshape_copy(a: &NdArray, newshape: &[usize]) -> NdArray {
    let expected: usize = if newshape.is_empty() {
        1
    } else {
        newshape.iter().product()
    };
    assert_eq!(
        a.len(),
        expected,
        "cannot reshape array of size {} into shape {:?}",
        a.len(),
        newshape
    );
    if a.is_c_contiguous() {
        return a.reshape_view(newshape);
    }
    let a = a.to_contiguous();
    a.reshape_view(newshape)
}

/// `np.reshape(a, newshape)` — exact dims (use `reshape_infer` for `-1`).
pub fn reshape(a: &NdArray, newshape: &[usize]) -> NdArray {
    reshape_copy(a, newshape)
}

/// `np.reshape(a, newshape)` allowing a single `-1` inferred dim.
pub fn reshape_infer(a: &NdArray, newshape: &[isize]) -> NdArray {
    let resolved = resolve_reshape(a.len(), newshape);
    reshape_copy(a, &resolved)
}

/// `np.ravel(a)` — 1-D view when contiguous, else contiguous copy.
pub fn ravel(a: &NdArray) -> NdArray {
    let n = a.len();
    if a.is_c_contiguous() {
        a.reshape_view(&[n])
    } else {
        a.to_contiguous().reshape_view(&[n])
    }
}

/// `np.swapaxes(a, axis1, axis2)` — O(1) strided view.
pub fn swapaxes(a: &NdArray, axis1: usize, axis2: usize) -> NdArray {
    a.swapaxes_view(axis1, axis2)
}

/// `np.moveaxis(a, source, destination)` — O(1) strided view.
pub fn moveaxis(a: &NdArray, source: usize, destination: usize) -> NdArray {
    let ndim = a.ndim();
    assert!(source < ndim && destination < ndim, "moveaxis: axis out of bounds");
    if source == destination {
        return a.clone();
    }
    let mut order: Vec<usize> = (0..ndim).collect();
    let axis = order.remove(source);
    order.insert(destination, axis);
    a.permute_axes_view(&order)
}

/// `np.expand_dims(a, axis)` — O(1) strided view.
pub fn expand_dims(a: &NdArray, axis: usize) -> NdArray {
    a.expand_dims_view(axis)
}

/// `np.squeeze(a, axis=…)` — O(1) strided view.
pub fn squeeze(a: &NdArray, axis: Option<usize>) -> NdArray {
    a.squeeze_view(axis)
}

/// Select one index along `axis` (NumPy integer indexing) — O(1) strided view.
pub fn index_axis(a: &NdArray, axis: usize, index: usize) -> NdArray {
    a.index_axis_view(axis, index)
}

/// Resolve a reshape spec that may contain a single `-1` (NumPy inference).
pub fn resolve_reshape(size: usize, newshape: &[isize]) -> Vec<usize> {
    let mut unknown = None;
    let mut out = Vec::with_capacity(newshape.len());
    for (i, &d) in newshape.iter().enumerate() {
        if d == -1 {
            assert!(unknown.is_none(), "can only specify one unknown dimension");
            unknown = Some(i);
            out.push(0);
        } else {
            assert!(d >= 0, "invalid reshape dim {d}");
            out.push(d as usize);
        }
    }

    let mut known_prod: usize = 1;
    let mut has_zero = false;
    for (i, &d) in newshape.iter().enumerate() {
        if Some(i) == unknown {
            continue;
        }
        if d == 0 {
            has_zero = true;
            break;
        }
        known_prod *= d as usize;
    }

    if let Some(idx) = unknown {
        assert!(!has_zero, "cannot infer -1 when another dim is 0");
        assert!(known_prod > 0, "cannot infer -1 with empty known product");
        assert_eq!(
            size % known_prod,
            0,
            "cannot reshape array of size {size} into {newshape:?}"
        );
        out[idx] = size / known_prod;
    } else if has_zero {
        assert_eq!(size, 0, "cannot reshape size {size} into {newshape:?}");
    } else {
        assert_eq!(
            size, known_prod,
            "cannot reshape array of size {size} into {newshape:?}"
        );
    }
    out
}

/// `np.broadcast_to(a, shape)` — materializes a contiguous copy.
pub fn broadcast_to(a: &NdArray, shape: &[usize]) -> NdArray {
    let out_shape = shape.to_vec();
    assert_eq!(
        crate::broadcast::broadcast_shapes(a.shape(), &out_shape),
        out_shape,
        "cannot broadcast {:?} to {:?}",
        a.shape(),
        shape
    );
    let n: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Fast path: broadcast (1, n) → (m, n) by repeating the row.
    if a.ndim() == 2
        && out_shape.len() == 2
        && a.shape()[0] == 1
        && a.shape()[1] == out_shape[1]
        && a.is_c_contiguous()
    {
        let m = out_shape[0];
        let width = out_shape[1];
        let row = a.as_slice().unwrap();
        let mut data = vec![0.0; n];
        for i in 0..m {
            data[i * width..(i + 1) * width].copy_from_slice(row);
        }
        return NdArray::from_shape_vec(&out_shape, data);
    }

    let src = a.to_contiguous();
    let src = src.as_slice().unwrap();
    let mut data = vec![0.0; n];
    let mut multi = vec![0usize; out_shape.len()];
    for (flat, dest) in data.iter_mut().enumerate() {
        crate::broadcast::unravel_index(flat, &out_shape, &mut multi);
        let src_i = crate::broadcast::broadcast_flat_index(a.shape(), &out_shape, &multi);
        *dest = src[src_i];
    }
    NdArray::from_shape_vec(&out_shape, data)
}

/// `np.concatenate(arrays, axis=axis)`
pub fn concatenate(arrays: &[&NdArray], axis: usize) -> NdArray {
    assert!(!arrays.is_empty(), "concatenate: need at least one array");
    let ndim = arrays[0].ndim();
    assert!(axis < ndim, "concatenate: axis {axis} out of bounds for ndim {ndim}");
    for a in arrays.iter().skip(1) {
        assert_eq!(a.ndim(), ndim, "concatenate: all inputs must have same ndim");
        for d in 0..ndim {
            if d != axis {
                assert_eq!(
                    a.shape()[d],
                    arrays[0].shape()[d],
                    "concatenate: shape mismatch on axis {d}"
                );
            }
        }
    }

    let mut out_shape = arrays[0].shape().to_vec();
    out_shape[axis] = arrays.iter().map(|a| a.shape()[axis]).sum();
    let out_len: usize = out_shape.iter().product();

    // Fast path: axis=0 C-contiguous → memcpy blocks.
    if axis == 0 && arrays.iter().all(|a| a.is_c_contiguous()) {
        let mut out = Vec::with_capacity(out_len);
        for a in arrays {
            out.extend_from_slice(a.as_slice().unwrap());
        }
        return NdArray::from_shape_vec(&out_shape, out);
    }

    let mut out = vec![0.0; out_len];

    let mut out_stride = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        out_stride[d] = out_stride[d + 1] * out_shape[d + 1];
    }

    let mut axis_offset = 0usize;
    for a in arrays {
        let a_shape = a.shape();
        let a_c = a.to_contiguous();
        let a_data = a_c.as_slice().unwrap();
        let mut a_stride = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            a_stride[d] = a_stride[d + 1] * a_shape[d + 1];
        }
        let a_axis = a_shape[axis];
        let block: usize = a_shape.iter().product();
        for flat in 0..block {
            let mut rem = flat;
            let mut out_flat = 0usize;
            for d in 0..ndim {
                let idx = rem / a_stride[d];
                rem %= a_stride[d];
                let out_idx = if d == axis { idx + axis_offset } else { idx };
                out_flat += out_idx * out_stride[d];
            }
            out[out_flat] = a_data[flat];
        }
        axis_offset += a_axis;
    }

    NdArray::from_shape_vec(&out_shape, out)
}

/// `np.stack(arrays, axis=axis)` — inserts a new axis.
pub fn stack(arrays: &[&NdArray], axis: usize) -> NdArray {
    assert!(!arrays.is_empty(), "stack: need at least one array");
    let base = arrays[0].shape();
    for a in arrays.iter().skip(1) {
        assert_eq!(a.shape(), base, "stack: all input shapes must match");
    }
    let ndim = base.len();
    assert!(
        axis <= ndim,
        "stack: axis {axis} out of bounds for result ndim {}",
        ndim + 1
    );

    // Fast path: stack on axis 0 for contiguous equal-shaped arrays.
    if axis == 0 && arrays.iter().all(|a| a.is_c_contiguous()) {
        let mut out_shape = Vec::with_capacity(ndim + 1);
        out_shape.push(arrays.len());
        out_shape.extend_from_slice(base);
        let mut out = Vec::with_capacity(NdArray::shape_len(&out_shape));
        for a in arrays {
            out.extend_from_slice(a.as_slice().unwrap());
        }
        return NdArray::from_shape_vec(&out_shape, out);
    }

    let expanded: Vec<NdArray> = arrays
        .iter()
        .map(|a| {
            let mut sh = Vec::with_capacity(ndim + 1);
            sh.extend_from_slice(&a.shape()[..axis]);
            sh.push(1);
            sh.extend_from_slice(&a.shape()[axis..]);
            reshape(a, &sh)
        })
        .collect();
    let refs: Vec<&NdArray> = expanded.iter().collect();
    concatenate(&refs, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{arange, ones};

    #[test]
    fn reshape_roundtrip() {
        let a = arange(0.0, 6.0, 1.0);
        let b = reshape(&a, &[2, 3]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.as_slice().unwrap(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reshape_infer_minus_one() {
        let a = arange(0.0, 6.0, 1.0);
        let b = reshape_infer(&a, &[-1, 3]);
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn concatenate_axis0() {
        let a = ones(&[2, 3]);
        let b = ones(&[1, 3]);
        let c = concatenate(&[&a, &b], 0);
        assert_eq!(c.shape(), &[3, 3]);
        assert!(c.iter().all(|x| x == 1.0));
    }

    #[test]
    fn stack_axis0() {
        let a = ones(&[2, 2]);
        let b = ones(&[2, 2]);
        let c = stack(&[&a, &b], 0);
        assert_eq!(c.shape(), &[2, 2, 2]);
    }

    #[test]
    fn broadcast_to_row() {
        let a = arange(1.0, 4.0, 1.0);
        let b = broadcast_to(&a, &[2, 3]);
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(
            b.as_slice().unwrap(),
            &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn expand_dims_squeeze_index_axis() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let e = expand_dims(&a, 0);
        assert_eq!(e.shape(), &[1, 2, 3]);
        let s = squeeze(&e, Some(0));
        assert_eq!(s.shape(), &[2, 3]);
        let col = index_axis(&a, 1, 2);
        assert_eq!(col.to_contiguous().as_slice().unwrap(), &[3.0, 6.0]);
    }

    #[test]
    fn swapaxes_2d() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = swapaxes(&a, 0, 1);
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b[[0, 1]], 4.0);
        assert_eq!(b.to_contiguous().as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}
