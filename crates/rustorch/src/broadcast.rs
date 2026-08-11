//! Torch-style broadcasting for contiguous row-major f32 tensors.

/// Align two shapes from the right; dims must match or one side is 1.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let ndim = a.len().max(b.len());
    let mut out = vec![1usize; ndim];
    for i in 0..ndim {
        let da = if i < ndim - a.len() {
            1
        } else {
            a[i - (ndim - a.len())]
        };
        let db = if i < ndim - b.len() {
            1
        } else {
            b[i - (ndim - b.len())]
        };
        assert!(
            da == db || da == 1 || db == 1,
            "cannot broadcast {a:?} and {b:?}"
        );
        out[i] = da.max(db);
    }
    out
}

fn numel(shape: &[usize]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    }
}

fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut s = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = s;
        s *= shape[i];
    }
    strides
}

fn pad_left(shape: &[usize], ndim: usize) -> Vec<usize> {
    let mut out = vec![1usize; ndim];
    let offset = ndim - shape.len();
    out[offset..].copy_from_slice(shape);
    out
}

fn increment_index(idx: &mut [usize], shape: &[usize]) {
    for d in (0..idx.len()).rev() {
        idx[d] += 1;
        if idx[d] < shape[d] {
            return;
        }
        idx[d] = 0;
    }
}

/// Expand `data` with `shape` to `out_shape` (owning copy).
pub fn expand_to(data: &[f32], shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
    assert_eq!(data.len(), numel(shape));
    let n = numel(out_shape);
    if shape == out_shape {
        return data.to_vec();
    }
    let ndim = out_shape.len();
    let src_shape = pad_left(shape, ndim);
    for i in 0..ndim {
        assert!(
            src_shape[i] == out_shape[i] || src_shape[i] == 1,
            "expand {shape:?} -> {out_shape:?}"
        );
    }
    let src_strides = contiguous_strides(&src_shape);
    let mut read_strides = src_strides;
    for i in 0..ndim {
        if src_shape[i] == 1 {
            read_strides[i] = 0;
        }
    }
    let mut out = vec![0.0f32; n];
    let mut idx = vec![0usize; ndim];
    for oi in 0..n {
        let mut offset = 0usize;
        for d in 0..ndim {
            offset += idx[d] * read_strides[d];
        }
        out[oi] = data[offset];
        increment_index(&mut idx, out_shape);
    }
    out
}

/// Sum `grad` (shaped as `from`) down to `to` by summing over broadcast axes.
pub fn reduce_sum_to(grad: &[f32], from: &[usize], to: &[usize]) -> Vec<f32> {
    assert_eq!(grad.len(), numel(from));
    if from == to {
        return grad.to_vec();
    }
    let ndim = from.len().max(to.len());
    let from_pad = pad_left(from, ndim);
    let to_pad = pad_left(to, ndim);
    // `from` should already be the broadcast result; pad if needed.
    let from_use = if from.len() == ndim {
        from.to_vec()
    } else {
        from_pad.clone()
    };
    assert_eq!(grad.len(), numel(&from_use));
    for i in 0..ndim {
        assert!(
            to_pad[i] == from_use[i] || to_pad[i] == 1,
            "reduce {from:?} -> {to:?}"
        );
    }
    let out_n = numel(to);
    let mut out = vec![0.0f32; out_n];
    let mut idx = vec![0usize; ndim];
    for &g in grad {
        if to.is_empty() {
            out[0] += g;
        } else {
            let offset_dims = ndim - to.len();
            let mut o = 0usize;
            let to_contig = contiguous_strides(to);
            for d in 0..to.len() {
                let coord = if to[d] == 1 {
                    0
                } else {
                    idx[d + offset_dims]
                };
                o += coord * to_contig[d];
            }
            out[o] += g;
        }
        increment_index(&mut idx, &from_use);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_vector_matrix() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), vec![3, 4]);
    }

    #[test]
    fn expand_and_reduce() {
        let a = vec![1.0f32, 2.0, 3.0];
        let e = expand_to(&a, &[3], &[2, 3]);
        assert_eq!(e, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        let g = vec![1.0f32; 6];
        let r = reduce_sum_to(&g, &[2, 3], &[3]);
        assert_eq!(r, vec![2.0, 2.0, 2.0]);
    }
}
