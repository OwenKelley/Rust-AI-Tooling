//! Element-wise / ufunc-style ops — mirrors `numpy` arithmetic & math ufuncs.
//!
//! Binary ops support NumPy broadcasting with fast paths for common shapes.

use crate::broadcast::{broadcast_flat_index, broadcast_shapes, unravel_index};
use crate::NdArray;

fn zip2(a: &NdArray, b: &NdArray, f: impl Fn(f64, f64) -> f64) -> NdArray {
    if a.shape() == b.shape() {
        let mut out = NdArray::zeros(a.shape());
        if let (Some(a_s), Some(b_s), Some(o_s)) =
            (a.as_slice(), b.as_slice(), out.as_slice_mut())
        {
            for i in 0..o_s.len() {
                o_s[i] = f(a_s[i], b_s[i]);
            }
            return out;
        }
        let n = a.len();
        for i in 0..n {
            out[i] = f(a.get_flat(i), b.get_flat(i));
        }
        return out;
    }

    // Fast path: (m,1) + (1,n) → (m,n) and variants with contiguous inputs.
    if let Some(out) = zip2_broadcast_2d(a, b, &f) {
        return out;
    }

    let out_shape = broadcast_shapes(a.shape(), b.shape());
    let n: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let mut out_data = vec![0.0; n];
    let a_c = a.to_contiguous();
    let b_c = b.to_contiguous();
    let a_s = a_c.as_slice().unwrap();
    let b_s = b_c.as_slice().unwrap();
    let mut multi = vec![0usize; out_shape.len()];
    for (flat, dest) in out_data.iter_mut().enumerate() {
        unravel_index(flat, &out_shape, &mut multi);
        let ai = broadcast_flat_index(a.shape(), &out_shape, &multi);
        let bi = broadcast_flat_index(b.shape(), &out_shape, &multi);
        *dest = f(a_s[ai], b_s[bi]);
    }
    NdArray::from_shape_vec(&out_shape, out_data)
}

/// Optimized outer-add style broadcasts for 2D.
fn zip2_broadcast_2d(
    a: &NdArray,
    b: &NdArray,
    f: &impl Fn(f64, f64) -> f64,
) -> Option<NdArray> {
    let (ash, bsh) = (a.shape(), b.shape());
    // (m,1) op (1,n) → (m,n)
    if ash.len() == 2 && bsh.len() == 2 && ash[1] == 1 && bsh[0] == 1 {
        let m = ash[0];
        let n = bsh[1];
        let a_c = a.to_contiguous();
        let b_c = b.to_contiguous();
        let col = a_c.as_slice()?;
        let row = b_c.as_slice()?;
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            let ai = col[i];
            let row_out = &mut out[i * n..(i + 1) * n];
            for j in 0..n {
                row_out[j] = f(ai, row[j]);
            }
        }
        return Some(NdArray::from_shape_vec(&[m, n], out));
    }
    // (m,n) op (1,n) or (m,n) op (m,1) — row/col broadcast onto matrix
    if ash.len() == 2 && bsh.len() == 2 && ash == &[ash[0], ash[1]] {
        let (m, n) = (ash[0], ash[1]);
        if bsh == [1, n] {
            let a_c = a.to_contiguous();
            let b_c = b.to_contiguous();
            let mat = a_c.as_slice()?;
            let row = b_c.as_slice()?;
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    out[i * n + j] = f(mat[i * n + j], row[j]);
                }
            }
            return Some(NdArray::from_shape_vec(&[m, n], out));
        }
        if bsh == [m, 1] {
            let a_c = a.to_contiguous();
            let b_c = b.to_contiguous();
            let mat = a_c.as_slice()?;
            let col = b_c.as_slice()?;
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                let bi = col[i];
                for j in 0..n {
                    out[i * n + j] = f(mat[i * n + j], bi);
                }
            }
            return Some(NdArray::from_shape_vec(&[m, n], out));
        }
    }
    // (m,) op (1,m) already covered by general; (n,) right-aligned with (m,n):
    if ash.len() == 2 && bsh.len() == 1 && bsh[0] == ash[1] {
        let (m, n) = (ash[0], ash[1]);
        let a_c = a.to_contiguous();
        let b_c = b.to_contiguous();
        let mat = a_c.as_slice()?;
        let row = b_c.as_slice()?;
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                out[i * n + j] = f(mat[i * n + j], row[j]);
            }
        }
        return Some(NdArray::from_shape_vec(&[m, n], out));
    }
    None
}

fn map1(a: &NdArray, f: impl Fn(f64) -> f64) -> NdArray {
    let mut out = NdArray::zeros(a.shape());
    if let (Some(a_s), Some(o_s)) = (a.as_slice(), out.as_slice_mut()) {
        for i in 0..o_s.len() {
            o_s[i] = f(a_s[i]);
        }
    } else {
        for i in 0..a.len() {
            out[i] = f(a.get_flat(i));
        }
    }
    out
}

/// `np.add(a, b)`
pub fn add(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x + y)
}

/// `np.subtract(a, b)`
pub fn subtract(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x - y)
}

/// `np.multiply(a, b)`
pub fn multiply(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x * y)
}

/// `np.divide(a, b)`
pub fn divide(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| x / y)
}

/// `np.power(a, b)` element-wise
pub fn power(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, f64::powf)
}

/// `np.maximum(a, b)`
pub fn maximum(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, f64::max)
}

/// `np.minimum(a, b)`
pub fn minimum(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, f64::min)
}

/// `np.sqrt(a)`
pub fn sqrt(a: &NdArray) -> NdArray {
    map1(a, f64::sqrt)
}

/// `np.exp(a)`
pub fn exp(a: &NdArray) -> NdArray {
    map1(a, f64::exp)
}

/// `np.log(a)` natural log
pub fn log(a: &NdArray) -> NdArray {
    map1(a, f64::ln)
}

/// `np.sin(a)`
pub fn sin(a: &NdArray) -> NdArray {
    map1(a, f64::sin)
}

/// `np.cos(a)`
pub fn cos(a: &NdArray) -> NdArray {
    map1(a, f64::cos)
}

/// `np.tan(a)`
pub fn tan(a: &NdArray) -> NdArray {
    map1(a, f64::tan)
}

/// `np.tanh(a)`
pub fn tanh(a: &NdArray) -> NdArray {
    map1(a, f64::tanh)
}

/// `np.negative(a)`
pub fn negative(a: &NdArray) -> NdArray {
    map1(a, |x| -x)
}

/// `np.abs(a)`
pub fn abs(a: &NdArray) -> NdArray {
    map1(a, f64::abs)
}

/// `np.sign(a)`
pub fn sign(a: &NdArray) -> NdArray {
    map1(a, |x| {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// `np.square(a)`
pub fn square(a: &NdArray) -> NdArray {
    map1(a, |x| x * x)
}

/// `np.reciprocal(a)`
pub fn reciprocal(a: &NdArray) -> NdArray {
    map1(a, |x| 1.0 / x)
}

/// `np.floor(a)`
pub fn floor(a: &NdArray) -> NdArray {
    map1(a, f64::floor)
}

/// `np.ceil(a)`
pub fn ceil(a: &NdArray) -> NdArray {
    map1(a, f64::ceil)
}

/// `np.trunc(a)`
pub fn trunc(a: &NdArray) -> NdArray {
    map1(a, f64::trunc)
}

/// `np.round(a)` — half away from zero via `f64::round`.
pub fn round(a: &NdArray) -> NdArray {
    map1(a, f64::round)
}

/// Comparison helpers return `1.0`/`0.0` float masks (NumPy returns bool).
pub fn greater(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| if x > y { 1.0 } else { 0.0 })
}

pub fn greater_equal(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| if x >= y { 1.0 } else { 0.0 })
}

pub fn less(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| if x < y { 1.0 } else { 0.0 })
}

pub fn less_equal(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| if x <= y { 1.0 } else { 0.0 })
}

pub fn equal(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| if x == y { 1.0 } else { 0.0 })
}

pub fn not_equal(a: &NdArray, b: &NdArray) -> NdArray {
    zip2(a, b, |x, y| if x != y { 1.0 } else { 0.0 })
}

/// `np.clip(a, min, max)`
pub fn clip(a: &NdArray, min: f64, max: f64) -> NdArray {
    map1(a, |x| x.clamp(min, max))
}

/// `np.where(cond, x, y)` — nonzero `cond` values are true.
pub fn where_(cond: &NdArray, x: &NdArray, y: &NdArray) -> NdArray {
    if cond.shape() == x.shape() && x.shape() == y.shape() {
        if let (Some(c), Some(xs), Some(ys)) = (cond.as_slice(), x.as_slice(), y.as_slice()) {
            let mut out = vec![0.0; c.len()];
            for i in 0..c.len() {
                out[i] = if c[i] != 0.0 { xs[i] } else { ys[i] };
            }
            return NdArray::from_shape_vec(cond.shape(), out);
        }
    }
    let s1 = broadcast_shapes(cond.shape(), x.shape());
    let out_shape = broadcast_shapes(&s1, y.shape());
    let n: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };
    let cc = cond.to_contiguous();
    let xc = x.to_contiguous();
    let yc = y.to_contiguous();
    let c_s = cc.as_slice().unwrap();
    let x_s = xc.as_slice().unwrap();
    let y_s = yc.as_slice().unwrap();
    let mut out_data = vec![0.0; n];
    let mut multi = vec![0usize; out_shape.len()];
    for (flat, dest) in out_data.iter_mut().enumerate() {
        unravel_index(flat, &out_shape, &mut multi);
        let ci = broadcast_flat_index(cond.shape(), &out_shape, &multi);
        let xi = broadcast_flat_index(x.shape(), &out_shape, &multi);
        let yi = broadcast_flat_index(y.shape(), &out_shape, &multi);
        *dest = if c_s[ci] != 0.0 { x_s[xi] } else { y_s[yi] };
    }
    NdArray::from_shape_vec(&out_shape, out_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{ones, seeded_uniform};
    use crate::test_util::assert_abs_diff_eq;

    #[test]
    fn add_ones() {
        let a = ones(&[2, 2]);
        let b = ones(&[2, 2]);
        let c = add(&a, &b);
        assert!(c.iter().all(|x| x == 2.0));
    }

    #[test]
    fn multiply_scale() {
        let a = seeded_uniform(&[4], 1, 0.0, 1.0);
        let b = ones(&[4]);
        let c = multiply(&a, &b);
        for i in 0..a.len() {
            assert_abs_diff_eq(a.get_flat(i), c.get_flat(i), 1e-12);
        }
    }

    #[test]
    fn add_broadcast_column_row() {
        let a = NdArray::from_shape_vec(&[3, 1], vec![1.0, 2.0, 3.0]);
        let b = NdArray::from_shape_vec(&[1, 4], vec![10.0, 20.0, 30.0, 40.0]);
        let c = add(&a, &b);
        assert_eq!(c.shape(), &[3, 4]);
        assert_eq!(c[[0, 0]], 11.0);
        assert_eq!(c[[2, 3]], 43.0);
    }

    #[test]
    fn clip_basic() {
        let a = NdArray::from_vec(vec![-2.0, 0.5, 3.0]);
        let c = clip(&a, 0.0, 1.0);
        assert_eq!(c.as_slice().unwrap(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn where_basic() {
        let cond = NdArray::from_vec(vec![1.0, 0.0, 2.0]);
        let x = ones(&[3]);
        let y = NdArray::from_elem(&[3], 5.0);
        let c = where_(&cond, &x, &y);
        assert_eq!(c.as_slice().unwrap(), &[1.0, 5.0, 1.0]);
    }
}
