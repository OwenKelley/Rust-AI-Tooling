//! Bridges between `rustorch::Tensor` and `rnumpy` / `rpandas`
//! (`torch.from_numpy` / `.numpy()` / `torch.tensor(df.values)`).

use rnumpy::{NdArray, NdArrayF32};
use rpandas::DataFrame;

use crate::tensor::Tensor;

/// `torch.from_numpy` for f64 `rnumpy::NdArray` (copies to contiguous f32).
pub fn from_numpy(a: &NdArray) -> Tensor {
    let a32 = a.astype_f32();
    from_numpy_f32(&a32)
}

/// `torch.from_numpy` for contiguous f32 `rnumpy::NdArrayF32` (copies storage).
pub fn from_numpy_f32(a: &NdArrayF32) -> Tensor {
    let sl = a.as_slice();
    let mut data = Vec::with_capacity(sl.len());
    unsafe {
        data.set_len(sl.len());
    }
    data.copy_from_slice(sl);
    Tensor::from_vec(data, a.shape(), false)
}

/// `torch.from_numpy` taking ownership of an `NdArrayF32` (no second copy).
pub fn from_numpy_f32_owned(a: NdArrayF32) -> Tensor {
    let (data, shape) = a.into_parts();
    Tensor::from_vec(data, &shape, false)
}

/// `tensor.numpy()` — returns a contiguous f64 `NdArray` (copy + cast).
pub fn to_numpy(t: &Tensor) -> NdArray {
    t.with_data(|sl| {
        let mut data = Vec::with_capacity(sl.len());
        for &x in sl {
            data.push(x as f64);
        }
        NdArray::from_shape_vec(&t.shape(), data)
    })
}

/// `tensor.numpy()` as f32 `NdArrayF32` (copy).
pub fn to_numpy_f32(t: &Tensor) -> NdArrayF32 {
    let shape = t.shape();
    t.with_data(|sl| {
        let mut data = Vec::with_capacity(sl.len());
        unsafe {
            data.set_len(sl.len());
        }
        data.copy_from_slice(sl);
        NdArrayF32::from_shape_vec(&shape, data)
    })
}

/// `torch.tensor(df.values)` for a numeric `DataFrame` (row-major `f32` copy).
pub fn from_dataframe(df: &DataFrame) -> Tensor {
    let names = df.column_names();
    let nrows = df.nrows();
    let ncols = names.len();
    let cols: Vec<Vec<f64>> = names.iter().map(|n| df.float_slice(n)).collect();
    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        for col in &cols {
            data.push(col[i] as f32);
        }
    }
    Tensor::from_vec(data, &[nrows, ncols], false)
}

/// Build a float `DataFrame` from a 2D tensor (`pd.DataFrame(tensor.numpy(), columns=...)`).
pub fn to_dataframe(t: &Tensor, names: &[&str]) -> DataFrame {
    let shape = t.shape();
    assert_eq!(shape.len(), 2, "to_dataframe: expected 2D tensor");
    assert_eq!(shape[1], names.len(), "to_dataframe: name count != ncols");
    DataFrame::from_numeric(names, &to_numpy(t))
}

impl Tensor {
    /// `tensor.numpy()` — f64 NumPy-shaped array.
    pub fn numpy(&self) -> NdArray {
        to_numpy(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::seeded_uniform;

    #[test]
    fn roundtrip_numpy_bridge() {
        let t = seeded_uniform(&[3, 4], 9, -1.0, 1.0);
        let a = to_numpy(&t);
        let t2 = from_numpy(&a);
        assert_eq!(t.shape(), t2.shape());
        let d1 = t.data();
        let d2 = t2.data();
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn roundtrip_dataframe_bridge() {
        let t = seeded_uniform(&[4, 3], 11, -1.0, 1.0);
        let df = to_dataframe(&t, &["a", "b", "c"]);
        let t2 = from_dataframe(&df);
        assert_eq!(t.shape(), t2.shape());
        let d1 = t.data();
        let d2 = t2.data();
        for i in 0..d1.len() {
            assert!((d1[i] - d2[i]).abs() < 1e-5);
        }
    }
}
