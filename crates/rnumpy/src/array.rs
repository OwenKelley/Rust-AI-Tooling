//! N-dimensional `f64` array with NumPy-like shape/strides.
//!
//! Storage is `Arc`-shared so transpose/slice/swapaxes/reshape are O(1) views
//! (no data copy). Contiguous kernels use dense slice fast paths.

use std::ops::{Index, IndexMut};
use std::sync::Arc;

/// N-dimensional `f64` array (NumPy `ndarray` analogue for this crate).
#[derive(Clone, Debug)]
pub struct NdArray {
    data: Arc<Vec<f64>>,
    shape: Vec<usize>,
    /// Element strides (not bytes).
    strides: Vec<isize>,
    offset: usize,
}

impl PartialEq for NdArray {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        if let (Some(a), Some(b)) = (self.as_slice(), other.as_slice()) {
            return a == b;
        }
        (0..self.len()).all(|i| self.get_flat(i) == other.get_flat(i))
    }
}

impl NdArray {
    pub fn shape_len(shape: &[usize]) -> usize {
        if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        }
    }

    pub fn row_major_strides(shape: &[usize]) -> Vec<isize> {
        let ndim = shape.len();
        if ndim == 0 {
            return Vec::new();
        }
        let mut strides = vec![1isize; ndim];
        for d in (0..ndim - 1).rev() {
            strides[d] = strides[d + 1] * shape[d + 1] as isize;
        }
        strides
    }

    fn from_parts(
        data: Arc<Vec<f64>>,
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Self {
        assert_eq!(shape.len(), strides.len(), "shape/strides rank mismatch");
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    fn from_vec_owned(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = Self::row_major_strides(&shape);
        Self::from_parts(Arc::new(data), shape, strides, 0)
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n = Self::shape_len(shape);
        Self::from_vec_owned(vec![0.0; n], shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> Self {
        Self::from_elem(shape, 1.0)
    }

    pub fn from_elem(shape: &[usize], fill_value: f64) -> Self {
        let n = Self::shape_len(shape);
        Self::from_vec_owned(vec![fill_value; n], shape.to_vec())
    }

    pub fn from_shape_vec(shape: &[usize], data: Vec<f64>) -> Self {
        let expected = Self::shape_len(shape);
        assert_eq!(
            data.len(),
            expected,
            "data len {} != shape product {}",
            data.len(),
            expected
        );
        Self::from_vec_owned(data, shape.to_vec())
    }

    pub fn from_vec(data: Vec<f64>) -> Self {
        let n = data.len();
        Self::from_shape_vec(&[n], data)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        Self::shape_len(&self.shape)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_c_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        self.strides == Self::row_major_strides(&self.shape)
    }

    pub fn as_slice(&self) -> Option<&[f64]> {
        if !self.is_c_contiguous() {
            return None;
        }
        let n = self.len();
        Some(&self.data[self.offset..self.offset + n])
    }

    pub fn as_slice_mut(&mut self) -> Option<&mut [f64]> {
        if !self.is_c_contiguous() {
            return None;
        }
        let n = self.len();
        let start = self.offset;
        let data = Arc::make_mut(&mut self.data);
        Some(&mut data[start..start + n])
    }

    pub fn as_slice_memory_order(&self) -> Option<&[f64]> {
        self.as_slice()
    }

    /// Contiguous row-major array; shares buffer when already a dense owner view.
    pub fn to_contiguous(&self) -> Self {
        if self.is_c_contiguous() {
            if self.offset == 0 && self.data.len() == self.len() {
                return self.clone();
            }
            return Self::from_shape_vec(&self.shape, self.as_slice().unwrap().to_vec());
        }
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        if self.ndim() == 2 {
            // Faster materialize for common 2D transpose/slice cases.
            let (r, c) = (self.shape[0], self.shape[1]);
            let rs = self.strides[0];
            let cs = self.strides[1];
            let base = self.offset as isize;
            out.resize(n, 0.0);
            for i in 0..r {
                let row = base + i as isize * rs;
                for j in 0..c {
                    out[i * c + j] = self.data[(row + j as isize * cs) as usize];
                }
            }
        } else {
            out.extend((0..n).map(|i| self.get_flat(i)));
        }
        Self::from_shape_vec(&self.shape, out)
    }

    /// O(1) reshape when C-contiguous (shared buffer).
    pub fn reshape_view(&self, newshape: &[usize]) -> Self {
        assert!(
            self.is_c_contiguous(),
            "reshape_view requires C-contiguous input"
        );
        let expected = Self::shape_len(newshape);
        assert_eq!(self.len(), expected, "reshape size mismatch");
        Self::from_parts(
            Arc::clone(&self.data),
            newshape.to_vec(),
            Self::row_major_strides(newshape),
            self.offset,
        )
    }

    pub fn data_index_flat(&self, flat: usize) -> usize {
        assert!(flat < self.len(), "flat index out of bounds");
        if self.shape.is_empty() {
            return self.offset;
        }
        // Fast path: C-contiguous
        if self.is_c_contiguous() {
            return self.offset + flat;
        }
        let mut rem = flat;
        let mut idx = self.offset as isize;
        for d in 0..self.ndim() {
            let stride_tail: usize = if d + 1 < self.ndim() {
                self.shape[d + 1..].iter().product()
            } else {
                1
            };
            let coord = rem / stride_tail;
            rem %= stride_tail;
            idx += coord as isize * self.strides[d];
        }
        assert!(idx >= 0, "negative data index");
        idx as usize
    }

    pub fn data_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.ndim());
        let mut idx = self.offset as isize;
        for (&coord, (&dim, &stride)) in indices
            .iter()
            .zip(self.shape.iter().zip(self.strides.iter()))
        {
            assert!(coord < dim, "index {coord} out of bounds for dim {dim}");
            idx += coord as isize * stride;
        }
        assert!(idx >= 0);
        idx as usize
    }

    pub fn get_flat(&self, flat: usize) -> f64 {
        self.data[self.data_index_flat(flat)]
    }

    pub fn get(&self, indices: &[usize]) -> f64 {
        self.data[self.data_index(indices)]
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> &mut f64 {
        let i = self.data_index(indices);
        &mut Arc::make_mut(&mut self.data)[i]
    }

    pub fn iter(&self) -> NdIter<'_> {
        NdIter {
            arr: self,
            pos: 0,
            end: self.len(),
            contig: self.as_slice(),
        }
    }

    pub fn sum(&self) -> f64 {
        if let Some(s) = self.as_slice() {
            return sum_slice(s);
        }
        self.iter().sum()
    }

    pub fn mean(&self) -> Option<f64> {
        let n = self.len();
        if n == 0 {
            None
        } else {
            Some(self.sum() / n as f64)
        }
    }

    /// O(1) reverse-axes transpose (shared buffer).
    pub fn transpose_view(&self) -> Self {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.reverse();
        strides.reverse();
        Self::from_parts(Arc::clone(&self.data), shape, strides, self.offset)
    }

    pub fn transpose_owned(&self) -> Self {
        self.transpose_view().to_contiguous()
    }

    /// O(1) axis swap (shared buffer).
    pub fn swapaxes_view(&self, axis1: usize, axis2: usize) -> Self {
        assert!(axis1 < self.ndim() && axis2 < self.ndim());
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(axis1, axis2);
        strides.swap(axis1, axis2);
        Self::from_parts(Arc::clone(&self.data), shape, strides, self.offset)
    }

    /// O(1) axis permutation (shared buffer).
    pub fn permute_axes_view(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), self.ndim());
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        for &ax in axes {
            shape.push(self.shape[ax]);
            strides.push(self.strides[ax]);
        }
        Self::from_parts(Arc::clone(&self.data), shape, strides, self.offset)
    }

    pub fn slice(&self, specs: &[AxisSlice]) -> Self {
        assert_eq!(specs.len(), self.ndim(), "slice rank must match ndim");
        let mut shape = Vec::with_capacity(self.ndim());
        let mut strides = Vec::with_capacity(self.ndim());
        let mut offset = self.offset as isize;
        for (ax, spec) in specs.iter().enumerate() {
            let dim = self.shape[ax] as isize;
            let (start, stop, step) = spec.resolve(dim);
            assert!(step != 0, "slice step cannot be 0");
            let len = if step > 0 {
                if start >= stop {
                    0
                } else {
                    (stop - start + step - 1) / step
                }
            } else if start <= stop {
                0
            } else {
                (start - stop - step - 1) / (-step)
            };
            let len = len.max(0) as usize;
            offset += start * self.strides[ax];
            shape.push(len);
            strides.push(self.strides[ax] * step);
        }
        assert!(offset >= 0);
        Self::from_parts(
            Arc::clone(&self.data),
            shape,
            strides,
            offset as usize,
        )
    }

    pub fn astype_f32(&self) -> NdArrayF32 {
        if let Some(s) = self.as_slice() {
            return NdArrayF32::from_shape_vec(
                &self.shape,
                s.iter().map(|&x| x as f32).collect(),
            );
        }
        let mut data = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            data.push(self.get_flat(i) as f32);
        }
        NdArrayF32::from_shape_vec(&self.shape, data)
    }
}

#[inline]
fn sum_slice(s: &[f64]) -> f64 {
    // Four-way accumulation helps LLVM vectorize.
    let mut a0 = 0.0;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut i = 0;
    let n = s.len();
    while i + 4 <= n {
        a0 += s[i];
        a1 += s[i + 1];
        a2 += s[i + 2];
        a3 += s[i + 3];
        i += 4;
    }
    let mut acc = a0 + a1 + a2 + a3;
    while i < n {
        acc += s[i];
        i += 1;
    }
    acc
}

/// Slice specification for one axis (`start:stop:step`).
#[derive(Clone, Copy, Debug)]
pub struct AxisSlice {
    pub start: Option<isize>,
    pub stop: Option<isize>,
    pub step: isize,
}

impl AxisSlice {
    pub fn all() -> Self {
        Self {
            start: None,
            stop: None,
            step: 1,
        }
    }

    pub fn new(start: Option<isize>, stop: Option<isize>, step: isize) -> Self {
        Self { start, stop, step }
    }

    fn resolve(self, dim: isize) -> (isize, isize, isize) {
        let step = self.step;
        let (default_start, default_stop) = if step > 0 {
            (0, dim)
        } else {
            (dim - 1, -1)
        };
        let mut start = self.start.unwrap_or(default_start);
        let mut stop = self.stop.unwrap_or(default_stop);
        if start < 0 {
            start += dim;
        }
        if stop < 0 && self.stop.is_some() {
            stop += dim;
        }
        if step > 0 {
            start = start.clamp(0, dim);
            stop = stop.clamp(0, dim);
        } else {
            start = start.clamp(-1, dim - 1);
            stop = stop.clamp(-1, dim - 1);
        }
        (start, stop, step)
    }
}

pub struct NdIter<'a> {
    arr: &'a NdArray,
    pos: usize,
    end: usize,
    contig: Option<&'a [f64]>,
}

impl Iterator for NdIter<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.end {
            return None;
        }
        let v = if let Some(s) = self.contig {
            s[self.pos]
        } else {
            self.arr.get_flat(self.pos)
        };
        self.pos += 1;
        Some(v)
    }
}

impl Index<usize> for NdArray {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        let i = self.data_index_flat(index);
        &self.data[i]
    }
}

impl IndexMut<usize> for NdArray {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let i = self.data_index_flat(index);
        &mut Arc::make_mut(&mut self.data)[i]
    }
}

impl Index<[usize; 0]> for NdArray {
    type Output = f64;

    fn index(&self, _index: [usize; 0]) -> &Self::Output {
        assert_eq!(self.ndim(), 0);
        &self.data[self.offset]
    }
}

impl IndexMut<[usize; 0]> for NdArray {
    fn index_mut(&mut self, _index: [usize; 0]) -> &mut Self::Output {
        assert_eq!(self.ndim(), 0);
        let off = self.offset;
        &mut Arc::make_mut(&mut self.data)[off]
    }
}

impl Index<[usize; 2]> for NdArray {
    type Output = f64;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        assert_eq!(self.ndim(), 2);
        let i = self.data_index(&index);
        &self.data[i]
    }
}

impl IndexMut<[usize; 2]> for NdArray {
    fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
        assert_eq!(self.ndim(), 2);
        let i = self.data_index(&index);
        &mut Arc::make_mut(&mut self.data)[i]
    }
}

/// Contiguous `f32` ND array (dtype companion to [`NdArray`]).
#[derive(Clone, Debug, PartialEq)]
pub struct NdArrayF32 {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl NdArrayF32 {
    pub fn zeros(shape: &[usize]) -> Self {
        let n = NdArray::shape_len(shape);
        Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn from_elem(shape: &[usize], fill: f32) -> Self {
        let n = NdArray::shape_len(shape);
        Self {
            data: vec![fill; n],
            shape: shape.to_vec(),
        }
    }

    pub fn from_shape_vec(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(data.len(), NdArray::shape_len(shape));
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn astype_f64(&self) -> NdArray {
        NdArray::from_shape_vec(
            &self.shape,
            self.data.iter().map(|&x| x as f64).collect(),
        )
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        Self::from_shape_vec(&self.shape, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_view_shares_buffer() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose_view();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.strides(), &[1, 3]);
        assert!(Arc::ptr_eq(&a.data, &t.data));
        assert!(!t.is_c_contiguous());
        let c = t.to_contiguous();
        assert_eq!(c.as_slice().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn slice_view() {
        let a = NdArray::from_shape_vec(&[4, 3], (0..12).map(|x| x as f64).collect());
        let s = a.slice(&[
            AxisSlice::new(Some(1), Some(3), 1),
            AxisSlice::new(None, None, 1),
        ]);
        assert_eq!(s.shape(), &[2, 3]);
        assert_eq!(s.get_flat(0), 3.0);
        assert!(Arc::ptr_eq(&a.data, &s.data));
    }

    #[test]
    fn reshape_view_o1() {
        let a = NdArray::from_vec((0..6).map(|x| x as f64).collect());
        let b = a.reshape_view(&[2, 3]);
        assert!(Arc::ptr_eq(&a.data, &b.data));
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn sum_matches() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.sum(), 10.0);
    }

    #[test]
    fn astype_roundtrip() {
        let a = NdArray::from_vec(vec![1.5, 2.5, 3.5]);
        let b = a.astype_f32();
        let c = b.astype_f64();
        assert_eq!(c.as_slice().unwrap(), &[1.5, 2.5, 3.5]);
    }
}
