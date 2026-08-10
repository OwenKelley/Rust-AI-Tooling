//! Strided tensor with typed storage and optional autograd metadata (views share storage).

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use crate::autograd::GradFn;
use crate::device::Device;
use crate::dtype::Dtype;

pub type TensorRef = Rc<RefCell<TensorInner>>;

/// Typed backing buffers. Float32/Float64 use `F32`; Int64/Bool use dedicated buffers.
#[derive(Debug, Clone)]
pub enum TensorStorage {
    F32(Rc<RefCell<Vec<f32>>>),
    I64(Rc<RefCell<Vec<i64>>>),
    /// 0/1 bytes.
    Bool(Rc<RefCell<Vec<u8>>>),
}

impl TensorStorage {
    pub fn f32(data: Vec<f32>) -> Self {
        Self::F32(Rc::new(RefCell::new(data)))
    }

    pub fn i64(data: Vec<i64>) -> Self {
        Self::I64(Rc::new(RefCell::new(data)))
    }

    pub fn bool_bytes(data: Vec<u8>) -> Self {
        Self::Bool(Rc::new(RefCell::new(data)))
    }

    pub fn len(&self) -> usize {
        match self {
            Self::F32(s) => s.borrow().len(),
            Self::I64(s) => s.borrow().len(),
            Self::Bool(s) => s.borrow().len(),
        }
    }

    pub fn strong_count(&self) -> usize {
        match self {
            Self::F32(s) => Rc::strong_count(s),
            Self::I64(s) => Rc::strong_count(s),
            Self::Bool(s) => Rc::strong_count(s),
        }
    }

    pub fn is_f32(&self) -> bool {
        matches!(self, Self::F32(_))
    }

    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => Rc::ptr_eq(a, b),
            (Self::I64(a), Self::I64(b)) => Rc::ptr_eq(a, b),
            (Self::Bool(a), Self::Bool(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// Shared mutable tensor storage with shape / strides / offset (PyTorch-style views).
#[derive(Debug)]
pub struct TensorInner {
    pub storage: TensorStorage,
    pub shape: Vec<usize>,
    /// Element strides (not byte strides).
    pub strides: Vec<isize>,
    pub offset: usize,
    pub device: Device,
    pub dtype: Dtype,
    pub requires_grad: bool,
    pub grad: Option<Vec<f32>>,
    pub grad_fn: Option<GradFn>,
}

/// Row-major (C-contiguous) element strides for `shape`.
pub fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    let mut s = 1isize;
    for i in (0..shape.len()).rev() {
        strides[i] = s;
        s = s.saturating_mul(shape[i] as isize);
    }
    strides
}

fn gather_indexed<T: Copy>(
    storage: &[T],
    shape: &[usize],
    strides: &[isize],
    offset: usize,
    numel: usize,
    contiguous: bool,
) -> Vec<T> {
    if numel == 0 {
        return Vec::new();
    }
    if contiguous {
        return storage[offset..offset + numel].to_vec();
    }
    let ndim = shape.len();
    let mut out = Vec::with_capacity(numel);
    let mut idx = vec![0usize; ndim];
    for flat in 0..numel {
        let mut rem = flat;
        for d in (0..ndim).rev() {
            let size = shape[d].max(1);
            idx[d] = rem % size;
            rem /= size;
        }
        let mut off = offset as isize;
        for d in 0..ndim {
            off += idx[d] as isize * strides[d];
        }
        out.push(storage[off as usize]);
    }
    out
}

impl TensorInner {
    /// Owned contiguous tensor from a dense `f32` buffer (Float32 or Float64 tag).
    pub fn new_contiguous(
        data: Vec<f32>,
        shape: Vec<usize>,
        device: Device,
        dtype: Dtype,
        requires_grad: bool,
        grad: Option<Vec<f32>>,
        grad_fn: Option<GradFn>,
    ) -> Self {
        let strides = row_major_strides(&shape);
        Self {
            storage: TensorStorage::f32(data),
            shape,
            strides,
            offset: 0,
            device,
            dtype,
            requires_grad,
            grad,
            grad_fn,
        }
    }

    pub fn numel(&self) -> usize {
        if self.shape.is_empty() {
            1
        } else {
            self.shape.iter().product()
        }
    }

    /// Standard contiguous check (size-1 dims may have any stride).
    pub fn is_contiguous(&self) -> bool {
        if self.numel() <= 1 {
            return true;
        }
        let mut expected = 1isize;
        for i in (0..self.shape.len()).rev() {
            let size = self.shape[i] as isize;
            if size != 1 {
                if self.strides[i] != expected {
                    return false;
                }
                expected = expected.saturating_mul(size);
            }
        }
        let end = self.storage_span_end();
        end <= self.storage.len()
    }

    /// One past the last storage index touched by this view (exclusive).
    fn storage_span_end(&self) -> usize {
        let n = self.numel();
        if n == 0 {
            return self.offset;
        }
        let mut max_off = self.offset as isize;
        for i in 0..self.shape.len() {
            if self.shape[i] > 0 {
                max_off += (self.shape[i] as isize - 1) * self.strides[i];
            }
        }
        (max_off as usize).saturating_add(1)
    }

    pub fn gather_f32(&self) -> Vec<f32> {
        match &self.storage {
            TensorStorage::F32(s) => gather_indexed(
                &s.borrow(),
                &self.shape,
                &self.strides,
                self.offset,
                self.numel(),
                self.is_contiguous(),
            ),
            TensorStorage::I64(_) => self
                .gather_i64()
                .into_iter()
                .map(|x| x as f32)
                .collect(),
            TensorStorage::Bool(_) => self
                .gather_bool_bytes()
                .into_iter()
                .map(|x| if x != 0 { 1.0 } else { 0.0 })
                .collect(),
        }
    }

    pub fn gather_i64(&self) -> Vec<i64> {
        match &self.storage {
            TensorStorage::I64(s) => gather_indexed(
                &s.borrow(),
                &self.shape,
                &self.strides,
                self.offset,
                self.numel(),
                self.is_contiguous(),
            ),
            TensorStorage::F32(_) => self
                .gather_f32()
                .into_iter()
                .map(|x| x as i64)
                .collect(),
            TensorStorage::Bool(_) => self
                .gather_bool_bytes()
                .into_iter()
                .map(|x| if x != 0 { 1 } else { 0 })
                .collect(),
        }
    }

    pub fn gather_bool_bytes(&self) -> Vec<u8> {
        match &self.storage {
            TensorStorage::Bool(s) => gather_indexed(
                &s.borrow(),
                &self.shape,
                &self.strides,
                self.offset,
                self.numel(),
                self.is_contiguous(),
            ),
            TensorStorage::F32(_) => self
                .gather_f32()
                .into_iter()
                .map(|x| if x != 0.0 { 1 } else { 0 })
                .collect(),
            TensorStorage::I64(_) => self
                .gather_i64()
                .into_iter()
                .map(|x| if x != 0 { 1 } else { 0 })
                .collect(),
        }
    }

    /// Gather logical elements in row-major order as `f32` (converts I64/Bool).
    pub fn to_contiguous_vec(&self) -> Vec<f32> {
        self.gather_f32()
    }

    /// Dense owned copy (compat for former `.data` clones); converts typed storage to f32.
    pub fn dense_data(&self) -> Vec<f32> {
        self.gather_f32()
    }

    /// Contiguous dense `f32` slice. Panics if not contiguous or not F32 storage.
    pub fn data_slice(&self) -> Ref<'_, [f32]> {
        assert!(
            self.is_contiguous(),
            "data_slice requires a contiguous tensor; call contiguous() first"
        );
        let offset = self.offset;
        let n = self.numel();
        match &self.storage {
            TensorStorage::F32(s) => Ref::map(s.borrow(), move |v| &v[offset..offset + n]),
            _ => panic!("data_slice requires F32 storage"),
        }
    }

    /// Ensure unique contiguous storage (copy if view / shared / non-contiguous).
    pub fn make_contiguous_unique(&mut self) {
        let shared = self.storage.strong_count() > 1;
        if self.is_contiguous() && !shared {
            return;
        }
        match &self.storage {
            TensorStorage::F32(_) => {
                let data = self.gather_f32();
                self.storage = TensorStorage::f32(data);
            }
            TensorStorage::I64(_) => {
                let data = self.gather_i64();
                self.storage = TensorStorage::i64(data);
            }
            TensorStorage::Bool(_) => {
                let data = self.gather_bool_bytes();
                self.storage = TensorStorage::bool_bytes(data);
            }
        }
        self.offset = 0;
        self.strides = row_major_strides(&self.shape);
    }

    /// Mutable dense `f32` slice after [`make_contiguous_unique`]. Panics for I64/Bool.
    pub fn data_mut_dense(&mut self) -> RefMut<'_, [f32]> {
        if !self.storage.is_f32() {
            panic!("data_mut_dense requires F32 storage");
        }
        self.make_contiguous_unique();
        let offset = self.offset;
        let n = self.numel();
        match &self.storage {
            TensorStorage::F32(s) => {
                RefMut::map(s.borrow_mut(), move |v| &mut v[offset..offset + n])
            }
            _ => unreachable!(),
        }
    }

    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            let n = self.numel();
            self.grad = Some(vec![0.0; n]);
        }
    }

    pub fn accumulate_grad(&mut self, g: &[f32]) {
        assert_eq!(g.len(), self.numel(), "grad size mismatch");
        if !self.requires_grad {
            return;
        }
        match &mut self.grad {
            Some(existing) => {
                for (e, &v) in existing.iter_mut().zip(g.iter()) {
                    *e += v;
                }
            }
            None => {
                self.grad = Some(g.to_vec());
            }
        }
    }
}

/// `torch.Tensor` analogue (CPU; views share `storage` for F32).
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) inner: TensorRef,
}

impl Tensor {
    pub(crate) fn from_inner(inner: TensorInner) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize], requires_grad: bool) -> Self {
        Self::from_vec_dtype(data, shape, requires_grad, Dtype::Float32)
    }

    pub fn from_vec_dtype(
        data: Vec<f32>,
        shape: &[usize],
        requires_grad: bool,
        dtype: Dtype,
    ) -> Self {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(data.len(), expected, "data length != shape product");
        match dtype {
            Dtype::Int64 => {
                let typed: Vec<i64> = data.iter().map(|&x| x as i64).collect();
                return Self::from_i64(typed, shape);
            }
            Dtype::Bool => {
                let typed: Vec<bool> = data.iter().map(|&x| x != 0.0).collect();
                return Self::from_bool(typed, shape);
            }
            Dtype::Float32 | Dtype::Float64 => {}
        }
        let rg = requires_grad && dtype.is_floating_point();
        Self::from_inner(TensorInner::new_contiguous(
            data,
            shape.to_vec(),
            Device::Cpu,
            dtype,
            rg,
            if rg {
                Some(vec![0.0; expected])
            } else {
                None
            },
            None,
        ))
    }

    /// Contiguous Int64 tensor with typed `i64` storage.
    pub fn from_i64(data: Vec<i64>, shape: &[usize]) -> Self {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(data.len(), expected, "data length != shape product");
        Self::from_inner(TensorInner {
            storage: TensorStorage::i64(data),
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: 0,
            device: Device::Cpu,
            dtype: Dtype::Int64,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        })
    }

    /// Contiguous Bool tensor with typed 0/1 `u8` storage.
    pub fn from_bool(data: Vec<bool>, shape: &[usize]) -> Self {
        let expected: usize = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
        assert_eq!(data.len(), expected, "data length != shape product");
        let bytes: Vec<u8> = data.into_iter().map(|b| if b { 1 } else { 0 }).collect();
        Self::from_inner(TensorInner {
            storage: TensorStorage::bool_bytes(bytes),
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            offset: 0,
            device: Device::Cpu,
            dtype: Dtype::Bool,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        })
    }

    pub fn shape(&self) -> Vec<usize> {
        self.inner.borrow().shape.clone()
    }

    pub fn ndim(&self) -> usize {
        self.inner.borrow().shape.len()
    }

    pub fn numel(&self) -> usize {
        self.inner.borrow().numel()
    }

    pub fn device(&self) -> Device {
        self.inner.borrow().device
    }

    pub fn dtype(&self) -> Dtype {
        self.inner.borrow().dtype
    }

    pub fn is_contiguous(&self) -> bool {
        self.inner.borrow().is_contiguous()
    }

    /// Clone if already contiguous; otherwise gather into a new owned tensor (preserves dtype/storage).
    pub fn contiguous(&self) -> Tensor {
        let t = self.inner.borrow();
        if t.is_contiguous() {
            return self.clone();
        }
        match &t.storage {
            TensorStorage::F32(_) => {
                let data = t.gather_f32();
                let rg = t.requires_grad;
                let dtype = t.dtype;
                let shape = t.shape.clone();
                drop(t);
                Self::from_vec_dtype(data, &shape, rg, dtype)
            }
            TensorStorage::I64(_) => {
                let data = t.gather_i64();
                let shape = t.shape.clone();
                drop(t);
                Self::from_i64(data, &shape)
            }
            TensorStorage::Bool(_) => {
                let data: Vec<bool> = t
                    .gather_bool_bytes()
                    .into_iter()
                    .map(|x| x != 0)
                    .collect();
                let shape = t.shape.clone();
                drop(t);
                Self::from_bool(data, &shape)
            }
        }
    }

    /// Contiguous tensor for dense kernels (clone if already contiguous).
    pub fn as_contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()
        }
    }

    /// Gather logical elements in row-major order as f32.
    pub fn to_contiguous_vec(&self) -> Vec<f32> {
        self.inner.borrow().to_contiguous_vec()
    }

    /// Typed Int64 logical elements.
    pub fn i64_data(&self) -> Vec<i64> {
        self.inner.borrow().gather_i64()
    }

    /// Typed Bool logical elements.
    pub fn bool_data(&self) -> Vec<bool> {
        self.inner
            .borrow()
            .gather_bool_bytes()
            .into_iter()
            .map(|x| x != 0)
            .collect()
    }

    /// Materialize unique contiguous storage for in-place mutation.
    pub fn make_contiguous_unique(&self) {
        self.inner.borrow_mut().make_contiguous_unique();
    }

    /// `tensor.to(device)` — CPU only; CUDA panics with a clear message (no runtime).
    pub fn to(&self, device: Device) -> Tensor {
        match device {
            Device::Cpu => self.clone(),
            Device::Cuda => panic!(
                "rtorch: Device::Cuda is API-only; no CUDA runtime. Use Device::Cpu."
            ),
        }
    }

    /// `tensor.to(dtype)` — casts values; Float* stay on F32 storage; Int64/Bool use typed buffers.
    pub fn to_dtype(&self, dtype: Dtype) -> Tensor {
        let t = self.inner.borrow();
        if t.dtype == dtype {
            return self.clone();
        }
        let shape = t.shape.clone();
        let from = t.dtype;
        let rg_src = t.requires_grad;

        let out = match (from, dtype) {
            (Dtype::Float32, Dtype::Float64) | (Dtype::Float64, Dtype::Float32) => {
                let data = crate::dtype::cast_f32_data(&t.dense_data(), from, dtype);
                let rg = dtype.is_floating_point() && rg_src;
                drop(t);
                Self::from_vec_dtype(data, &shape, rg, dtype)
            }
            (Dtype::Float32 | Dtype::Float64, Dtype::Int64) => {
                let data: Vec<i64> = t.dense_data().iter().map(|&x| x as i64).collect();
                drop(t);
                Self::from_i64(data, &shape)
            }
            (Dtype::Float32 | Dtype::Float64, Dtype::Bool) => {
                let data: Vec<bool> = t.dense_data().iter().map(|&x| x != 0.0).collect();
                drop(t);
                Self::from_bool(data, &shape)
            }
            (Dtype::Int64, Dtype::Float32) => {
                let data: Vec<f32> = t.gather_i64().iter().map(|&x| x as f32).collect();
                drop(t);
                Self::from_vec_dtype(data, &shape, false, Dtype::Float32)
            }
            (Dtype::Int64, Dtype::Float64) => {
                let data: Vec<f32> = t.gather_i64().iter().map(|&x| x as f32).collect();
                drop(t);
                Self::from_vec_dtype(data, &shape, false, Dtype::Float64)
            }
            (Dtype::Bool, Dtype::Float32) => {
                let data: Vec<f32> = t
                    .gather_bool_bytes()
                    .iter()
                    .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                    .collect();
                drop(t);
                Self::from_vec_dtype(data, &shape, false, Dtype::Float32)
            }
            (Dtype::Bool, Dtype::Float64) => {
                let data: Vec<f32> = t
                    .gather_bool_bytes()
                    .iter()
                    .map(|&x| if x != 0 { 1.0 } else { 0.0 })
                    .collect();
                drop(t);
                Self::from_vec_dtype(data, &shape, false, Dtype::Float64)
            }
            (Dtype::Int64, Dtype::Bool) => {
                let data: Vec<bool> = t.gather_i64().iter().map(|&x| x != 0).collect();
                drop(t);
                Self::from_bool(data, &shape)
            }
            (Dtype::Bool, Dtype::Int64) => {
                let data: Vec<i64> = t
                    .gather_bool_bytes()
                    .iter()
                    .map(|&x| if x != 0 { 1 } else { 0 })
                    .collect();
                drop(t);
                Self::from_i64(data, &shape)
            }
            _ => unreachable!("same dtype handled above"),
        };
        out
    }

    /// `tensor.cpu()`
    pub fn cpu(&self) -> Tensor {
        self.to(Device::Cpu)
    }

    /// `tensor.float()`
    pub fn float(&self) -> Tensor {
        self.to_dtype(Dtype::Float32)
    }

    /// `tensor.double()` / `tensor.to(torch.float64)`
    pub fn double(&self) -> Tensor {
        self.to_dtype(Dtype::Float64)
    }

    /// `tensor.long()` / `tensor.to(torch.int64)`
    pub fn long(&self) -> Tensor {
        self.to_dtype(Dtype::Int64)
    }

    /// `tensor.bool()`
    pub fn bool_(&self) -> Tensor {
        self.to_dtype(Dtype::Bool)
    }

    pub fn requires_grad(&self) -> bool {
        self.inner.borrow().requires_grad
    }

    /// Borrow dense f32 data without cloning (materializes a contiguous clone if needed).
    pub fn with_data<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let c = self.as_contiguous();
        let inner = c.inner.borrow();
        let slice = inner.data_slice();
        f(&slice)
    }

    /// Accumulate gradient from another tensor's data without an extra clone.
    pub(crate) fn accumulate_from(&self, src: &Tensor) {
        let g = src.to_contiguous_vec();
        self.inner.borrow_mut().accumulate_grad(&g);
    }

    pub fn set_requires_grad(&self, flag: bool) {
        let mut t = self.inner.borrow_mut();
        if flag {
            assert!(
                t.dtype.is_floating_point(),
                "only floating-point tensors can require grad"
            );
        }
        t.requires_grad = flag;
        if flag && t.grad.is_none() {
            let n = t.numel();
            t.grad = Some(vec![0.0; n]);
        }
        if !flag {
            t.grad = None;
            t.grad_fn = None;
        }
    }

    /// Clones logical storage as f32; prefer [`Self::with_data`] in hot paths.
    pub fn data(&self) -> Vec<f32> {
        self.to_contiguous_vec()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.inner.borrow().grad.clone()
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().zero_grad();
    }

    /// Scalar checksum for parity (sum of finite values via dense f32 view).
    pub fn checksum(&self) -> f64 {
        self.inner
            .borrow()
            .dense_data()
            .iter()
            .map(|&x| if x.is_finite() { x as f64 } else { 0.0 })
            .sum()
    }

    /// Grad checksum (0 if no grad).
    pub fn grad_checksum(&self) -> f64 {
        match &self.inner.borrow().grad {
            Some(g) => g
                .iter()
                .map(|&x| if x.is_finite() { x as f64 } else { 0.0 })
                .sum(),
            None => 0.0,
        }
    }

    pub fn item(&self) -> f32 {
        let t = self.inner.borrow();
        assert_eq!(t.numel(), 1, "item: tensor must have one element");
        t.dense_data()[0]
    }

    /// Detach: new tensor sharing data values without grad history.
    pub fn detach(&self) -> Tensor {
        let t = self.inner.borrow();
        let out = TensorInner {
            storage: t.storage.clone(),
            shape: t.shape.clone(),
            strides: t.strides.clone(),
            offset: t.offset,
            device: t.device,
            dtype: t.dtype,
            requires_grad: false,
            grad: None,
            grad_fn: None,
        };
        Self::from_inner(out)
    }

    /// Overwrite F32 storage from a contiguous `src` (same numel). Leaf only.
    pub fn copy_from_slice(&self, src: &[f32]) {
        let mut t = self.inner.borrow_mut();
        assert!(t.grad_fn.is_none(), "copy_from_slice: leaf only");
        assert_eq!(src.len(), t.numel());
        let mut d = t.data_mut_dense();
        d.copy_from_slice(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_shape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert_eq!(t.shape(), vec![2, 2]);
        assert!((t.checksum() - 10.0).abs() < 1e-6);
        assert!(t.is_contiguous());
    }

    #[test]
    fn dtype_float32() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2], false);
        assert_eq!(t.dtype(), Dtype::Float32);
        assert!(t.dtype().is_floating_point());
        assert_eq!(t.dtype().type_str(), "float32");
        let f = t.float().to_dtype(Dtype::Float32);
        assert_eq!(f.dtype(), Dtype::Float32);
        assert!((f.checksum() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn dtype_float64_roundtrip() {
        let t = Tensor::from_vec(vec![1.0, 2.0], &[2], false);
        let d = t.double();
        assert_eq!(d.dtype(), Dtype::Float64);
        assert!(d.dtype().is_floating_point());
        assert!(matches!(
            d.inner.borrow().storage,
            TensorStorage::F32(_)
        ));
        let f = d.float();
        assert_eq!(f.dtype(), Dtype::Float32);
        assert!((f.checksum() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn dtype_long_bool_roundtrip() {
        let t = Tensor::from_vec(vec![1.7, -2.3, 0.0], &[3], false);
        let i = t.long();
        assert_eq!(i.dtype(), Dtype::Int64);
        assert!(matches!(
            i.inner.borrow().storage,
            TensorStorage::I64(_)
        ));
        assert_eq!(i.i64_data(), vec![1i64, -2, 0]);
        assert_eq!(i.data(), vec![1.0, -2.0, 0.0]);
        let b = t.bool_();
        assert_eq!(b.dtype(), Dtype::Bool);
        assert!(matches!(
            b.inner.borrow().storage,
            TensorStorage::Bool(_)
        ));
        assert_eq!(b.bool_data(), vec![true, true, false]);
        assert_eq!(b.data(), vec![1.0, 1.0, 0.0]);
        let back = i.float();
        assert_eq!(back.dtype(), Dtype::Float32);
        assert!(matches!(
            back.inner.borrow().storage,
            TensorStorage::F32(_)
        ));
        assert_eq!(back.data(), vec![1.0, -2.0, 0.0]);
    }

    #[test]
    fn from_i64_to_float32() {
        let t = Tensor::from_i64(vec![1, -2, 0], &[3]);
        assert_eq!(t.dtype(), Dtype::Int64);
        assert!(matches!(
            t.inner.borrow().storage,
            TensorStorage::I64(_)
        ));
        let f = t.to_dtype(Dtype::Float32);
        assert_eq!(f.dtype(), Dtype::Float32);
        assert_eq!(f.data(), vec![1.0, -2.0, 0.0]);
    }

    #[test]
    fn transpose_view_shares_storage() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let v = crate::ops::transpose_data(&t);
        assert_eq!(v.shape(), vec![2, 2]);
        assert_eq!(v.data(), vec![1.0, 3.0, 2.0, 4.0]);
        assert!(t
            .inner
            .borrow()
            .storage
            .ptr_eq(&v.inner.borrow().storage));
    }
}
