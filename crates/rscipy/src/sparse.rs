//! Sparse matrices — mirrors common `scipy.sparse` CSR/CSC entry points.
//!
//! Local CSR/CSC storage only (no third-party sparse crates). Built to
//! interoperate with `rnumpy::NdArray` for dense conversion and SpMV.
//!
//! Storage vectors are `Arc`-backed so CSR↔CSC transpose is O(1) (share buffers,
//! swap dimensions) matching SciPy's cheap format reinterpret.

use std::sync::Arc;

use rnumpy::NdArray;

/// Compressed Sparse Row matrix (`scipy.sparse.csr_matrix`).
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Arc<Vec<f64>>,
    pub indices: Arc<Vec<usize>>,
    pub indptr: Arc<Vec<usize>>,
}

/// Compressed Sparse Column matrix (`scipy.sparse.csc_matrix`).
#[derive(Debug, Clone)]
pub struct CscMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub data: Arc<Vec<f64>>,
    pub indices: Arc<Vec<usize>>,
    pub indptr: Arc<Vec<usize>>,
}

fn arc_vecs(
    data: Vec<f64>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
) -> (Arc<Vec<f64>>, Arc<Vec<usize>>, Arc<Vec<usize>>) {
    (Arc::new(data), Arc::new(indices), Arc::new(indptr))
}

impl CsrMatrix {
    /// Number of stored (explicit) nonzeros.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    /// Validate CSR structure.
    pub fn check(&self) {
        assert_eq!(self.indptr.len(), self.nrows + 1, "csr: indptr length");
        assert_eq!(self.data.len(), self.indices.len(), "csr: data/indices");
        assert_eq!(self.indptr[0], 0, "csr: indptr[0] must be 0");
        assert_eq!(*self.indptr.last().unwrap(), self.data.len(), "csr: indptr end");
        for i in 0..self.nrows {
            assert!(self.indptr[i] <= self.indptr[i + 1]);
            for p in self.indptr[i]..self.indptr[i + 1] {
                assert!(self.indices[p] < self.ncols, "csr: col index OOB");
            }
        }
    }
}

impl CscMatrix {
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.nrows, self.ncols)
    }

    pub fn check(&self) {
        assert_eq!(self.indptr.len(), self.ncols + 1, "csc: indptr length");
        assert_eq!(self.data.len(), self.indices.len(), "csc: data/indices");
        assert_eq!(self.indptr[0], 0, "csc: indptr[0] must be 0");
        assert_eq!(*self.indptr.last().unwrap(), self.data.len(), "csc: indptr end");
        for j in 0..self.ncols {
            assert!(self.indptr[j] <= self.indptr[j + 1]);
            for p in self.indptr[j]..self.indptr[j + 1] {
                assert!(self.indices[p] < self.nrows, "csc: row index OOB");
            }
        }
    }
}

/// `scipy.sparse.csr_matrix(dense)` — drop exact zeros.
pub fn csr_from_dense(a: &NdArray) -> CsrMatrix {
    assert_eq!(a.ndim(), 2, "csr_from_dense: expected 2D");
    let nrows = a.shape()[0];
    let ncols = a.shape()[1];
    let ac = a.to_contiguous();
    let s = ac.as_slice().unwrap();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(nrows + 1);
    indptr.push(0);
    for i in 0..nrows {
        for j in 0..ncols {
            let v = s[i * ncols + j];
            if v != 0.0 {
                data.push(v);
                indices.push(j);
            }
        }
        indptr.push(data.len());
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CsrMatrix {
        nrows,
        ncols,
        data,
        indices,
        indptr,
    }
}

/// `scipy.sparse.csc_matrix(dense)` — drop exact zeros.
pub fn csc_from_dense(a: &NdArray) -> CscMatrix {
    assert_eq!(a.ndim(), 2, "csc_from_dense: expected 2D");
    let nrows = a.shape()[0];
    let ncols = a.shape()[1];
    let ac = a.to_contiguous();
    let s = ac.as_slice().unwrap();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(ncols + 1);
    indptr.push(0);
    for j in 0..ncols {
        for i in 0..nrows {
            let v = s[i * ncols + j];
            if v != 0.0 {
                data.push(v);
                indices.push(i);
            }
        }
        indptr.push(data.len());
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CscMatrix {
        nrows,
        ncols,
        data,
        indices,
        indptr,
    }
}

/// CSR → dense.
pub fn csr_to_dense(a: &CsrMatrix) -> NdArray {
    let mut out = vec![0.0; a.nrows * a.ncols];
    for i in 0..a.nrows {
        for p in a.indptr[i]..a.indptr[i + 1] {
            out[i * a.ncols + a.indices[p]] = a.data[p];
        }
    }
    NdArray::from_shape_vec(&[a.nrows, a.ncols], out)
}

/// CSC → dense.
pub fn csc_to_dense(a: &CscMatrix) -> NdArray {
    let mut out = vec![0.0; a.nrows * a.ncols];
    for j in 0..a.ncols {
        for p in a.indptr[j]..a.indptr[j + 1] {
            out[a.indices[p] * a.ncols + j] = a.data[p];
        }
    }
    NdArray::from_shape_vec(&[a.nrows, a.ncols], out)
}

/// `scipy.sparse.eye(n, format='csr')`.
pub fn eye_csr(n: usize) -> CsrMatrix {
    let (data, indices, indptr) = arc_vecs(
        vec![1.0; n],
        (0..n).collect(),
        (0..=n).collect(),
    );
    CsrMatrix {
        nrows: n,
        ncols: n,
        data,
        indices,
        indptr,
    }
}

/// `scipy.sparse.eye(n, format='csc')`.
pub fn eye_csc(n: usize) -> CscMatrix {
    let (data, indices, indptr) = arc_vecs(
        vec![1.0; n],
        (0..n).collect(),
        (0..=n).collect(),
    );
    CscMatrix {
        nrows: n,
        ncols: n,
        data,
        indices,
        indptr,
    }
}

/// `scipy.sparse.diags(diagonals, offsets, shape)` for a single main diagonal.
pub fn diags_csr(diag: &[f64], n: usize) -> CsrMatrix {
    assert_eq!(diag.len(), n, "diags_csr: diag length must match n");
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n + 1);
    indptr.push(0);
    for i in 0..n {
        if diag[i] != 0.0 {
            data.push(diag[i]);
            indices.push(i);
        }
        indptr.push(data.len());
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CsrMatrix {
        nrows: n,
        ncols: n,
        data,
        indices,
        indptr,
    }
}

/// CSR transpose as CSC (`A.T` with `format='csc'`).
///
/// CSR storage of `A` is identical to CSC storage of `A.T` with dims swapped.
/// `Arc` clone is O(1).
pub fn csr_transpose(a: &CsrMatrix) -> CscMatrix {
    CscMatrix {
        nrows: a.ncols,
        ncols: a.nrows,
        data: Arc::clone(&a.data),
        indices: Arc::clone(&a.indices),
        indptr: Arc::clone(&a.indptr),
    }
}

/// CSR → CSC conversion (sorts row indices within columns).
pub fn csr_to_csc(a: &CsrMatrix) -> CscMatrix {
    let nnz = a.nnz();
    let mut col_counts = vec![0usize; a.ncols];
    for &j in a.indices.iter() {
        col_counts[j] += 1;
    }
    let mut indptr = Vec::with_capacity(a.ncols + 1);
    indptr.push(0);
    for c in &col_counts {
        indptr.push(indptr.last().unwrap() + c);
    }
    let mut data = vec![0.0; nnz];
    let mut indices = vec![0usize; nnz];
    let mut next = indptr[..a.ncols].to_vec();
    for i in 0..a.nrows {
        for p in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[p];
            let dest = next[j];
            data[dest] = a.data[p];
            indices[dest] = i;
            next[j] += 1;
        }
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CscMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        data,
        indices,
        indptr,
    }
}

/// CSC → CSR conversion.
pub fn csc_to_csr(a: &CscMatrix) -> CsrMatrix {
    let nnz = a.nnz();
    let mut row_counts = vec![0usize; a.nrows];
    for &i in a.indices.iter() {
        row_counts[i] += 1;
    }
    let mut indptr = Vec::with_capacity(a.nrows + 1);
    indptr.push(0);
    for c in &row_counts {
        indptr.push(indptr.last().unwrap() + c);
    }
    let mut data = vec![0.0; nnz];
    let mut indices = vec![0usize; nnz];
    let mut next = indptr[..a.nrows].to_vec();
    for j in 0..a.ncols {
        for p in a.indptr[j]..a.indptr[j + 1] {
            let i = a.indices[p];
            let dest = next[i];
            data[dest] = a.data[p];
            indices[dest] = j;
            next[i] += 1;
        }
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        data,
        indices,
        indptr,
    }
}

/// CSR @ dense vector → dense vector (`scipy` SpMV).
pub fn csr_matvec(a: &CsrMatrix, x: &NdArray) -> NdArray {
    assert_eq!(x.ndim(), 1, "csr_matvec: x must be 1D");
    assert_eq!(x.len(), a.ncols, "csr_matvec: width mismatch");
    let xc = x.to_contiguous();
    let xs = xc.as_slice().unwrap();
    let mut y = vec![0.0; a.nrows];
    let data = a.data.as_slice();
    let indices = a.indices.as_slice();
    let indptr = a.indptr.as_slice();
    for i in 0..a.nrows {
        let mut s = 0.0;
        for p in indptr[i]..indptr[i + 1] {
            s += data[p] * xs[indices[p]];
        }
        y[i] = s;
    }
    NdArray::from_vec(y)
}

/// CSR @ dense matrix (m×k) → dense (nrows×k).
///
/// Special-cases common RHS widths and uses a dense accumulator row.
pub fn csr_matmat(a: &CsrMatrix, b: &NdArray) -> NdArray {
    assert_eq!(b.ndim(), 2, "csr_matmat: B must be 2D");
    assert_eq!(b.shape()[0], a.ncols, "csr_matmat: inner dim mismatch");
    let k = b.shape()[1];
    let bc = b.to_contiguous();
    let bs = bc.as_slice().unwrap();
    let mut y = vec![0.0; a.nrows * k];
    let data = a.data.as_slice();
    let indices = a.indices.as_slice();
    let indptr = a.indptr.as_slice();

    match k {
        1 => {
            for i in 0..a.nrows {
                let mut s = 0.0;
                for p in indptr[i]..indptr[i + 1] {
                    s += data[p] * bs[indices[p]];
                }
                y[i] = s;
            }
        }
        4 => {
            for i in 0..a.nrows {
                let mut s0 = 0.0;
                let mut s1 = 0.0;
                let mut s2 = 0.0;
                let mut s3 = 0.0;
                for p in indptr[i]..indptr[i + 1] {
                    let v = data[p];
                    let brow = indices[p] * 4;
                    s0 += v * bs[brow];
                    s1 += v * bs[brow + 1];
                    s2 += v * bs[brow + 2];
                    s3 += v * bs[brow + 3];
                }
                let yrow = i * 4;
                y[yrow] = s0;
                y[yrow + 1] = s1;
                y[yrow + 2] = s2;
                y[yrow + 3] = s3;
            }
        }
        8 => {
            for i in 0..a.nrows {
                let mut acc = [0.0; 8];
                for p in indptr[i]..indptr[i + 1] {
                    let v = data[p];
                    let brow = indices[p] * 8;
                    for c in 0..8 {
                        acc[c] += v * bs[brow + c];
                    }
                }
                let yrow = i * 8;
                y[yrow..yrow + 8].copy_from_slice(&acc);
            }
        }
        _ => {
            // Dense row accumulator: touch each RHS column once per row nnz.
            let mut acc = vec![0.0; k];
            for i in 0..a.nrows {
                acc.fill(0.0);
                for p in indptr[i]..indptr[i + 1] {
                    let v = data[p];
                    let brow = indices[p] * k;
                    for c in 0..k {
                        acc[c] += v * bs[brow + c];
                    }
                }
                let yrow = i * k;
                y[yrow..yrow + k].copy_from_slice(&acc);
            }
        }
    }
    NdArray::from_shape_vec(&[a.nrows, k], y)
}

/// CSC @ dense vector.
pub fn csc_matvec(a: &CscMatrix, x: &NdArray) -> NdArray {
    assert_eq!(x.ndim(), 1, "csc_matvec: x must be 1D");
    assert_eq!(x.len(), a.ncols, "csc_matvec: width mismatch");
    let xc = x.to_contiguous();
    let xs = xc.as_slice().unwrap();
    let mut y = vec![0.0; a.nrows];
    for j in 0..a.ncols {
        let xj = xs[j];
        if xj == 0.0 {
            continue;
        }
        for p in a.indptr[j]..a.indptr[j + 1] {
            y[a.indices[p]] += a.data[p] * xj;
        }
    }
    NdArray::from_vec(y)
}

/// Elementwise scale: `a * scalar` (CSR).
pub fn csr_scale(a: &CsrMatrix, alpha: f64) -> CsrMatrix {
    let data: Vec<f64> = a.data.iter().map(|v| v * alpha).collect();
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        data: Arc::new(data),
        indices: Arc::clone(&a.indices),
        indptr: Arc::clone(&a.indptr),
    }
}

/// CSR + CSR (same shape). Unsorted duplicate columns possible; we coalesce.
pub fn csr_add(a: &CsrMatrix, b: &CsrMatrix) -> CsrMatrix {
    assert_eq!(a.nrows, b.nrows);
    assert_eq!(a.ncols, b.ncols);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(a.nrows + 1);
    indptr.push(0);
    let mut row_acc = vec![0.0; a.ncols];
    let mut touched = Vec::new();
    let mut mark = vec![0u32; a.ncols];
    let mut stamp = 1u32;

    for i in 0..a.nrows {
        if stamp == u32::MAX {
            mark.fill(0);
            stamp = 1;
        }
        touched.clear();
        for p in a.indptr[i]..a.indptr[i + 1] {
            let j = a.indices[p];
            if mark[j] != stamp {
                mark[j] = stamp;
                row_acc[j] = 0.0;
                touched.push(j);
            }
            row_acc[j] += a.data[p];
        }
        for p in b.indptr[i]..b.indptr[i + 1] {
            let j = b.indices[p];
            if mark[j] != stamp {
                mark[j] = stamp;
                row_acc[j] = 0.0;
                touched.push(j);
            }
            row_acc[j] += b.data[p];
        }
        touched.sort_unstable();
        for &j in &touched {
            let v = row_acc[j];
            if v != 0.0 {
                data.push(v);
                indices.push(j);
            }
        }
        indptr.push(data.len());
        stamp += 1;
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CsrMatrix {
        nrows: a.nrows,
        ncols: a.ncols,
        data,
        indices,
        indptr,
    }
}

/// Frobenius norm of CSR data.
pub fn csr_frobenius_norm(a: &CsrMatrix) -> f64 {
    a.data.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Build CSR by dropping entries with `|a_ij| < thresh`.
pub fn csr_from_threshold(a: &NdArray, thresh: f64) -> CsrMatrix {
    assert_eq!(a.ndim(), 2, "csr_from_threshold: expected 2D");
    let nrows = a.shape()[0];
    let ncols = a.shape()[1];
    let ac = a.to_contiguous();
    let s = ac.as_slice().unwrap();
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(nrows + 1);
    indptr.push(0);
    for i in 0..nrows {
        for j in 0..ncols {
            let v = s[i * ncols + j];
            if v.abs() >= thresh {
                data.push(v);
                indices.push(j);
            }
        }
        indptr.push(data.len());
    }
    let (data, indices, indptr) = arc_vecs(data, indices, indptr);
    CsrMatrix {
        nrows,
        ncols,
        data,
        indices,
        indptr,
    }
}

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// `scipy.sparse.linalg.spsolve(A, b)` for square CSR `A`.
///
/// v1: densifies then uses `rnumpy::solve` (correct for modest `n`).
pub fn spsolve(a: &CsrMatrix, b: &NdArray) -> NdArray {
    assert_eq!(a.nrows, a.ncols, "spsolve: square matrix required");
    assert_eq!(b.ndim(), 1, "spsolve: 1D rhs for v1");
    assert_eq!(b.len(), a.nrows, "spsolve: rhs length mismatch");
    let dense = csr_to_dense(a);
    rnumpy::solve(&dense, b)
}

/// `scipy.sparse.linalg.cg(A, b, rtol=tol)` — conjugate gradient for SPD CSR.
///
/// Returns the approximate solution (no convergence info object yet).
pub fn cg(a: &CsrMatrix, b: &NdArray, tol: f64, maxiter: Option<usize>) -> NdArray {
    assert_eq!(a.nrows, a.ncols, "cg: square matrix required");
    assert_eq!(b.ndim(), 1, "cg: expected 1D rhs");
    assert_eq!(b.len(), a.nrows, "cg: rhs length mismatch");
    let n = a.nrows;
    let maxiter = maxiter.unwrap_or(n.saturating_mul(10).max(1));
    let bc = b.to_contiguous();
    let b_s = bc.as_slice().unwrap();

    let mut x = vec![0.0; n];
    let mut r = b_s.to_vec();
    let mut p = r.clone();
    let mut rsold = dot_f64(&r, &r);
    let bnorm = rsold.sqrt().max(1.0);

    for _ in 0..maxiter {
        if rsold.sqrt() <= tol * bnorm {
            break;
        }
        let ap = csr_matvec(a, &NdArray::from_vec(p.clone()));
        let ap_s = ap.as_slice().unwrap();
        let pap = dot_f64(&p, ap_s);
        assert!(pap.abs() > 0.0, "cg: breakdown (pᵀAp = 0)");
        let alpha = rsold / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap_s[i];
        }
        let rsnew = dot_f64(&r, &r);
        if rsnew.sqrt() <= tol * bnorm {
            break;
        }
        let beta = rsnew / rsold;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rsold = rsnew;
    }
    NdArray::from_vec(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnumpy::seeded_uniform;

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    fn spd_dense(n: usize, seed: u64) -> NdArray {
        let mut m = seeded_uniform(&[n, n], seed, -1.0, 1.0);
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (m[[i, j]] + m[[j, i]]);
                m[[i, j]] = v;
                m[[j, i]] = v;
            }
            m[[i, i]] += (n as f64) + 1.0;
        }
        m
    }

    #[test]
    fn csr_add_diag() {
        let a = eye_csr(3);
        let b = csr_scale(&a, 2.0);
        let c = csr_add(&a, &b);
        let d = csr_to_dense(&c);
        for i in 0..3 {
            assert_close(d[[i, i]], 3.0, 1e-12);
        }
    }

    #[test]
    fn csr_to_csc_roundtrip() {
        let a = seeded_uniform(&[4, 4], 7, -1.0, 1.0);
        let csr = csr_from_threshold(&a, 0.25);
        let csc = csr_to_csc(&csr);
        let back = csc_to_csr(&csc);
        let d1 = csr_to_dense(&csr);
        let d2 = csr_to_dense(&back);
        for i in 0..4 {
            for j in 0..4 {
                assert_close(d1[[i, j]], d2[[i, j]], 1e-12);
            }
        }
    }

    #[test]
    fn csr_transpose_matches_dense() {
        let a = seeded_uniform(&[3, 5], 9, -1.0, 1.0);
        let csr = csr_from_threshold(&a, 0.2);
        let at = csr_transpose(&csr);
        let d = csr_to_dense(&csr);
        let dt = csc_to_dense(&at);
        for i in 0..3 {
            for j in 0..5 {
                assert_close(dt[[j, i]], d[[i, j]], 1e-12);
            }
        }
    }

    #[test]
    fn csr_transpose_shares_storage() {
        let a = eye_csr(4);
        let at = csr_transpose(&a);
        assert!(Arc::ptr_eq(&a.data, &at.data));
        assert!(Arc::ptr_eq(&a.indices, &at.indices));
        assert!(Arc::ptr_eq(&a.indptr, &at.indptr));
    }

    #[test]
    fn spsolve_identity() {
        let a = eye_csr(4);
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let x = spsolve(&a, &b);
        assert_eq!(x.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cg_spd() {
        let dense = spd_dense(6, 3);
        let csr = csr_from_dense(&dense);
        let b = seeded_uniform(&[6], 4, -1.0, 1.0);
        let x = cg(&csr, &b, 1e-10, Some(200));
        let ax = csr_matvec(&csr, &x);
        for i in 0..6 {
            assert_close(ax[i], b[i], 1e-6);
        }
    }
}
