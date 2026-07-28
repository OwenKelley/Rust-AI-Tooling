//! Linear algebra — mirrors common `numpy` / `np.linalg` entry points.
//!
//! GEMM is in-house (`gemm`). Solve/inv/QR/SVD/eigh use local algorithms only.

use crate::creation::eye;
use crate::gemm::{dot_f64, gemm_rowmajor, gemv_rowmajor, gevm_rowmajor};
use crate::NdArray;

fn contig(a: &NdArray) -> NdArray {
    a.to_contiguous()
}

fn require_slice(a: &NdArray) -> NdArray {
    // Owned contiguous buffer so callers can borrow a stable slice.
    contig(a)
}

/// `np.transpose(a)` — O(1) strided view (NumPy-like).
pub fn transpose(a: &NdArray) -> NdArray {
    a.transpose_view()
}

/// `np.matmul(a, b)` for 2D matrices.
pub fn matmul(a: &NdArray, b: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "matmul: expected 2D A");
    assert_eq!(b.ndim(), 2, "matmul: expected 2D B");
    let m = a.shape()[0];
    let k = a.shape()[1];
    let k2 = b.shape()[0];
    let n = b.shape()[1];
    assert_eq!(k, k2, "matmul: inner dims must match");
    let ac = require_slice(a);
    let bc = require_slice(b);
    let data = gemm_rowmajor(ac.as_slice().unwrap(), bc.as_slice().unwrap(), m, k, n);
    NdArray::from_shape_vec(&[m, n], data)
}

/// `np.dot(a, b)` — vectors → scalar-as-0d; matrices → matmul.
pub fn dot(a: &NdArray, b: &NdArray) -> NdArray {
    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            let ac = require_slice(a);
            let bc = require_slice(b);
            let s = dot_f64(ac.as_slice().unwrap(), bc.as_slice().unwrap());
            NdArray::from_elem(&[], s)
        }
        (2, 2) => matmul(a, b),
        (2, 1) => {
            let m = a.shape()[0];
            let k = a.shape()[1];
            assert_eq!(b.len(), k, "dot: matrix-vector width mismatch");
            let ac = require_slice(a);
            let bc = require_slice(b);
            let data = gemv_rowmajor(ac.as_slice().unwrap(), bc.as_slice().unwrap(), m, k);
            NdArray::from_shape_vec(&[m], data)
        }
        (1, 2) => {
            let k = a.len();
            let k2 = b.shape()[0];
            let n = b.shape()[1];
            assert_eq!(k, k2, "dot: vector-matrix width mismatch");
            let ac = require_slice(a);
            let bc = require_slice(b);
            let data = gevm_rowmajor(ac.as_slice().unwrap(), bc.as_slice().unwrap(), k, n);
            NdArray::from_shape_vec(&[n], data)
        }
        _ => panic!("dot: unsupported ndims {} and {}", a.ndim(), b.ndim()),
    }
}

/// `np.trace(a)` for 2D.
pub fn trace(a: &NdArray) -> f64 {
    assert_eq!(a.ndim(), 2, "trace: expected 2D");
    let m = a.shape()[0].min(a.shape()[1]);
    let mut s = 0.0;
    for i in 0..m {
        s += a[[i, i]];
    }
    s
}

/// `np.linalg.norm(a)` — vector 2-norm / matrix Frobenius (default NumPy).
pub fn norm(a: &NdArray) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// `np.linalg.solve(a, b)` for square `a` and 1D/2D `b`.
pub fn solve(a: &NdArray, b: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "solve: A must be 2D");
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "solve: A must be square");
    assert!(b.ndim() == 1 || b.ndim() == 2, "solve: b must be 1D or 2D");
    assert_eq!(b.shape()[0], n, "solve: b rows must match A");

    let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };
    let ac = require_slice(a);
    let bc = require_slice(b);
    let a_s = ac.as_slice().unwrap();
    let b_s = bc.as_slice().unwrap();
    let mut aug = vec![0.0; n * (n + nrhs)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + nrhs) + j] = a_s[i * n + j];
        }
        for k in 0..nrhs {
            aug[i * (n + nrhs) + n + k] = if b.ndim() == 1 {
                b_s[i]
            } else {
                b_s[i * nrhs + k]
            };
        }
    }

    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[col * (n + nrhs) + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * (n + nrhs) + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        assert!(best > 0.0, "solve: singular matrix");
        if pivot != col {
            for j in 0..(n + nrhs) {
                aug.swap(col * (n + nrhs) + j, pivot * (n + nrhs) + j);
            }
        }
        let diag = aug[col * (n + nrhs) + col];
        for r in (col + 1)..n {
            let factor = aug[r * (n + nrhs) + col] / diag;
            aug[r * (n + nrhs) + col] = 0.0;
            for j in (col + 1)..(n + nrhs) {
                aug[r * (n + nrhs) + j] -= factor * aug[col * (n + nrhs) + j];
            }
        }
    }

    let mut x = vec![0.0; n * nrhs];
    for k in 0..nrhs {
        for i in (0..n).rev() {
            let mut s = aug[i * (n + nrhs) + n + k];
            for j in (i + 1)..n {
                s -= aug[i * (n + nrhs) + j] * x[j * nrhs + k];
            }
            x[i * nrhs + k] = s / aug[i * (n + nrhs) + i];
        }
    }

    if b.ndim() == 1 {
        NdArray::from_vec(x)
    } else {
        NdArray::from_shape_vec(&[n, nrhs], x)
    }
}

/// `np.linalg.inv(a)` for square `a`.
pub fn inv(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "inv: expected 2D");
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "inv: expected square");
    solve(a, &eye(n))
}

/// `np.linalg.det(a)` via Gaussian elimination with partial pivoting.
pub fn det(a: &NdArray) -> f64 {
    assert_eq!(a.ndim(), 2, "det: expected 2D");
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "det: expected square");
    let mut m = require_slice(a).as_slice().unwrap().to_vec();
    let mut sign = 1.0;
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[col * n + col].abs();
        for r in (col + 1)..n {
            let v = m[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best == 0.0 {
            return 0.0;
        }
        if pivot != col {
            sign = -sign;
            for j in 0..n {
                m.swap(col * n + j, pivot * n + j);
            }
        }
        let diag = m[col * n + col];
        for r in (col + 1)..n {
            let factor = m[r * n + col] / diag;
            m[r * n + col] = 0.0;
            for j in (col + 1)..n {
                m[r * n + j] -= factor * m[col * n + j];
            }
        }
    }
    let mut d = sign;
    for i in 0..n {
        d *= m[i * n + i];
    }
    d
}

/// `np.linalg.qr(a)` — thin QR via Householder reflections. Returns `(Q, R)`.
pub fn qr(a: &NdArray) -> (NdArray, NdArray) {
    assert_eq!(a.ndim(), 2, "qr: expected 2D");
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n);
    let mut r = require_slice(a).as_slice().unwrap().to_vec(); // m x n row-major
    let mut q = eye(m).as_slice().unwrap().to_vec(); // m x m

    for j in 0..k {
        // Householder on column j from row j downward.
        let mut norm_sq = 0.0;
        for i in j..m {
            let v = r[i * n + j];
            norm_sq += v * v;
        }
        let mut norm = norm_sq.sqrt();
        if norm == 0.0 {
            continue;
        }
        // Sign choice for stability.
        if r[j * n + j] > 0.0 {
            norm = -norm;
        }
        let u0 = r[j * n + j] - norm;
        let mut u_norm_sq = u0 * u0;
        for i in (j + 1)..m {
            let v = r[i * n + j];
            u_norm_sq += v * v;
        }
        if u_norm_sq == 0.0 {
            continue;
        }
        let scale = 2.0 / u_norm_sq;

        // Apply H to R from the left: columns j..n
        for col in j..n {
            let mut dot = u0 * r[j * n + col];
            for i in (j + 1)..m {
                dot += r[i * n + j] * r[i * n + col];
            }
            // u for i>j is r[i,j] before overwrite; save u vector first
            let _ = dot;
        }
        // Build u explicitly
        let mut u = vec![0.0; m - j];
        u[0] = u0;
        for i in (j + 1)..m {
            u[i - j] = r[i * n + j];
        }
        for col in j..n {
            let mut dot = 0.0;
            for i in 0..u.len() {
                dot += u[i] * r[(j + i) * n + col];
            }
            let factor = scale * dot;
            for i in 0..u.len() {
                r[(j + i) * n + col] -= factor * u[i];
            }
        }
        // Apply H to Q from the right: Q := Q H (so columns transform)
        for row in 0..m {
            let mut dot = 0.0;
            for i in 0..u.len() {
                dot += u[i] * q[row * m + (j + i)];
            }
            let factor = scale * dot;
            for i in 0..u.len() {
                q[row * m + (j + i)] -= factor * u[i];
            }
        }
        // Zero below diagonal in R column j
        r[j * n + j] = norm;
        for i in (j + 1)..m {
            r[i * n + j] = 0.0;
        }
    }

    // Thin Q: first k columns; thin R: first k rows
    let mut q_thin = vec![0.0; m * k];
    for i in 0..m {
        for j in 0..k {
            q_thin[i * k + j] = q[i * m + j];
        }
    }
    let mut r_thin = vec![0.0; k * n];
    for i in 0..k {
        for j in 0..n {
            r_thin[i * n + j] = r[i * n + j];
        }
    }
    (
        NdArray::from_shape_vec(&[m, k], q_thin),
        NdArray::from_shape_vec(&[k, n], r_thin),
    )
}

/// Jacobi eigenvalue algorithm for symmetric matrices. Returns ascending eigenvalues.
fn jacobi_eigh_sym(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let tol = 1e-14;
    for _ in 0..(n * n * 30).max(100) {
        // Find largest off-diagonal
        let mut p = 0;
        let mut q = 1;
        let mut max = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = m[i * n + j].abs();
                if val > max {
                    max = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max < tol {
            break;
        }
        let app = m[p * n + p];
        let aqq = m[q * n + q];
        let apq = m[p * n + q];
        let theta = 0.5 * (aqq - app) / apq;
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            -1.0 / (-theta + (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // Rotate
        m[p * n + p] = app - t * apq;
        m[q * n + q] = aqq + t * apq;
        m[p * n + q] = 0.0;
        m[q * n + p] = 0.0;
        for i in 0..n {
            if i != p && i != q {
                let aip = m[i * n + p];
                let aiq = m[i * n + q];
                m[i * n + p] = c * aip - s * aiq;
                m[p * n + i] = m[i * n + p];
                m[i * n + q] = c * aiq + s * aip;
                m[q * n + i] = m[i * n + q];
            }
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip - s * viq;
            v[i * n + q] = s * vip + c * viq;
        }
    }
    let mut evals: Vec<(f64, usize)> = (0..n).map(|i| (m[i * n + i], i)).collect();
    evals.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = vec![0.0; n * n];
    for (new_i, &(val, old_i)) in evals.iter().enumerate() {
        eigenvalues.push(val);
        for r in 0..n {
            eigenvectors[r * n + new_i] = v[r * n + old_i];
        }
    }
    (eigenvalues, eigenvectors)
}

/// `np.linalg.eigvalsh(a)` — eigenvalues of a symmetric matrix (ascending).
pub fn eigvalsh(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2);
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n);
    let ac = require_slice(a);
    let (evals, _) = jacobi_eigh_sym(ac.as_slice().unwrap(), n);
    NdArray::from_vec(evals)
}

/// `np.linalg.eigh(a)` — `(eigenvalues, eigenvectors)` for symmetric `a`.
pub fn eigh(a: &NdArray) -> (NdArray, NdArray) {
    assert_eq!(a.ndim(), 2);
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n);
    let ac = require_slice(a);
    let (evals, evecs) = jacobi_eigh_sym(ac.as_slice().unwrap(), n);
    (
        NdArray::from_vec(evals),
        NdArray::from_shape_vec(&[n, n], evecs),
    )
}

/// `np.linalg.svd(a, compute_uv=False)` — singular values descending.
pub fn svdvals(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2);
    let m = a.shape()[0];
    let n = a.shape()[1];
    let ac = require_slice(a);
    let a_s = ac.as_slice().unwrap();
    // Form Gram matrix of the smaller side.
    let (gram, dim) = if n <= m {
        // A^T A (n x n)
        let mut g = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let mut s = 0.0;
                for r in 0..m {
                    s += a_s[r * n + i] * a_s[r * n + j];
                }
                g[i * n + j] = s;
                g[j * n + i] = s;
            }
        }
        (g, n)
    } else {
        // A A^T (m x m)
        let mut g = vec![0.0; m * m];
        for i in 0..m {
            for j in i..m {
                let mut s = 0.0;
                for c in 0..n {
                    s += a_s[i * n + c] * a_s[j * n + c];
                }
                g[i * m + j] = s;
                g[j * m + i] = s;
            }
        }
        (g, m)
    };
    let (evals, _) = jacobi_eigh_sym(&gram, dim);
    let mut svals: Vec<f64> = evals
        .into_iter()
        .map(|e| e.max(0.0).sqrt())
        .collect();
    svals.sort_by(|a, b| b.total_cmp(a)); // descending
    // Truncate to min(m,n)
    svals.truncate(m.min(n));
    NdArray::from_vec(svals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{eye, ones, seeded_uniform};
    use crate::ops::{abs as np_abs, subtract};
    use crate::test_util::assert_abs_diff_eq;

    #[test]
    fn matmul_eye() {
        let a = ones(&[3, 3]);
        let i = eye(3);
        let c = matmul(&a, &i);
        for x in c.iter() {
            assert_abs_diff_eq(x, 1.0, 1e-12);
        }
    }

    #[test]
    fn matmul_parallel_path() {
        let a = ones(&[128, 128]);
        let b = eye(128);
        let c = matmul(&a, &b);
        for x in c.iter() {
            assert_abs_diff_eq(x, 1.0, 1e-9);
        }
    }

    #[test]
    fn dot_vectors() {
        let a = ones(&[3]);
        let b = ones(&[3]);
        let s = dot(&a, &b);
        assert_abs_diff_eq(s[[]], 3.0, 1e-12);
    }

    #[test]
    fn transpose_view_o1() {
        let a = NdArray::from_shape_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = transpose(&a);
        assert!(!t.is_c_contiguous());
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t[[2, 1]], 6.0);
    }

    #[test]
    fn solve_identity() {
        let a = eye(3);
        let b = NdArray::from_vec(vec![1.0, 2.0, 3.0]);
        let x = solve(&a, &b);
        assert_eq!(x.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn inv_roundtrip() {
        let mut a = seeded_uniform(&[4, 4], 7, -1.0, 1.0);
        for i in 0..4 {
            a[[i, i]] += 4.0;
        }
        let ai = inv(&a);
        let i = matmul(&a, &ai);
        let eye4 = eye(4);
        let err = subtract(&i, &eye4);
        for x in err.iter() {
            assert_abs_diff_eq(x, 0.0, 1e-9);
        }
    }

    #[test]
    fn det_eye() {
        assert_abs_diff_eq(det(&eye(5)), 1.0, 1e-12);
    }

    #[test]
    fn norm_vector() {
        let a = NdArray::from_vec(vec![3.0, 4.0]);
        assert_abs_diff_eq(norm(&a), 5.0, 1e-12);
    }

    #[test]
    fn qr_reconstructs() {
        let a = seeded_uniform(&[5, 3], 3, -1.0, 1.0);
        let (q, r) = qr(&a);
        let recon = matmul(&q, &r);
        let err = np_abs(&subtract(&recon, &a));
        assert!(err.iter().all(|x| x < 1e-10));
    }

    #[test]
    fn eigvalsh_spd() {
        let mut s = seeded_uniform(&[4, 4], 1, -1.0, 1.0);
        for i in 0..4 {
            for j in 0..i {
                let v = 0.5 * (s[[i, j]] + s[[j, i]]);
                s[[i, j]] = v;
                s[[j, i]] = v;
            }
            s[[i, i]] += 4.0;
        }
        let w = eigvalsh(&s);
        assert_eq!(w.len(), 4);
        assert!(w.iter().all(|x| x > 0.0));
    }

    #[test]
    fn svdvals_eye() {
        let s = svdvals(&eye(3));
        assert_eq!(s.as_slice().unwrap(), &[1.0, 1.0, 1.0]);
    }
}
