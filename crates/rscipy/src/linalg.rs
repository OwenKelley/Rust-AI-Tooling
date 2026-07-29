//! Linear algebra — mirrors common `scipy.linalg` entry points.
//!
//! Built on `rnumpy::NdArray`. Algorithms are local (no BLAS/LAPACK crates).

use rnumpy::{eye, matmul, qr, solve, svdvals, NdArray};

fn contig(a: &NdArray) -> NdArray {
    a.to_contiguous()
}

/// Norm order for [`norm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormOrd {
    /// Default: vector 2-norm / matrix Frobenius.
    Default,
    Fro,
    One,
    Inf,
    /// Vector Euclidean / matrix spectral (largest singular value).
    Two,
}

/// `scipy.linalg.norm(a, ord=...)`.
pub fn norm(a: &NdArray) -> f64 {
    norm_ord(a, NormOrd::Default)
}

/// Norm with explicit order.
pub fn norm_ord(a: &NdArray, ord: NormOrd) -> f64 {
    match a.ndim() {
        1 => match ord {
            NormOrd::Default | NormOrd::Two | NormOrd::Fro => {
                a.iter().map(|x| x * x).sum::<f64>().sqrt()
            }
            NormOrd::One => a.iter().map(|x| x.abs()).sum(),
            NormOrd::Inf => a.iter().map(|x| x.abs()).fold(0.0_f64, f64::max),
        },
        2 => {
            let m = a.shape()[0];
            let n = a.shape()[1];
            let ac = contig(a);
            let s = ac.as_slice().unwrap();
            match ord {
                NormOrd::Default | NormOrd::Fro => {
                    s.iter().map(|x| x * x).sum::<f64>().sqrt()
                }
                NormOrd::One => {
                    let mut best: f64 = 0.0;
                    for j in 0..n {
                        let mut col = 0.0;
                        for i in 0..m {
                            col += s[i * n + j].abs();
                        }
                        best = best.max(col);
                    }
                    best
                }
                NormOrd::Inf => {
                    let mut best: f64 = 0.0;
                    for i in 0..m {
                        let mut row = 0.0;
                        for j in 0..n {
                            row += s[i * n + j].abs();
                        }
                        best = best.max(row);
                    }
                    best
                }
                NormOrd::Two => {
                    let sv = svdvals(a);
                    sv.iter().fold(0.0_f64, f64::max)
                }
            }
        }
        _ => panic!("norm: expected 1D or 2D"),
    }
}

/// Shared GE with partial pivoting. Returns packed LU (m×n) and pivot vector
/// where `piv[i]` is the row index that was swapped into position `i`
/// (SciPy `lu_factor` 0-based pivot semantics for square: row i exchanged with piv[i]).
fn lu_factor_raw(a: &NdArray) -> (Vec<f64>, Vec<i64>, usize, usize) {
    assert_eq!(a.ndim(), 2, "lu_factor: expected 2D");
    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = m.min(n);
    let mut lu = contig(a).as_slice().unwrap().to_vec();
    // SciPy piv[i] = row that was interchanged with row i (after previous swaps).
    let mut piv = vec![0i64; k];

    for col in 0..k {
        let mut pivot = col;
        let mut best = lu[col * n + col].abs();
        for r in (col + 1)..m {
            let v = lu[r * n + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        assert!(best > 0.0, "lu_factor: singular / zero pivot");
        piv[col] = pivot as i64;
        if pivot != col {
            for j in 0..n {
                lu.swap(col * n + j, pivot * n + j);
            }
        }
        let diag = lu[col * n + col];
        for r in (col + 1)..m {
            let factor = lu[r * n + col] / diag;
            lu[r * n + col] = factor;
            let row = r * n;
            let prow = col * n;
            for j in (col + 1)..n {
                lu[row + j] -= factor * lu[prow + j];
            }
        }
    }
    (lu, piv, m, n)
}

/// `scipy.linalg.lu_factor(a)` → `(lu, piv)`.
pub fn lu_factor(a: &NdArray) -> (NdArray, NdArray) {
    let (lu, piv, m, n) = lu_factor_raw(a);
    (
        NdArray::from_shape_vec(&[m, n], lu),
        NdArray::from_vec(piv.iter().map(|&p| p as f64).collect()),
    )
}

/// `scipy.linalg.lu(a)` with partial pivoting.
///
/// Returns `(P, L, U)` such that `A = P @ L @ U` (SciPy convention).
pub fn lu(a: &NdArray) -> (NdArray, NdArray, NdArray) {
    let (lu_data, piv, m, n) = lu_factor_raw(a);
    let k = m.min(n);

    // Reconstruct permutation: start with identity, apply swaps piv[i]↔i in order.
    let mut perm: Vec<usize> = (0..m).collect();
    for i in 0..k {
        let j = piv[i] as usize;
        perm.swap(i, j);
    }
    // After swaps, row i of LU came from original row perm[i].
    // P_ge @ A = L@U with P_ge[i, perm[i]] = 1.
    // SciPy A = P @ L @ U ⇒ P = P_ge^T ⇒ P[perm[i], i] = 1.
    let mut p = vec![0.0; m * m];
    for i in 0..m {
        p[perm[i] * m + i] = 1.0;
    }

    let mut l = vec![0.0; m * k];
    let mut u = vec![0.0; k * n];
    for i in 0..m {
        for j in 0..k {
            if i > j {
                l[i * k + j] = lu_data[i * n + j];
            } else if i == j {
                l[i * k + j] = 1.0;
            }
        }
    }
    for i in 0..k {
        for j in 0..n {
            if i <= j {
                u[i * n + j] = lu_data[i * n + j];
            }
        }
    }

    (
        NdArray::from_shape_vec(&[m, m], p),
        NdArray::from_shape_vec(&[m, k], l),
        NdArray::from_shape_vec(&[k, n], u),
    )
}

/// `scipy.linalg.cholesky(a, lower=True)` for SPD `a`.
pub fn cholesky(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "cholesky: expected 2D");
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "cholesky: expected square");
    let ac = contig(a);
    let a_s = ac.as_slice().unwrap();
    let mut l = vec![0.0; n * n];

    for i in 0..n {
        for j in 0..=i {
            let mut s = a_s[i * n + j];
            for p in 0..j {
                s -= l[i * n + p] * l[j * n + p];
            }
            if i == j {
                assert!(s > 0.0, "cholesky: matrix not SPD");
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    NdArray::from_shape_vec(&[n, n], l)
}

/// `scipy.linalg.solve_triangular(a, b, lower=...)`.
pub fn solve_triangular(a: &NdArray, b: &NdArray, lower: bool) -> NdArray {
    assert_eq!(a.ndim(), 2, "solve_triangular: A must be 2D");
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "solve_triangular: A must be square");
    assert!(
        b.ndim() == 1 || b.ndim() == 2,
        "solve_triangular: b must be 1D or 2D"
    );
    assert_eq!(b.shape()[0], n, "solve_triangular: b rows must match A");

    let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };
    let ac = contig(a);
    let bc = contig(b);
    let a_s = ac.as_slice().unwrap();
    let b_s = bc.as_slice().unwrap();
    let mut x = vec![0.0; n * nrhs];
    for i in 0..n {
        for k in 0..nrhs {
            x[i * nrhs + k] = if b.ndim() == 1 {
                b_s[i]
            } else {
                b_s[i * nrhs + k]
            };
        }
    }

    if lower {
        for k in 0..nrhs {
            for i in 0..n {
                let mut s = x[i * nrhs + k];
                for j in 0..i {
                    s -= a_s[i * n + j] * x[j * nrhs + k];
                }
                let diag = a_s[i * n + i];
                assert!(diag != 0.0, "solve_triangular: zero diagonal");
                x[i * nrhs + k] = s / diag;
            }
        }
    } else {
        for k in 0..nrhs {
            for i in (0..n).rev() {
                let mut s = x[i * nrhs + k];
                for j in (i + 1)..n {
                    s -= a_s[i * n + j] * x[j * nrhs + k];
                }
                let diag = a_s[i * n + i];
                assert!(diag != 0.0, "solve_triangular: zero diagonal");
                x[i * nrhs + k] = s / diag;
            }
        }
    }

    if b.ndim() == 1 {
        NdArray::from_vec(x)
    } else {
        NdArray::from_shape_vec(&[n, nrhs], x)
    }
}

/// `scipy.linalg.lstsq(a, b)` — least squares via thin QR.
///
/// Singular values / rank come from the R diagonal (rank-revealing QR estimate),
/// avoiding a separate Jacobi SVD on `A`.
pub fn lstsq(a: &NdArray, b: &NdArray) -> (NdArray, NdArray, usize, NdArray) {
    assert_eq!(a.ndim(), 2, "lstsq: A must be 2D");
    let m = a.shape()[0];
    let n = a.shape()[1];
    assert!(b.ndim() == 1 || b.ndim() == 2, "lstsq: b must be 1D or 2D");
    assert_eq!(b.shape()[0], m, "lstsq: b rows must match A");

    let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };
    let (q, r) = qr(a);
    let k = m.min(n);

    let qt = rnumpy::transpose(&q);
    let bc = if b.ndim() == 1 {
        let data = contig(b).as_slice().unwrap().to_vec();
        NdArray::from_shape_vec(&[m, 1], data)
    } else {
        contig(b)
    };
    let qty = matmul(&qt, &bc);

    let mut x = vec![0.0; n * nrhs];
    let rc = contig(&r);
    let r_s = rc.as_slice().unwrap();
    let qty_c = contig(&qty);
    let qty_s = qty_c.as_slice().unwrap();

    // σ ≈ |diag(R)| for thin QR (exact when A is well-conditioned / R diagonal
    // dominates). Sorted descending to match SciPy's SVD ordering.
    let mut s_vals: Vec<f64> = (0..k).map(|i| r_s[i * n + i].abs()).collect();
    s_vals.sort_by(|a, b| b.total_cmp(a));
    let s_max = s_vals.first().copied().unwrap_or(0.0);
    let thresh = (m.max(n) as f64) * f64::EPSILON * s_max;
    let rank = s_vals
        .iter()
        .filter(|&&v| v > thresh)
        .count()
        .max(if s_max > 0.0 { 1 } else { 0 })
        .min(k);

    let solve_n = rank.min(k);
    for col in 0..nrhs {
        let mut y = vec![0.0; solve_n];
        for i in (0..solve_n).rev() {
            let mut sum = qty_s[i * nrhs + col];
            for j in (i + 1)..solve_n {
                sum -= r_s[i * n + j] * y[j];
            }
            let diag = r_s[i * n + i];
            assert!(diag.abs() > 1e-14, "lstsq: rank-deficient");
            y[i] = sum / diag;
        }
        for i in 0..solve_n {
            x[i * nrhs + col] = y[i];
        }
    }

    let x_arr = if b.ndim() == 1 {
        NdArray::from_vec((0..n).map(|i| x[i * nrhs]).collect())
    } else {
        NdArray::from_shape_vec(&[n, nrhs], x.clone())
    };

    // For full-column-rank overdetermined LS: ||b - Ax||^2 = ||b||^2 - ||Q^T b||^2
    // (Q has orthonormal columns spanning range(A)).
    let residuals = if m > n && rank == n {
        let bcmp = if b.ndim() == 1 {
            contig(b)
        } else {
            contig(b)
        };
        let b_s = bcmp.as_slice().unwrap();
        let mut res = vec![0.0; nrhs];
        for col in 0..nrhs {
            let mut bnorm = 0.0;
            if b.ndim() == 1 {
                for i in 0..m {
                    let v = b_s[i];
                    bnorm += v * v;
                }
            } else {
                for i in 0..m {
                    let v = b_s[i * nrhs + col];
                    bnorm += v * v;
                }
            }
            let mut ynorm = 0.0;
            for i in 0..n {
                let v = qty_s[i * nrhs + col];
                ynorm += v * v;
            }
            res[col] = (bnorm - ynorm).max(0.0);
        }
        NdArray::from_vec(res)
    } else {
        NdArray::from_vec(vec![])
    };

    (x_arr, residuals, rank, NdArray::from_vec(s_vals))
}

/// `scipy.linalg.expm(a)` via scaling-and-squaring + Padé approximant (order 6).
pub fn expm(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "expm: expected 2D");
    let n = a.shape()[0];
    assert_eq!(a.shape()[1], n, "expm: expected square");

    // Scale so ||A||_1 / 2^s is small.
    let mut a1 = norm_ord(a, NormOrd::One);
    let mut s = 0usize;
    while a1 > 0.5 {
        a1 *= 0.5;
        s += 1;
    }
    let scale = (1u64 << s) as f64;
    let ac = contig(a);
    let mut ascaled = ac.as_slice().unwrap().to_vec();
    for v in ascaled.iter_mut() {
        *v /= scale;
    }
    let a_s = NdArray::from_shape_vec(&[n, n], ascaled);

    // Padé (6,6): U = A (b1 I + b3 A2 + b5 A4), V = b0 I + b2 A2 + b4 A4 + b6 A6
    // expm ≈ (V-U)^{-1} (V+U)
    let a2 = matmul(&a_s, &a_s);
    let a4 = matmul(&a2, &a2);
    let a6 = matmul(&a4, &a2);

    let b0 = 1.0;
    let b1 = 0.5;
    let b2 = 1.0 / 10.0;
    let b3 = 1.0 / 120.0;
    let b4 = 1.0 / 1680.0;
    let b5 = 1.0 / 30240.0;
    let b6 = 1.0 / 665280.0;
    let eye_n = eye(n);
    let mut u_mat = vec![0.0; n * n];
    let mut v_mat = vec![0.0; n * n];
    let a2s = contig(&a2).as_slice().unwrap().to_vec();
    let a4s = contig(&a4).as_slice().unwrap().to_vec();
    let a6s = contig(&a6).as_slice().unwrap().to_vec();
    let as_s = contig(&a_s).as_slice().unwrap().to_vec();
    let e_s = contig(&eye_n).as_slice().unwrap().to_vec();

    // V = b0 I + b2 A2 + b4 A4 + b6 A6
    // U = A (b1 I + b3 A2 + b5 A4)
    for i in 0..n * n {
        v_mat[i] = b0 * e_s[i] + b2 * a2s[i] + b4 * a4s[i] + b6 * a6s[i];
    }
    let mut tmp = vec![0.0; n * n];
    for i in 0..n * n {
        tmp[i] = b1 * e_s[i] + b3 * a2s[i] + b5 * a4s[i];
    }
    // U = A @ tmp
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += as_s[i * n + k] * tmp[k * n + j];
            }
            u_mat[i * n + j] = sum;
        }
    }

    let mut vpu = vec![0.0; n * n];
    let mut vmu = vec![0.0; n * n];
    for i in 0..n * n {
        vpu[i] = v_mat[i] + u_mat[i];
        vmu[i] = v_mat[i] - u_mat[i];
    }
    let vmu_a = NdArray::from_shape_vec(&[n, n], vmu);
    let vpu_a = NdArray::from_shape_vec(&[n, n], vpu);
    let mut r = solve(&vmu_a, &vpu_a);

    // Square s times
    for _ in 0..s {
        r = matmul(&r, &r);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnumpy::{matmul, seeded_uniform, transpose};

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    fn spd(n: usize) -> NdArray {
        let mut a = seeded_uniform(&[n, n], 7, -1.0, 1.0);
        for i in 0..n {
            for j in 0..i {
                let v = 0.5 * (a[[i, j]] + a[[j, i]]);
                a[[i, j]] = v;
                a[[j, i]] = v;
            }
            a[[i, i]] += n as f64;
        }
        a
    }

    #[test]
    fn cholesky_recovers() {
        let a = spd(5);
        let l = cholesky(&a);
        let a2 = matmul(&l, &transpose(&l));
        for i in 0..5 {
            for j in 0..5 {
                assert_close(a2[[i, j]], a[[i, j]], 1e-10);
            }
        }
    }

    #[test]
    fn lu_recovers() {
        let a = seeded_uniform(&[4, 4], 3, -1.0, 1.0);
        let (p, l, u) = lu(&a);
        let lu_m = matmul(&l, &u);
        let a2 = matmul(&p, &lu_m);
        for i in 0..4 {
            for j in 0..4 {
                assert_close(a2[[i, j]], a[[i, j]], 1e-10);
            }
        }
    }

    #[test]
    fn lu_factor_matches_lu_pack() {
        let a = seeded_uniform(&[3, 3], 5, -1.0, 1.0);
        let (lu_m, _piv) = lu_factor(&a);
        let (_, l, u) = lu(&a);
        // packed: below diag = L, upper = U
        for i in 0..3 {
            for j in 0..3 {
                if i > j {
                    assert_close(lu_m[[i, j]], l[[i, j]], 1e-12);
                } else {
                    assert_close(lu_m[[i, j]], u[[i, j]], 1e-12);
                }
            }
        }
    }

    #[test]
    fn solve_triangular_lower() {
        let a = spd(4);
        let l = cholesky(&a);
        let b = seeded_uniform(&[4], 11, -1.0, 1.0);
        let x = solve_triangular(&l, &b, true);
        let bx = matmul(
            &l,
            &NdArray::from_shape_vec(&[4, 1], x.as_slice().unwrap().to_vec()),
        );
        for i in 0..4 {
            assert_close(bx[[i, 0]], b[i], 1e-10);
        }
    }

    #[test]
    fn lstsq_exact_square() {
        let mut a = seeded_uniform(&[5, 5], 2, -1.0, 1.0);
        for i in 0..5 {
            a[[i, i]] += 5.0;
        }
        let b = seeded_uniform(&[5], 4, -1.0, 1.0);
        let (x, _, rank, s) = lstsq(&a, &b);
        assert_eq!(rank, 5);
        assert_eq!(s.len(), 5);
        let ax = matmul(
            &a,
            &NdArray::from_shape_vec(&[5, 1], x.as_slice().unwrap().to_vec()),
        );
        for i in 0..5 {
            assert_close(ax[[i, 0]], b[i], 1e-8);
        }
    }

    #[test]
    fn expm_zero_is_eye() {
        let a = NdArray::zeros(&[3, 3]);
        let e = expm(&a);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_close(e[[i, j]], expected, 1e-12);
            }
        }
    }

    #[test]
    fn norm_ords() {
        let v = NdArray::from_vec(vec![3.0, 4.0]);
        assert_close(norm_ord(&v, NormOrd::Two), 5.0, 1e-12);
        assert_close(norm_ord(&v, NormOrd::One), 7.0, 1e-12);
        assert_close(norm_ord(&v, NormOrd::Inf), 4.0, 1e-12);
    }
}
