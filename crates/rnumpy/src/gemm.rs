//! In-house GEMM / dot kernels using only `std` (no rayon, no BLAS crates).
//!
//! Storage may still come from `ndarray`; these routines own the flops.

use std::thread;

/// Row-parallel threshold: below this, serial blocked GEMM is usually faster.
const PARALLEL_ROWS: usize = 96;
/// Cache blocking size (tunable).
const BLOCK: usize = 64;

/// `C[m,n] = A[m,k] @ B[k,n]` for row-major contiguous buffers.
pub fn gemm_rowmajor(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), k * n, "B shape mismatch");
    let mut c = vec![0.0; m * n];
    if m >= PARALLEL_ROWS && n >= 32 && k >= 32 {
        gemm_parallel(a, b, &mut c, m, k, n);
    } else {
        gemm_blocked(a, b, &mut c, m, k, n);
    }
    c
}

/// Dot product of two equal-length vectors.
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    // Manual accumulators help the compiler keep more FMA pipelines busy.
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let n = a.len();
    let mut i = 0;
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    let mut s = s0 + s1 + s2 + s3;
    while i < n {
        s += a[i] * b[i];
        i += 1;
    }
    s
}

/// Matrix-vector: `y[m] = A[m,k] @ x[k]`.
pub fn gemv_rowmajor(a: &[f64], x: &[f64], m: usize, k: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k);
    assert_eq!(x.len(), k);
    let mut y = vec![0.0; m];
    for i in 0..m {
        y[i] = dot_f64(&a[i * k..(i + 1) * k], x);
    }
    y
}

/// Vector-matrix: `y[n] = x[k] @ B[k,n]`.
pub fn gevm_rowmajor(x: &[f64], b: &[f64], k: usize, n: usize) -> Vec<f64> {
    assert_eq!(x.len(), k);
    assert_eq!(b.len(), k * n);
    let mut y = vec![0.0; n];
    for p in 0..k {
        let xp = x[p];
        let brow = &b[p * n..(p + 1) * n];
        for j in 0..n {
            y[j] += xp * brow[j];
        }
    }
    y
}

fn gemm_parallel(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let workers = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .clamp(1, m);
    let chunk = m.div_ceil(workers);

    thread::scope(|scope| {
        let mut row = 0;
        let mut parts: Vec<&mut [f64]> = Vec::with_capacity(workers);
        let mut rest = &mut c[..];
        while row < m {
            let rows = chunk.min(m - row);
            let (head, tail) = rest.split_at_mut(rows * n);
            parts.push(head);
            rest = tail;
            row += rows;
        }

        let mut row_start = 0;
        for part in parts {
            let rows = part.len() / n;
            let a_rows = &a[row_start * k..(row_start + rows) * k];
            scope.spawn(move || {
                gemm_blocked(a_rows, b, part, rows, k, n);
            });
            row_start += rows;
        }
    });
}

fn gemm_blocked(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    let bs = BLOCK;
    let mut i0 = 0;
    while i0 < m {
        let i1 = (i0 + bs).min(m);
        let mut j0 = 0;
        while j0 < n {
            let j1 = (j0 + bs).min(n);
            let mut p0 = 0;
            while p0 < k {
                let p1 = (p0 + bs).min(k);
                for i in i0..i1 {
                    let a_row = &a[i * k + p0..i * k + p1];
                    let c_row = &mut c[i * n + j0..i * n + j1];
                    for (pp, &ap) in a_row.iter().enumerate() {
                        let p = p0 + pp;
                        let b_row = &b[p * n + j0..p * n + j1];
                        // Unrolled inner saxpy-ish update.
                        let mut j = 0;
                        let width = j1 - j0;
                        while j + 4 <= width {
                            c_row[j] += ap * b_row[j];
                            c_row[j + 1] += ap * b_row[j + 1];
                            c_row[j + 2] += ap * b_row[j + 2];
                            c_row[j + 3] += ap * b_row[j + 3];
                            j += 4;
                        }
                        while j < width {
                            c_row[j] += ap * b_row[j];
                            j += 1;
                        }
                    }
                }
                p0 = p1;
            }
            j0 = j1;
        }
        i0 = i1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_matches_naive() {
        let m = 5;
        let k = 4;
        let n = 3;
        let a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.5).collect();
        let got = gemm_rowmajor(&a, &b, m, k, n);
        let mut expect = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                expect[i * n + j] = s;
            }
        }
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-9, "{g} vs {e}");
        }
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }
}
