//! Contiguous row-major f32 GEMM via `matrixmultiply` (portable default).
//!
//! With the `parallel` feature (default), mid/large GEMMs use a Rayon thread pool
//! instead of spawning fresh OS threads per call.

use matrixmultiply::sgemm;

const MR: usize = 4;
const PARALLEL_FLOPS: u64 = 4_000_000;
const PARALLEL_MAX_WORKERS: usize = 8;

fn parallel_workers(m: usize, flops: u64) -> usize {
    if flops < PARALLEL_FLOPS || m < MR * 4 {
        return 1;
    }
    let hw = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    hw.min(m / MR).min(PARALLEL_MAX_WORKERS).max(1)
}

/// `C[m,n] = A[m,k] @ B[k,n]` for contiguous row-major buffers.
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), k * n, "B shape mismatch");
    let mut c = crate::bufpool::take_f32(m * n);
    gemm_nn_into(a, b, &mut c, m, k, n, 0.0);
    c
}

/// `C[m,n] = A[m,k] @ B^T` where `B` is contiguous `[n,k]` (Linear: `X @ W^T`).
pub fn gemm_f32_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), n * k, "B shape mismatch");
    let mut c = crate::bufpool::take_f32(m * n);
    gemm_nt_into(a, b, &mut c, m, k, n, 0.0);
    c
}

/// `C[m,n] = A^T @ B` where `A` is contiguous `[k,m]` and `B` is `[k,n]`.
pub fn gemm_f32_tn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), k * m, "A shape mismatch");
    assert_eq!(b.len(), k * n, "B shape mismatch");
    let mut c = crate::bufpool::take_f32(m * n);
    gemm_tn_into(a, b, &mut c, m, k, n, 0.0);
    c
}

/// `C = A @ B + beta * C` (row-major).
pub fn gemm_nn_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    let flops = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    let workers = parallel_workers(m, flops);
    if workers <= 1 {
        sgemm_nn_beta(a, b, c, m, k, n, beta);
    } else {
        sgemm_nn_parallel(a, b, c, m, k, n, workers, beta);
    }
}

/// `C = A @ B^T + beta * C` (`B` stored `[n,k]`).
pub fn gemm_nt_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    assert_eq!(c.len(), m * n);
    let flops = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    let workers = parallel_workers(m, flops);
    if workers <= 1 {
        sgemm_nt_beta(a, b, c, m, k, n, beta);
    } else {
        sgemm_nt_parallel(a, b, c, m, k, n, workers, beta);
    }
}

/// `C = A^T @ B + beta * C` (`A` stored `[k,m]`).
pub fn gemm_tn_into(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    assert_eq!(a.len(), k * m);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    let flops = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    // Split along output rows (`m`); TN packs A as `[k,m]` so row tiles are contiguous in C.
    let workers = parallel_workers(m, flops);
    if workers <= 1 {
        sgemm_tn_beta(a, b, c, m, k, n, beta);
    } else {
        sgemm_tn_parallel(a, b, c, m, k, n, workers, beta);
    }
}

/// Recycle a GEMM output buffer when the caller no longer needs it.
pub fn recycle_gemm_buf(buf: Vec<f32>) {
    crate::bufpool::recycle_f32(buf);
}

#[inline]
fn sgemm_nn_beta(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
fn sgemm_nt_beta(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[inline]
fn sgemm_tn_beta(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, beta: f32) {
    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            n as isize,
            1,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

fn sgemm_nn_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    workers: usize,
    beta: f32,
) {
    let chunk = ((m + workers - 1) / workers).max(MR);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let mut ranges = Vec::with_capacity(workers);
        let mut row0 = 0usize;
        while row0 < m {
            let rows = chunk.min(m - row0);
            ranges.push((row0, rows));
            row0 += rows;
        }
        let c_addr = c.as_mut_ptr() as usize;
        ranges.into_par_iter().for_each(|(row0, rows)| {
            let part = unsafe {
                std::slice::from_raw_parts_mut((c_addr as *mut f32).add(row0 * n), rows * n)
            };
            let a_rows = &a[row0 * k..(row0 + rows) * k];
            sgemm_nn_beta(a_rows, b, part, rows, k, n, beta);
        });
        return;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = beta;
        std::thread::scope(|scope| {
            let mut rest = &mut c[..];
            let mut row0 = 0usize;
            for _ in 0..workers {
                if row0 >= m {
                    break;
                }
                let rows = chunk.min(m - row0);
                let (part, next) = rest.split_at_mut(rows * n);
                let a_rows = &a[row0 * k..(row0 + rows) * k];
                scope.spawn(move || sgemm_nn_beta(a_rows, b, part, rows, k, n, beta));
                rest = next;
                row0 += rows;
            }
        });
    }
}

fn sgemm_nt_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    workers: usize,
    beta: f32,
) {
    let chunk = ((m + workers - 1) / workers).max(MR);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let mut ranges = Vec::with_capacity(workers);
        let mut row0 = 0usize;
        while row0 < m {
            let rows = chunk.min(m - row0);
            ranges.push((row0, rows));
            row0 += rows;
        }
        let c_addr = c.as_mut_ptr() as usize;
        ranges.into_par_iter().for_each(|(row0, rows)| {
            let part = unsafe {
                std::slice::from_raw_parts_mut((c_addr as *mut f32).add(row0 * n), rows * n)
            };
            let a_rows = &a[row0 * k..(row0 + rows) * k];
            sgemm_nt_beta(a_rows, b, part, rows, k, n, beta);
        });
        return;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = beta;
        std::thread::scope(|scope| {
            let mut rest = &mut c[..];
            let mut row0 = 0usize;
            for _ in 0..workers {
                if row0 >= m {
                    break;
                }
                let rows = chunk.min(m - row0);
                let (part, next) = rest.split_at_mut(rows * n);
                let a_rows = &a[row0 * k..(row0 + rows) * k];
                scope.spawn(move || sgemm_nt_beta(a_rows, b, part, rows, k, n, beta));
                rest = next;
                row0 += rows;
            }
        });
    }
}

fn sgemm_tn_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    workers: usize,
    beta: f32,
) {
    // A is `[k,m]` column-major-ish for rows of A^T; each output row `i` uses A[:, i].
    let chunk = ((m + workers - 1) / workers).max(MR);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        let mut ranges = Vec::with_capacity(workers);
        let mut row0 = 0usize;
        while row0 < m {
            let rows = chunk.min(m - row0);
            ranges.push((row0, rows));
            row0 += rows;
        }
        let c_addr = c.as_mut_ptr() as usize;
        let a_addr = a.as_ptr() as usize;
        ranges.into_par_iter().for_each(|(row0, rows)| {
            let part = unsafe {
                std::slice::from_raw_parts_mut((c_addr as *mut f32).add(row0 * n), rows * n)
            };
            unsafe {
                sgemm(
                    rows,
                    k,
                    n,
                    1.0,
                    (a_addr as *const f32).add(row0),
                    1,
                    m as isize,
                    b.as_ptr(),
                    n as isize,
                    1,
                    beta,
                    part.as_mut_ptr(),
                    n as isize,
                    1,
                );
            }
        });
        return;
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = (workers, chunk, beta);
        sgemm_tn_beta(a, b, c, m, k, n, beta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
        c
    }

    fn naive_nt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[j * k + p];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    fn naive_tn(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[p * m + i];
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn gemm_matches_naive() {
        for &(m, k, n) in &[(3, 4, 5), (16, 16, 16), (64, 32, 48), (65, 17, 19)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02 - 0.3).collect();
            let got = gemm_f32(&a, &b, m, k, n);
            let exp = naive(&a, &b, m, k, n);
            for (g, e) in got.iter().zip(exp.iter()) {
                let tol = 1e-3 * e.abs().max(1.0);
                assert!((g - e).abs() < tol, "{g} vs {e} shape=({m},{k},{n})");
            }
        }
    }

    #[test]
    fn gemm_nt_matches_naive() {
        for &(m, k, n) in &[(3, 4, 5), (16, 16, 16), (128, 64, 32), (65, 17, 19)] {
            let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let b: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.02 - 0.3).collect();
            let got = gemm_f32_nt(&a, &b, m, k, n);
            let exp = naive_nt(&a, &b, m, k, n);
            for (g, e) in got.iter().zip(exp.iter()) {
                let tol = 1e-3 * e.abs().max(1.0);
                assert!((g - e).abs() < tol, "{g} vs {e} nt=({m},{k},{n})");
            }
        }
    }

    #[test]
    fn gemm_tn_matches_naive() {
        for &(m, k, n) in &[(3, 4, 5), (16, 16, 16), (128, 64, 32), (65, 17, 19)] {
            let a: Vec<f32> = (0..k * m).map(|i| (i as f32) * 0.01 - 0.5).collect();
            let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02 - 0.3).collect();
            let got = gemm_f32_tn(&a, &b, m, k, n);
            let exp = naive_tn(&a, &b, m, k, n);
            for (g, e) in got.iter().zip(exp.iter()) {
                let tol = 1e-3 * e.abs().max(1.0);
                assert!((g - e).abs() < tol, "{g} vs {e} tn=({m},{k},{n})");
            }
        }
    }

    #[test]
    fn gemm_acc_beta() {
        let m = 8usize;
        let k = 8usize;
        let n = 8usize;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.02).collect();
        let base = gemm_f32(&a, &b, m, k, n);
        let mut c = base.clone();
        gemm_nn_into(&a, &b, &mut c, m, k, n, 1.0);
        for i in 0..c.len() {
            assert!((c[i] - 2.0 * base[i]).abs() < 1e-4);
        }
    }
}
