//! Contiguous row-major f32 GEMM (`C = A @ B`), local/`std` only.
//!
//! Packs NR-wide B column panels + 4×8 microkernel (AVX2/FMA when available).
//! Parallel row split only for very large GEMMs (spawn cost dominates otherwise).

use std::thread;

const MR: usize = 4;
const NR: usize = 8;
/// Prefer serial AVX2 until GEMMs are large enough that spawn pays off.
const PARALLEL_FLOPS: u64 = 64_000_000;

#[derive(Clone, Copy)]
enum Isa {
    Scalar,
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2Fma,
}

fn detect_isa() -> Isa {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Isa::Avx2Fma;
        }
    }
    Isa::Scalar
}

/// `C[m,n] = A[m,k] @ B[k,n]` for contiguous row-major buffers.
pub fn gemm_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), k * n, "B shape mismatch");
    let mut c = vec![0.0f32; m * n];
    let isa = detect_isa();
    let flops = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
    if flops >= PARALLEL_FLOPS && m >= MR * 8 {
        gemm_bpanel_parallel(a, b, &mut c, m, k, n, isa);
    } else {
        gemm_bpanel(a, b, &mut c, m, k, n, isa);
    }
    c
}

fn gemm_bpanel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, isa: Isa) {
    let mut b_panel = vec![0.0f32; k * NR];
    let mut j = 0;
    while j + NR <= n {
        for p in 0..k {
            let src = &b[p * n + j..p * n + j + NR];
            b_panel[p * NR..(p + 1) * NR].copy_from_slice(src);
        }
        gemm_bpanel_rows(a, &b_panel, c, 0, m, k, n, j, isa);
        j += NR;
    }
    if j < n {
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p];
                for jj in j..n {
                    c[i * n + jj] += av * b[p * n + jj];
                }
            }
        }
    }
}

fn gemm_bpanel_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    isa: Isa,
) {
    let workers = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(m / MR)
        .max(1);
    if workers <= 1 {
        gemm_bpanel(a, b, c, m, k, n, isa);
        return;
    }
    let chunk = ((m + workers - 1) / workers).max(MR);
    thread::scope(|scope| {
        let mut rest = &mut c[..];
        let mut row0 = 0usize;
        for _ in 0..workers {
            if row0 >= m {
                break;
            }
            let rows = chunk.min(m - row0);
            let (part, next) = rest.split_at_mut(rows * n);
            let a_rows = &a[row0 * k..(row0 + rows) * k];
            scope.spawn(move || {
                gemm_bpanel(a_rows, b, part, rows, k, n, isa);
            });
            rest = next;
            row0 += rows;
        }
    });
}

fn gemm_bpanel_rows(
    a: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    i0: usize,
    m_rows: usize,
    k: usize,
    n: usize,
    j: usize,
    isa: Isa,
) {
    let mut i = i0;
    let i_end = i0 + m_rows;
    while i + MR <= i_end {
        match isa {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Isa::Avx2Fma => unsafe {
                microkernel_4x8_avx2(a, b_panel, c, i, j, k, n);
            },
            Isa::Scalar => microkernel_4x8_scalar(a, b_panel, c, i, j, k, n),
        }
        i += MR;
    }
    while i < i_end {
        for jj in 0..NR {
            let mut s = c[i * n + j + jj];
            for p in 0..k {
                s += a[i * k + p] * b_panel[p * NR + jj];
            }
            c[i * n + j + jj] = s;
        }
        i += 1;
    }
}

fn microkernel_4x8_scalar(
    a: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    i: usize,
    j: usize,
    k: usize,
    n: usize,
) {
    let mut acc = [[0.0f32; NR]; MR];
    for p in 0..k {
        let base = p * NR;
        let b0 = b_panel[base];
        let b1 = b_panel[base + 1];
        let b2 = b_panel[base + 2];
        let b3 = b_panel[base + 3];
        let b4 = b_panel[base + 4];
        let b5 = b_panel[base + 5];
        let b6 = b_panel[base + 6];
        let b7 = b_panel[base + 7];
        for r in 0..MR {
            let ar = a[(i + r) * k + p];
            acc[r][0] += ar * b0;
            acc[r][1] += ar * b1;
            acc[r][2] += ar * b2;
            acc[r][3] += ar * b3;
            acc[r][4] += ar * b4;
            acc[r][5] += ar * b5;
            acc[r][6] += ar * b6;
            acc[r][7] += ar * b7;
        }
    }
    for r in 0..MR {
        let row = &mut c[(i + r) * n + j..(i + r) * n + j + NR];
        for t in 0..NR {
            row[t] += acc[r][t];
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x8_avx2(
    a: &[f32],
    b_panel: &[f32],
    c: &mut [f32],
    i: usize,
    j: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut c0 = _mm256_setzero_ps();
    let mut c1 = _mm256_setzero_ps();
    let mut c2 = _mm256_setzero_ps();
    let mut c3 = _mm256_setzero_ps();

    let a0 = a.as_ptr().add(i * k);
    let a1 = a.as_ptr().add((i + 1) * k);
    let a2 = a.as_ptr().add((i + 2) * k);
    let a3 = a.as_ptr().add((i + 3) * k);
    let bp = b_panel.as_ptr();

    for p in 0..k {
        let bv = _mm256_loadu_ps(bp.add(p * NR));
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a0.add(p)), bv, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a1.add(p)), bv, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a2.add(p)), bv, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*a3.add(p)), bv, c3);
    }

    let cp = c.as_mut_ptr();
    _mm256_storeu_ps(cp.add(i * n + j), _mm256_add_ps(_mm256_loadu_ps(cp.add(i * n + j)), c0));
    _mm256_storeu_ps(
        cp.add((i + 1) * n + j),
        _mm256_add_ps(_mm256_loadu_ps(cp.add((i + 1) * n + j)), c1),
    );
    _mm256_storeu_ps(
        cp.add((i + 2) * n + j),
        _mm256_add_ps(_mm256_loadu_ps(cp.add((i + 2) * n + j)), c2),
    );
    _mm256_storeu_ps(
        cp.add((i + 3) * n + j),
        _mm256_add_ps(_mm256_loadu_ps(cp.add((i + 3) * n + j)), c3),
    );
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
}
