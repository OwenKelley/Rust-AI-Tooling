//! In-house GEMM / dot kernels using only `std` (no rayon, no BLAS crates).
//!
//! Strategy:
//! - Medium sizes: pack each NR-wide B column panel once, then 4×8 AVX2/FMA microkernel
//! - Large sizes: Goto-style A/B packing + optional `std::thread` row parallelism
//! - Dot: AVX2+FMA reduction

use std::thread;

/// Microkernel rows / cols (register block).
const MR: usize = 4;
const NR: usize = 8;

/// Cache blocks for Goto path (tunable).
const MC: usize = 128;
const NC: usize = 256;
const KC: usize = 256;

/// Parallelize only for large GEMMs; medium sizes stay serial (spawn cost dominates).
const PARALLEL_FLOPS: u64 = 8_000_000;

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

/// `C[m,n] = A[m,k] @ B[k,n]` for row-major contiguous buffers.
pub fn gemm_rowmajor(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    assert_eq!(a.len(), m * k, "A shape mismatch");
    assert_eq!(b.len(), k * n, "B shape mismatch");
    let mut c = vec![0.0; m * n];
    let isa = detect_isa();
    let flops = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);

    // B-panel packing wins through a few thousand dims; Goto for huge.
    if m <= 1536 && n <= 1536 && k <= 1536 {
        if flops >= PARALLEL_FLOPS && m >= MR * 8 {
            gemm_bpanel_parallel(a, b, &mut c, m, k, n, isa);
        } else {
            gemm_bpanel(a, b, &mut c, m, k, n, isa);
        }
    } else if flops >= PARALLEL_FLOPS && m >= MR * 4 {
        gemm_goto_parallel(a, b, &mut c, m, k, n, isa);
    } else {
        gemm_goto(a, b, &mut c, m, k, n, isa);
    }
    c
}

/// Dot product of two equal-length vectors.
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    match detect_isa() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2Fma => unsafe { dot_avx2(a, b) },
        Isa::Scalar => dot_scalar(a, b),
    }
}

fn dot_scalar(a: &[f64], b: &[f64]) -> f64 {
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f64], b: &[f64]) -> f64 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    while i + 8 <= n {
        let va0 = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb0 = _mm256_loadu_pd(b.as_ptr().add(i));
        let va1 = _mm256_loadu_pd(a.as_ptr().add(i + 4));
        let vb1 = _mm256_loadu_pd(b.as_ptr().add(i + 4));
        acc0 = _mm256_fmadd_pd(va0, vb0, acc0);
        acc1 = _mm256_fmadd_pd(va1, vb1, acc1);
        i += 8;
    }
    acc0 = _mm256_add_pd(acc0, acc1);
    while i + 4 <= n {
        let va = _mm256_loadu_pd(a.as_ptr().add(i));
        let vb = _mm256_loadu_pd(b.as_ptr().add(i));
        acc0 = _mm256_fmadd_pd(va, vb, acc0);
        i += 4;
    }
    let mut tmp = [0.0f64; 4];
    _mm256_storeu_pd(tmp.as_mut_ptr(), acc0);
    let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    while i < n {
        s += *a.get_unchecked(i) * *b.get_unchecked(i);
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
        let mut j = 0;
        while j + 4 <= n {
            y[j] += xp * brow[j];
            y[j + 1] += xp * brow[j + 1];
            y[j + 2] += xp * brow[j + 2];
            y[j + 3] += xp * brow[j + 3];
            j += 4;
        }
        while j < n {
            y[j] += xp * brow[j];
            j += 1;
        }
    }
    y
}

/// Pack `B[:, j..j+NR]` into contiguous `k × NR` (row-major panels).
fn pack_b_cols(b: &[f64], n: usize, j: usize, k: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), k * NR);
    for p in 0..k {
        let src = p * n + j;
        let dst = p * NR;
        out[dst..dst + NR].copy_from_slice(&b[src..src + NR]);
    }
}

/// Medium GEMM: one packed B panel per NR columns, then 4×8 kernels over rows.
fn gemm_bpanel(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    isa: Isa,
) {
    let mut b_panel = vec![0.0; k * NR];
    let mut j = 0;
    while j + NR <= n {
        pack_b_cols(b, n, j, k, &mut b_panel);
        gemm_bpanel_rows(a, &b_panel, c, 0, m, k, n, j, isa);
        j += NR;
    }
    // Leftover columns.
    while j < n {
        for i in 0..m {
            let mut s = c[i * n + j];
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
        j += 1;
    }
}

/// Parallel over row slabs. Each worker runs full B-panel GEMM on its rows
/// (re-packs B independently — avoids spawning inside the column loop).
fn gemm_bpanel_parallel(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    isa: Isa,
) {
    let workers = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .clamp(1, m.max(1));
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
                gemm_bpanel(a_rows, b, part, rows, k, n, isa);
            });
            row_start += rows;
        }
    });
}

fn gemm_bpanel_rows(
    a: &[f64],
    b_panel: &[f64],
    c: &mut [f64],
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
                microkernel_4x8_packed_avx2(a, b_panel, c, i, j, k, n);
            },
            Isa::Scalar => microkernel_4x8_packed_scalar(a, b_panel, c, i, j, k, n),
        }
        i += MR;
    }
    while i < i_end {
        for jj in j..j + NR {
            let mut s = c[i * n + jj];
            for p in 0..k {
                s += a[i * k + p] * b_panel[p * NR + (jj - j)];
            }
            c[i * n + jj] = s;
        }
        i += 1;
    }
}

fn microkernel_4x8_packed_scalar(
    a: &[f64],
    b_panel: &[f64],
    c: &mut [f64],
    i: usize,
    j: usize,
    k: usize,
    n: usize,
) {
    let mut acc = [[0.0f64; NR]; MR];
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
unsafe fn microkernel_4x8_packed_avx2(
    a: &[f64],
    b_panel: &[f64],
    c: &mut [f64],
    i: usize,
    j: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut c0 = [
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    ];
    let mut c1 = [
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
        _mm256_setzero_pd(),
    ];

    let a0 = a.as_ptr().add(i * k);
    let a1 = a.as_ptr().add((i + 1) * k);
    let a2 = a.as_ptr().add((i + 2) * k);
    let a3 = a.as_ptr().add((i + 3) * k);
    let bp = b_panel.as_ptr();

    // Prefetch + k-unroll by 4; keep only 8 C accumulators in registers.
    let mut p = 0;
    while p + 4 <= k {
        for q in 0..4 {
            let pp = p + q;
            let bptr = bp.add(pp * NR);
            let b_lo = _mm256_loadu_pd(bptr);
            let b_hi = _mm256_loadu_pd(bptr.add(4));
            let r0 = _mm256_broadcast_sd(&*a0.add(pp));
            let r1 = _mm256_broadcast_sd(&*a1.add(pp));
            let r2 = _mm256_broadcast_sd(&*a2.add(pp));
            let r3 = _mm256_broadcast_sd(&*a3.add(pp));
            c0[0] = _mm256_fmadd_pd(r0, b_lo, c0[0]);
            c1[0] = _mm256_fmadd_pd(r0, b_hi, c1[0]);
            c0[1] = _mm256_fmadd_pd(r1, b_lo, c0[1]);
            c1[1] = _mm256_fmadd_pd(r1, b_hi, c1[1]);
            c0[2] = _mm256_fmadd_pd(r2, b_lo, c0[2]);
            c1[2] = _mm256_fmadd_pd(r2, b_hi, c1[2]);
            c0[3] = _mm256_fmadd_pd(r3, b_lo, c0[3]);
            c1[3] = _mm256_fmadd_pd(r3, b_hi, c1[3]);
        }
        if p + 16 < k {
            _mm_prefetch(bp.add((p + 16) * NR) as *const i8, _MM_HINT_T0);
        }
        p += 4;
    }
    while p < k {
        let bptr = bp.add(p * NR);
        let b_lo = _mm256_loadu_pd(bptr);
        let b_hi = _mm256_loadu_pd(bptr.add(4));
        let r0 = _mm256_broadcast_sd(&*a0.add(p));
        let r1 = _mm256_broadcast_sd(&*a1.add(p));
        let r2 = _mm256_broadcast_sd(&*a2.add(p));
        let r3 = _mm256_broadcast_sd(&*a3.add(p));
        c0[0] = _mm256_fmadd_pd(r0, b_lo, c0[0]);
        c1[0] = _mm256_fmadd_pd(r0, b_hi, c1[0]);
        c0[1] = _mm256_fmadd_pd(r1, b_lo, c0[1]);
        c1[1] = _mm256_fmadd_pd(r1, b_hi, c1[1]);
        c0[2] = _mm256_fmadd_pd(r2, b_lo, c0[2]);
        c1[2] = _mm256_fmadd_pd(r2, b_hi, c1[2]);
        c0[3] = _mm256_fmadd_pd(r3, b_lo, c0[3]);
        c1[3] = _mm256_fmadd_pd(r3, b_hi, c1[3]);
        p += 1;
    }

    for r in 0..4 {
        let row = c.as_mut_ptr().add((i + r) * n + j);
        let cur_lo = _mm256_loadu_pd(row);
        let cur_hi = _mm256_loadu_pd(row.add(4));
        _mm256_storeu_pd(row, _mm256_add_pd(cur_lo, c0[r]));
        _mm256_storeu_pd(row.add(4), _mm256_add_pd(cur_hi, c1[r]));
    }
}

fn gemm_goto_parallel(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    isa: Isa,
) {
    let workers = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .clamp(1, m.max(1));
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
                gemm_goto(a_rows, b, part, rows, k, n, isa);
            });
            row_start += rows;
        }
    });
}

fn pack_b(b: &[f64], n: usize, k0: usize, kc: usize, j0: usize, nc: usize, out: &mut [f64]) {
    debug_assert_eq!(out.len(), kc * nc);
    for p in 0..kc {
        let src = (k0 + p) * n + j0;
        let dst = p * nc;
        out[dst..dst + nc].copy_from_slice(&b[src..src + nc]);
    }
}

fn pack_a(a: &[f64], k: usize, i0: usize, mc: usize, k0: usize, kc: usize, out: &mut [f64]) {
    let mut o = 0;
    let mut ii = 0;
    while ii < mc {
        let rows = MR.min(mc - ii);
        for p in 0..kc {
            let col = k0 + p;
            for r in 0..rows {
                out[o + r] = a[(i0 + ii + r) * k + col];
            }
            for r in rows..MR {
                out[o + r] = 0.0;
            }
            o += MR;
        }
        ii += MR;
    }
}

fn gemm_goto(
    a: &[f64],
    b: &[f64],
    c: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
    isa: Isa,
) {
    let mut a_pack = vec![0.0; MC.div_ceil(MR) * MR * KC];
    let mut b_pack = vec![0.0; KC * NC];

    let mut j0 = 0;
    while j0 < n {
        let nc = NC.min(n - j0);
        let mut p0 = 0;
        while p0 < k {
            let kc = KC.min(k - p0);
            pack_b(b, n, p0, kc, j0, nc, &mut b_pack[..kc * nc]);

            let mut i0 = 0;
            while i0 < m {
                let mc = MC.min(m - i0);
                pack_a(a, k, i0, mc, p0, kc, &mut a_pack);

                let mut ir = 0;
                while ir < mc {
                    let mr = MR.min(mc - ir);
                    let a_panel =
                        &a_pack[(ir / MR) * (MR * kc)..(ir / MR + 1) * (MR * kc)];

                    let mut jr = 0;
                    while jr < nc {
                        let nr = NR.min(nc - jr);
                        let c_ptr = (i0 + ir) * n + (j0 + jr);
                        if mr == MR && nr == NR {
                            match isa {
                                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                                Isa::Avx2Fma => unsafe {
                                    microkernel_4x8_goto_avx2(
                                        a_panel,
                                        &b_pack,
                                        kc,
                                        nc,
                                        jr,
                                        &mut c[c_ptr..],
                                        n,
                                    );
                                },
                                Isa::Scalar => {
                                    microkernel_4x8_goto_scalar(
                                        a_panel,
                                        &b_pack,
                                        kc,
                                        nc,
                                        jr,
                                        &mut c[c_ptr..],
                                        n,
                                    );
                                }
                            }
                        } else {
                            microkernel_edge(
                                a_panel,
                                &b_pack,
                                kc,
                                nc,
                                jr,
                                mr,
                                nr,
                                &mut c[c_ptr..],
                                n,
                            );
                        }
                        jr += NR;
                    }
                    ir += MR;
                }
                i0 += MC;
            }
            p0 += KC;
        }
        j0 += NC;
    }
}

fn microkernel_4x8_goto_scalar(
    a_panel: &[f64],
    b_pack: &[f64],
    kc: usize,
    b_ld: usize,
    jr: usize,
    c: &mut [f64],
    ldc: usize,
) {
    let mut acc = [[0.0f64; NR]; MR];
    for p in 0..kc {
        let a0 = a_panel[p * MR];
        let a1 = a_panel[p * MR + 1];
        let a2 = a_panel[p * MR + 2];
        let a3 = a_panel[p * MR + 3];
        let base = p * b_ld + jr;
        let bv = [
            b_pack[base],
            b_pack[base + 1],
            b_pack[base + 2],
            b_pack[base + 3],
            b_pack[base + 4],
            b_pack[base + 5],
            b_pack[base + 6],
            b_pack[base + 7],
        ];
        for t in 0..NR {
            acc[0][t] += a0 * bv[t];
            acc[1][t] += a1 * bv[t];
            acc[2][t] += a2 * bv[t];
            acc[3][t] += a3 * bv[t];
        }
    }
    for r in 0..MR {
        for t in 0..NR {
            c[r * ldc + t] += acc[r][t];
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_4x8_goto_avx2(
    a_panel: &[f64],
    b_pack: &[f64],
    kc: usize,
    b_ld: usize,
    jr: usize,
    c: &mut [f64],
    ldc: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut c0 = [_mm256_setzero_pd(); 4];
    let mut c1 = [_mm256_setzero_pd(); 4];

    for p in 0..kc {
        let bp = b_pack.as_ptr().add(p * b_ld + jr);
        let b_lo = _mm256_loadu_pd(bp);
        let b_hi = _mm256_loadu_pd(bp.add(4));
        let ap = a_panel.as_ptr().add(p * MR);
        for r in 0..4 {
            let ar = _mm256_broadcast_sd(&*ap.add(r));
            c0[r] = _mm256_fmadd_pd(ar, b_lo, c0[r]);
            c1[r] = _mm256_fmadd_pd(ar, b_hi, c1[r]);
        }
    }

    for r in 0..4 {
        let row = c.as_mut_ptr().add(r * ldc);
        let cur_lo = _mm256_loadu_pd(row);
        let cur_hi = _mm256_loadu_pd(row.add(4));
        _mm256_storeu_pd(row, _mm256_add_pd(cur_lo, c0[r]));
        _mm256_storeu_pd(row.add(4), _mm256_add_pd(cur_hi, c1[r]));
    }
}

fn microkernel_edge(
    a_panel: &[f64],
    b_pack: &[f64],
    kc: usize,
    b_ld: usize,
    jr: usize,
    mr: usize,
    nr: usize,
    c: &mut [f64],
    ldc: usize,
) {
    for r in 0..mr {
        for jj in 0..nr {
            let mut s = 0.0;
            for p in 0..kc {
                s += a_panel[p * MR + r] * b_pack[p * b_ld + jr + jj];
            }
            c[r * ldc + jj] += s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn gemm_matches_naive_small() {
        let m = 5;
        let k = 4;
        let n = 3;
        let a: Vec<f64> = (0..m * k).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.5).collect();
        let got = gemm_rowmajor(&a, &b, m, k, n);
        let expect = naive(&a, &b, m, k, n);
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-9, "{g} vs {e}");
        }
    }

    #[test]
    fn gemm_matches_naive_blocked_sizes() {
        for &(m, k, n) in &[(17, 19, 23), (64, 64, 64), (128, 96, 80), (255, 257, 128)] {
            let a: Vec<f64> = (0..m * k)
                .map(|i| ((i * 37) % 97) as f64 * 0.01)
                .collect();
            let b: Vec<f64> = (0..k * n)
                .map(|i| ((i * 53) % 89) as f64 * 0.01)
                .collect();
            let got = gemm_rowmajor(&a, &b, m, k, n);
            let expect = naive(&a, &b, m, k, n);
            let mut max_err: f64 = 0.0;
            for (g, e) in got.iter().zip(expect.iter()) {
                max_err = max_err.max((g - e).abs());
            }
            assert!(
                max_err < 1e-8 * (k as f64).sqrt(),
                "m={m} k={k} n={n} max_err={max_err}"
            );
        }
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot_f64(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }
}
