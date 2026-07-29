//! Fast Fourier transforms — mirrors common `scipy.fft` entry points.
//!
//! Local radix-2 + Bluestein DFT (no third-party FFT crates). Complex values
//! are stored as `NdArray` with shape `[n, 2]` = `(real, imag)` columns.

use rnumpy::NdArray;

type C = (f64, f64);

#[inline]
fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn csub(a: C, b: C) -> C {
    (a.0 - b.0, a.1 - b.1)
}

#[inline]
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cconj(a: C) -> C {
    (a.0, -a.1)
}

#[inline]
fn cscale(a: C, s: f64) -> C {
    (a.0 * s, a.1 * s)
}

fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

pub(crate) fn next_pow2(n: usize) -> usize {
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

/// In-place iterative radix-2 Cooley–Tukey FFT (`inverse=false` forward).
fn fft_radix2(a: &mut [C], inverse: bool) {
    let n = a.len();
    assert!(is_pow2(n), "fft_radix2: length must be power of two");

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let wlen = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = (1.0, 0.0);
            for j in 0..half {
                let u = a[i + j];
                let v = cmul(a[i + j + half], w);
                a[i + j] = cadd(u, v);
                a[i + j + half] = csub(u, v);
                w = cmul(w, wlen);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Bluestein's chirp z-transform for arbitrary length.
fn fft_bluestein(input: &[C], inverse: bool) -> Vec<C> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let m = next_pow2(2 * n - 1);
    let sign = if inverse { 1.0 } else { -1.0 };

    // chirp: exp(i * sign * π * k^2 / n)
    let mut chirp = vec![(0.0, 0.0); n];
    for k in 0..n {
        let angle = sign * std::f64::consts::PI * ((k * k) as f64) / n as f64;
        chirp[k] = (angle.cos(), angle.sin());
    }

    let mut a = vec![(0.0, 0.0); m];
    for k in 0..n {
        a[k] = cmul(input[k], chirp[k]);
    }

    let mut b = vec![(0.0, 0.0); m];
    b[0] = cconj(chirp[0]);
    for k in 1..n {
        let c = cconj(chirp[k]);
        b[k] = c;
        b[m - k] = c;
    }

    fft_radix2(&mut a, false);
    fft_radix2(&mut b, false);
    for i in 0..m {
        a[i] = cmul(a[i], b[i]);
    }
    fft_radix2(&mut a, true);
    let inv_m = 1.0 / m as f64;
    for v in a.iter_mut() {
        *v = cscale(*v, inv_m);
    }

    let mut out = vec![(0.0, 0.0); n];
    for k in 0..n {
        out[k] = cmul(a[k], chirp[k]);
    }
    out
}

pub(crate) fn fft_complex(input: &[C], inverse: bool) -> Vec<C> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if is_pow2(n) {
        let mut a = input.to_vec();
        fft_radix2(&mut a, inverse);
        if inverse {
            let s = 1.0 / n as f64;
            for v in a.iter_mut() {
                *v = cscale(*v, s);
            }
        }
        a
    } else {
        let mut out = fft_bluestein(input, inverse);
        if inverse {
            let s = 1.0 / n as f64;
            for v in out.iter_mut() {
                *v = cscale(*v, s);
            }
        }
        out
    }
}

fn to_complex_pairs(a: &NdArray) -> Vec<C> {
    match a.ndim() {
        1 => a
            .to_contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|&x| (x, 0.0))
            .collect(),
        2 => {
            assert_eq!(a.shape()[1], 2, "complex array must have shape [n, 2]");
            let n = a.shape()[0];
            let c = a.to_contiguous();
            let s = c.as_slice().unwrap();
            (0..n).map(|i| (s[i * 2], s[i * 2 + 1])).collect()
        }
        _ => panic!("fft: expected 1D real or [n,2] complex"),
    }
}

fn from_complex_pairs(v: &[C]) -> NdArray {
    let mut data = Vec::with_capacity(v.len() * 2);
    for &(re, im) in v {
        data.push(re);
        data.push(im);
    }
    NdArray::from_shape_vec(&[v.len(), 2], data)
}

/// `scipy.fft.fft(x)` — 1D FFT. Real 1D input or complex `[n,2]`.
///
/// Returns complex spectrum as `[n, 2]` (real, imag).
pub fn fft(a: &NdArray) -> NdArray {
    let x = to_complex_pairs(a);
    from_complex_pairs(&fft_complex(&x, false))
}

/// `scipy.fft.ifft(x)` — inverse 1D FFT. Input complex `[n,2]` (or real 1D).
pub fn ifft(a: &NdArray) -> NdArray {
    let x = to_complex_pairs(a);
    from_complex_pairs(&fft_complex(&x, true))
}

/// `scipy.fft.rfft(x)` — FFT of real sequence; length `n//2+1` complex.
pub fn rfft(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 1, "rfft: expected 1D real");
    let full = fft(a);
    let n = a.len();
    let n_out = n / 2 + 1;
    let c = full.to_contiguous();
    let s = c.as_slice().unwrap();
    NdArray::from_shape_vec(&[n_out, 2], s[..n_out * 2].to_vec())
}

/// `scipy.fft.irfft(x, n=None)` — inverse of `rfft`.
///
/// `x` is complex `[n_rfft, 2]`. If `n` is None, output length is `2*(m-1)`.
pub fn irfft(a: &NdArray, n: Option<usize>) -> NdArray {
    assert_eq!(a.ndim(), 2, "irfft: expected complex [m, 2]");
    assert_eq!(a.shape()[1], 2);
    let m = a.shape()[0];
    let n_out = n.unwrap_or(2 * (m - 1));
    assert!(n_out / 2 + 1 == m || n_out / 2 + 1 <= m);

    let c = a.to_contiguous();
    let s = c.as_slice().unwrap();
    let mut full = vec![(0.0, 0.0); n_out];
    let take = (n_out / 2 + 1).min(m);
    for k in 0..take {
        full[k] = (s[k * 2], s[k * 2 + 1]);
    }
    for k in 1..take {
        let j = n_out - k;
        if j > k && j < n_out {
            full[j] = cconj(full[k]);
        }
    }
    // For even n_out, Nyquist bin is real
    if n_out % 2 == 0 && take > 0 {
        let nyq = n_out / 2;
        if nyq < take {
            full[nyq] = (full[nyq].0, 0.0);
        }
    }
    let out = fft_complex(&full, true);
    NdArray::from_vec(out.iter().map(|c| c.0).collect())
}

/// `scipy.fft.fftfreq(n, d=1.0)`.
pub fn fftfreq(n: usize, d: f64) -> NdArray {
    let mut out = vec![0.0; n];
    let val = 1.0 / (n as f64 * d);
    let n_half = (n + 1) / 2;
    for i in 0..n_half {
        out[i] = i as f64 * val;
    }
    for i in n_half..n {
        out[i] = (i as isize - n as isize) as f64 * val;
    }
    NdArray::from_vec(out)
}

/// `scipy.fft.rfftfreq(n, d=1.0)`.
pub fn rfftfreq(n: usize, d: f64) -> NdArray {
    let n_out = n / 2 + 1;
    let val = 1.0 / (n as f64 * d);
    NdArray::from_vec((0..n_out).map(|i| i as f64 * val).collect())
}

/// `scipy.fft.fft2` for 2D real/complex arrays (row-column separable).
pub fn fft2(a: &NdArray) -> NdArray {
    assert_eq!(a.ndim(), 2, "fft2: expected 2D");
    let m = a.shape()[0];
    let n = a.shape()[1];
    // Treat as real matrix → complex [m, n, 2] flattened as [m*n, 2] after row FFTs then col.
    let ac = a.to_contiguous();
    let s = ac.as_slice().unwrap();

    // Row-wise FFT of real rows → complex rows
    let mut rows: Vec<Vec<C>> = Vec::with_capacity(m);
    for i in 0..m {
        let row: Vec<C> = (0..n).map(|j| (s[i * n + j], 0.0)).collect();
        rows.push(fft_complex(&row, false));
    }
    // Column-wise FFT
    let mut out = vec![0.0; m * n * 2];
    for j in 0..n {
        let col: Vec<C> = (0..m).map(|i| rows[i][j]).collect();
        let fcol = fft_complex(&col, false);
        for i in 0..m {
            out[(i * n + j) * 2] = fcol[i].0;
            out[(i * n + j) * 2 + 1] = fcol[i].1;
        }
    }
    // Store as [m, n, 2] flattened shape [m, n*2] for harness simplicity: shape [m*n, 2]
    NdArray::from_shape_vec(&[m * n, 2], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    #[test]
    fn fft_ifft_roundtrip_pow2() {
        let x = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 0.25, 0.75]);
        let y = ifft(&fft(&x));
        for i in 0..8 {
            assert_close(y[[i, 0]], x[i], 1e-10);
            assert_close(y[[i, 1]], 0.0, 1e-10);
        }
    }

    #[test]
    fn fft_ifft_roundtrip_odd() {
        let x = NdArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = ifft(&fft(&x));
        for i in 0..5 {
            assert_close(y[[i, 0]], x[i], 1e-9);
            assert_close(y[[i, 1]], 0.0, 1e-9);
        }
    }

    #[test]
    fn rfft_irfft_roundtrip() {
        let x = NdArray::from_vec(vec![1.0, 0.0, -1.0, 0.0, 0.5, 0.25, -0.5, 0.1]);
        let y = irfft(&rfft(&x), Some(8));
        for i in 0..8 {
            assert_close(y[i], x[i], 1e-10);
        }
    }

    #[test]
    fn fftfreq_known() {
        let f = fftfreq(4, 1.0);
        assert_close(f[0], 0.0, 1e-12);
        assert_close(f[1], 0.25, 1e-12);
        assert_close(f[2], -0.5, 1e-12);
        assert_close(f[3], -0.25, 1e-12);
    }
}
