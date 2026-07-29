//! Signal processing — mirrors common `scipy.signal` entry points.
//!
//! Local implementations only (direct convolution; windows; detrend).

use crate::fft::{fft_complex, next_pow2};
use rnumpy::NdArray;

/// Convolution mode matching `numpy/scipy` (`full`, `same`, `valid`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvolveMode {
    Full,
    Same,
    Valid,
}

fn parse_mode(mode: &str) -> ConvolveMode {
    match mode {
        "full" => ConvolveMode::Full,
        "same" => ConvolveMode::Same,
        "valid" => ConvolveMode::Valid,
        other => panic!("convolve: unknown mode '{other}'"),
    }
}

/// Direct 1D convolution — `scipy.signal.convolve(a, v, mode=...)`.
pub fn convolve(a: &NdArray, v: &NdArray, mode: &str) -> NdArray {
    assert_eq!(a.ndim(), 1, "convolve: a must be 1D");
    assert_eq!(v.ndim(), 1, "convolve: v must be 1D");
    let aa = a.to_contiguous();
    let vv = v.to_contiguous();
    let x = aa.as_slice().unwrap();
    let h = vv.as_slice().unwrap();
    let n = x.len();
    let m = h.len();
    assert!(n > 0 && m > 0);

    let full_len = n + m - 1;
    let mut full = vec![0.0; full_len];
    for i in 0..n {
        for j in 0..m {
            full[i + j] += x[i] * h[j];
        }
    }
    slice_mode(&full, n, m, parse_mode(mode))
}

/// FFT-based 1D convolution (`scipy.signal.fftconvolve`) for real vectors.
///
/// Pads to the next power of two so the radix-2 path is used (avoids Bluestein),
/// then takes the leading `n+m-1` samples of the circular convolution.
pub fn fftconvolve(a: &NdArray, v: &NdArray, mode: &str) -> NdArray {
    assert_eq!(a.ndim(), 1);
    assert_eq!(v.ndim(), 1);
    let aa = a.to_contiguous();
    let vv = v.to_contiguous();
    let x = aa.as_slice().unwrap();
    let h = vv.as_slice().unwrap();
    let n = x.len();
    let m = h.len();
    let full_len = n + m - 1;
    let nfft = next_pow2(full_len);

    let mut xp = vec![(0.0, 0.0); nfft];
    let mut hp = vec![(0.0, 0.0); nfft];
    for i in 0..n {
        xp[i] = (x[i], 0.0);
    }
    for i in 0..m {
        hp[i] = (h[i], 0.0);
    }
    let mut xf = fft_complex(&xp, false);
    let hf = fft_complex(&hp, false);
    for k in 0..nfft {
        let (ar, ai) = xf[k];
        let (br, bi) = hf[k];
        xf[k] = (ar * br - ai * bi, ar * bi + ai * br);
    }
    let y = fft_complex(&xf, true);
    let full: Vec<f64> = (0..full_len).map(|i| y[i].0).collect();
    slice_mode(&full, n, m, parse_mode(mode))
}

fn slice_mode(full: &[f64], n: usize, m: usize, mode: ConvolveMode) -> NdArray {
    let full_len = full.len();
    match mode {
        ConvolveMode::Full => NdArray::from_vec(full.to_vec()),
        ConvolveMode::Same => {
            let out_len = n;
            let start = (full_len - out_len) / 2;
            NdArray::from_vec(full[start..start + out_len].to_vec())
        }
        ConvolveMode::Valid => {
            let out_len = n.abs_diff(m) + 1;
            let start = n.min(m) - 1;
            NdArray::from_vec(full[start..start + out_len].to_vec())
        }
    }
}

/// 1D correlation — `scipy.signal.correlate(a, v, mode=...)`.
///
/// For real inputs this is `convolve(a, reverse(v))`.
pub fn correlate(a: &NdArray, v: &NdArray, mode: &str) -> NdArray {
    assert_eq!(a.ndim(), 1);
    assert_eq!(v.ndim(), 1);
    let vv = v.to_contiguous();
    let h = vv.as_slice().unwrap();
    let rev = NdArray::from_vec(h.iter().rev().copied().collect());
    convolve(a, &rev, mode)
}

/// `scipy.signal.windows.hann(M, sym=True)`.
pub fn hann(m: usize, sym: bool) -> NdArray {
    if m == 0 {
        return NdArray::from_vec(vec![]);
    }
    if m == 1 {
        return NdArray::from_vec(vec![1.0]);
    }
    let (length, scale) = if sym {
        (m, m - 1)
    } else {
        (m + 1, m)
    };
    let mut w = Vec::with_capacity(m);
    for i in 0..length.min(m + if sym { 0 } else { 1 }) {
        if !sym && i == m {
            break;
        }
        let t = i as f64;
        w.push(0.5 - 0.5 * (2.0 * std::f64::consts::PI * t / scale as f64).cos());
    }
    if !sym {
        w.truncate(m);
    }
    NdArray::from_vec(w)
}

/// `scipy.signal.windows.hamming(M, sym=True)`.
pub fn hamming(m: usize, sym: bool) -> NdArray {
    if m == 0 {
        return NdArray::from_vec(vec![]);
    }
    if m == 1 {
        return NdArray::from_vec(vec![1.0]);
    }
    let scale = if sym { m - 1 } else { m };
    let mut w = Vec::with_capacity(m);
    for i in 0..m {
        let t = i as f64;
        w.push(0.54 - 0.46 * (2.0 * std::f64::consts::PI * t / scale as f64).cos());
    }
    NdArray::from_vec(w)
}

/// `scipy.signal.windows.blackman(M, sym=True)`.
pub fn blackman(m: usize, sym: bool) -> NdArray {
    if m == 0 {
        return NdArray::from_vec(vec![]);
    }
    if m == 1 {
        return NdArray::from_vec(vec![1.0]);
    }
    let scale = if sym { m - 1 } else { m };
    let mut w = Vec::with_capacity(m);
    for i in 0..m {
        let t = i as f64;
        let ang = 2.0 * std::f64::consts::PI * t / scale as f64;
        w.push(0.42 - 0.5 * ang.cos() + 0.08 * (2.0 * ang).cos());
    }
    NdArray::from_vec(w)
}

/// `scipy.signal.detrend(data, type='linear'|'constant')` for 1D.
pub fn detrend(a: &NdArray, kind: &str) -> NdArray {
    assert_eq!(a.ndim(), 1, "detrend: expected 1D");
    let c = a.to_contiguous();
    let s = c.as_slice().unwrap();
    let n = s.len();
    assert!(n > 0);
    match kind {
        "constant" => {
            let mean = s.iter().sum::<f64>() / n as f64;
            NdArray::from_vec(s.iter().map(|&x| x - mean).collect())
        }
        "linear" => {
            // Least-squares fit y = a + b t, t = 0..n-1
            let mut sum_t = 0.0;
            let mut sum_tt = 0.0;
            let mut sum_y = 0.0;
            let mut sum_ty = 0.0;
            for (i, &y) in s.iter().enumerate() {
                let t = i as f64;
                sum_t += t;
                sum_tt += t * t;
                sum_y += y;
                sum_ty += t * y;
            }
            let nf = n as f64;
            let den = nf * sum_tt - sum_t * sum_t;
            let b = if den == 0.0 {
                0.0
            } else {
                (nf * sum_ty - sum_t * sum_y) / den
            };
            let a0 = (sum_y - b * sum_t) / nf;
            NdArray::from_vec(
                s.iter()
                    .enumerate()
                    .map(|(i, &y)| y - (a0 + b * i as f64))
                    .collect(),
            )
        }
        other => panic!("detrend: unknown type '{other}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    #[test]
    fn convolve_full_known() {
        let a = NdArray::from_vec(vec![1.0, 2.0, 3.0]);
        let v = NdArray::from_vec(vec![0.0, 1.0, 0.5]);
        let y = convolve(&a, &v, "full");
        // [0,1,0.5] * [1,2,3] full = [0,1,2.5,4,1.5]
        assert_eq!(y.len(), 5);
        assert_close(y[0], 0.0, 1e-12);
        assert_close(y[1], 1.0, 1e-12);
        assert_close(y[2], 2.5, 1e-12);
        assert_close(y[3], 4.0, 1e-12);
        assert_close(y[4], 1.5, 1e-12);
    }

    #[test]
    fn fftconvolve_matches_direct() {
        let a = NdArray::from_vec(vec![1.0, -1.0, 2.0, 0.5, 3.0, -0.25, 1.5, 0.0]);
        let v = NdArray::from_vec(vec![1.0, 0.5, -0.5]);
        let d = convolve(&a, &v, "full");
        let f = fftconvolve(&a, &v, "full");
        assert_eq!(d.len(), f.len());
        for i in 0..d.len() {
            assert_close(d[i], f[i], 1e-9);
        }
    }

    #[test]
    fn detrend_constant() {
        let a = NdArray::from_vec(vec![3.0, 3.0, 3.0, 3.0]);
        let y = detrend(&a, "constant");
        for i in 0..4 {
            assert_close(y[i], 0.0, 1e-12);
        }
    }

    #[test]
    fn hann_symmetric_ends() {
        let w = hann(5, true);
        assert_close(w[0], 0.0, 1e-12);
        assert_close(w[4], 0.0, 1e-12);
        assert_close(w[2], 1.0, 1e-12);
    }
}
