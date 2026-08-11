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

type C = (f64, f64);

#[inline]
fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn cdiv(a: C, b: C) -> C {
    let n = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / n, (a.1 * b.0 - a.0 * b.1) / n)
}

/// Multiply polynomial (high→low coeffs) by `(z − root)`.
fn poly_mul_root(p: &[C], root: C) -> Vec<C> {
    let mut q = vec![(0.0, 0.0); p.len() + 1];
    for i in 0..p.len() {
        q[i] = cadd(q[i], p[i]);
        q[i + 1] = cadd(q[i + 1], cmul(p[i], (-root.0, -root.1)));
    }
    q
}

fn poly_from_roots(roots: &[C]) -> Vec<f64> {
    let mut p = vec![(1.0, 0.0)];
    for &r in roots {
        p = poly_mul_root(&p, r);
    }
    p.into_iter().map(|c| c.0).collect()
}

/// Analog Butterworth prototype — `scipy.signal.buttap(N)`.
fn buttap(n: usize) -> (Vec<C>, f64) {
    assert!(n > 0, "buttap: order must be > 0");
    let mut poles = Vec::with_capacity(n);
    let mut m = -(n as i64) + 1;
    while m < n as i64 {
        let ang = std::f64::consts::PI * m as f64 / (2.0 * n as f64);
        poles.push((-ang.cos(), -ang.sin()));
        m += 2;
    }
    (poles, 1.0)
}

fn bilinear_zpk(z: &[C], p: &[C], k: f64, fs: f64) -> (Vec<C>, Vec<C>, f64) {
    let fs2 = 2.0 * fs;
    let degree = p.len() as i64 - z.len() as i64;
    let mut zz: Vec<C> = z
        .iter()
        .map(|&zi| cdiv((fs2 + zi.0, zi.1), (fs2 - zi.0, -zi.1)))
        .collect();
    for _ in 0..degree.max(0) {
        zz.push((-1.0, 0.0));
    }
    let pz: Vec<C> = p
        .iter()
        .map(|&pi| cdiv((fs2 + pi.0, pi.1), (fs2 - pi.0, -pi.1)))
        .collect();
    let mut num = (1.0, 0.0);
    for &zi in z {
        num = cmul(num, (fs2 - zi.0, -zi.1));
    }
    let mut den = (1.0, 0.0);
    for &pi in p {
        den = cmul(den, (fs2 - pi.0, -pi.1));
    }
    let kz = cmul((k, 0.0), cdiv(num, den)).0;
    (zz, pz, kz)
}

/// `scipy.signal.butter(N, Wn, btype=..., analog=False, output='ba')`.
///
/// `wn` is normalized to Nyquist (`1.0` = Nyquist). Supports `"lowpass"` / `"highpass"`.
pub fn butter(order: usize, wn: f64, btype: &str) -> (NdArray, NdArray) {
    assert!(order > 0, "butter: order must be > 0");
    assert!(wn > 0.0 && wn < 1.0, "butter: wn must be in (0, 1)");
    let fs = 2.0;
    let warped = 2.0 * fs * (std::f64::consts::PI * wn / fs).tan();
    let (poles0, gain0) = buttap(order);
    let (z, p, k) = match btype {
        "low" | "lowpass" => {
            let p: Vec<C> = poles0
                .iter()
                .map(|&pi| (pi.0 * warped, pi.1 * warped))
                .collect();
            let k = gain0 * warped.powi(order as i32);
            (Vec::<C>::new(), p, k)
        }
        "high" | "highpass" => {
            // SciPy `lp2hp`: zeros at 0, poles = wo/p, k *= real(prod(-p)).
            let mut prod = (1.0, 0.0);
            for &pi in &poles0 {
                prod = cmul(prod, (-pi.0, -pi.1));
            }
            let k = gain0 * prod.0;
            let z = vec![(0.0, 0.0); order];
            let p: Vec<C> = poles0
                .iter()
                .map(|&pi| cdiv((warped, 0.0), pi))
                .collect();
            (z, p, k)
        }
        other => panic!("butter: unsupported btype '{other}' (use lowpass/highpass)"),
    };

    let (zd, pd, kd) = bilinear_zpk(&z, &p, k, fs);
    let mut b = poly_from_roots(&zd);
    let a = poly_from_roots(&pd);
    for c in &mut b {
        *c *= kd;
    }
    let a0 = a[0];
    assert!(a0.abs() > 0.0, "butter: degenerate denominator");
    let b: Vec<f64> = b.iter().map(|c| c / a0).collect();
    let a: Vec<f64> = a.iter().map(|c| c / a0).collect();
    (NdArray::from_vec(b), NdArray::from_vec(a))
}

/// `scipy.signal.lfilter(b, a, x)` — Direct Form II Transposed, 1D.
///
/// Optional `zi` is the initial delay state (length `max(len(b),len(a))-1`).
/// Returns `(y, zf)` final state when `zi` is provided; otherwise just `y` via [`lfilter`].
pub fn lfilter_zi_run(
    b: &NdArray,
    a: &NdArray,
    x: &NdArray,
    zi: Option<&[f64]>,
) -> (NdArray, Vec<f64>) {
    assert_eq!(b.ndim(), 1);
    assert_eq!(a.ndim(), 1);
    assert_eq!(x.ndim(), 1);
    let bb = b.to_contiguous();
    let aa = a.to_contiguous();
    let xx = x.to_contiguous();
    let b_s = bb.as_slice().unwrap();
    let a_s = aa.as_slice().unwrap();
    let x_s = xx.as_slice().unwrap();
    assert!(!a_s.is_empty() && a_s[0] != 0.0);
    let n = x_s.len();
    let nb = b_s.len();
    let na = a_s.len();
    let a0 = a_s[0];
    let b_n: Vec<f64> = b_s.iter().map(|v| v / a0).collect();
    let a_n: Vec<f64> = a_s.iter().map(|v| v / a0).collect();
    let order = (nb - 1).max(na - 1);
    let mut z = match zi {
        Some(s) => {
            assert_eq!(s.len(), order, "lfilter: zi length mismatch");
            s.to_vec()
        }
        None => vec![0.0; order],
    };
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut yi = if nb > 0 { b_n[0] * x_s[i] } else { 0.0 };
        if order > 0 {
            yi += z[0];
        }
        for j in 0..order {
            let mut zj = if j + 1 < order { z[j + 1] } else { 0.0 };
            if j + 1 < nb {
                zj += b_n[j + 1] * x_s[i];
            }
            if j + 1 < na {
                zj -= a_n[j + 1] * yi;
            }
            z[j] = zj;
        }
        y[i] = yi;
    }
    (NdArray::from_vec(y), z)
}

/// `scipy.signal.lfilter(b, a, x)` with zero initial state.
pub fn lfilter(b: &NdArray, a: &NdArray, x: &NdArray) -> NdArray {
    lfilter_zi_run(b, a, x, None).0
}

/// `scipy.signal.lfilter_zi(b, a)` — steady-state delay for a unit step.
pub fn lfilter_zi(b: &NdArray, a: &NdArray) -> NdArray {
    let bb = b.to_contiguous();
    let aa = a.to_contiguous();
    let b_s = bb.as_slice().unwrap();
    let a_s = aa.as_slice().unwrap();
    let a0 = a_s[0];
    let n = (b_s.len() - 1).max(a_s.len() - 1);
    if n == 0 {
        return NdArray::from_vec(vec![]);
    }
    let mut b_n = vec![0.0; n + 1];
    let mut a_n = vec![0.0; n + 1];
    for i in 0..b_s.len().min(n + 1) {
        b_n[i] = b_s[i] / a0;
    }
    for i in 0..a_s.len().min(n + 1) {
        a_n[i] = a_s[i] / a0;
    }
    // scipy.linalg.companion(a) then I - companion.T
    let mut comp = vec![0.0; n * n];
    for j in 0..n {
        comp[j] = -a_n[j + 1];
    }
    for i in 1..n {
        comp[i * n + (i - 1)] = 1.0;
    }
    let mut imina = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let ct = comp[j * n + i];
            imina[i * n + j] = if i == j { 1.0 - ct } else { -ct };
        }
    }
    let mut rhs = vec![0.0; n];
    for j in 0..n {
        rhs[j] = b_n[j + 1] - a_n[j + 1] * b_n[0];
    }
    rnumpy::solve(
        &NdArray::from_shape_vec(&[n, n], imina),
        &NdArray::from_vec(rhs),
    )
}

/// Odd extension pad (SciPy `odd_ext`) by `n` samples each side.
fn odd_ext(x: &[f64], n: usize) -> Vec<f64> {
    assert!(x.len() > 1, "filtfilt: signal too short for padding");
    let mut left = Vec::with_capacity(n);
    for i in 0..n {
        let idx = (i + 1).min(x.len() - 1);
        left.push(2.0 * x[0] - x[idx]);
    }
    left.reverse();
    let mut right = Vec::with_capacity(n);
    let last = x.len() - 1;
    for i in 0..n {
        let idx = last.saturating_sub(i + 1);
        right.push(2.0 * x[last] - x[idx]);
    }
    let mut out = Vec::with_capacity(x.len() + 2 * n);
    out.extend_from_slice(&left);
    out.extend_from_slice(x);
    out.extend_from_slice(&right);
    out
}

/// `scipy.signal.filtfilt(b, a, x, method='pad', padtype='odd')` for 1D.
pub fn filtfilt(b: &NdArray, a: &NdArray, x: &NdArray) -> NdArray {
    assert_eq!(x.ndim(), 1);
    let xx = x.to_contiguous();
    let x_s = xx.as_slice().unwrap();
    let n = x_s.len();
    let ntaps = b.len().max(a.len());
    let edge = 3 * ntaps;
    assert!(n > edge, "filtfilt: length must exceed pad edge");
    let ext = odd_ext(x_s, edge);
    let zi = lfilter_zi(b, a);
    let zi_s = zi.as_slice().unwrap();
    let zi_fwd: Vec<f64> = zi_s.iter().map(|v| v * ext[0]).collect();
    let (y, _) = lfilter_zi_run(b, a, &NdArray::from_vec(ext), Some(&zi_fwd));
    let y_s = y.as_slice().unwrap();
    let rev: Vec<f64> = y_s.iter().rev().copied().collect();
    let zi_bwd: Vec<f64> = zi_s.iter().map(|v| v * rev[0]).collect();
    let (y2, _) = lfilter_zi_run(b, a, &NdArray::from_vec(rev), Some(&zi_bwd));
    let y2_s = y2.as_slice().unwrap();
    let out: Vec<f64> = y2_s.iter().rev().copied().skip(edge).take(n).collect();
    NdArray::from_vec(out)
}

/// `scipy.signal.welch(x, fs=1.0, window='hann', nperseg=..., noverlap=None, scaling='density')`.
///
/// Returns `(f, Pxx)` 1D arrays. Uses Hann window, `scaling='density'`, onesided.
pub fn welch(
    x: &NdArray,
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
) -> (NdArray, NdArray) {
    assert_eq!(x.ndim(), 1);
    assert!(nperseg > 0 && nperseg <= x.len());
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    assert!(noverlap < nperseg);
    let step = nperseg - noverlap;
    let xc = x.to_contiguous();
    let xs = xc.as_slice().unwrap();
    let win = hann(nperseg, false);
    let w = win.as_slice().unwrap();
    let mut u = 0.0; // window power
    for &wi in w {
        u += wi * wi;
    }
    let n_freqs = nperseg / 2 + 1;
    let mut acc = vec![0.0; n_freqs];
    let mut nseg = 0usize;
    let mut start = 0usize;
    while start + nperseg <= xs.len() {
        let mut seg = vec![0.0; nperseg];
        for i in 0..nperseg {
            seg[i] = xs[start + i] * w[i];
        }
        let spec = crate::fft::rfft(&NdArray::from_vec(seg));
        let s = spec.as_slice().unwrap();
        for k in 0..n_freqs {
            let re = s[k * 2];
            let im = s[k * 2 + 1];
            let mut p = re * re + im * im;
            // onesided density scaling (SciPy)
            if k > 0 && (nperseg % 2 == 1 || k < n_freqs - 1) {
                p *= 2.0;
            }
            acc[k] += p;
        }
        nseg += 1;
        start += step;
    }
    assert!(nseg > 0, "welch: no segments");
    let scale = fs * u * nseg as f64;
    for v in &mut acc {
        *v /= scale;
    }
    let freqs = crate::fft::rfftfreq(nperseg, 1.0 / fs);
    (freqs, NdArray::from_vec(acc))
}

/// `scipy.signal.stft` (onesided, Hann, `return_onesided=True`) — magnitude spectrum.
///
/// Returns `(f, t, Zxx_abs)` where `Zxx_abs` is shape `[n_freqs, n_frames]`.
pub fn stft(
    x: &NdArray,
    fs: f64,
    nperseg: usize,
    noverlap: Option<usize>,
) -> (NdArray, NdArray, NdArray) {
    assert_eq!(x.ndim(), 1);
    assert!(nperseg > 0 && nperseg <= x.len());
    let noverlap = noverlap.unwrap_or(nperseg / 2);
    assert!(noverlap < nperseg);
    let step = nperseg - noverlap;
    let xc = x.to_contiguous();
    let xs = xc.as_slice().unwrap();
    let win = hann(nperseg, false);
    let w = win.as_slice().unwrap();
    let n_freqs = nperseg / 2 + 1;
    let n_frames = (xs.len() - noverlap) / step;
    let mut z = vec![0.0; n_freqs * n_frames];
    let scale = w.iter().map(|v| v.abs()).sum::<f64>();
    for (frame, start) in (0..n_frames).map(|i| (i, i * step)) {
        let mut seg = vec![0.0; nperseg];
        for i in 0..nperseg {
            seg[i] = xs[start + i] * w[i];
        }
        let spec = crate::fft::rfft(&NdArray::from_vec(seg));
        let s = spec.as_slice().unwrap();
        for k in 0..n_freqs {
            let re = s[k * 2] / scale;
            let im = s[k * 2 + 1] / scale;
            z[k * n_frames + frame] = (re * re + im * im).sqrt();
        }
    }
    let freqs = crate::fft::rfftfreq(nperseg, 1.0 / fs);
    let times: Vec<f64> = (0..n_frames)
        .map(|i| (i * step) as f64 / fs)
        .collect();
    (
        freqs,
        NdArray::from_vec(times),
        NdArray::from_shape_vec(&[n_freqs, n_frames], z),
    )
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

    #[test]
    fn butter_lowpass_dc_gain() {
        let (b, a) = butter(4, 0.2, "lowpass");
        // At DC (z=1), H ≈ sum(b)/sum(a) ≈ 1
        let sb: f64 = b.iter().sum();
        let sa: f64 = a.iter().sum();
        assert_close(sb / sa, 1.0, 1e-6);
    }

    #[test]
    fn filtfilt_smooths() {
        let (b, a) = butter(3, 0.1, "lowpass");
        let mut x = vec![0.0; 128];
        for i in 0..128 {
            x[i] = if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        let y = filtfilt(&b, &a, &NdArray::from_vec(x));
        let mean_abs: f64 = y.iter().map(|v| v.abs()).sum::<f64>() / y.len() as f64;
        assert!(mean_abs < 0.5, "expected lowpass to attenuate alternating signal");
    }

    #[test]
    fn welch_positive() {
        let x = NdArray::from_vec((0..256).map(|i| (i as f64 * 0.1).sin()).collect());
        let (_f, pxx) = welch(&x, 1.0, 64, Some(32));
        assert!(pxx.iter().all(|v| v >= 0.0));
    }
}
