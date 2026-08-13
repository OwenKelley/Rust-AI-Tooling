//! Numerical integration — mirrors common `scipy.integrate` entry points.
//!
//! Local quadrature and ODE IVP (RK45 / Dormand–Prince). No third-party crates.

use rnumpy::NdArray;

/// `scipy.integrate.trapezoid(y, x=None, dx=1.0)` for 1D.
pub fn trapezoid(y: &NdArray, x: Option<&NdArray>, dx: f64) -> f64 {
    assert_eq!(y.ndim(), 1, "trapezoid: y must be 1D");
    let yc = y.to_contiguous();
    let ys = yc.as_slice().unwrap();
    let n = ys.len();
    assert!(n >= 2, "trapezoid: need at least 2 samples");

    match x {
        None => {
            let mut s = 0.0;
            for i in 0..n - 1 {
                s += ys[i] + ys[i + 1];
            }
            0.5 * dx * s
        }
        Some(xv) => {
            assert_eq!(xv.ndim(), 1);
            let xc = xv.to_contiguous();
            let xs = xc.as_slice().unwrap();
            assert_eq!(xs.len(), n, "trapezoid: x/y length mismatch");
            let mut s = 0.0;
            for i in 0..n - 1 {
                s += (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]);
            }
            0.5 * s
        }
    }
}

/// `scipy.integrate.simpson(y, x=None, dx=1.0)` for 1D (composite Simpson).
pub fn simpson(y: &NdArray, x: Option<&NdArray>, dx: f64) -> f64 {
    assert_eq!(y.ndim(), 1, "simpson: y must be 1D");
    let yc = y.to_contiguous();
    let ys = yc.as_slice().unwrap();
    let n = ys.len();
    assert!(n >= 2, "simpson: need at least 2 samples");

    match x {
        None => simpson_uniform(ys, dx),
        Some(xv) => {
            let xc = xv.to_contiguous();
            let xs = xc.as_slice().unwrap();
            assert_eq!(xs.len(), n);
            simpson_nonuniform(ys, xs)
        }
    }
}

fn simpson_uniform(y: &[f64], dx: f64) -> f64 {
    let n = y.len();
    if n == 2 {
        return 0.5 * dx * (y[0] + y[1]);
    }
    if n % 2 == 1 {
        // Odd number of samples → even intervals: classic composite Simpson.
        basic_simpson_even_spaced(y, 0, n - 2, dx)
    } else {
        // Even number of samples → SciPy: Simpson on [0..N-3] + Cartwright last interval.
        let mut result = basic_simpson_even_spaced(y, 0, n - 3, dx);
        // h0 = h1 = dx
        let alpha = 5.0 / 12.0;
        let beta = 2.0 / 3.0;
        let eta = 1.0 / 12.0;
        result += dx * (alpha * y[n - 1] + beta * y[n - 2] - eta * y[n - 3]);
        result
    }
}

/// Even-spaced `_basic_simpson(y, start, stop, dx)` from SciPy.
fn basic_simpson_even_spaced(y: &[f64], start: usize, stop: usize, dx: f64) -> f64 {
    let mut s = 0.0;
    let mut i = start;
    while i <= stop {
        s += y[i] + 4.0 * y[i + 1] + y[i + 2];
        i += 2;
    }
    s * dx / 3.0
}

fn simpson_nonuniform(y: &[f64], x: &[f64]) -> f64 {
    // SciPy-style uneven Simpson (pairs of intervals + trailing trapezoid).
    let n = y.len();
    if n == 2 {
        return 0.5 * (x[1] - x[0]) * (y[0] + y[1]);
    }
    let mut result = 0.0;
    let mut i = 0;
    while i + 1 < n {
        if i + 2 < n {
            let h0 = x[i + 1] - x[i];
            let h1 = x[i + 2] - x[i + 1];
            assert!(h0 > 0.0 && h1 > 0.0, "simpson: x must be increasing");
            let hsum = h0 + h1;
            let h0divh1 = h0 / h1;
            result += hsum / 6.0
                * ((2.0 - 1.0 / h0divh1) * y[i]
                    + (hsum / h0) * (hsum / h1) * y[i + 1]
                    + (2.0 - h0divh1) * y[i + 2]);
            i += 2;
        } else {
            result += 0.5 * (x[i + 1] - x[i]) * (y[i] + y[i + 1]);
            i += 1;
        }
    }
    result
}

/// `scipy.integrate.cumulative_trapezoid(y, x=None, dx=1.0, initial=None)`.
///
/// If `initial` is `Some(v)`, prepend `v` (SciPy `initial=v`); length = n.
/// If `None`, length = n-1 (SciPy default).
pub fn cumulative_trapezoid(
    y: &NdArray,
    x: Option<&NdArray>,
    dx: f64,
    initial: Option<f64>,
) -> NdArray {
    assert_eq!(y.ndim(), 1);
    let yc = y.to_contiguous();
    let ys = yc.as_slice().unwrap();
    let n = ys.len();
    assert!(n >= 2);

    let mut partial = Vec::with_capacity(n - 1);
    let mut acc = 0.0;
    match x {
        None => {
            for i in 0..n - 1 {
                acc += 0.5 * dx * (ys[i] + ys[i + 1]);
                partial.push(acc);
            }
        }
        Some(xv) => {
            let xc = xv.to_contiguous();
            let xs = xc.as_slice().unwrap();
            assert_eq!(xs.len(), n);
            for i in 0..n - 1 {
                acc += 0.5 * (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]);
                partial.push(acc);
            }
        }
    }
    match initial {
        Some(v) => {
            let mut out = Vec::with_capacity(n);
            out.push(v);
            out.extend(partial.into_iter().map(|p| p + v));
            NdArray::from_vec(out)
        }
        None => NdArray::from_vec(partial),
    }
}

/// Adaptive Simpson quadrature — lightweight stand-in for `scipy.integrate.quad`
/// on smooth scalar integrands over `[a, b]`.
pub fn quad<F>(mut f: F, a: f64, b: f64, eps: f64) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    assert!(a.is_finite() && b.is_finite());
    let fa = f(a);
    let fb = f(b);
    let (val, err) = adaptive_simpson(&mut f, a, b, fa, fb, eps, 0);
    (val, err)
}

fn simpson_rule(fa: f64, fm: f64, fb: f64, h: f64) -> f64 {
    (h / 6.0) * (fa + 4.0 * fm + fb)
}

fn adaptive_simpson<F>(
    f: &mut F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    eps: f64,
    depth: usize,
) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    let m = 0.5 * (a + b);
    let fm = f(m);
    let h = b - a;
    let s = simpson_rule(fa, fm, fb, h);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    let s2 = simpson_rule(fa, flm, fm, h * 0.5) + simpson_rule(fm, frm, fb, h * 0.5);
    let err = ((s2 - s) / 15.0).abs();
    if depth > 40 || err <= eps {
        return (s2 + (s2 - s) / 15.0, err);
    }
    let (left, e1) = adaptive_simpson(f, a, m, fa, fm, eps * 0.5, depth + 1);
    let (right, e2) = adaptive_simpson(f, m, b, fm, fb, eps * 0.5, depth + 1);
    (left + right, e1 + e2)
}

/// `scipy.integrate.dblquad(func, a, b, gfun, hfun)`.
///
/// Computes ∫_a^b dx ∫_{g(x)}^{h(x)} dy `f(y, x)` (SciPy argument order: `y` then `x`).
pub fn dblquad<F, G, H>(mut f: F, a: f64, b: f64, mut g: G, mut h: H, eps: f64) -> (f64, f64)
where
    F: FnMut(f64, f64) -> f64,
    G: FnMut(f64) -> f64,
    H: FnMut(f64) -> f64,
{
    assert!(a.is_finite() && b.is_finite());
    // Outer adaptive Simpson over x; each node runs an inner quad over y.
    let outer = |x: f64| {
        let y0 = g(x);
        let y1 = h(x);
        if (y1 - y0).abs() < 1e-15 {
            return 0.0;
        }
        let (lo, hi, sign) = if y1 >= y0 {
            (y0, y1, 1.0)
        } else {
            (y1, y0, -1.0)
        };
        let (inner, _) = quad(|y| f(y, x), lo, hi, eps * 0.5);
        sign * inner
    };
    quad(outer, a, b, eps)
}

/// Result of `solve_ivp` (subset of SciPy's `OdeResult`).
#[derive(Debug, Clone)]
pub struct OdeResult {
    pub t: Vec<f64>,
    /// Shape conceptually `(n_states, n_times)` stored row-major by state.
    pub y: Vec<Vec<f64>>,
    pub success: bool,
    pub nfev: usize,
}

impl OdeResult {
    /// Flattened checksum helper: sum of all y values.
    pub fn y_sum(&self) -> f64 {
        self.y.iter().flat_map(|row| row.iter()).sum()
    }
}

/// `scipy.integrate.solve_ivp(..., method='RK45')` for autonomous / non-autonomous ODEs.
///
/// `f(t, y) -> dy/dt` with `y` as `&[f64]`. Integrates from `t_span=(t0,tf)` and
/// returns states at `t_eval` (must be within the span, sorted).
pub fn solve_ivp_rk45<F>(
    f: F,
    t_span: (f64, f64),
    y0: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
) -> OdeResult
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    solve_ivp_adaptive(f, t_span, y0, t_eval, rtol, atol, OdeMethod::Rk45)
}

/// `scipy.integrate.solve_ivp(..., method='RK23')` (Bogacki–Shampine 3(2)).
pub fn solve_ivp_rk23<F>(
    f: F,
    t_span: (f64, f64),
    y0: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
) -> OdeResult
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    solve_ivp_adaptive(f, t_span, y0, t_eval, rtol, atol, OdeMethod::Rk23)
}

#[derive(Clone, Copy)]
enum OdeMethod {
    Rk45,
    Rk23,
}

fn solve_ivp_adaptive<F>(
    mut f: F,
    t_span: (f64, f64),
    y0: &[f64],
    t_eval: &[f64],
    rtol: f64,
    atol: f64,
    method: OdeMethod,
) -> OdeResult
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let (t0, tf) = t_span;
    assert!(tf > t0, "solve_ivp: need tf > t0");
    let n = y0.len();
    assert!(n > 0);
    assert!(!t_eval.is_empty());
    for w in t_eval.windows(2) {
        assert!(w[1] >= w[0], "t_eval must be nondecreasing");
    }
    assert!(t_eval[0] >= t0 - 1e-15 && t_eval[t_eval.len() - 1] <= tf + 1e-15);

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut nfev = 0usize;
    let mut h = ((tf - t0) / 100.0).abs().max(1e-6);

    let mut tout = Vec::with_capacity(t_eval.len());
    let mut yout: Vec<Vec<f64>> = vec![Vec::with_capacity(t_eval.len()); n];
    let mut eval_i = 0usize;

    while eval_i < t_eval.len() && (t_eval[eval_i] - t).abs() <= 1e-14 {
        tout.push(t_eval[eval_i]);
        for s in 0..n {
            yout[s].push(y[s]);
        }
        eval_i += 1;
    }

    let max_steps = 1_000_000usize;
    let mut steps = 0usize;
    let mut success = true;
    let err_exp = match method {
        OdeMethod::Rk45 => -0.2,
        OdeMethod::Rk23 => -1.0 / 3.0,
    };

    while eval_i < t_eval.len() && steps < max_steps {
        steps += 1;
        if t >= tf {
            success = false;
            break;
        }
        let t_target = t_eval[eval_i].min(tf);
        h = h.min(tf - t).min(t_target - t);
        if h <= 0.0 {
            if (t - t_eval[eval_i]).abs() <= 1e-12 {
                tout.push(t_eval[eval_i]);
                for s in 0..n {
                    yout[s].push(y[s]);
                }
                eval_i += 1;
                continue;
            }
            success = false;
            break;
        }

        let (y_new, y_err, fev) = match method {
            OdeMethod::Rk45 => rk45_step(&mut f, t, &y, h, n),
            OdeMethod::Rk23 => rk23_step(&mut f, t, &y, h, n),
        };
        nfev += fev;

        let mut err = 0.0_f64;
        for i in 0..n {
            let sc = atol + rtol * y[i].abs().max(y_new[i].abs());
            let e = ((y_new[i] - y_err[i]) / sc).abs();
            err = err.max(e);
        }

        let accept = err <= 1.0;
        if accept {
            t += h;
            y = y_new;
            while eval_i < t_eval.len() && t_eval[eval_i] <= t + 1e-12 {
                tout.push(t_eval[eval_i]);
                for s in 0..n {
                    yout[s].push(y[s]);
                }
                eval_i += 1;
            }
        }

        let factor = if err == 0.0 {
            2.0
        } else {
            0.9 * err.powf(err_exp)
        };
        let factor = factor.clamp(0.2, 5.0);
        h *= factor;
        if h < 1e-16 {
            success = false;
            break;
        }
    }

    if eval_i < t_eval.len() {
        success = false;
    }

    OdeResult {
        t: tout,
        y: yout,
        success,
        nfev,
    }
}

fn rk45_step<F>(f: &mut F, t: f64, y: &[f64], h: f64, n: usize) -> (Vec<f64>, Vec<f64>, usize)
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let mut nfev = 0usize;
    let k1 = f(t, y);
    nfev += 1;
    assert_eq!(k1.len(), n);

    let y2 = axpy(y, &k1, h * (1.0 / 5.0));
    let k2 = f(t + h * (1.0 / 5.0), &y2);
    nfev += 1;

    let y3 = comb(y, &[(&k1, h * (3.0 / 40.0)), (&k2, h * (9.0 / 40.0))]);
    let k3 = f(t + h * (3.0 / 10.0), &y3);
    nfev += 1;

    let y4 = comb(
        y,
        &[
            (&k1, h * (44.0 / 45.0)),
            (&k2, h * (-56.0 / 15.0)),
            (&k3, h * (32.0 / 9.0)),
        ],
    );
    let k4 = f(t + h * (4.0 / 5.0), &y4);
    nfev += 1;

    let y5 = comb(
        y,
        &[
            (&k1, h * (19372.0 / 6561.0)),
            (&k2, h * (-25360.0 / 2187.0)),
            (&k3, h * (64448.0 / 6561.0)),
            (&k4, h * (-212.0 / 729.0)),
        ],
    );
    let k5 = f(t + h * (8.0 / 9.0), &y5);
    nfev += 1;

    let y6 = comb(
        y,
        &[
            (&k1, h * (9017.0 / 3168.0)),
            (&k2, h * (-355.0 / 33.0)),
            (&k3, h * (46732.0 / 5247.0)),
            (&k4, h * (49.0 / 176.0)),
            (&k5, h * (-5103.0 / 18656.0)),
        ],
    );
    let k6 = f(t + h, &y6);
    nfev += 1;

    let y_new = comb(
        y,
        &[
            (&k1, h * (35.0 / 384.0)),
            (&k3, h * (500.0 / 1113.0)),
            (&k4, h * (125.0 / 192.0)),
            (&k5, h * (-2187.0 / 6784.0)),
            (&k6, h * (11.0 / 84.0)),
        ],
    );
    let k7 = f(t + h, &y_new);
    nfev += 1;

    let y_err = comb(
        y,
        &[
            (&k1, h * (5179.0 / 57600.0)),
            (&k3, h * (7571.0 / 16695.0)),
            (&k4, h * (393.0 / 640.0)),
            (&k5, h * (-92097.0 / 339200.0)),
            (&k6, h * (187.0 / 2100.0)),
            (&k7, h * (1.0 / 40.0)),
        ],
    );
    (y_new, y_err, nfev)
}

/// Bogacki–Shampine 3(2) step.
fn rk23_step<F>(f: &mut F, t: f64, y: &[f64], h: f64, n: usize) -> (Vec<f64>, Vec<f64>, usize)
where
    F: FnMut(f64, &[f64]) -> Vec<f64>,
{
    let mut nfev = 0usize;
    let k1 = f(t, y);
    nfev += 1;
    assert_eq!(k1.len(), n);

    let y2 = axpy(y, &k1, h * 0.5);
    let k2 = f(t + 0.5 * h, &y2);
    nfev += 1;

    let y3 = axpy(y, &k2, h * 0.75);
    let k3 = f(t + 0.75 * h, &y3);
    nfev += 1;

    let y_new = comb(
        y,
        &[
            (&k1, h * (2.0 / 9.0)),
            (&k2, h * (1.0 / 3.0)),
            (&k3, h * (4.0 / 9.0)),
        ],
    );
    let k4 = f(t + h, &y_new);
    nfev += 1;

    let y_err = comb(
        y,
        &[
            (&k1, h * (7.0 / 24.0)),
            (&k2, h * (1.0 / 4.0)),
            (&k3, h * (1.0 / 3.0)),
            (&k4, h * (1.0 / 8.0)),
        ],
    );
    (y_new, y_err, nfev)
}

fn axpy(y: &[f64], k: &[f64], a: f64) -> Vec<f64> {
    y.iter().zip(k.iter()).map(|(&yi, &ki)| yi + a * ki).collect()
}

fn comb(y: &[f64], terms: &[(&[f64], f64)]) -> Vec<f64> {
    let n = y.len();
    let mut out = y.to_vec();
    for &(k, a) in terms {
        for i in 0..n {
            out[i] += a * k[i];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    #[test]
    fn trapezoid_linear() {
        let x = NdArray::from_vec(vec![0.0, 1.0, 2.0]);
        let y = NdArray::from_vec(vec![0.0, 1.0, 2.0]);
        assert_close(trapezoid(&y, Some(&x), 1.0), 2.0, 1e-12);
    }

    #[test]
    fn simpson_parabola() {
        // ∫_0^2 x^2 dx = 8/3
        let x = NdArray::from_vec(vec![0.0, 1.0, 2.0]);
        let y = NdArray::from_vec(vec![0.0, 1.0, 4.0]);
        assert_close(simpson(&y, Some(&x), 1.0), 8.0 / 3.0, 1e-12);
    }

    #[test]
    fn cumulative_trapezoid_initial() {
        let y = NdArray::from_vec(vec![1.0, 1.0, 1.0]);
        let c = cumulative_trapezoid(&y, None, 1.0, Some(0.0));
        assert_eq!(c.len(), 3);
        assert_close(c[0], 0.0, 1e-12);
        assert_close(c[1], 1.0, 1e-12);
        assert_close(c[2], 2.0, 1e-12);
    }

    #[test]
    fn quad_gaussian() {
        let (v, _) = quad(|x| (-x * x).exp(), 0.0, 1.0, 1e-10);
        // √π/2 * erf(1)
        let expected = std::f64::consts::PI.sqrt() / 2.0 * crate::special::erf_scalar(1.0);
        assert_close(v, expected, 1e-7);
    }

    #[test]
    fn dblquad_unit_square() {
        // ∫_0^1 ∫_0^1 1 dy dx = 1
        let (v, _) = dblquad(|_y, _x| 1.0, 0.0, 1.0, |_| 0.0, |_| 1.0, 1e-8);
        assert_close(v, 1.0, 1e-6);
    }

    #[test]
    fn solve_ivp_exp_decay() {
        // y' = -y, y(0)=1 → y(t)=e^{-t}
        let t_eval: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let r = solve_ivp_rk45(
            |_t, y| vec![-y[0]],
            (0.0, 1.0),
            &[1.0],
            &t_eval,
            1e-6,
            1e-9,
        );
        assert!(r.success);
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(r.y[0][i], (-t).exp(), 1e-4);
        }
    }

    #[test]
    fn solve_ivp_rk23_exp_decay() {
        let t_eval: Vec<f64> = (0..11).map(|i| i as f64 * 0.1).collect();
        let r = solve_ivp_rk23(
            |_t, y| vec![-y[0]],
            (0.0, 1.0),
            &[1.0],
            &t_eval,
            1e-6,
            1e-9,
        );
        assert!(r.success);
        for (i, &t) in t_eval.iter().enumerate() {
            assert_close(r.y[0][i], (-t).exp(), 5e-4);
        }
    }
}
