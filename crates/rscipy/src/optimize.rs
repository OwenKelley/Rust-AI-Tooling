//! Optimization — mirrors common `scipy.optimize` entry points.
//!
//! Local implementations only (no third-party crates).

/// Result mirror of SciPy's `OptimizeResult` (subset used by parity).
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub success: bool,
    pub nit: usize,
    pub nfev: usize,
}

/// `scipy.optimize.minimize(..., method='Nelder-Mead')`.
pub fn minimize_nelder_mead<F>(
    mut f: F,
    x0: &[f64],
    maxiter: usize,
    xatol: f64,
    fatol: f64,
) -> OptimizeResult
where
    F: FnMut(&[f64]) -> f64,
{
    let n = x0.len();
    assert!(n > 0, "nelder_mead: empty x0");

    // Initial simplex: x0 and x0 + step*e_i
    let mut simplex = vec![x0.to_vec(); n + 1];
    let mut fvals = vec![0.0; n + 1];
    let mut nfev = 0usize;
    fvals[0] = f(&simplex[0]);
    nfev += 1;
    for i in 0..n {
        simplex[i + 1] = x0.to_vec();
        let step = if x0[i].abs() > 1e-3 {
            0.05 * x0[i].abs()
        } else {
            0.00025
        };
        simplex[i + 1][i] += step;
        fvals[i + 1] = f(&simplex[i + 1]);
        nfev += 1;
    }

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    let mut nit = 0usize;
    loop {
        // Order by fvals ascending
        let mut order: Vec<usize> = (0..n + 1).collect();
        order.sort_by(|&i, &j| fvals[i].partial_cmp(&fvals[j]).unwrap());

        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        // Centroid of all but worst
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Convergence
        let mut xmax: f64 = 0.0;
        for i in 0..=n {
            for j in 0..n {
                xmax = xmax.max((simplex[i][j] - simplex[best][j]).abs());
            }
        }
        let mut fmax: f64 = 0.0;
        for i in 0..=n {
            fmax = fmax.max((fvals[i] - fvals[best]).abs());
        }
        if xmax <= xatol && fmax <= fatol {
            return OptimizeResult {
                x: simplex[best].clone(),
                fun: fvals[best],
                success: true,
                nit,
                nfev,
            };
        }
        if nit >= maxiter {
            return OptimizeResult {
                x: simplex[best].clone(),
                fun: fvals[best],
                success: false,
                nit,
                nfev,
            };
        }
        nit += 1;

        // Reflect
        let mut xr = vec![0.0; n];
        for j in 0..n {
            xr[j] = centroid[j] + alpha * (centroid[j] - simplex[worst][j]);
        }
        let fr = f(&xr);
        nfev += 1;

        if fr < fvals[best] {
            // Expand
            let mut xe = vec![0.0; n];
            for j in 0..n {
                xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]);
            }
            let fe = f(&xe);
            nfev += 1;
            if fe < fr {
                simplex[worst] = xe;
                fvals[worst] = fe;
            } else {
                simplex[worst] = xr;
                fvals[worst] = fr;
            }
        } else if fr < fvals[second_worst] {
            simplex[worst] = xr;
            fvals[worst] = fr;
        } else {
            // Contract
            let (xc, fc) = if fr < fvals[worst] {
                // Outside
                let mut xc = vec![0.0; n];
                for j in 0..n {
                    xc[j] = centroid[j] + rho * (xr[j] - centroid[j]);
                }
                let fc = f(&xc);
                nfev += 1;
                (xc, fc)
            } else {
                // Inside
                let mut xc = vec![0.0; n];
                for j in 0..n {
                    xc[j] = centroid[j] + rho * (simplex[worst][j] - centroid[j]);
                }
                let fc = f(&xc);
                nfev += 1;
                (xc, fc)
            };
            if fc < fvals[worst].min(fr) {
                simplex[worst] = xc;
                fvals[worst] = fc;
            } else {
                // Shrink toward best
                for i in 0..=n {
                    if i == best {
                        continue;
                    }
                    for j in 0..n {
                        simplex[i][j] =
                            simplex[best][j] + sigma * (simplex[i][j] - simplex[best][j]);
                    }
                    fvals[i] = f(&simplex[i]);
                    nfev += 1;
                }
            }
        }
    }
}

fn project_bounds(x: &mut [f64], bounds: &[(f64, f64)]) {
    for i in 0..x.len() {
        let (lo, hi) = bounds[i];
        if x[i] < lo {
            x[i] = lo;
        }
        if x[i] > hi {
            x[i] = hi;
        }
    }
}

/// `scipy.optimize.minimize(..., method='L-BFGS-B')` with box bounds.
///
/// `grad` must be provided (analytical). `bounds[i] = (low, high)`.
pub fn minimize_lbfgsb<F, G>(
    mut f: F,
    mut grad: G,
    x0: &[f64],
    bounds: &[(f64, f64)],
    maxiter: usize,
    m_hist: usize,
    gtol: f64,
) -> OptimizeResult
where
    F: FnMut(&[f64]) -> f64,
    G: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    assert_eq!(bounds.len(), n);
    let m_hist = m_hist.max(1).min(20);

    let mut x = x0.to_vec();
    project_bounds(&mut x, bounds);
    let mut fx = f(&x);
    let mut nfev = 1usize;
    let mut g = grad(&x);
    let mut nit = 0usize;

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    loop {
        let mut pg_norm: f64 = 0.0;
        for i in 0..n {
            let mut gi = g[i];
            if x[i] <= bounds[i].0 && gi > 0.0 {
                gi = 0.0;
            }
            if x[i] >= bounds[i].1 && gi < 0.0 {
                gi = 0.0;
            }
            pg_norm = pg_norm.max(gi.abs());
        }
        if pg_norm < gtol {
            return OptimizeResult {
                x,
                fun: fx,
                success: true,
                nit,
                nfev,
            };
        }
        if nit >= maxiter {
            return OptimizeResult {
                x,
                fun: fx,
                success: fx < 1e-6,
                nit,
                nfev,
            };
        }

        let mut q = g.clone();
        let mut alpha_coef = vec![0.0; s_hist.len()];
        for i in (0..s_hist.len()).rev() {
            let mut dot = 0.0;
            for j in 0..n {
                dot += s_hist[i][j] * q[j];
            }
            alpha_coef[i] = rho_hist[i] * dot;
            for j in 0..n {
                q[j] -= alpha_coef[i] * y_hist[i][j];
            }
        }
        let mut gamma = 1.0;
        if let (Some(s_last), Some(y_last)) = (s_hist.last(), y_hist.last()) {
            let mut ys = 0.0;
            let mut yy = 0.0;
            for j in 0..n {
                ys += y_last[j] * s_last[j];
                yy += y_last[j] * y_last[j];
            }
            if yy > 1e-16 {
                gamma = ys / yy;
            }
        }
        let mut rdir = vec![0.0; n];
        for j in 0..n {
            rdir[j] = gamma * q[j];
        }
        for i in 0..s_hist.len() {
            let mut dot = 0.0;
            for j in 0..n {
                dot += y_hist[i][j] * rdir[j];
            }
            let beta = rho_hist[i] * dot;
            for j in 0..n {
                rdir[j] += s_hist[i][j] * (alpha_coef[i] - beta);
            }
        }
        let mut p = vec![0.0; n];
        for j in 0..n {
            p[j] = -rdir[j];
        }
        // Zero components that point outside at bounds
        for j in 0..n {
            if x[j] <= bounds[j].0 && p[j] < 0.0 {
                p[j] = 0.0;
            }
            if x[j] >= bounds[j].1 && p[j] > 0.0 {
                p[j] = 0.0;
            }
        }

        let mut gtp = 0.0;
        for j in 0..n {
            gtp += g[j] * p[j];
        }
        if gtp >= 0.0 {
            for j in 0..n {
                p[j] = -g[j];
                if x[j] <= bounds[j].0 && p[j] < 0.0 {
                    p[j] = 0.0;
                }
                if x[j] >= bounds[j].1 && p[j] > 0.0 {
                    p[j] = 0.0;
                }
            }
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
        }

        let x_old = x.clone();
        let g_old = g.clone();
        let fx_old = fx;
        let mut step = 1.0;
        let c1 = 1e-4;
        let mut accepted = false;
        for _ in 0..30 {
            for j in 0..n {
                x[j] = x_old[j] + step * p[j];
            }
            project_bounds(&mut x, bounds);
            fx = f(&x);
            nfev += 1;
            // Armijo using actual displacement (handles projection)
            let mut gdx = 0.0;
            for j in 0..n {
                gdx += g_old[j] * (x[j] - x_old[j]);
            }
            if fx <= fx_old + c1 * gdx {
                accepted = true;
                break;
            }
            // Also accept plain decrease if directional estimate is unreliable
            if fx < fx_old && step < 1e-8 {
                accepted = true;
                break;
            }
            step *= 0.5;
            if step < 1e-20 {
                break;
            }
        }
        if !accepted || (x.iter().zip(x_old.iter()).all(|(a, b)| (a - b).abs() < 1e-16)) {
            // Tiny step: try pure projected steepest descent once
            s_hist.clear();
            y_hist.clear();
            rho_hist.clear();
            step = 1e-3;
            for j in 0..n {
                x[j] = x_old[j] - step * g_old[j];
            }
            project_bounds(&mut x, bounds);
            fx = f(&x);
            nfev += 1;
            if fx >= fx_old {
                x = x_old;
                fx = fx_old;
                return OptimizeResult {
                    x,
                    fun: fx,
                    success: fx < 1e-8,
                    nit,
                    nfev,
                };
            }
        }

        g = grad(&x);
        nit += 1;

        let mut s_vec = vec![0.0; n];
        let mut y_vec = vec![0.0; n];
        let mut ys = 0.0;
        for j in 0..n {
            s_vec[j] = x[j] - x_old[j];
            y_vec[j] = g[j] - g_old[j];
            ys += y_vec[j] * s_vec[j];
        }
        if ys > 1e-16 {
            if s_hist.len() == m_hist {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
            s_hist.push(s_vec);
            y_hist.push(y_vec);
            rho_hist.push(1.0 / ys);
        }
    }
}

/// `scipy.optimize.least_squares` via Levenberg–Marquardt (unbounded).
///
/// `resid(x) -> r` (m,), `jac(x) -> J` row-major flattened m*n or as m vectors of n.
/// Here `jac` returns row-major `m * n` values.
pub fn least_squares<R, J>(
    mut resid: R,
    mut jac: J,
    x0: &[f64],
    m_resid: usize,
    maxiter: usize,
    ftol: f64,
    xtol: f64,
    gtol: f64,
) -> OptimizeResult
where
    R: FnMut(&[f64]) -> Vec<f64>,
    J: FnMut(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = resid(&x);
    assert_eq!(r.len(), m_resid);
    let mut nfev = 1usize;
    let mut cost = 0.5 * r.iter().map(|v| v * v).sum::<f64>();
    let mut lambda = 1e-3;
    let mut nit = 0usize;

    loop {
        let jflat = jac(&x);
        assert_eq!(jflat.len(), m_resid * n);

        // g = J^T r
        let mut g = vec![0.0; n];
        for i in 0..m_resid {
            for j in 0..n {
                g[j] += jflat[i * n + j] * r[i];
            }
        }
        let gnorm = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if gnorm < gtol {
            return OptimizeResult {
                x,
                fun: cost,
                success: true,
                nit,
                nfev,
            };
        }
        if nit >= maxiter {
            return OptimizeResult {
                x,
                fun: cost,
                success: false,
                nit,
                nfev,
            };
        }

        // Solve (J^T J + lambda I) p = -J^T r
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let mut s = 0.0;
                for k in 0..m_resid {
                    s += jflat[k * n + i] * jflat[k * n + j];
                }
                a[i * n + j] = s;
                a[j * n + i] = s;
            }
            a[i * n + i] += lambda;
        }
        let mut rhs = vec![0.0; n];
        for j in 0..n {
            rhs[j] = -g[j];
        }
        // Gaussian elimination
        let p = solve_dense(n, &mut a, &rhs);

        let mut x_new = vec![0.0; n];
        for j in 0..n {
            x_new[j] = x[j] + p[j];
        }
        let r_new = resid(&x_new);
        nfev += 1;
        let cost_new = 0.5 * r_new.iter().map(|v| v * v).sum::<f64>();

        let mut dx: f64 = 0.0;
        for j in 0..n {
            dx = dx.max(p[j].abs());
        }
        let f_rel = if cost > 0.0 {
            ((cost - cost_new) / cost).abs()
        } else {
            0.0
        };

        if cost_new < cost {
            // Accept
            let gain = cost - cost_new;
            let _ = gain;
            x = x_new;
            r = r_new;
            cost = cost_new;
            lambda = (lambda * 0.3).max(1e-12);
            nit += 1;
            if f_rel < ftol || dx < xtol {
                return OptimizeResult {
                    x,
                    fun: cost,
                    success: true,
                    nit,
                    nfev,
                };
            }
        } else {
            lambda = (lambda * 10.0).min(1e12);
            nit += 1;
        }
    }
}

fn solve_dense(n: usize, a: &mut [f64], b: &[f64]) -> Vec<f64> {
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[col * (n + 1) + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * (n + 1) + col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-18 {
            // Singular — return zero step
            return vec![0.0; n];
        }
        if pivot != col {
            for j in 0..=n {
                aug.swap(col * (n + 1) + j, pivot * (n + 1) + j);
            }
        }
        let diag = aug[col * (n + 1) + col];
        for r in (col + 1)..n {
            let factor = aug[r * (n + 1) + col] / diag;
            for j in col..=n {
                aug[r * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            s -= aug[i * (n + 1) + j] * x[j];
        }
        x[i] = s / aug[i * (n + 1) + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        let d = (a - b).abs();
        assert!(d <= eps, "|{a} - {b}| = {d} > {eps}");
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        (1.0 - a).powi(2) + 100.0 * (b - a * a).powi(2)
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        let (a, b) = (x[0], x[1]);
        vec![
            -2.0 * (1.0 - a) - 400.0 * a * (b - a * a),
            200.0 * (b - a * a),
        ]
    }

    #[test]
    fn nelder_mead_rosenbrock() {
        let r = minimize_nelder_mead(rosenbrock, &[-1.2, 1.0], 2000, 1e-8, 1e-8);
        assert!(r.success);
        assert_close(r.x[0], 1.0, 1e-4);
        assert_close(r.x[1], 1.0, 1e-4);
    }

    #[test]
    fn lbfgsb_rosenbrock() {
        let bounds = [(-2.0, 2.0), (-2.0, 2.0)];
        let r = minimize_lbfgsb(
            rosenbrock,
            rosenbrock_grad,
            &[-1.2, 1.0],
            &bounds,
            2000,
            10,
            1e-6,
        );
        assert!(
            r.x[0] > 0.99 && r.x[1] > 0.99 && r.fun < 1e-6,
            "got x={:?} fun={} success={} nit={}",
            r.x,
            r.fun,
            r.success,
            r.nit
        );
    }

    #[test]
    fn least_squares_linear() {
        // Fit y = 2x + 3 for points x=0,1,2 → residual r_i = (a*x_i + b) - y_i
        let xs = [0.0, 1.0, 2.0];
        let ys = [3.0, 5.0, 7.0];
        let resid = |p: &[f64]| {
            xs.iter()
                .zip(ys.iter())
                .map(|(&x, &y)| p[0] * x + p[1] - y)
                .collect::<Vec<_>>()
        };
        let jac = |_p: &[f64]| {
            let mut j = Vec::with_capacity(6);
            for &x in &xs {
                j.push(x);
                j.push(1.0);
            }
            j
        };
        let r = least_squares(resid, jac, &[0.0, 0.0], 3, 50, 1e-12, 1e-12, 1e-12);
        assert!(r.success);
        assert_close(r.x[0], 2.0, 1e-6);
        assert_close(r.x[1], 3.0, 1e-6);
    }
}
