//! AMP scaffolding (`torch.cuda.amp`) — CPU f32 only in v1 (no FP16 kernels).

use std::cell::Cell;

use crate::ops::{full, mul};
use crate::optim::{Adam, SGD};
use crate::tensor::Tensor;

thread_local! {
    static AUTOCAST: Cell<bool> = const { Cell::new(false) };
}

/// Whether the current thread is inside [`autocast`].
pub fn is_autocast_enabled() -> bool {
    AUTOCAST.with(|c| c.get())
}

fn set_autocast(enabled: bool) {
    AUTOCAST.with(|c| c.set(enabled));
}

/// RAII guard for [`autocast`].
pub struct AutocastGuard {
    prev: bool,
}

impl AutocastGuard {
    pub fn new(enabled: bool) -> Self {
        let prev = is_autocast_enabled();
        set_autocast(enabled);
        Self { prev }
    }
}

impl Drop for AutocastGuard {
    fn drop(&mut self) {
        set_autocast(self.prev);
    }
}

/// `with torch.autocast(device_type='cpu'|'cuda', enabled=...):`
///
/// v1 still computes in f32; the flag is for API shape / future FP16 kernels.
pub fn autocast<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let _guard = AutocastGuard::new(enabled);
    f()
}

/// `torch.cuda.amp.GradScaler` (CPU-friendly scale/unscale of float grads).
pub struct GradScaler {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: u64,
    growth_tracker: u64,
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl GradScaler {
    pub fn new() -> Self {
        Self {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_tracker: 0,
        }
    }

    /// `scaler.scale(loss)` — multiply loss by the current scale.
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        let s = full(&[], self.scale, false);
        mul(loss, &s)
    }

    /// Divide parameter grads by `scale` (in place). Returns false if any non-finite grad.
    pub fn unscale_grads(&self, params: &[Tensor]) -> bool {
        let inv = 1.0 / self.scale;
        let mut ok = true;
        for p in params {
            let mut inner = p.inner.borrow_mut();
            if let Some(g) = inner.grad.as_mut() {
                for v in g.iter_mut() {
                    *v *= inv;
                    if !v.is_finite() {
                        ok = false;
                    }
                }
            }
        }
        ok
    }

    /// `scaler.step(optimizer)` for SGD (unscale → step if finite → update scale).
    pub fn step_sgd(&mut self, opt: &SGD) {
        let ok = self.unscale_grads(&opt.params);
        if ok {
            opt.step();
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
            }
        } else {
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
        }
    }

    /// `scaler.step(optimizer)` for Adam.
    pub fn step_adam(&mut self, opt: &mut Adam) {
        let ok = self.unscale_grads(&opt.params);
        if ok {
            opt.step();
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
            }
        } else {
            self.scale *= self.backoff_factor;
            self.growth_tracker = 0;
        }
    }

    /// `scaler.update()` — no-op when growth is handled in `step_*` (kept for API shape).
    pub fn update(&mut self) {}

    pub fn get_scale(&self) -> f32 {
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{seeded_uniform, sum};
    use crate::optim::SGD;

    #[test]
    fn scaler_scales_and_unscales() {
        let x = seeded_uniform(&[4], 1, -1.0, 1.0);
        x.set_requires_grad(true);
        let mut scaler = GradScaler::new();
        scaler.scale = 4.0;
        let y = scaler.scale_loss(&sum(&x));
        y.backward();
        let g_scaled = x.grad().unwrap();
        assert!((g_scaled[0] - 4.0).abs() < 1e-4);
        let opt = SGD::new(vec![x.clone()], 0.1);
        assert!(scaler.unscale_grads(&opt.params));
        let g = x.grad().unwrap();
        assert!((g[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn autocast_flag() {
        assert!(!is_autocast_enabled());
        autocast(true, || {
            assert!(is_autocast_enabled());
        });
        assert!(!is_autocast_enabled());
    }
}
