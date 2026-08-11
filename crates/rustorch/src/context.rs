//! Grad mode (`torch.is_grad_enabled` / `torch.no_grad`).

use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|c| c.set(enabled));
}

/// RAII guard restoring previous grad-enabled flag on drop.
pub struct NoGradGuard {
    prev: bool,
}

impl NoGradGuard {
    pub fn new() -> Self {
        let prev = is_grad_enabled();
        set_grad_enabled(false);
        Self { prev }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_grad_enabled(self.prev);
    }
}

/// `with torch.no_grad(): ...`
pub fn no_grad<R>(f: impl FnOnce() -> R) -> R {
    let _guard = NoGradGuard::new();
    f()
}
