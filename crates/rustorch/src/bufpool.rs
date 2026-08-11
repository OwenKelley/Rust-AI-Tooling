//! Small thread-local freelist for temporary `f32` buffers (train hot paths).

use std::cell::RefCell;

thread_local! {
    static F32_POOL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
}

/// Take a buffer with at least `n` elements (length set to `n`, values uninitialized).
pub fn take_f32(n: usize) -> Vec<f32> {
    F32_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let mut buf = if let Some(i) = pool.iter().position(|v| v.capacity() >= n) {
            pool.swap_remove(i)
        } else {
            Vec::with_capacity(n.max(64))
        };
        buf.clear();
        if buf.capacity() < n {
            buf.reserve(n - buf.capacity());
        }
        unsafe {
            buf.set_len(n);
        }
        buf
    })
}

/// Return a buffer to the pool (capacity retained).
pub fn recycle_f32(mut buf: Vec<f32>) {
    if buf.capacity() == 0 {
        return;
    }
    buf.clear();
    F32_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if pool.len() < 32 {
            pool.push(buf);
        }
    });
}

/// Run `f` with a pooled scratch buffer of length `n`.
pub fn with_f32_buf<R>(n: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    let mut buf = take_f32(n);
    let out = f(&mut buf);
    recycle_f32(buf);
    out
}
