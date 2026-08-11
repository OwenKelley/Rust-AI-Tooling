//! Optimizers — SGD / Adam / AdamW and StepLR.

use crate::tensor::Tensor;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn adam_update_avx2(
    d: *mut f32,
    mi: *mut f32,
    vi: *mut f32,
    g: *mut f32,
    n: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    b1t: f32,
    b2t: f32,
    zero_grad: bool,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let v_beta1 = _mm256_set1_ps(beta1);
    let v_beta2 = _mm256_set1_ps(beta2);
    let v_omb1 = _mm256_set1_ps(1.0 - beta1);
    let v_omb2 = _mm256_set1_ps(1.0 - beta2);
    let v_lr = _mm256_set1_ps(lr);
    let v_eps = _mm256_set1_ps(eps);
    let v_b1t = _mm256_set1_ps(b1t);
    let v_b2t = _mm256_set1_ps(b2t);
    let v_zero = _mm256_setzero_ps();

    let mut j = 0usize;
    while j + 8 <= n {
        let gv = _mm256_loadu_ps(g.add(j));
        let mut mv = _mm256_loadu_ps(mi.add(j));
        let mut vv = _mm256_loadu_ps(vi.add(j));
        mv = _mm256_fmadd_ps(v_omb1, gv, _mm256_mul_ps(v_beta1, mv));
        vv = _mm256_fmadd_ps(v_omb2, _mm256_mul_ps(gv, gv), _mm256_mul_ps(v_beta2, vv));
        _mm256_storeu_ps(mi.add(j), mv);
        _mm256_storeu_ps(vi.add(j), vv);
        let mhat = _mm256_div_ps(mv, v_b1t);
        let vhat = _mm256_div_ps(vv, v_b2t);
        let denom = _mm256_add_ps(_mm256_sqrt_ps(vhat), v_eps);
        let step = _mm256_div_ps(_mm256_mul_ps(v_lr, mhat), denom);
        let dv = _mm256_loadu_ps(d.add(j));
        _mm256_storeu_ps(d.add(j), _mm256_sub_ps(dv, step));
        if zero_grad {
            _mm256_storeu_ps(g.add(j), v_zero);
        }
        j += 8;
    }
    while j < n {
        let gj = *g.add(j);
        let miv = beta1 * *mi.add(j) + (1.0 - beta1) * gj;
        let viv = beta2 * *vi.add(j) + (1.0 - beta2) * gj * gj;
        *mi.add(j) = miv;
        *vi.add(j) = viv;
        let mhat = miv / b1t;
        let vhat = viv / b2t;
        *d.add(j) -= lr * mhat / (vhat.sqrt() + eps);
        if zero_grad {
            *g.add(j) = 0.0;
        }
        j += 1;
    }
}

#[inline]
unsafe fn adam_update_ptrs(
    d: *mut f32,
    mi: *mut f32,
    vi: *mut f32,
    g: *mut f32,
    n: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    b1t: f32,
    b2t: f32,
    zero_grad: bool,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if n >= 8 && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            adam_update_avx2(d, mi, vi, g, n, lr, beta1, beta2, eps, b1t, b2t, zero_grad);
            return;
        }
    }
    let mut j = 0usize;
    while j + 8 <= n {
        for t in 0..8 {
            let i = j + t;
            let gj = *g.add(i);
            let miv = beta1 * *mi.add(i) + (1.0 - beta1) * gj;
            let viv = beta2 * *vi.add(i) + (1.0 - beta2) * gj * gj;
            *mi.add(i) = miv;
            *vi.add(i) = viv;
            let mhat = miv / b1t;
            let vhat = viv / b2t;
            *d.add(i) -= lr * mhat / (vhat.sqrt() + eps);
            if zero_grad {
                *g.add(i) = 0.0;
            }
        }
        j += 8;
    }
    while j < n {
        let gj = *g.add(j);
        let miv = beta1 * *mi.add(j) + (1.0 - beta1) * gj;
        let viv = beta2 * *vi.add(j) + (1.0 - beta2) * gj * gj;
        *mi.add(j) = miv;
        *vi.add(j) = viv;
        let mhat = miv / b1t;
        let vhat = viv / b2t;
        *d.add(j) -= lr * mhat / (vhat.sqrt() + eps);
        if zero_grad {
            *g.add(j) = 0.0;
        }
        j += 1;
    }
}

/// `torch.optim.SGD(params, lr=...)` without momentum.
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    pub fn step(&self) {
        for p in &self.params {
            let mut inner = p.inner.borrow_mut();
            let g_ptr = match &inner.grad {
                Some(g) => g.as_ptr(),
                None => continue,
            };
            let n = inner.grad.as_ref().unwrap().len();
            {
                let mut d = inner.data_mut_dense();
                for j in 0..n {
                    unsafe {
                        *d.get_unchecked_mut(j) -= self.lr * *g_ptr.add(j);
                    }
                }
            }
        }
    }

    pub fn step_and_zero_grad(&self) {
        self.step();
        self.zero_grad();
    }
}

/// `torch.optim.Adam(params, lr=..., betas=(0.9, 0.999), eps=1e-8)`.
pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self::with_betas(params, lr, 0.9, 0.999, 1e-8)
    }

    pub fn with_betas(params: Vec<Tensor>, lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        Self {
            params,
            lr,
            beta1,
            beta2,
            eps,
            t: 0,
            m,
            v,
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn step_inner(&mut self, zero_grad: bool) {
        self.t += 1;
        let t = self.t as f32;
        let b1t = 1.0 - self.beta1.powf(t);
        let b2t = 1.0 - self.beta2.powf(t);
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        const PARALLEL_ELEMS: usize = 16_384;
        for (i, p) in self.params.iter().enumerate() {
            let mut inner = p.inner.borrow_mut();
            let (g_ptr, n) = match &mut inner.grad {
                Some(g) => (g.as_mut_ptr(), g.len()),
                None => continue,
            };
            let mi = self.m[i].as_mut_ptr();
            let vi = self.v[i].as_mut_ptr();
            let mut d = inner.data_mut_dense();
            debug_assert_eq!(d.len(), n);
            let d_ptr = d.as_mut_ptr();
            #[cfg(feature = "parallel")]
            {
                if n >= PARALLEL_ELEMS {
                    use rayon::prelude::*;
                    let chunk = (n / rayon::current_num_threads().max(1)).max(4096);
                    let ranges: Vec<(usize, usize)> = {
                        let mut v = Vec::new();
                        let mut s = 0usize;
                        while s < n {
                            let e = (s + chunk).min(n);
                            v.push((s, e - s));
                            s = e;
                        }
                        v
                    };
                    let d_addr = d_ptr as usize;
                    let mi_addr = mi as usize;
                    let vi_addr = vi as usize;
                    let g_addr = g_ptr as usize;
                    ranges.into_par_iter().for_each(|(off, len)| unsafe {
                        adam_update_ptrs(
                            (d_addr as *mut f32).add(off),
                            (mi_addr as *mut f32).add(off),
                            (vi_addr as *mut f32).add(off),
                            (g_addr as *mut f32).add(off),
                            len,
                            lr,
                            beta1,
                            beta2,
                            eps,
                            b1t,
                            b2t,
                            zero_grad,
                        );
                    });
                    continue;
                }
            }
            unsafe {
                adam_update_ptrs(
                    d_ptr, mi, vi, g_ptr, n, lr, beta1, beta2, eps, b1t, b2t, zero_grad,
                );
            }
        }
    }

    pub fn step(&mut self) {
        self.step_inner(false);
    }

    /// Adam update then clear parameter grads in the same pass (avoids a second fill).
    pub fn step_and_zero_grad(&mut self) {
        self.step_inner(true);
    }

    /// Snapshot of Adam hyperparams + moment buffers.
    pub fn state_dict(&self) -> AdamStateDict {
        AdamStateDict {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            step: self.t,
            exp_avg: self.m.clone(),
            exp_avg_sq: self.v.clone(),
            params: self
                .params
                .iter()
                .map(|p| p.inner.borrow().dense_data())
                .collect(),
        }
    }

    pub fn load_state_dict(&mut self, sd: &AdamStateDict) {
        assert_eq!(sd.exp_avg.len(), self.params.len());
        assert_eq!(sd.exp_avg_sq.len(), self.params.len());
        assert_eq!(sd.params.len(), self.params.len());
        self.lr = sd.lr;
        self.beta1 = sd.beta1;
        self.beta2 = sd.beta2;
        self.eps = sd.eps;
        self.t = sd.step;
        self.m = sd.exp_avg.clone();
        self.v = sd.exp_avg_sq.clone();
        for (p, data) in self.params.iter().zip(sd.params.iter()) {
            assert_eq!(p.numel(), data.len());
            p.inner.borrow_mut().data_mut_dense().copy_from_slice(data);
        }
    }
}

/// Serializable Adam optimizer state.
#[derive(Clone, Debug)]
pub struct AdamStateDict {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub step: u64,
    pub exp_avg: Vec<Vec<f32>>,
    pub exp_avg_sq: Vec<Vec<f32>>,
    pub params: Vec<Vec<f32>>,
}

/// `torch.optim.AdamW` — Adam with decoupled weight decay.
pub struct AdamW {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: u64,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

impl AdamW {
    pub fn new(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            m,
            v,
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    fn step_inner(&mut self, zero_grad: bool) {
        self.t += 1;
        let t = self.t as f32;
        let b1t = 1.0 - self.beta1.powf(t);
        let b2t = 1.0 - self.beta2.powf(t);
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;
        for (i, p) in self.params.iter().enumerate() {
            let mut inner = p.inner.borrow_mut();
            let (g_ptr, n) = match &mut inner.grad {
                Some(g) => (g.as_mut_ptr(), g.len()),
                None => continue,
            };
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            {
                let mut d = inner.data_mut_dense();
                for w in d.iter_mut() {
                    *w -= lr * wd * *w;
                }
                unsafe {
                    adam_update_ptrs(
                        d.as_mut_ptr(),
                        mi.as_mut_ptr(),
                        vi.as_mut_ptr(),
                        g_ptr,
                        n,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        b1t,
                        b2t,
                        zero_grad,
                    );
                }
            }
        }
    }

    pub fn step(&mut self) {
        self.step_inner(false);
    }

    pub fn step_and_zero_grad(&mut self) {
        self.step_inner(true);
    }
}

/// `torch.optim.lr_scheduler.StepLR` — decay `lr` by `gamma` every `step_size` steps.
pub struct StepLR<'a> {
    pub step_size: usize,
    pub gamma: f32,
    pub last_epoch: usize,
    lr: &'a mut f32,
    initial_lr: f32,
}

impl<'a> StepLR<'a> {
    pub fn new(lr: &'a mut f32, step_size: usize, gamma: f32) -> Self {
        let initial_lr = *lr;
        Self {
            step_size,
            gamma,
            last_epoch: 0,
            lr,
            initial_lr,
        }
    }

    pub fn step(&mut self) {
        self.last_epoch += 1;
        let factor = self.gamma.powi((self.last_epoch / self.step_size) as i32);
        *self.lr = self.initial_lr * factor;
    }
}

/// `torch.optim.lr_scheduler.MultiStepLR`.
pub struct MultiStepLR<'a> {
    pub milestones: Vec<usize>,
    pub gamma: f32,
    pub last_epoch: usize,
    lr: &'a mut f32,
    initial_lr: f32,
}

impl<'a> MultiStepLR<'a> {
    pub fn new(lr: &'a mut f32, milestones: Vec<usize>, gamma: f32) -> Self {
        let initial_lr = *lr;
        Self {
            milestones,
            gamma,
            last_epoch: 0,
            lr,
            initial_lr,
        }
    }

    pub fn step(&mut self) {
        self.last_epoch += 1;
        let n = self
            .milestones
            .iter()
            .filter(|&&m| m <= self.last_epoch)
            .count();
        *self.lr = self.initial_lr * self.gamma.powi(n as i32);
    }
}

/// `torch.optim.lr_scheduler.CosineAnnealingLR`.
pub struct CosineAnnealingLR<'a> {
    pub t_max: usize,
    pub eta_min: f32,
    pub last_epoch: usize,
    lr: &'a mut f32,
    initial_lr: f32,
}

impl<'a> CosineAnnealingLR<'a> {
    pub fn new(lr: &'a mut f32, t_max: usize, eta_min: f32) -> Self {
        let initial_lr = *lr;
        Self {
            t_max,
            eta_min,
            last_epoch: 0,
            lr,
            initial_lr,
        }
    }

    pub fn step(&mut self) {
        self.last_epoch += 1;
        let t = self.last_epoch as f32;
        let t_max = self.t_max.max(1) as f32;
        *self.lr = self.eta_min
            + (self.initial_lr - self.eta_min)
                * (1.0 + (std::f32::consts::PI * t / t_max).cos())
                * 0.5;
    }
}
