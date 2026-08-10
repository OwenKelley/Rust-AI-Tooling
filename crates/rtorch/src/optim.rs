//! Optimizers — SGD / Adam / AdamW and StepLR.

use crate::tensor::Tensor;

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
            let g = match inner.grad.clone() {
                Some(g) => g,
                None => continue,
            };
            let mut d = inner.data_mut_dense();
            for (w, gw) in d.iter_mut().zip(g.iter()) {
                *w -= self.lr * gw;
            }
        }
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

    pub fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;
        let b1t = 1.0 - self.beta1.powf(t);
        let b2t = 1.0 - self.beta2.powf(t);
        for (i, p) in self.params.iter().enumerate() {
            let mut inner = p.inner.borrow_mut();
            let g = match &inner.grad {
                Some(g) => g.clone(),
                None => continue,
            };
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            for j in 0..g.len() {
                mi[j] = self.beta1 * mi[j] + (1.0 - self.beta1) * g[j];
                vi[j] = self.beta2 * vi[j] + (1.0 - self.beta2) * g[j] * g[j];
                let mhat = mi[j] / b1t;
                let vhat = vi[j] / b2t;
                inner.data_mut_dense()[j] -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }
        }
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

impl AdamStateDict {
    pub fn checksum(&self) -> f64 {
        let mut acc = self.lr as f64
            + self.beta1 as f64
            + self.beta2 as f64
            + self.eps as f64
            + self.step as f64;
        for buf in self
            .exp_avg
            .iter()
            .chain(self.exp_avg_sq.iter())
            .chain(self.params.iter())
        {
            for &v in buf {
                if v.is_finite() {
                    acc += v as f64;
                }
            }
        }
        acc
    }
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

    pub fn step(&mut self) {
        self.t += 1;
        let t = self.t as f32;
        let b1t = 1.0 - self.beta1.powf(t);
        let b2t = 1.0 - self.beta2.powf(t);
        for (i, p) in self.params.iter().enumerate() {
            let mut inner = p.inner.borrow_mut();
            let g = match &inner.grad {
                Some(g) => g.clone(),
                None => continue,
            };
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            {
                let mut d = inner.data_mut_dense();
                for j in 0..g.len() {
                    // Decoupled weight decay (PyTorch AdamW default).
                    d[j] -= self.lr * self.weight_decay * d[j];
                    mi[j] = self.beta1 * mi[j] + (1.0 - self.beta1) * g[j];
                    vi[j] = self.beta2 * vi[j] + (1.0 - self.beta2) * g[j] * g[j];
                    let mhat = mi[j] / b1t;
                    let vhat = vi[j] / b2t;
                    d[j] -= self.lr * mhat / (vhat.sqrt() + self.eps);
                }
            }
        }
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
        // Match PyTorch: `__init__` calls `step()` once → `last_epoch == 0`.
        Self {
            step_size,
            gamma,
            last_epoch: 0,
            lr,
            initial_lr,
        }
    }

    /// Match PyTorch after construction: Nth user `step` uses epoch `N`.
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

    /// Match PyTorch after construction: Nth user `step` uses epoch `N`.
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
        // Match PyTorch: `__init__` calls `step()` once → `last_epoch == 0`, lr unchanged.
        Self {
            t_max,
            eta_min,
            last_epoch: 0,
            lr,
            initial_lr,
        }
    }

    /// Match PyTorch closed form after construction: Nth user `step` uses epoch `N`.
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
