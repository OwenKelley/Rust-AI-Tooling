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
            for (w, gw) in inner.data.iter_mut().zip(g.iter()) {
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
                inner.data[j] -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }
        }
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
            for j in 0..g.len() {
                // Decoupled weight decay (PyTorch AdamW default).
                inner.data[j] -= self.lr * self.weight_decay * inner.data[j];
                mi[j] = self.beta1 * mi[j] + (1.0 - self.beta1) * g[j];
                vi[j] = self.beta2 * vi[j] + (1.0 - self.beta2) * g[j] * g[j];
                let mhat = mi[j] / b1t;
                let vhat = vi[j] / b2t;
                inner.data[j] -= self.lr * mhat / (vhat.sqrt() + self.eps);
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
        Self {
            step_size,
            gamma,
            last_epoch: 0,
            lr,
            initial_lr,
        }
    }

    /// Match PyTorch `StepLR`: after the Nth call, `lr = initial * gamma ** ((N-1) // step_size)`.
    pub fn step(&mut self) {
        let factor = self.gamma.powi((self.last_epoch / self.step_size) as i32);
        *self.lr = self.initial_lr * factor;
        self.last_epoch += 1;
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

    /// Match PyTorch closed form: `lr = initial * gamma ** bisect_right(milestones, epoch)`.
    pub fn step(&mut self) {
        let n = self
            .milestones
            .iter()
            .filter(|&&m| m <= self.last_epoch)
            .count();
        *self.lr = self.initial_lr * self.gamma.powi(n as i32);
        self.last_epoch += 1;
    }
}
