//! Dropout module.

use std::cell::Cell;

use crate::functional;
use crate::nn::Module;
use crate::tensor::Tensor;

/// `torch.nn.Dropout(p=0.5)`. Seeded RNG for parity when training.
pub struct Dropout {
    pub p: f32,
    pub train: Cell<bool>,
    pub seed: Cell<u64>,
}

impl Dropout {
    pub fn new(p: f32, seed: u64) -> Self {
        Self {
            p,
            train: Cell::new(true),
            seed: Cell::new(seed),
        }
    }

    pub fn train(&self) {
        self.train.set(true);
    }

    pub fn eval(&self) {
        self.train.set(false);
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Tensor {
        let seed = self.seed.get();
        // Advance seed each forward so successive calls differ (like torch RNG).
        self.seed.set(seed.wrapping_add(0x9E37_79B9));
        functional::dropout(input, self.p, self.train.get(), seed)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
