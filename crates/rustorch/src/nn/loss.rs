//! Loss modules.

use crate::functional;
use crate::tensor::Tensor;

pub struct MSELoss;

impl MSELoss {
    pub fn forward(&self, input: &Tensor, target: &Tensor) -> Tensor {
        functional::mse_loss(input, target)
    }
}

/// `torch.nn.CrossEntropyLoss` (mean reduction). Targets are class indices.
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn forward(&self, input: &Tensor, target: &[usize]) -> Tensor {
        functional::cross_entropy(input, target)
    }
}
