//! `torch.nn` modules.

mod activation;
mod batchnorm;
mod container;
mod conv;
mod dropout;
mod embedding;
mod linear;
mod loss;
mod norm;
mod pool;

pub use activation::{ReLU, Sigmoid, Softmax, GELU, Tanh};
pub use batchnorm::BatchNorm1d;
pub use container::{ModuleList, Sequential};
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use loss::{CrossEntropyLoss, MSELoss};
pub use norm::LayerNorm;
pub use pool::{max_pool2d, Flatten, MaxPool2d};

use crate::tensor::Tensor;

/// Minimal `nn.Module` trait.
pub trait Module {
    fn forward(&self, input: &Tensor) -> Tensor;

    fn parameters(&self) -> Vec<Tensor>;

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}
