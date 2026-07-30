//! `torch.nn` modules.

mod activation;
mod attention;
mod batchnorm;
mod container;
mod conv;
mod dropout;
mod embedding;
mod linear;
mod loss;
mod norm;
mod pool;
mod rnn;
pub mod state;
mod transformer;

pub use activation::{LeakyReLU, ReLU, Sigmoid, Softmax, GELU, SiLU, Tanh};
pub use attention::MultiheadAttention;
pub use batchnorm::{BatchNorm1d, BatchNorm2d};
pub use container::{ModuleList, Sequential};
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use loss::{CrossEntropyLoss, MSELoss};
pub use norm::LayerNorm;
pub use pool::{
    adaptive_avg_pool2d, avg_pool2d, max_pool2d, AdaptiveAvgPool2d, AvgPool2d, Flatten, MaxPool2d,
};
pub use rnn::{GRU, LSTM};
pub use state::{load_state_dict, state_dict, state_dict_checksum, StateDict};
pub use transformer::{
    generate_square_subsequent_mask, TransformerActivation, TransformerDecoder,
    TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer,
};

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
