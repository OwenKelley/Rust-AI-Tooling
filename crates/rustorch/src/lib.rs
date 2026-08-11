//! RusTorch (`rustorch`) — PyTorch-shaped tensor + autograd API for Rust (CPU f32).
//!
//! Portable by default (`matrixmultiply` GEMM). Enable the `parallel` feature
//! (on by default) for a Rayon thread pool on mid/large matmuls. Disable with
//! `--no-default-features` for minimal embeds.

pub mod amp;
pub mod autograd;
pub mod bridge;
pub mod broadcast;
pub mod bufpool;
pub mod context;
pub mod cpu_kernels;
pub mod data;
pub mod device;
pub mod dtype;
pub mod functional;
pub mod gemm;
pub mod math_kernels;
pub mod nested;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tensor;
pub mod tensor_ops;

pub use amp::{autocast, is_autocast_enabled, AutocastGuard, GradScaler};
pub use bridge::{
    from_dataframe, from_numpy, from_numpy_f32, from_numpy_f32_owned, to_dataframe, to_numpy,
    to_numpy_f32,
};
pub use context::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard};
pub use data::{default_collate, DataLoader, RandomSampler, SequentialSampler, TensorDataset};
pub use device::Device;
pub use dtype::Dtype;
pub use nested::{nested_tensor, NestedTensor};
pub use functional::{
    cross_entropy, dropout, fused_linear_relu, gelu, leaky_relu, linear, linear_cross_entropy,
    log_softmax, mse_loss, relu, scaled_dot_product_attention, scaled_dot_product_attention_masked,
    sigmoid, silu, softmax, tanh,
};
pub use autograd::{apply_function, grad, gradcheck_max_error, square_function, FunctionCtx};
pub use nn::{
    adaptive_avg_pool2d, avg_pool2d, generate_square_subsequent_mask, load_state_dict, max_pool2d,
    state_dict, state_dict_checksum, AdaptiveAvgPool2d, AvgPool2d, BatchNorm1d, BatchNorm2d, Conv2d,
    CrossEntropyLoss, Dropout, Embedding, Flatten, GRU, LSTM, LayerNorm, LeakyReLU, Linear,
    MaxPool2d, Module, ModuleList, MultiheadAttention, ReLU, Sequential, Sigmoid, Softmax,
    StateDict, TransformerActivation, TransformerDecoder, TransformerDecoderLayer,
    TransformerEncoder, TransformerEncoderLayer, GELU, MSELoss, SiLU, Tanh,
};
pub use ops::{
    abs, add, add_, bmm, cat, chunk, clamp, div, exp, fill_, full, gather_rows, index_select, log,
    matmul, mean, mul, mul_, narrow, neg, ones, permute, pow, randn, relu_, reshape, seeded_uniform,
    select, shuffle_rows_inplace, stack, sub, sub_, sum, transpose, zero_, zeros,
};
pub use optim::{Adam, AdamStateDict, AdamW, CosineAnnealingLR, MultiStepLR, SGD, StepLR};
pub use tensor::Tensor;
