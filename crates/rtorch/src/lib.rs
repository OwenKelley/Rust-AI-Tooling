//! `rtorch` — PyTorch-shaped tensor + autograd API for Rust (CPU f32).
//!
//! Local/`std` only. Names mirror PyTorch for parity harnesses.

pub mod autograd;
pub mod broadcast;
pub mod context;
pub mod functional;
pub mod gemm;
pub mod math_kernels;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod tensor;

pub use context::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard};
pub use functional::{
    cross_entropy, dropout, gelu, linear, log_softmax, mse_loss, relu, sigmoid, softmax, tanh,
};
pub use nn::{
    max_pool2d, BatchNorm1d, Conv2d, CrossEntropyLoss, Dropout, Embedding, Flatten, LayerNorm,
    Linear, MaxPool2d, Module, ModuleList, ReLU, Sequential, Sigmoid, Softmax, GELU, MSELoss, Tanh,
};
pub use ops::{
    abs, add, cat, clamp, div, exp, full, index_select, log, matmul, mean, mul, neg, ones, pow,
    randn, reshape, seeded_uniform, stack, sub, sum, transpose, zeros,
};
pub use optim::{Adam, AdamW, MultiStepLR, SGD, StepLR};
pub use tensor::Tensor;
