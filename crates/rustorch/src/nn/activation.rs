//! Activation modules.

use crate::functional;
use crate::nn::Module;
use crate::tensor::Tensor;

pub struct ReLU;

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::relu(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

pub struct LeakyReLU {
    pub negative_slope: f32,
}

impl LeakyReLU {
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::leaky_relu(input, self.negative_slope)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

pub struct Sigmoid;

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::sigmoid(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

/// Softmax along last dim (2D).
pub struct Softmax;

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::softmax(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

pub struct Tanh;

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::tanh(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

/// GELU with tanh approximation (`approximate='tanh'`).
pub struct GELU;

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::gelu(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

pub struct SiLU;

impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Tensor {
        functional::silu(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
