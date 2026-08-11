//! `nn.Sequential` / `ModuleList`.

use crate::nn::Module;
use crate::tensor::Tensor;

/// Ordered list of modules, applied in sequence.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Self { layers }
    }

    pub fn push(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer);
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = Vec::new();
        for layer in &self.layers {
            ps.extend(layer.parameters());
        }
        ps
    }
}

/// Flat list of modules (parameters collected; no automatic forward).
pub struct ModuleList {
    pub modules: Vec<Box<dyn Module>>,
}

impl ModuleList {
    pub fn new(modules: Vec<Box<dyn Module>>) -> Self {
        Self { modules }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut ps = Vec::new();
        for m in &self.modules {
            ps.extend(m.parameters());
        }
        ps
    }
}
