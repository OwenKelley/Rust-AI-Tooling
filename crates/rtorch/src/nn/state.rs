//! `state_dict` / `load_state_dict` helpers.

use crate::tensor::Tensor;

/// Ordered parameter snapshot: name → contiguous f32 buffer.
pub type StateDict = Vec<(String, Vec<f32>)>;

/// Collect parameter tensors into a state dict (copies data).
pub fn state_dict(named: &[(&str, &Tensor)]) -> StateDict {
    named
        .iter()
        .map(|(name, t)| ((*name).to_string(), t.data()))
        .collect()
}

/// Copy buffers from `sd` into matching named parameter tensors.
pub fn load_state_dict(named: &[(&str, &Tensor)], sd: &StateDict) {
    for (name, data) in sd {
        let (_, t) = named
            .iter()
            .find(|(n, _)| *n == name.as_str())
            .unwrap_or_else(|| panic!("load_state_dict: missing key '{name}'"));
        assert_eq!(
            t.numel(),
            data.len(),
            "load_state_dict: size mismatch for '{name}'"
        );
        t.inner.borrow_mut().data.copy_from_slice(data);
    }
}

/// Checksum of all values in a state dict (parity harness).
pub fn state_dict_checksum(sd: &StateDict) -> f64 {
    let mut acc = 0.0f64;
    for (_, data) in sd {
        for &v in data {
            if v.is_finite() {
                acc += v as f64;
            }
        }
    }
    acc
}
