//! Device placement (`torch.device`) — CPU compute; CUDA is API-only in v1.

/// `torch.device` analogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Device {
    #[default]
    Cpu,
    /// Present for API parity. Moving tensors here panics (no CUDA runtime).
    Cuda,
}

impl Device {
    pub fn is_cuda(self) -> bool {
        matches!(self, Device::Cuda)
    }

    pub fn type_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.type_str())
    }
}
