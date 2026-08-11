//! Tensor dtypes (`torch.dtype`) — Float32/Float64 on f32 buffers; Int64/Bool use typed storage.

/// `torch.dtype` analogue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum Dtype {
    #[default]
    Float32,
    Float64,
    Int64,
    Bool,
}

impl Dtype {
    pub fn is_floating_point(self) -> bool {
        matches!(self, Dtype::Float32 | Dtype::Float64)
    }

    pub fn type_str(self) -> &'static str {
        match self {
            Dtype::Float32 => "float32",
            Dtype::Float64 => "float64",
            Dtype::Int64 => "int64",
            Dtype::Bool => "bool",
        }
    }
}

impl std::fmt::Display for Dtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.type_str())
    }
}

/// Cast element values between dtypes when both sides use f32-encoded values
/// (Float32/Float64 tagging, or legacy f32 encodings). Typed Int64/Bool casts
/// go through `Tensor::to_dtype` instead.
pub fn cast_f32_data(data: &[f32], from: Dtype, to: Dtype) -> Vec<f32> {
    if from == to {
        return data.to_vec();
    }
    data.iter()
        .map(|&x| cast_scalar(x, from, to))
        .collect()
}

fn cast_scalar(x: f32, from: Dtype, to: Dtype) -> f32 {
    // Normalize source interpretation, then emit destination encoding.
    let as_f = match from {
        Dtype::Float32 | Dtype::Float64 => x,
        Dtype::Int64 => (x as i64) as f32,
        Dtype::Bool => {
            if x != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    };
    match to {
        Dtype::Float32 | Dtype::Float64 => as_f,
        Dtype::Int64 => (as_f as i64) as f32,
        Dtype::Bool => {
            if as_f != 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}
