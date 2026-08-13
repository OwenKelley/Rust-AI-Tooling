//! Arrow-like arrays (PyArrow-shaped, contiguous buffers).

#[derive(Debug, Clone, PartialEq)]
pub enum Array {
    Float64(Float64Array),
    Int64(Int64Array),
    Boolean(BooleanArray),
    Utf8(StringArray),
    TimestampNs(Int64Array),
    ListFloat64(ListFloat64Array),
    DictionaryUtf8(DictionaryUtf8Array),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Float64Array {
    pub values: Vec<f64>,
    /// `true` = null
    pub nulls: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Int64Array {
    pub values: Vec<i64>,
    pub nulls: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BooleanArray {
    pub values: Vec<bool>,
    pub nulls: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StringArray {
    pub values: Vec<Option<String>>,
}

/// Variable-size list of f64 (offsets length = n+1).
#[derive(Debug, Clone, PartialEq)]
pub struct ListFloat64Array {
    pub offsets: Vec<i32>,
    pub values: Vec<f64>,
    pub nulls: Vec<bool>,
}

impl ListFloat64Array {
    pub fn from_slices(lists: &[Vec<f64>]) -> Self {
        let mut offsets = vec![0i32];
        let mut values = Vec::new();
        for l in lists {
            values.extend_from_slice(l);
            offsets.push(values.len() as i32);
        }
        Self {
            offsets,
            values,
            nulls: vec![false; lists.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn null_count(&self) -> usize {
        self.nulls.iter().filter(|&&n| n).count()
    }
}

/// Dictionary-encoded utf8: `indices[i]` indexes `dictionary` (nulls independent).
#[derive(Debug, Clone, PartialEq)]
pub struct DictionaryUtf8Array {
    pub indices: Vec<i32>,
    pub nulls: Vec<bool>,
    pub dictionary: Vec<String>,
}

impl DictionaryUtf8Array {
    pub fn from_values(values: &[Option<&str>]) -> Self {
        let mut dictionary = Vec::new();
        let mut map = std::collections::HashMap::new();
        let mut indices = Vec::with_capacity(values.len());
        let mut nulls = Vec::with_capacity(values.len());
        for v in values {
            match v {
                None => {
                    indices.push(0);
                    nulls.push(true);
                }
                Some(s) => {
                    let idx = *map.entry(s.to_string()).or_insert_with(|| {
                        let i = dictionary.len() as i32;
                        dictionary.push(s.to_string());
                        i
                    });
                    indices.push(idx);
                    nulls.push(false);
                }
            }
        }
        Self {
            indices,
            nulls,
            dictionary,
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn null_count(&self) -> usize {
        self.nulls.iter().filter(|&&n| n).count()
    }

    pub fn value(&self, i: usize) -> Option<&str> {
        if self.nulls[i] {
            None
        } else {
            Some(self.dictionary[self.indices[i] as usize].as_str())
        }
    }
}

impl Float64Array {
    pub fn from_slice(v: &[f64]) -> Self {
        Self {
            values: v.to_vec(),
            nulls: vec![false; v.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn null_count(&self) -> usize {
        self.nulls.iter().filter(|&&n| n).count()
    }
}

impl Int64Array {
    pub fn from_slice(v: &[i64]) -> Self {
        Self {
            values: v.to_vec(),
            nulls: vec![false; v.len()],
        }
    }

    pub fn from_nullable(v: &[Option<i64>]) -> Self {
        let mut values = Vec::with_capacity(v.len());
        let mut nulls = Vec::with_capacity(v.len());
        for x in v {
            match x {
                Some(n) => {
                    values.push(*n);
                    nulls.push(false);
                }
                None => {
                    values.push(0);
                    nulls.push(true);
                }
            }
        }
        Self { values, nulls }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn null_count(&self) -> usize {
        self.nulls.iter().filter(|&&n| n).count()
    }
}

impl BooleanArray {
    pub fn from_slice(v: &[bool]) -> Self {
        Self {
            values: v.to_vec(),
            nulls: vec![false; v.len()],
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn null_count(&self) -> usize {
        self.nulls.iter().filter(|&&n| n).count()
    }
}

impl StringArray {
    pub fn from_vec(v: Vec<Option<String>>) -> Self {
        Self { values: v }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn null_count(&self) -> usize {
        self.values.iter().filter(|v| v.is_none()).count()
    }
}

impl Array {
    pub fn len(&self) -> usize {
        match self {
            Self::Float64(a) => a.len(),
            Self::Int64(a) | Self::TimestampNs(a) => a.len(),
            Self::Boolean(a) => a.len(),
            Self::Utf8(a) => a.len(),
            Self::ListFloat64(a) => a.len(),
            Self::DictionaryUtf8(a) => a.len(),
        }
    }

    pub fn null_count(&self) -> usize {
        match self {
            Self::Float64(a) => a.null_count(),
            Self::Int64(a) | Self::TimestampNs(a) => a.null_count(),
            Self::Boolean(a) => a.null_count(),
            Self::Utf8(a) => a.null_count(),
            Self::ListFloat64(a) => a.null_count(),
            Self::DictionaryUtf8(a) => a.null_count(),
        }
    }

    /// Parity checksum: sum of non-null numeric / utf8-len codes.
    pub fn checksum(&self) -> f64 {
        match self {
            Self::Float64(a) => a
                .values
                .iter()
                .zip(a.nulls.iter())
                .map(|(&v, &n)| if n { 0.0 } else { v })
                .sum(),
            Self::Int64(a) | Self::TimestampNs(a) => a
                .values
                .iter()
                .zip(a.nulls.iter())
                .map(|(&v, &n)| if n { 0.0 } else { v as f64 })
                .sum(),
            Self::Boolean(a) => a
                .values
                .iter()
                .zip(a.nulls.iter())
                .map(|(&v, &n)| if n { 0.0 } else { f64::from(u8::from(v)) })
                .sum(),
            Self::Utf8(a) => a
                .values
                .iter()
                .map(|v| match v {
                    Some(s) => s.len() as f64 + s.bytes().map(|b| b as f64).sum::<f64>(),
                    None => 0.0,
                })
                .sum(),
            Self::ListFloat64(a) => {
                a.values.iter().sum::<f64>() + a.offsets.iter().map(|&o| o as f64).sum::<f64>()
            }
            Self::DictionaryUtf8(a) => {
                let dict_sum: f64 = a
                    .dictionary
                    .iter()
                    .map(|s| s.len() as f64 + s.bytes().map(|b| b as f64).sum::<f64>())
                    .sum();
                let idx_sum: f64 = a
                    .indices
                    .iter()
                    .zip(a.nulls.iter())
                    .map(|(&i, &n)| if n { 0.0 } else { i as f64 })
                    .sum();
                dict_sum + idx_sum
            }
        }
    }
}

/// Validity bitmap (LSB of first byte = index 0). Empty if `null_count == 0`.
pub fn validity_bitmap(nulls: &[bool]) -> Vec<u8> {
    if !nulls.iter().any(|&n| n) {
        return Vec::new();
    }
    let n_bytes = (nulls.len() + 7) / 8;
    let mut out = vec![0u8; n_bytes];
    for (i, &is_null) in nulls.iter().enumerate() {
        if !is_null {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

pub fn bit_is_set(bitmap: &[u8], i: usize) -> bool {
    if bitmap.is_empty() {
        return true;
    }
    (bitmap[i / 8] & (1 << (i % 8))) != 0
}
