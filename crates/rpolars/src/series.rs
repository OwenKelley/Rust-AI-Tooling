//! Named column — mirrors `polars.Series`.

use rarrow::{
    Array, BooleanArray, Float64Array, Int64Array, StringArray,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: String,
    pub data: Array,
}

impl Series {
    pub fn new(name: impl Into<String>, data: Array) -> Self {
        Self {
            name: name.into(),
            data,
        }
    }

    pub fn from_f64(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self::new(name, Array::Float64(Float64Array::from_slice(&values)))
    }

    pub fn from_i64(name: impl Into<String>, values: Vec<Option<i64>>) -> Self {
        Self::new(name, Array::Int64(Int64Array::from_nullable(&values)))
    }

    pub fn from_bool(name: impl Into<String>, values: Vec<bool>) -> Self {
        Self::new(name, Array::Boolean(BooleanArray::from_slice(&values)))
    }

    pub fn from_utf8(name: impl Into<String>, values: Vec<Option<String>>) -> Self {
        Self::new(name, Array::Utf8(StringArray::from_vec(values)))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn rename(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn checksum(&self) -> f64 {
        self.name.len() as f64 + self.data.checksum()
    }

    pub fn take(&self, indices: &[usize]) -> Series {
        Series::new(self.name.clone(), take_array(&self.data, indices))
    }

    pub fn slice(&self, offset: usize, length: usize) -> Series {
        let end = (offset + length).min(self.len());
        let start = offset.min(self.len());
        let indices: Vec<usize> = (start..end).collect();
        self.take(&indices)
    }
}

pub(crate) fn take_array(data: &Array, indices: &[usize]) -> Array {
    match data {
        Array::Float64(a) => Array::Float64(Float64Array {
            values: indices.iter().map(|&i| a.values[i]).collect(),
            nulls: indices.iter().map(|&i| a.nulls[i]).collect(),
        }),
        Array::Int64(a) => Array::Int64(Int64Array {
            values: indices.iter().map(|&i| a.values[i]).collect(),
            nulls: indices.iter().map(|&i| a.nulls[i]).collect(),
        }),
        Array::Boolean(a) => Array::Boolean(BooleanArray {
            values: indices.iter().map(|&i| a.values[i]).collect(),
            nulls: indices.iter().map(|&i| a.nulls[i]).collect(),
        }),
        Array::Utf8(a) => Array::Utf8(StringArray {
            values: indices.iter().map(|&i| a.values[i].clone()).collect(),
        }),
        Array::TimestampNs(a) => Array::TimestampNs(Int64Array {
            values: indices.iter().map(|&i| a.values[i]).collect(),
            nulls: indices.iter().map(|&i| a.nulls[i]).collect(),
        }),
        Array::ListFloat64(a) => {
            use rarrow::ListFloat64Array;
            let mut offsets = vec![0i32];
            let mut values = Vec::new();
            let mut nulls = Vec::new();
            for &i in indices {
                nulls.push(a.nulls[i]);
                let start = a.offsets[i] as usize;
                let end = a.offsets[i + 1] as usize;
                values.extend_from_slice(&a.values[start..end]);
                offsets.push(values.len() as i32);
            }
            Array::ListFloat64(ListFloat64Array {
                offsets,
                values,
                nulls,
            })
        }
        Array::DictionaryUtf8(a) => Array::DictionaryUtf8(rarrow::DictionaryUtf8Array {
            indices: indices.iter().map(|&i| a.indices[i]).collect(),
            nulls: indices.iter().map(|&i| a.nulls[i]).collect(),
            dictionary: a.dictionary.clone(),
        }),
    }
}

