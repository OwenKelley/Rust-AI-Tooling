//! RecordBatch / Table.

use crate::array::Array;
use crate::schema::{DataType, Field, Schema};

#[derive(Debug, Clone, PartialEq)]
pub struct RecordBatch {
    pub schema: Schema,
    pub columns: Vec<Array>,
}

impl RecordBatch {
    pub fn try_new(schema: Schema, columns: Vec<Array>) -> Self {
        assert_eq!(schema.fields.len(), columns.len(), "schema/columns mismatch");
        let n = columns.first().map(|c| c.len()).unwrap_or(0);
        for (i, c) in columns.iter().enumerate() {
            assert_eq!(c.len(), n, "column {i} length mismatch");
            assert_eq!(
                dtype_of(c),
                schema.fields[i].data_type,
                "column {i} dtype mismatch"
            );
        }
        Self { schema, columns }
    }

    pub fn num_rows(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }

    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    pub fn checksum(&self) -> f64 {
        let mut s = self.num_rows() as f64 + self.num_columns() as f64;
        for c in &self.columns {
            s += c.checksum();
        }
        s
    }
}

fn dtype_of(a: &Array) -> DataType {
    match a {
        Array::Float64(_) => DataType::Float64,
        Array::Int64(_) => DataType::Int64,
        Array::Boolean(_) => DataType::Boolean,
        Array::Utf8(_) => DataType::Utf8,
        Array::TimestampNs(_) => DataType::TimestampNs,
        Array::ListFloat64(_) => DataType::ListFloat64,
        Array::DictionaryUtf8(_) => DataType::DictionaryUtf8,
    }
}

/// Build a batch from named columns (infers schema; nullable if any nulls).
pub fn batch_from_columns(cols: Vec<(String, Array)>) -> RecordBatch {
    // Always mark fields nullable for IPC (matches pyarrow defaults / wider interop).
    let fields: Vec<Field> = cols
        .iter()
        .enumerate()
        .map(|(i, (n, a))| {
            let mut f = Field::new(n.clone(), dtype_of(a), true);
            if matches!(f.data_type, DataType::DictionaryUtf8) {
                f.dict_id = Some(i as i64);
            }
            f
        })
        .collect();
    let columns: Vec<Array> = cols.into_iter().map(|(_, a)| a).collect();
    RecordBatch::try_new(Schema::new(fields), columns)
}
