//! Convert between `rpandas::DataFrame` and `rarrow::RecordBatch`.

use rarrow::{
    Array, BooleanArray, DataType, Field, Float64Array, Int64Array, RecordBatch, Schema,
    StringArray,
};

use crate::frame::{Column, DataFrame};

/// `DataFrame` → Arrow `RecordBatch` (numeric / bool / utf8 columns).
pub fn dataframe_to_record_batch(df: &DataFrame) -> RecordBatch {
    let mut fields = Vec::new();
    let mut columns = Vec::new();
    for (name, col) in df.columns_ref() {
        match col {
            Column::Float64(a) => {
                let c = a.to_contiguous();
                let vals = c.as_slice().unwrap().to_vec();
                fields.push(Field::new(name.clone(), DataType::Float64, true));
                columns.push(Array::Float64(Float64Array {
                    values: vals,
                    nulls: vec![false; a.len()],
                }));
            }
            Column::Int64 { values, nulls } => {
                fields.push(Field::new(name.clone(), DataType::Int64, true));
                columns.push(Array::Int64(Int64Array {
                    values: values.clone(),
                    nulls: nulls.clone(),
                }));
            }
            Column::Bool { values, nulls } => {
                fields.push(Field::new(name.clone(), DataType::Boolean, true));
                columns.push(Array::Boolean(BooleanArray {
                    values: values.clone(),
                    nulls: nulls.clone(),
                }));
            }
            Column::Utf8 { values, nulls } => {
                fields.push(Field::new(name.clone(), DataType::Utf8, true));
                let vals: Vec<Option<String>> = values
                    .iter()
                    .zip(nulls.iter())
                    .map(|(s, &n)| if n { None } else { Some(s.clone()) })
                    .collect();
                columns.push(Array::Utf8(StringArray { values: vals }));
            }
        }
    }
    RecordBatch::try_new(Schema::new(fields), columns)
}

/// Arrow `RecordBatch` → `DataFrame`.
pub fn record_batch_to_dataframe(batch: &RecordBatch) -> DataFrame {
    let mut cols = Vec::new();
    for (field, col) in batch.schema.fields.iter().zip(batch.columns.iter()) {
        let c = match col {
            Array::Float64(a) => Column::Float64(rnumpy::NdArray::from_vec(a.values.clone())),
            Array::Int64(a) => Column::Int64 {
                values: a.values.clone(),
                nulls: a.nulls.clone(),
            },
            Array::Boolean(a) => Column::Bool {
                values: a.values.clone(),
                nulls: a.nulls.clone(),
            },
            Array::Utf8(a) => {
                let mut values = Vec::new();
                let mut nulls = Vec::new();
                for v in &a.values {
                    match v {
                        Some(s) => {
                            values.push(s.clone());
                            nulls.push(false);
                        }
                        None => {
                            values.push(String::new());
                            nulls.push(true);
                        }
                    }
                }
                Column::Utf8 { values, nulls }
            }
            Array::TimestampNs(a) => Column::Int64 {
                values: a.values.clone(),
                nulls: a.nulls.clone(),
            },
            Array::ListFloat64(_) => {
                panic!("record_batch_to_dataframe: ListFloat64 not supported")
            }
            Array::DictionaryUtf8(a) => {
                let mut values = Vec::new();
                let mut nulls = Vec::new();
                for i in 0..a.len() {
                    if a.nulls[i] {
                        values.push(String::new());
                        nulls.push(true);
                    } else {
                        values.push(a.dictionary[a.indices[i] as usize].clone());
                        nulls.push(false);
                    }
                }
                Column::Utf8 { values, nulls }
            }
        };
        cols.push((field.name.clone(), c));
    }
    DataFrame::from_columns(cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rnumpy::NdArray;

    #[test]
    fn df_arrow_roundtrip() {
        let df = DataFrame::from_columns(vec![
            (
                "a".into(),
                Column::Float64(NdArray::from_vec(vec![1.0, 2.0])),
            ),
            (
                "b".into(),
                Column::Int64 {
                    values: vec![1, 0],
                    nulls: vec![false, true],
                },
            ),
        ]);
        let batch = dataframe_to_record_batch(&df);
        let back = record_batch_to_dataframe(&batch);
        assert_eq!(back.checksum(), df.checksum());
    }
}
