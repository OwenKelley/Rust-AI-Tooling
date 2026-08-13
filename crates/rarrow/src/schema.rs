//! Schema / Field / DataType (PyArrow-shaped).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Float64,
    Int64,
    Boolean,
    Utf8,
    /// Nanoseconds since UNIX epoch (Arrow `timestamp[ns]`).
    TimestampNs,
    /// List of float64 values (Arrow `list<item: double>`).
    ListFloat64,
    /// Dictionary-encoded utf8 (Arrow indices + string dictionary).
    DictionaryUtf8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    /// Dictionary id when `data_type == DictionaryUtf8`.
    pub dict_id: Option<i64>,
}

impl Field {
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        let dict_id = if matches!(data_type, DataType::DictionaryUtf8) {
            Some(0)
        } else {
            None
        };
        Self {
            name: name.into(),
            data_type,
            nullable,
            dict_id,
        }
    }

    pub fn with_dict_id(mut self, id: i64) -> Self {
        self.dict_id = Some(id);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    pub fields: Vec<Field>,
}

impl Schema {
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    pub fn empty() -> Self {
        Self { fields: Vec::new() }
    }
}
