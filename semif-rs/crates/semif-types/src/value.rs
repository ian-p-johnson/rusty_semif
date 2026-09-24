//! Python-`json`-shaped value model.
//!
//! `Int` keeps its canonical decimal digits as text so integers beyond i64/u64
//! round-trip exactly like Python's arbitrary-precision ints. Floats carry f64.
//! NaN/Infinity parse (Python `json.loads` accepts them) and are rejected later
//! at validation, exactly like `json.dumps(..., allow_nan=False)`.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum PyValue {
    Null,
    Bool(bool),
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<PyValue>),
    Object(Vec<(String, PyValue)>),
}

impl PyValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PyValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[PyValue]> {
        match self {
            PyValue::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, PyValue)]> {
        match self {
            PyValue::Object(entries) => Some(entries),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&PyValue> {
        match self {
            PyValue::Object(entries) => entries
                .iter()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    pub fn is_finite_number(&self) -> bool {
        match self {
            PyValue::Int(_) => true,
            PyValue::Float(f) => f.is_finite(),
            _ => false,
        }
    }
}

impl fmt::Display for PyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyValue::Null => write!(f, "None"),
            PyValue::Bool(true) => write!(f, "True"),
            PyValue::Bool(false) => write!(f, "False"),
            PyValue::Int(digits) => write!(f, "{digits}"),
            PyValue::Float(value) => write!(f, "{value:?}"),
            PyValue::Str(text) => write!(f, "{text:?}"),
            PyValue::Array(_) | PyValue::Object(_) => write!(f, "…"),
        }
    }
}
