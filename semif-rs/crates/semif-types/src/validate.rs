//! Row validation, operation-for-operation with `semif_phase1.core.validate_row`.
//!
//! Check order and message strings are pinned by `semif-rs/fixtures/rows.jsonl`;
//! do not reorder or reword.

use crate::LETTERS;
use crate::value::PyValue;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("{message}")]
pub struct ValidationError {
    pub message: String,
}

fn error(message: impl Into<String>) -> ValidationError {
    ValidationError {
        message: message.into(),
    }
}

fn contains_non_finite(value: &PyValue) -> bool {
    match value {
        PyValue::Float(f) => !f.is_finite(),
        PyValue::Array(items) => items.iter().any(contains_non_finite),
        PyValue::Object(entries) => entries.iter().any(|(_, v)| contains_non_finite(v)),
        _ => false,
    }
}

/// Validate one decision row; identical check order to the Python scorer.
pub fn validate_row(row: &PyValue) -> Result<(), ValidationError> {
    let entries = row
        .as_object()
        .ok_or_else(|| error("Row is missing fields: ['id', 'options', 'question', 'state']"))?;
    let required = ["id", "state", "question", "options"];
    let mut missing: Vec<&str> = required
        .iter()
        .filter(|key| !entries.iter().any(|(name, _)| name == *key))
        .copied()
        .collect();
    if !missing.is_empty() {
        missing.sort_unstable();
        let quoted: Vec<String> = missing.iter().map(|key| format!("'{key}'")).collect();
        return Err(error(format!(
            "Row is missing fields: [{}]",
            quoted.join(", ")
        )));
    }
    for key in ["id", "question"] {
        let nonempty_string = row
            .get(key)
            .and_then(PyValue::as_str)
            .is_some_and(|text| !text.is_empty());
        if !nonempty_string {
            return Err(error("id and question must be nonempty strings"));
        }
    }
    let state = row.get("state").expect("checked above");
    let state_ok = matches!(
        state,
        PyValue::Str(_) | PyValue::Object(_) | PyValue::Array(_)
    ) && !matches!(state, PyValue::Str(s) if s.is_empty())
        && !matches!(state, PyValue::Object(entries) if entries.is_empty())
        && !matches!(state, PyValue::Array(items) if items.is_empty());
    if !state_ok {
        return Err(error("state must be a nonempty string, object, or array"));
    }
    if contains_non_finite(state) {
        return Err(error("state must be finite JSON-compatible data"));
    }
    let Some(options) = row.get("options").and_then(PyValue::as_array) else {
        return Err(error("options must contain 2-16 entries"));
    };
    if options.len() < 2 || options.len() > LETTERS.len() {
        return Err(error("options must contain 2-16 entries"));
    }
    let mut ids: Vec<&str> = Vec::with_capacity(options.len());
    for option in options {
        let Some(fields) = option.as_object() else {
            return Err(error("Each option needs string id and description fields"));
        };
        let id = fields
            .iter()
            .find(|(name, _)| name == "id")
            .and_then(|(_, value)| value.as_str());
        let description = fields
            .iter()
            .find(|(name, _)| name == "description")
            .and_then(|(_, value)| value.as_str());
        match (id, description) {
            (Some(_), Some(_)) => ids.push(id.expect("checked")),
            _ => return Err(error("Each option needs string id and description fields")),
        }
    }
    let unique = ids.iter().collect::<std::collections::HashSet<_>>().len();
    if unique != ids.len() {
        return Err(error("Option IDs must be unique"));
    }
    Ok(())
}
