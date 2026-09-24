//! State-prefix extraction, matching `semif_phase1.shared._state_prefix`
//! (serial and shared modes share this logic; serial.py's duplicate raises a
//! combined message — the shared.py messages are pinned here).

use crate::prompt::{direct_payload, render_prompt};
use crate::pyjson::dumps_payload;
use crate::tokenizer::{ReferenceTokenizer, ScorerError};
use semif_types::{DIRECT_SYSTEM, PyValue};

fn synthetic_row(state: &PyValue) -> PyValue {
    PyValue::Object(vec![
        ("id".into(), PyValue::Str("prefix-only".into())),
        ("state".into(), state.clone()),
        (
            "question".into(),
            PyValue::Str("prefix boundary placeholder".into()),
        ),
        (
            "options".into(),
            PyValue::Array(vec![
                PyValue::Object(vec![
                    ("id".into(), PyValue::Str("yes".into())),
                    ("description".into(), PyValue::Str("Yes".into())),
                ]),
                PyValue::Object(vec![
                    ("id".into(), PyValue::Str("no".into())),
                    ("description".into(), PyValue::Str("No".into())),
                ]),
            ]),
        ),
    ])
}

fn strip_last_char(text: &str) -> Option<&str> {
    let mut chars = text.char_indices();
    chars.next()?;
    let (last_index, _) = chars.next_back()?;
    Some(&text[..last_index])
}

/// Build the token prefix that ends right after the serialized evidence.
pub fn state_prefix(
    tokenizer: &ReferenceTokenizer,
    state: &PyValue,
) -> Result<Vec<u32>, ScorerError> {
    let row = synthetic_row(state);
    semif_types::validate_row(&row).map_err(|error| ScorerError::Message(error.to_string()))?;
    let payload_text = dumps_payload(&direct_payload(&row))
        .map_err(|_| ScorerError::Message("Evidence serialization changed".into()))?;
    let prompt = render_prompt(DIRECT_SYSTEM, &payload_text);

    let evidence_json =
        dumps_payload(&PyValue::Object(vec![("evidence".into(), state.clone())]))
            .map_err(|_| ScorerError::Message("Evidence serialization changed".into()))?;
    let evidence = strip_last_char(&evidence_json)
        .ok_or_else(|| ScorerError::Message("Evidence serialization changed".into()))?;

    if prompt.matches(&payload_text).count() != 1 || !payload_text.starts_with(evidence) {
        return Err(ScorerError::Message(
            "Cannot locate the unmodified evidence payload in the chat template".into(),
        ));
    }
    let index = prompt
        .find(&payload_text)
        .ok_or_else(|| ScorerError::Message("Evidence serialization changed".into()))?;
    let text = format!("{}{}", &prompt[..index], evidence);
    let ids = tokenizer.encode(&text)?;
    if ids.is_empty() {
        return Err(ScorerError::Message(
            "Cannot establish a deterministic evidence prefix".into(),
        ));
    }
    Ok(ids[..ids.len() - 1].to_vec())
}
