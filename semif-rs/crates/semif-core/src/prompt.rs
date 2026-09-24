//! Pinned chat-template rendering and payload construction.
//!
//! Reproduces the Qwen3.5 `chat_template.jinja` path actually exercised by the
//! scorer: string system+user messages (each `render_content(...)|trim`-ed),
//! then `add_generation_prompt` with `enable_thinking=false`. Byte-equality is
//! gated by `fixtures/prompts.jsonl` (307 rendered prompts).

use crate::digest;
use crate::parser::ParseError;
use crate::pyjson::dumps_payload;
use semif_types::{DIRECT_SYSTEM, LETTERS, PyValue, validate::ValidationError};

#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error("Out of range float values are not JSON compliant")]
    NonFinitePayload,
    #[error(transparent)]
    Parse(#[from] ParseError),
}

/// `direct_messages` payload: `{"evidence": state, "criterion": question,
/// "options": [{"letter": "A", "description": ...}, ...]}` in insertion order.
pub fn direct_payload(row: &PyValue) -> PyValue {
    let state = row.get("state").cloned().unwrap_or(PyValue::Null);
    let question = row
        .get("question")
        .and_then(PyValue::as_str)
        .unwrap_or_default();
    let options: Vec<PyValue> = row
        .get("options")
        .and_then(PyValue::as_array)
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(index, option)| {
            let description = option
                .get("description")
                .and_then(PyValue::as_str)
                .unwrap_or_default();
            PyValue::Object(vec![
                ("letter".into(), PyValue::Str(LETTERS[index].to_string())),
                ("description".into(), PyValue::Str(description.into())),
            ])
        })
        .collect();
    PyValue::Object(vec![
        ("evidence".into(), state),
        ("criterion".into(), PyValue::Str(question.into())),
        ("options".into(), PyValue::Array(options)),
    ])
}

/// Python `str.strip()` equivalent for the `|trim` filter (payload strings
/// always begin `{` and end `}`, so this is a guard, not a transformation).
fn python_trim(text: &str) -> &str {
    text.trim()
}

/// Render one prompt the way `apply_chat_template(..., enable_thinking=False)` does.
pub fn render_prompt(system_content: &str, user_content: &str) -> String {
    format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        python_trim(system_content),
        python_trim(user_content),
    )
}

/// Validate a row and render its full prompt; returns (prompt, prompt_sha256).
pub fn build_prompt(row: &PyValue) -> Result<(String, String), PromptError> {
    semif_types::validate_row(row)?;
    let payload = direct_payload(row);
    let payload_text = dumps_payload(&payload).map_err(|_| PromptError::NonFinitePayload)?;
    let prompt = render_prompt(DIRECT_SYSTEM, &payload_text);
    let hash = digest(&prompt);
    Ok((prompt, hash))
}

/// Debug helper used by tests: the payload text for one row.
pub fn payload_text(row: &PyValue) -> Result<String, PromptError> {
    let payload = direct_payload(row);
    dumps_payload(&payload).map_err(|_| PromptError::NonFinitePayload)
}

#[cfg(test)]
mod tests {
    use super::{build_prompt, render_prompt};
    use crate::parser::parse;
    use semif_types::DIRECT_SYSTEM;

    const ROW: &str = r#"{"id": "s1", "state": "Evidence.", "question": "True?",
        "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}"#;

    #[test]
    fn shape_matches_pinned_template() {
        let row = parse(ROW).unwrap();
        let (prompt, _) = build_prompt(&row).unwrap();
        let expected = format!(
            "<|im_start|>system\n{DIRECT_SYSTEM}<|im_end|>\n<|im_start|>user\n{{\"evidence\": \"Evidence.\", \"criterion\": \"True?\", \"options\": [{{\"letter\": \"A\", \"description\": \"Yes.\"}}, {{\"letter\": \"B\", \"description\": \"No.\"}}]}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn trim_is_applied_like_jinja() {
        assert_eq!(render_prompt("  x  ", "  y  "), render_prompt("x", "y"));
    }
}
