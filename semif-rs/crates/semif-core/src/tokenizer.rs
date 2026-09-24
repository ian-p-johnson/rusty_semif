//! Reference-tokenizer harness: truncation and padding detached at load
//! (the shipped `tokenizer.json` bakes in a 512-token truncation that Python's
//! bare `encode` never applies — a proven Stage 2 trap from the von port).

use crate::PROMPT_VERSION;
use crate::digest;
use crate::parser::ParseError;
use crate::prompt::{PromptError, build_prompt};
use semif_types::{LETTERS, PyValue};
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, thiserror::Error)]
pub enum ScorerError {
    #[error("tokenizer load failed: {0}")]
    Load(String),
    #[error(transparent)]
    Prompt(#[from] PromptError),
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error("{0}")]
    Message(String),
}

pub struct ReferenceTokenizer {
    inner: Tokenizer,
}

impl ReferenceTokenizer {
    pub fn from_checkpoint_dir(dir: &Path) -> Result<Self, ScorerError> {
        let path = dir.join("tokenizer.json");
        let mut inner = Tokenizer::from_file(&path)
            .map_err(|error| ScorerError::Load(format!("{path:?}: {error}")))?;
        let _ = inner.with_truncation(None);
        let _ = inner.with_padding(None);
        Ok(Self { inner })
    }

    /// `tokenizer.encode(text, add_special_tokens=False)`
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, ScorerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| ScorerError::Load(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// `tokenizer.decode([id])`
    pub fn decode_one(&self, id: u32) -> Result<String, ScorerError> {
        self.inner
            .decode(&[id], false)
            .map_err(|error| ScorerError::Load(error.to_string()))
    }
}

/// `_slot_ids`: every answer letter must be one exact round-trip token, no collisions.
fn slot_ids(tokenizer: &ReferenceTokenizer, count: usize) -> Result<Vec<u32>, ScorerError> {
    let mut result = Vec::with_capacity(count);
    for letter in &LETTERS[..count] {
        let encoded = tokenizer.encode(&letter.to_string())?;
        if encoded.len() != 1 || tokenizer.decode_one(encoded[0])? != letter.to_string() {
            return Err(ScorerError::Message(format!(
                "Answer slot {letter:?} is not one exact round-trip token"
            )));
        }
        result.push(encoded[0]);
    }
    if result
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != result.len()
    {
        return Err(ScorerError::Message("Answer-slot tokens collide".into()));
    }
    Ok(result)
}

/// `direct.encode_prompt`: budget, slots, and prefix-stability checks.
/// Returns (ids, slots, prompt_sha256).
pub fn encode_prompt(
    tokenizer: &ReferenceTokenizer,
    row: &PyValue,
    max_tokens: usize,
) -> Result<(Vec<u32>, Vec<u32>, String), ScorerError> {
    let (prompt, prompt_hash) = build_prompt(row)?;
    let ids = tokenizer.encode(&prompt)?;
    let row_id = row.get("id").and_then(PyValue::as_str).unwrap_or_default();
    if ids.is_empty() || ids.len() > max_tokens {
        return Err(ScorerError::Message(format!(
            "Row {row_id}: {} input tokens exceed limit {max_tokens}; no truncation allowed",
            ids.len()
        )));
    }
    let option_count = row
        .get("options")
        .and_then(PyValue::as_array)
        .map(<[PyValue]>::len)
        .unwrap_or(0);
    let slots = slot_ids(tokenizer, option_count)?;
    for (letter, token) in LETTERS.iter().zip(&slots) {
        let mut extended = prompt.clone();
        extended.push(*letter);
        let mut expected = ids.clone();
        expected.push(*token);
        if tokenizer.encode(&extended)? != expected {
            return Err(ScorerError::Message(format!(
                "Answer boundary changes tokenization for slot {letter}"
            )));
        }
    }
    debug_assert_eq!(digest(&prompt), prompt_hash);
    Ok((ids, slots, prompt_hash))
}

/// Row envelope identity shared by every mode's result (prompt version stamp).
pub fn prompt_version() -> &'static str {
    PROMPT_VERSION
}
