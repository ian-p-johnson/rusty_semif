//! Engine seam: modes request one result row per input row.
//!
//! Stage 1 ships only the structure-faithful stub; real backends (llama.cpp,
//! tch/CUDA) land in Stages 2–3 behind this trait. The stub still runs the
//! full prompt/token/slot pipeline so identity fields (`prompt_sha256`,
//! `input_tokens`) are real; only the readout is synthetic.

use semif_core::tokenizer::{ReferenceTokenizer, ScorerError, encode_prompt};
use semif_types::PyValue;

pub struct ScoreContext<'a> {
    pub tokenizer: &'a ReferenceTokenizer,
    pub max_tokens: usize,
    pub source: String,
    pub revision: String,
    pub dtype: String,
}

pub trait Engine {
    fn score(&self, row: &PyValue, context: &ScoreContext<'_>) -> Result<PyValue, ScorerError>;
}

/// Uniform-probability stub: `probabilities = 1/K`, `option_logits = 0.0`.
/// The envelope mirrors `direct.score` minus timing fields; byte parity is
/// pinned by `fixtures/stubs.jsonl`.
pub struct StubEngine;

impl Engine for StubEngine {
    fn score(&self, row: &PyValue, context: &ScoreContext<'_>) -> Result<PyValue, ScorerError> {
        let (ids, _slots, prompt_hash) = encode_prompt(context.tokenizer, row, context.max_tokens)?;
        let options = row
            .get("options")
            .and_then(PyValue::as_array)
            .ok_or_else(|| ScorerError::Message("options must contain 2-16 entries".into()))?;
        let option_ids: Vec<PyValue> = options
            .iter()
            .map(|option| {
                PyValue::Str(
                    option
                        .get("id")
                        .and_then(PyValue::as_str)
                        .unwrap_or_default()
                        .into(),
                )
            })
            .collect();
        let count = option_ids.len() as f64;
        let uniform = 1.0 / count;
        let row_id = row.get("id").and_then(PyValue::as_str).unwrap_or_default();
        Ok(PyValue::Object(vec![
            ("id".into(), PyValue::Str(row_id.into())),
            ("option_ids".into(), PyValue::Array(option_ids)),
            (
                "probabilities".into(),
                PyValue::Array(vec![PyValue::Float(uniform); options.len()]),
            ),
            (
                "option_logits".into(),
                PyValue::Array(vec![PyValue::Float(0.0); options.len()]),
            ),
            ("input_tokens".into(), PyValue::Int(ids.len().to_string())),
            ("prompt_sha256".into(), PyValue::Str(prompt_hash)),
            (
                "prompt_version".into(),
                PyValue::Str(semif_core::PROMPT_VERSION.into()),
            ),
            (
                "model".into(),
                PyValue::Object(vec![
                    ("source".into(), PyValue::Str(context.source.clone())),
                    ("revision".into(), PyValue::Str(context.revision.clone())),
                    ("dtype".into(), PyValue::Str(context.dtype.clone())),
                    (
                        "serving_config".into(),
                        PyValue::Str("stub-uniform-v1".into()),
                    ),
                ]),
            ),
            ("readout".into(), PyValue::Str("stub-uniform-v1".into())),
            (
                "probability_status".into(),
                PyValue::Str("stub; not a model score".into()),
            ),
        ]))
    }
}
