//! Engine seam: the CLI dispatches per backend and mode.
//!
//! Stage 2: `LlamacppEngine` (real) for `--backend llamacpp`; the stub engine
//! remains the placeholder for torch/mlx until Stages 3+.

use semif_core::tokenizer::{ReferenceTokenizer, ScorerError};
use semif_types::PyValue;

pub struct ScoreContext<'a> {
    pub tokenizer: &'a ReferenceTokenizer,
    pub max_tokens: usize,
    pub source: String,
    pub revision: String,
    pub dtype: String,
}

pub trait Engine {
    fn score_direct(
        &self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError>;
    fn score_serial(
        &mut self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError>;
    fn score_shared(
        &mut self,
        rows: &[PyValue],
        context: &ScoreContext<'_>,
    ) -> Result<(Vec<PyValue>, Option<PyValue>), ScorerError>;
}

/// Uniform-probability stub: `probabilities = 1/K`, `option_logits = 0.0`.
/// The envelope mirrors `direct.score` minus timing fields; byte parity is
/// pinned by `fixtures/stubs.jsonl`. Serial/shared reuse the same envelope
/// (no timing block) until real backends land for those backends.
pub struct StubEngine;

impl StubEngine {
    fn stub_row(&self, row: &PyValue, context: &ScoreContext<'_>) -> Result<PyValue, ScorerError> {
        let (ids, _slots, prompt_hash) =
            semif_core::tokenizer::encode_prompt(context.tokenizer, row, context.max_tokens)?;
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
        let uniform = 1.0 / option_ids.len() as f64;
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

impl Engine for StubEngine {
    fn score_direct(
        &self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        self.stub_row(row, context)
    }

    fn score_serial(
        &mut self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        self.stub_row(row, context)
    }

    fn score_shared(
        &mut self,
        rows: &[PyValue],
        context: &ScoreContext<'_>,
    ) -> Result<(Vec<PyValue>, Option<PyValue>), ScorerError> {
        Ok((
            rows.iter()
                .map(|row| self.stub_row(row, context))
                .collect::<Result<Vec<_>, _>>()?,
            None,
        ))
    }
}
