//! Stage 2 CPU engine: llama.cpp over the wheel's bundled `libllama`.
//!
//! Loaded via `dlopen` so the Rust engine executes the *same* native library
//! as the Python oracle backend — the parity gate then measures the port, not
//! llama.cpp version drift.

pub mod backend;
pub mod ffi;
pub mod scoring;

use backend::Engine;
use semif_core::tokenizer::{ReferenceTokenizer, ScorerError};
use semif_engine::{Engine as EngineTrait, ScoreContext};
use semif_types::PyValue;
use std::path::{Path, PathBuf};

pub struct LlamacppEngine {
    engine: Engine,
    metadata: PyValue,
    serial_state: scoring::SerialState,
}

impl LlamacppEngine {
    pub fn new(
        library_path: &Path,
        gguf: &Path,
        tokenizer: &ReferenceTokenizer,
        threads: usize,
        max_tokens: usize,
        source: &str,
        revision: &str,
    ) -> Result<Self, ScorerError> {
        let engine = Engine::new(library_path, gguf, threads, max_tokens)?;
        engine.verify_vocabulary(tokenizer)?;
        let metadata = scoring::metadata(&engine, source, revision);
        Ok(LlamacppEngine {
            engine,
            metadata,
            serial_state: scoring::SerialState::default(),
        })
    }

    pub fn default_library_path() -> Result<PathBuf, ScorerError> {
        ffi::default_library_path().ok_or_else(|| {
            ScorerError::Message(
                "llama.cpp library not found; set SEMIF_LLAMA_LIB to the wheel's libllama.so"
                    .into(),
            )
        })
    }
}

impl EngineTrait for LlamacppEngine {
    fn score_direct(
        &self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        scoring::score_direct(
            &self.engine,
            context.tokenizer,
            row,
            &self.metadata,
            context.max_tokens,
        )
    }

    fn score_serial(
        &mut self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        scoring::score_serial(
            &self.engine,
            &mut self.serial_state,
            context.tokenizer,
            row,
            &self.metadata,
            context.max_tokens,
        )
    }

    fn score_shared(
        &mut self,
        rows: &[PyValue],
        context: &ScoreContext<'_>,
    ) -> Result<(Vec<PyValue>, Option<PyValue>), ScorerError> {
        let (results, timing) = scoring::score_shared(
            &self.engine,
            context.tokenizer,
            rows,
            &self.metadata,
            context.max_tokens,
        )?;
        Ok((results, Some(timing)))
    }
}
