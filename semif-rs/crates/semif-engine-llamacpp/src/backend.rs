//! The llama.cpp scoring engine, operation-for-operation with
//! `semif_phase1.llamacpp_backend` (single context, one sequence, chunked
//! decode, whole-sequence state save/restore for branch replication — the
//! hybrid linear-attention memory supports neither sequence copies nor
//! partial tail removal).

use crate::ffi::{self, LlamaBatch, LlamaPos, LlamaToken, Symbols};
use semif_core::tokenizer::{ReferenceTokenizer, ScorerError, encode_prompt};
use semif_types::{LETTERS, PyValue};
use std::ffi::CString;
use std::path::Path;

const DECODE_CHUNK: usize = 512;

pub struct Engine {
    pub symbols: Symbols,
    pub model: *mut std::ffi::c_void,
    pub context: *mut std::ffi::c_void,
    pub memory: *mut std::ffi::c_void,
    pub vocab: *const std::ffi::c_void,
    pub context_tokens: usize,
    pub vocab_size: usize,
    pub gguf_bytes: u64,
    pub gguf_sha256: String,
    pub gguf_name: String,
    pub threads: i32,
    pub max_prompt_tokens: usize,
}

struct OwnedCString {
    pointer: CString,
}

impl OwnedCString {
    fn new(text: &str) -> Result<Self, ScorerError> {
        Ok(OwnedCString {
            pointer: CString::new(text.replace('\0', ""))
                .map_err(|_| ScorerError::Message("GGUF path/text contained a NUL byte".into()))?,
        })
    }
}

impl Engine {
    /// `llamacpp_backend._Engine` + `load_model` file/hash handling.
    pub fn new(
        library_path: &Path,
        gguf: &Path,
        threads: usize,
        max_prompt_tokens: usize,
    ) -> Result<Self, ScorerError> {
        let library = ffi::load_library(library_path).map_err(ScorerError::Message)?;
        let symbols = ffi::load_symbols(library).map_err(ScorerError::Message)?;
        let checksum = gguf_sha256(gguf)?;
        let gguf_bytes = std::fs::metadata(gguf)
            .map_err(|error| ScorerError::Message(format!("GGUF metadata: {error}")))?
            .len();
        let path_text = OwnedCString::new(&gguf.to_string_lossy())?;
        unsafe {
            (symbols.llama_backend_init)();
            (symbols.llama_log_set)(Some(ffi::noop_log), std::ptr::null_mut());
            let mut model_params = (symbols.llama_model_default_params)();
            model_params.n_gpu_layers = 0;
            let model =
                (symbols.llama_model_load_from_file)(path_text.pointer.as_ptr(), model_params);
            if model.is_null() {
                return Err(ScorerError::Message(format!(
                    "llama.cpp failed to load the GGUF checkpoint: {gguf:?}"
                )));
            }
            let window = max_prompt_tokens + 64;
            let mut context_params = (symbols.llama_context_default_params)();
            context_params.n_ctx = window as u32;
            context_params.n_seq_max = 1;
            context_params.n_outputs_max = 1;
            context_params.n_threads = threads as i32;
            context_params.n_threads_batch = threads as i32;
            let context = (symbols.llama_init_from_model)(model, context_params);
            if context.is_null() {
                (symbols.llama_model_free)(model);
                return Err(ScorerError::Message(
                    "llama.cpp failed to create the scoring context".into(),
                ));
            }
            let memory = (symbols.llama_get_memory)(context);
            if memory.is_null() {
                (symbols.llama_free)(context);
                (symbols.llama_model_free)(model);
                return Err(ScorerError::Message(
                    "llama.cpp returned no context memory".into(),
                ));
            }
            let vocab = (symbols.llama_model_get_vocab)(model);
            let context_tokens = (symbols.llama_n_ctx)(context).max(0) as usize;
            let vocab_size = (symbols.llama_n_vocab)(vocab).max(0) as usize;
            Ok(Engine {
                symbols,
                model,
                context,
                memory,
                vocab,
                context_tokens,
                vocab_size,
                gguf_bytes,
                gguf_sha256: checksum,
                gguf_name: gguf
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                threads: threads as i32,
                max_prompt_tokens,
            })
        }
    }

    /// `_gguf_tokenize`: GGUF-vocabulary tokenization of UTF-8 text.
    pub fn gguf_tokenize(&self, text: &str) -> Result<Vec<LlamaToken>, ScorerError> {
        let data = text.as_bytes();
        unsafe {
            let needed = (self.symbols.llama_tokenize)(
                self.vocab,
                data.as_ptr().cast(),
                data.len() as i32,
                std::ptr::null_mut(),
                0,
                false,
                true,
            );
            // Python: `if needed < 0: needed = -needed` — the negative return IS the count.
            let capacity = needed.unsigned_abs() as usize;
            let mut tokens: Vec<LlamaToken> = vec![0; capacity];
            let written = (self.symbols.llama_tokenize)(
                self.vocab,
                data.as_ptr().cast(),
                data.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                false,
                true,
            );
            if written < 0 {
                return Err(ScorerError::Message(
                    "The GGUF tokenizer rejected the prompt text".into(),
                ));
            }
            tokens.truncate(written as usize);
            Ok(tokens)
        }
    }

    /// `_gguf_piece`: render one token back to bytes.
    pub fn gguf_piece(&self, token: LlamaToken) -> Result<Vec<u8>, ScorerError> {
        let mut buffer = [0u8; 64];
        let written = unsafe {
            (self.symbols.llama_token_to_piece)(
                self.vocab,
                token,
                buffer.as_mut_ptr(),
                buffer.len() as i32,
                0,
                true,
            )
        };
        if written < 0 {
            return Err(ScorerError::Message(
                "The GGUF tokenizer cannot render a token".into(),
            ));
        }
        Ok(buffer[..written as usize].to_vec())
    }

    /// `_verify_vocabulary`: GGUF vocabulary must agree with the reference
    /// tokenizer, and every answer letter must be one shared single token.
    pub fn verify_vocabulary(&self, tokenizer: &ReferenceTokenizer) -> Result<(), ScorerError> {
        let probe = PyValue::Object(vec![
            ("id".into(), PyValue::Str("vocabulary-probe".into())),
            ("state".into(), PyValue::Str("probe evidence".into())),
            ("question".into(), PyValue::Str("probe criterion?".into())),
            (
                "options".into(),
                PyValue::Array(vec![
                    PyValue::Object(vec![
                        ("id".into(), PyValue::Str("yes".into())),
                        ("description".into(), PyValue::Str("Yes.".into())),
                    ]),
                    PyValue::Object(vec![
                        ("id".into(), PyValue::Str("no".into())),
                        ("description".into(), PyValue::Str("No.".into())),
                    ]),
                ]),
            ),
        ]);
        let rendered = render_row_prompt(&probe)?;
        let reference: Vec<LlamaToken> = tokenizer
            .encode(&rendered)?
            .iter()
            .map(|&t| t as LlamaToken)
            .collect();
        if self.gguf_tokenize(&rendered)? != reference {
            return Err(ScorerError::Message(
                "The GGUF vocabulary disagrees with the reference tokenizer".into(),
            ));
        }
        for letter in LETTERS {
            let encoded = tokenizer.encode(&letter.to_string())?;
            if encoded.len() != 1
                || self.gguf_piece(encoded[0] as LlamaToken)? != letter.to_string().into_bytes()
            {
                return Err(ScorerError::Message(format!(
                    "Answer slot {letter:?} is not a shared single token"
                )));
            }
        }
        Ok(())
    }

    /// `encode_verified`: reference encoding, then GGUF re-tokenization agreement.
    pub fn encode_verified(
        &self,
        tokenizer: &ReferenceTokenizer,
        row: &PyValue,
        max_tokens: usize,
    ) -> Result<(Vec<LlamaToken>, Vec<LlamaToken>, String), ScorerError> {
        let (ids, slots, prompt_hash) = encode_prompt(tokenizer, row, max_tokens)?;
        let rendered = render_row_prompt(row)?;
        let reference: Vec<LlamaToken> = ids.iter().map(|&t| t as LlamaToken).collect();
        if self.gguf_tokenize(&rendered)? != reference {
            let id = row.get("id").and_then(PyValue::as_str).unwrap_or_default();
            return Err(ScorerError::Message(format!(
                "Row {id}: GGUF tokenization disagrees with the reference tokenizer"
            )));
        }
        let llama_ids: Vec<LlamaToken> = ids.iter().map(|&t| t as LlamaToken).collect();
        let llama_slots: Vec<LlamaToken> = slots.iter().map(|&t| t as LlamaToken).collect();
        Ok((llama_ids, llama_slots, prompt_hash))
    }

    /// `_decode`: chunked decode over sequence `sequence`, logits on final token.
    fn decode(
        &self,
        tokens: &[LlamaToken],
        start: usize,
        sequence: i32,
        want_logits: bool,
    ) -> Result<Option<Vec<f32>>, ScorerError> {
        if tokens.is_empty() {
            return Err(ScorerError::Message(
                "Refusing to decode an empty token list".into(),
            ));
        }
        let total = tokens.len();
        for offset in (0..total).step_by(DECODE_CHUNK) {
            let end = (offset + DECODE_CHUNK).min(total);
            let chunk = &tokens[offset..end];
            unsafe {
                let mut batch: LlamaBatch =
                    (self.symbols.llama_batch_init)(chunk.len() as i32, 0, 1);
                for (index, &token) in chunk.iter().enumerate() {
                    *batch.token.add(index) = token;
                    *batch.pos.add(index) = (start + offset + index) as LlamaPos;
                    *batch.n_seq_id.add(index) = 1;
                    **batch.seq_id.add(index) = sequence;
                    *batch.logits.add(index) = (want_logits && offset + index == total - 1) as i8;
                }
                batch.n_tokens = chunk.len() as i32;
                let status = (self.symbols.llama_decode)(self.context, batch);
                (self.symbols.llama_batch_free)(batch);
                if status != 0 {
                    return Err(ScorerError::Message(
                        "llama_decode failed; raise --max-tokens if prompts grew".into(),
                    ));
                }
            }
        }
        if !want_logits {
            return Ok(None);
        }
        let pointer = unsafe { (self.symbols.llama_get_logits_ith)(self.context, -1) };
        if pointer.is_null() {
            return Err(ScorerError::Message(
                "llama.cpp returned no logits for the flagged position".into(),
            ));
        }
        let mut logits = Vec::with_capacity(self.vocab_size);
        for index in 0..self.vocab_size {
            logits.push(unsafe { *pointer.add(index) });
        }
        Ok(Some(logits))
    }

    pub fn clear(&self) {
        unsafe { (self.symbols.llama_memory_clear)(self.memory, false) };
    }

    pub fn prefill(&self, prefix: &[LlamaToken]) -> Result<(), ScorerError> {
        self.decode(prefix, 0, 0, false).map(|_| ())
    }

    /// `save_state`: snapshot sequence 0 (bytes + size).
    pub fn save_state(&self) -> Result<Vec<u8>, ScorerError> {
        let size = unsafe { (self.symbols.llama_state_seq_get_size)(self.context, 0) };
        if size == 0 {
            return Err(ScorerError::Message(
                "llama.cpp returned an empty prefix state".into(),
            ));
        }
        let mut buffer = vec![0u8; size];
        let written = unsafe {
            (self.symbols.llama_state_seq_get_data)(self.context, buffer.as_mut_ptr(), size, 0)
        };
        if written != size {
            return Err(ScorerError::Message(
                "llama.cpp wrote an incomplete prefix state".into(),
            ));
        }
        Ok(buffer)
    }

    /// `restore_state`: drop sequence 0, then load the snapshot.
    pub fn restore_state(&self, state: &[u8]) -> Result<(), ScorerError> {
        let removed = unsafe { (self.symbols.llama_memory_seq_rm)(self.memory, 0, -1, -1) };
        if !removed {
            return Err(ScorerError::Message(
                "llama.cpp could not drop the previous scored branch".into(),
            ));
        }
        let written = unsafe {
            (self.symbols.llama_state_seq_set_data)(self.context, state.as_ptr(), state.len(), 0)
        };
        if written == 0 {
            return Err(ScorerError::Message(
                "llama.cpp could not restore the saved prefix state".into(),
            ));
        }
        Ok(())
    }

    pub fn branch_logits(
        &self,
        prefix_length: usize,
        suffix: &[LlamaToken],
    ) -> Result<Vec<f32>, ScorerError> {
        Ok(self
            .decode(suffix, prefix_length, 0, true)?
            .expect("branch logits requested"))
    }

    pub fn full_logits(&self, tokens: &[LlamaToken]) -> Result<Vec<f32>, ScorerError> {
        self.clear();
        Ok(self
            .decode(tokens, 0, 0, true)?
            .expect("full logits requested"))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                (self.symbols.llama_free)(self.context);
            }
            if !self.model.is_null() {
                (self.symbols.llama_model_free)(self.model);
            }
        }
    }
}

fn gguf_sha256(path: &Path) -> Result<String, ScorerError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut hasher = Sha256::new();
    let mut file = std::fs::File::open(path)
        .map_err(|error| ScorerError::Message(format!("GGUF open failed: {error}")))?;
    let mut block = vec![0u8; 8 << 20];
    loop {
        let read = file
            .read(&mut block)
            .map_err(|e| ScorerError::Message(e.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&block[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// Re-render the full prompt for a row (same bytes `encode_prompt` hashed).
pub fn render_row_prompt(row: &PyValue) -> Result<String, ScorerError> {
    let (prompt, _hash) = semif_core::prompt::build_prompt(row).map_err(|e| match e {
        semif_core::prompt::PromptError::Parse(inner) => ScorerError::Parse(inner),
        other => ScorerError::Message(other.to_string()),
    })?;
    Ok(prompt)
}
