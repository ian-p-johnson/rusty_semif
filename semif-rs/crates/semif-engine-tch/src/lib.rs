//! Stage 3 CUDA/BF16 engine: the TorchScript-traced Qwen3.5 graphs executed
//! through `tch` against the oracle venv's own libtorch.
//!
//! Two artifacts, one per mode family:
//! - **direct/serial/shared** — `qwen35-direct-{context}.pt`, traced from
//!   `(input_ids[1,W], length[1])` with right padding and the length gather.
//! - **reranker** — `qwen3reranker-{context}.pt`, traced from
//!   `(input_ids[B,W], attention_mask[B,W], position_ids[B,W])`. Python's
//!   `score_pair_batch` left-pads every option of a row to that row's widest
//!   pair and lets the model default to `position_ids = arange(width)`, so a
//!   shorter option is evaluated at shifted RoPE positions. The Rust layout
//!   reproduces that shift instead of removing it: each option is right-aligned
//!   in the fixed width, masked, and given `index - (W - row_width)`.
//!
//! Route decision (Stage 3, operator-approved): torch.jit.trace of the pinned
//! models replaces the hand-written 32-layer hybrid stack. Serial/shared run as
//! documented fresh-recompute (new serving_config) until a hand-written cache
//! exists. Because trace fusion legitimately changes kernels, the numeric gate
//! is **bit-exactness against the exporter's own trace-check capture**, plus
//! decision agreement against the eager Python run.

use semif_core::tokenizer::ScorerError;
use semif_engine::{Engine as EngineTrait, ScoreContext};
use semif_types::PyValue;
use std::path::Path;

/// Which artifact this engine carries; the two refuse each other's modes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EngineKind {
    Direct,
    Reranker,
}

pub struct TchEngine {
    module: tch::CModule,
    device: tch::Device,
    metadata: PyValue,
    trace_width: usize,
    pad_id: i64,
    kind: EngineKind,
    /// Reranker yes/no vocabulary positions, read from the trace-check record.
    no_id: i64,
    yes_id: i64,
}

/// Device selection for a trace context name.
pub fn device_for_context(context: &str) -> tch::Device {
    if context == "cudabf16" {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}

/// AGENTS.md: one visible CUDA GPU per scorer process, exactly like
/// `semif_phase1.core.resolve_device("cuda")`.
pub fn require_single_gpu() -> Result<(), ScorerError> {
    let count = tch::Cuda::device_count();
    if count != 1 {
        return Err(ScorerError::Message(format!(
            "Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES (saw {count})"
        )));
    }
    Ok(())
}

/// Locate the wheel's `libtorch_cuda.so`. `cargo test` runs with CWD set to the
/// *package* directory, so a bare `../.venv` probe misses: walk upward instead.
fn find_libtorch_cuda() -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("SEMIF_TORCH_LIB") {
        let path = std::path::PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let mut directory = std::env::current_dir().ok()?;
    loop {
        for python in ["python3.13", "python3.12", "python3.11", "python3.10"] {
            let candidate = directory
                .join(".venv/lib")
                .join(python)
                .join("site-packages/torch/lib/libtorch_cuda.so");
            if candidate.exists() {
                return Some(candidate);
            }
        }
        let fallback = directory.join("target/debug/libtorch_cuda.so");
        if fallback.exists() {
            return Some(fallback);
        }
        if !directory.pop() {
            return None;
        }
    }
}

/// CUDA kernels register via static initializers, so libtorch_cuda must be
/// loaded (RTLD_GLOBAL) before any CUDA tensor work — the same thing Python's
/// `_load_global_deps` does at import time. Idempotent; keep handles alive.
fn preload_libtorch_cuda() {
    match find_libtorch_cuda() {
        Some(path) => {
            // RTLD_GLOBAL like Python's _load_global_deps; keep the handle alive.
            match unsafe { libloading::os::unix::Library::open(Some(&path), 0x101) } {
                Ok(library) => std::mem::forget(library),
                Err(error) => eprintln!("warning: failed to preload {path:?}: {error}"),
            }
        }
        None => {
            eprintln!("warning: CUDA requested but libtorch_cuda not found; set SEMIF_TORCH_LIB")
        }
    }
}

/// Minimal flat-JSON integer field reader (avoids pulling serde into the engine).
fn json_int_field(text: &str, key: &str) -> usize {
    let pattern = format!("\"{key}\": ");
    let start = text
        .find(&pattern)
        .unwrap_or_else(|| panic!("field {key} in trace-check"))
        + pattern.len();
    let digits: String = text[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().expect("integer field")
}

/// The trace-check sidecar sitting next to an artifact, parsed with the same
/// flat reader that supplies the trace width and padding token.
fn read_check(artifact: &Path, name: &str) -> Result<String, ScorerError> {
    let path = artifact.parent().unwrap_or(Path::new(".")).join(name);
    std::fs::read_to_string(&path)
        .map_err(|error| ScorerError::Message(format!("{path:?}: {error}")))
}

fn base_metadata(
    source: &str,
    revision: &str,
    dtype: &str,
    device_tag: &str,
    artifact: &Path,
) -> PyValue {
    PyValue::Object(vec![
        ("source".into(), PyValue::Str(source.into())),
        ("revision".into(), PyValue::Str(revision.into())),
        ("dtype".into(), PyValue::Str(dtype.into())),
        ("device_tag".into(), PyValue::Str(device_tag.into())),
        (
            "artifact".into(),
            PyValue::Str(
                artifact
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            ),
        ),
        (
            "engine".into(),
            PyValue::Str("semif-rs tch (TorchScript trace)".into()),
        ),
    ])
}

impl TchEngine {
    /// Load a traced direct-mode module (`qwen35-direct-{context}.pt`).
    pub fn new(
        artifact: &std::path::Path,
        device: tch::Device,
        source: &str,
        revision: &str,
        dtype: &str,
        device_tag: &str,
    ) -> Result<Self, ScorerError> {
        if device != tch::Device::Cpu {
            preload_libtorch_cuda();
        }
        let module = if device == tch::Device::Cpu {
            tch::CModule::load(String::from(artifact.to_string_lossy()))
        } else {
            tch::CModule::load_on_device(String::from(artifact.to_string_lossy()), device)
        }
        .map_err(|error| ScorerError::Message(format!("tch load failed: {error}")))?;
        let check = read_check(artifact, &format!("trace-check-{device_tag}.json"))?;
        Ok(TchEngine {
            module,
            device,
            metadata: base_metadata(source, revision, dtype, device_tag, artifact),
            trace_width: json_int_field(&check, "trace_width"),
            pad_id: json_int_field(&check, "pad_id") as i64,
            kind: EngineKind::Direct,
            no_id: 0,
            yes_id: 0,
        })
    }

    /// Load a traced reranker module (`qwen3reranker-{context}.pt`).
    ///
    /// The sidecar also pins the yes/no vocabulary positions, so the engine
    /// never re-derives the answer contract from the tokenizer at load time —
    /// `answer_ids` re-verifies it per run instead.
    pub fn new_reranker(
        artifact: &std::path::Path,
        device: tch::Device,
        source: &str,
        revision: &str,
        dtype: &str,
        device_tag: &str,
    ) -> Result<Self, ScorerError> {
        if device != tch::Device::Cpu {
            preload_libtorch_cuda();
        }
        let module = if device == tch::Device::Cpu {
            tch::CModule::load(String::from(artifact.to_string_lossy()))
        } else {
            tch::CModule::load_on_device(String::from(artifact.to_string_lossy()), device)
        }
        .map_err(|error| ScorerError::Message(format!("tch load failed: {error}")))?;
        let check = read_check(artifact, &format!("trace-check-reranker-{device_tag}.json"))?;
        Ok(TchEngine {
            module,
            device,
            metadata: base_metadata(source, revision, dtype, device_tag, artifact),
            trace_width: json_int_field(&check, "trace_width"),
            pad_id: json_int_field(&check, "pad_id") as i64,
            kind: EngineKind::Reranker,
            no_id: json_int_field(&check, "no_id") as i64,
            yes_id: json_int_field(&check, "yes_id") as i64,
        })
    }

    /// Last-position full-vocabulary logits (f32) for one encoded row.
    fn full_logits(&self, ids: &[u32]) -> Result<Vec<f32>, ScorerError> {
        if ids.len() > self.trace_width {
            return Err(ScorerError::Message(format!(
                "row needs {} tokens, trace width is {}",
                ids.len(),
                self.trace_width
            )));
        }
        let mut padded: Vec<i64> = ids.iter().map(|&token| token as i64).collect();
        padded.resize(self.trace_width, self.pad_id);
        let input = tch::Tensor::from_slice(&padded)
            .unsqueeze(0)
            .to(self.device);
        let length = tch::Tensor::from_slice(&[ids.len() as i64]).to(self.device);
        let output: Result<tch::Tensor, ScorerError> = tch::no_grad(|| {
            self.module
                .forward_ts(&[input, length])
                .map_err(|error| ScorerError::Message(format!("tch forward failed: {error}")))
        });
        let output = output?;
        let flat = output.reshape([-1]).to_kind(tch::Kind::Float);
        let size = flat.numel() as usize;
        let mut values = vec![0f32; size];
        flat.copy_data(&mut values, size);
        Ok(values)
    }

    /// One forward over a row's option pairs, returning per-option vocabulary
    /// logits (f32). `row_width` is the widest pair in the row — the width
    /// Python would have padded this batch to.
    ///
    /// Layout per option: `[pad] * (W - L) + ids` so the real block ends at
    /// `W - 1` (making `logits[:, -1, :]` the last real token), with
    /// `position_ids = index - (W - row_width)` on real tokens so they land on
    /// the same RoPE offsets Python's width-`row_width` batch gives them.
    pub fn reranker_vocab(
        &self,
        pairs: &[Vec<u32>],
        row_width: usize,
    ) -> Result<Vec<Vec<f32>>, ScorerError> {
        if self.kind != EngineKind::Reranker {
            return Err(ScorerError::Message(
                "engine does not carry a reranker artifact".into(),
            ));
        }
        let width = self.trace_width;
        let longest = pairs.iter().map(Vec::len).max().unwrap_or(0);
        if pairs.is_empty() || longest == 0 {
            return Err(ScorerError::Message("Pair batch is empty".into()));
        }
        let row_width = row_width.max(longest);
        if longest > width {
            return Err(ScorerError::Message(format!(
                "row needs {longest} tokens, trace width is {width}"
            )));
        }

        let mut padded = Vec::with_capacity(pairs.len() * width);
        let mut mask = Vec::with_capacity(pairs.len() * width);
        for ids in pairs {
            let left = width - ids.len();
            padded.extend(std::iter::repeat_n(self.pad_id, left));
            padded.extend(ids.iter().map(|&token| token as i64));
            mask.extend(std::iter::repeat_n(0i64, left));
            mask.extend(std::iter::repeat_n(1i64, ids.len()));
        }

        let offset = width - row_width;
        let one_row: Vec<i64> = (0..width)
            .map(|index| {
                if index < offset {
                    1
                } else {
                    index as i64 - offset as i64
                }
            })
            .collect();
        let position: Vec<i64> = (0..pairs.len())
            .flat_map(|_| one_row.iter().copied())
            .collect();

        let shape = [pairs.len() as i64, width as i64];
        let input_ids = tch::Tensor::from_slice(&padded)
            .reshape(shape)
            .to(self.device);
        let attention_mask = tch::Tensor::from_slice(&mask)
            .reshape(shape)
            .to(self.device);
        let position_ids = tch::Tensor::from_slice(&position)
            .reshape(shape)
            .to(self.device);
        let output: tch::Tensor = tch::no_grad(|| {
            self.module
                .forward_ts(&[input_ids, attention_mask, position_ids])
                .map_err(|error| ScorerError::Message(format!("tch forward failed: {error}")))
        })?;
        let flat = output.reshape([-1]).to_kind(tch::Kind::Float);
        let total = flat.numel() as usize;
        let mut values = vec![0f32; total];
        flat.copy_data(&mut values, total);
        let stride_vocab = total / pairs.len();
        Ok(values.chunks(stride_vocab).map(<[f32]>::to_vec).collect())
    }
}

fn logsumexp_f32(values: &[f32]) -> f64 {
    let peak = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let peak = peak as f64;
    let mut total = 0.0f32;
    for value in values {
        total += (*value - peak as f32).exp();
    }
    peak + (total.ln()) as f64
}

fn logsumexp_f64(values: &[f64]) -> f64 {
    let peak = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0f64;
    for value in values {
        total += (*value - peak).exp();
    }
    peak + total.ln()
}

/// Two-value f32 softmax, mirroring `Tensor.softmax(-1)` over `[no, yes]`.
fn softmax2_yes(no: f32, yes: f32) -> f32 {
    let peak = no.max(yes);
    let denominator = (no - peak).exp() + (yes - peak).exp();
    (yes - peak).exp() / denominator
}

impl EngineTrait for TchEngine {
    fn score_direct(
        &self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        if self.kind != EngineKind::Direct {
            return Err(ScorerError::Message(
                "engine carries the reranker artifact; use --mode reranker".into(),
            ));
        }
        let (ids, slots, prompt_hash) =
            semif_core::tokenizer::encode_prompt(context.tokenizer, row, context.max_tokens)?;
        let vocabulary = self.full_logits(&ids)?;
        let selected: Vec<f64> = slots
            .iter()
            .map(|slot| vocabulary[*slot as usize] as f64)
            .collect();
        let probabilities = semif_core::softmax::softmax(&selected)
            .map_err(|e| ScorerError::Message(e.to_string()))?;
        let vocab_mass = logsumexp_f32(&vocabulary);
        let slot_mass = logsumexp_f64(&selected);
        let allowed_token_mass = (slot_mass - vocab_mass).exp();
        let full_vocab_argmax_id =
            vocabulary
                .iter()
                .enumerate()
                .fold(0usize, |best, (index, value)| {
                    if *value > vocabulary[best] {
                        index
                    } else {
                        best
                    }
                });
        let logits_sha256 = sha256_f32_le(&vocabulary);
        let row_id = row.get("id").and_then(PyValue::as_str).unwrap_or_default();
        let option_ids: Vec<PyValue> = row
            .get("options")
            .and_then(PyValue::as_array)
            .map(|options| {
                options
                    .iter()
                    .map(|o| {
                        PyValue::Str(
                            o.get("id")
                                .and_then(PyValue::as_str)
                                .unwrap_or_default()
                                .into(),
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut metadata_entries = match &self.metadata {
            PyValue::Object(entries) => entries.clone(),
            _ => Vec::new(),
        };
        metadata_entries.push((
            "serving_config".into(),
            PyValue::Str("tch-trace-direct-v1".into()),
        ));

        Ok(PyValue::Object(vec![
            ("id".into(), PyValue::Str(row_id.into())),
            ("option_ids".into(), PyValue::Array(option_ids)),
            (
                "probabilities".into(),
                PyValue::Array(probabilities.into_iter().map(PyValue::Float).collect()),
            ),
            (
                "option_logits".into(),
                PyValue::Array(selected.into_iter().map(PyValue::Float).collect()),
            ),
            (
                "allowed_token_mass".into(),
                PyValue::Float(allowed_token_mass),
            ),
            (
                "full_vocab_argmax_id".into(),
                PyValue::Int(full_vocab_argmax_id.to_string()),
            ),
            ("logits_sha256".into(), PyValue::Str(logits_sha256)),
            ("input_tokens".into(), PyValue::Int(ids.len().to_string())),
            ("prompt_sha256".into(), PyValue::Str(prompt_hash)),
            (
                "prompt_version".into(),
                PyValue::Str(semif_core::PROMPT_VERSION.into()),
            ),
            ("model".into(), PyValue::Object(metadata_entries)),
            (
                "readout".into(),
                PyValue::Str(
                    "traced full-vocabulary last-position logits; no generated tokens".into(),
                ),
            ),
            (
                "probability_status".into(),
                PyValue::Str(
                    "conditional option score; uncalibrated as decision confidence".into(),
                ),
            ),
        ]))
    }

    fn score_serial(
        &mut self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        self.score_direct(row, context)
    }

    fn score_shared(
        &mut self,
        rows: &[PyValue],
        context: &ScoreContext<'_>,
    ) -> Result<(Vec<PyValue>, Option<PyValue>), ScorerError> {
        let results = rows
            .iter()
            .map(|row| self.score_direct(row, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok((results, None))
    }

    fn score_reranker(
        &self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        if self.kind != EngineKind::Reranker {
            return Err(ScorerError::Message(
                "engine does not carry a reranker artifact".into(),
            ));
        }
        semif_types::validate_row(row).map_err(|error| ScorerError::Message(error.to_string()))?;
        let started = std::time::Instant::now();
        let (no_id, yes_id) = semif_core::reranker::answer_ids(context.tokenizer)?;
        // Python derives the answer tokens at runtime; the exporter recorded
        // them at export time. Both must agree, or the trace's gathered columns
        // would silently describe different tokens than the tokenizer does.
        if no_id as i64 != self.no_id || yes_id as i64 != self.yes_id {
            return Err(ScorerError::Message(format!(
                "reranker answer tokens drifted from the trace export: tokenizer (no={no_id}, \
                 yes={yes_id}) vs export (no={}, yes={})",
                self.no_id, self.yes_id
            )));
        }
        let options = row
            .get("options")
            .and_then(PyValue::as_array)
            .ok_or_else(|| ScorerError::Message("options must contain 2-16 entries".into()))?;

        let mut pairs = Vec::with_capacity(options.len());
        let mut prompt_hashes = Vec::with_capacity(options.len());
        let mut lengths = Vec::with_capacity(options.len());
        for option in options {
            let (ids, hash) = semif_core::reranker::encode_pair(
                context.tokenizer,
                row,
                option,
                context.max_tokens,
            )?;
            lengths.push(ids.len());
            pairs.push(ids);
            prompt_hashes.push(hash);
        }
        let row_width = lengths.iter().copied().max().unwrap_or(0);
        if row_width > self.trace_width {
            return Err(ScorerError::Message(format!(
                "row needs {row_width} tokens, trace width is {}",
                self.trace_width
            )));
        }

        let forward_started = std::time::Instant::now();
        let vocabulary = self.reranker_vocab(&pairs, row_width)?;
        // The readout copies the result to the host, which synchronizes the
        // device, so this duration is honest rather than launch-only.
        let forward_seconds = forward_started.elapsed().as_secs_f64();

        let mut log_odds = Vec::with_capacity(pairs.len());
        let mut binary_relevance = Vec::with_capacity(pairs.len());
        for logits in &vocabulary {
            let no = *logits.get(no_id as usize).ok_or_else(|| {
                ScorerError::Message("reranker no-token outside the vocabulary".into())
            })?;
            let yes = *logits.get(yes_id as usize).ok_or_else(|| {
                ScorerError::Message("reranker yes-token outside the vocabulary".into())
            })?;
            log_odds.push((yes - no) as f64);
            binary_relevance.push(f64::from(softmax2_yes(no, yes)));
        }
        let probabilities = semif_core::softmax::softmax(&log_odds)
            .map_err(|error| ScorerError::Message(error.to_string()))?;

        let row_id = row.get("id").and_then(PyValue::as_str).unwrap_or_default();
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
        let total_seconds = started.elapsed().as_secs_f64();

        let mut metadata_entries = match &self.metadata {
            PyValue::Object(entries) => entries.clone(),
            _ => Vec::new(),
        };
        metadata_entries.push((
            "serving_config".into(),
            PyValue::Str("tch-trace-reranker-v1".into()),
        ));

        Ok(PyValue::Object(vec![
            ("id".into(), PyValue::Str(row_id.into())),
            ("option_ids".into(), PyValue::Array(option_ids)),
            (
                "probabilities".into(),
                PyValue::Array(probabilities.into_iter().map(PyValue::Float).collect()),
            ),
            (
                "option_logits".into(),
                PyValue::Array(log_odds.into_iter().map(PyValue::Float).collect()),
            ),
            (
                "independent_binary_relevance".into(),
                PyValue::Array(binary_relevance.into_iter().map(PyValue::Float).collect()),
            ),
            (
                "input_tokens".into(),
                PyValue::Int(lengths.iter().sum::<usize>().to_string()),
            ),
            (
                "max_option_input_tokens".into(),
                PyValue::Int(row_width.to_string()),
            ),
            ("forward_seconds".into(), PyValue::Float(forward_seconds)),
            ("total_seconds".into(), PyValue::Float(total_seconds)),
            (
                "option_prompt_sha256".into(),
                PyValue::Array(prompt_hashes.into_iter().map(PyValue::Str).collect()),
            ),
            (
                "pair_batches".into(),
                PyValue::Array(vec![PyValue::Object(vec![
                    (
                        "pair_batch_size".into(),
                        PyValue::Int(options.len().to_string()),
                    ),
                    ("forward_seconds".into(), PyValue::Float(forward_seconds)),
                    // The logical batch is `row_width` wide; this engine pads
                    // every row to the fixed trace width instead. Both numbers
                    // are reported so the difference is visible, not implied.
                    (
                        "padded_tokens".into(),
                        PyValue::Int((options.len() * self.trace_width).to_string()),
                    ),
                    (
                        "trace_width".into(),
                        PyValue::Int(self.trace_width.to_string()),
                    ),
                ])]),
            ),
            (
                "prompt_version".into(),
                PyValue::Str(semif_types::RERANKER_PROMPT_VERSION.into()),
            ),
            ("model".into(), PyValue::Object(metadata_entries)),
            (
                "readout".into(),
                PyValue::Str(
                    "native yes/no log-odds per option, normalized only for relative comparison"
                        .into(),
                ),
            ),
            (
                "probability_status".into(),
                PyValue::Str(
                    "relative option compatibility; uncalibrated as categorical probability".into(),
                ),
            ),
        ]))
    }
}

fn sha256_f32_le(values: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>()
}
