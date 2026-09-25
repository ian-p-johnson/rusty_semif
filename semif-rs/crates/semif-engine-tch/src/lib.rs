//! Stage 3 CUDA/BF16 engine: the TorchScript-traced Qwen3.5 direct-mode graph
//! executed through `tch` against the oracle venv's own libtorch.
//!
//! Route decision (Stage 3, operator-approved): torch.jit.trace of the pinned
//! model replaces the hand-written 32-layer hybrid stack. Direct mode is
//! faithfully traced; serial/shared run as documented fresh-recompute (new
//! serving_config) until a hand-written cache exists. Gates are unchanged:
//! CPU fp32 ≤1e-4 vs `logits-cpufp32.jsonl`; CUDA BF16 snap ≥99% + decision
//! ≥99% vs the device-tagged `logits-cuda-bf16-rtx5070ti-laptop.jsonl`.

use semif_core::tokenizer::{ScorerError, encode_prompt};
use semif_engine::{Engine as EngineTrait, ScoreContext};
use semif_types::PyValue;
use std::path::Path;

pub struct TchEngine {
    module: tch::CModule,
    device: tch::Device,
    metadata: PyValue,
    trace_width: usize,
    pad_id: i64,
}

/// Device selection for a trace context name.
pub fn device_for_context(context: &str) -> tch::Device {
    if context == "cudabf16" {
        tch::Device::Cuda(0)
    } else {
        tch::Device::Cpu
    }
}

impl TchEngine {
    /// Load a traced direct-mode module (`qwen35-direct-{context}.pt`).
    /// CUDA kernels register via static initializers, so libtorch_cuda must be
    /// loaded (RTLD_GLOBAL) before any CUDA tensor work — the same thing
    /// Python's `_load_global_deps` does at import time.
    pub fn new(
        artifact: &std::path::Path,
        device: tch::Device,
        source: &str,
        revision: &str,
        dtype: &str,
        device_tag: &str,
    ) -> Result<Self, ScorerError> {
        if device != tch::Device::Cpu {
            let candidates = [
                std::path::PathBuf::from(
                    ".venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so",
                ),
                std::path::PathBuf::from(
                    "../.venv/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so",
                ),
                std::path::PathBuf::from("target/debug/libtorch_cuda.so"),
            ];
            let path = std::env::var("SEMIF_TORCH_LIB")
                .map(std::path::PathBuf::from)
                .ok()
                .or_else(|| candidates.into_iter().find(|p| p.exists()));
            if let Some(path) = path {
                // RTLD_GLOBAL like Python's _load_global_deps; keep the handle forever.
                match unsafe { libloading::os::unix::Library::open(Some(&path), 0x101) } {
                    Ok(library) => std::mem::forget(library),
                    Err(error) => eprintln!("warning: failed to preload {path:?}: {error}"),
                }
            } else {
                eprintln!(
                    "warning: CUDA requested but libtorch_cuda not found; set SEMIF_TORCH_LIB"
                );
            }
        }
        let module = if device == tch::Device::Cpu {
            tch::CModule::load(String::from(artifact.to_string_lossy()))
        } else {
            tch::CModule::load_on_device(String::from(artifact.to_string_lossy()), device)
        }
        .map_err(|error| ScorerError::Message(format!("tch load failed: {error}")))?;
        let metadata = PyValue::Object(vec![
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
        ]);
        let check_json = std::fs::read_to_string(
            artifact
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("trace-check-{device_tag}.json")),
        )
        .map_err(|error| ScorerError::Message(format!("trace-check missing: {error}")))?;
        let trace_width = json_int_field(&check_json, "trace_width");
        let pad_id = json_int_field(&check_json, "pad_id") as i64;
        Ok(TchEngine {
            module,
            device,
            metadata,
            trace_width,
            pad_id,
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
}

/// Minimal flat-JSON integer field reader (avoids pulling serde into the engine).
fn json_int_field(text: &str, key: &str) -> usize {
    let pattern = format!("\"{key}\": ");
    let start = text.find(&pattern).expect("field in trace-check") + pattern.len();
    let digits: String = text[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().expect("integer field")
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

impl EngineTrait for TchEngine {
    fn score_direct(
        &self,
        row: &PyValue,
        context: &ScoreContext<'_>,
    ) -> Result<PyValue, ScorerError> {
        let (ids, slots, prompt_hash) = encode_prompt(context.tokenizer, row, context.max_tokens)?;
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
        let logits_sha256 = {
            use sha2::{Digest, Sha256};
            let mut hasher = Sha256::new();
            for value in &vocabulary {
                hasher.update(value.to_le_bytes());
            }
            let digest = hasher.finalize();
            digest
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
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
}
