//! Result-row construction and mode flows, mirroring
//! `llamacpp_backend._result` / `score` / `SerialPrefixScorer` / `score_shared`.

use crate::backend::Engine;
use crate::ffi::LlamaToken;
use semif_core::digest;
use semif_core::tokenizer::ScorerError;
use semif_types::PyValue;

/// `_logsumexp(numpy.asarray(selected))` — Python floats, so numpy makes an
/// f64 array: peak/shift/exp/sum all in f64.
fn logsumexp_f64(values: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    for value in values {
        if *value > peak {
            peak = *value;
        }
    }
    let mut total = 0.0f64;
    for value in values {
        total += (*value - peak).exp();
    }
    peak + total.ln()
}

/// `_logsumexp(vocabulary)` — the raw engine vocabulary is an f32 array:
/// shift/exp/sum in f32, lifted to f64 only at the end. numpy's pairwise
/// summation differs from a sequential fold by ~1e-7 relative on 248k
/// positive terms, far inside the 1e-4 scalar gate.
fn logsumexp_f32(values: &[f32]) -> f64 {
    let mut peak = f32::NEG_INFINITY;
    for value in values {
        if *value > peak {
            peak = *value;
        }
    }
    let peak = peak as f64;
    let mut total = 0.0f32;
    for value in values {
        total += (*value - peak as f32).exp();
    }
    peak + (total.ln()) as f64
}

/// `int(vocabulary.argmax())`: first maximal index.
fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn result_row(
    row: &PyValue,
    ids: &[LlamaToken],
    slots: &[LlamaToken],
    prompt_hash: &str,
    vocabulary: &[f32],
    metadata: &PyValue,
    serving_config: &str,
    readout: &str,
) -> Result<PyValue, ScorerError> {
    let selected: Vec<f64> = slots
        .iter()
        .map(|slot| vocabulary[*slot as usize] as f64)
        .collect();
    let probabilities =
        semif_core::softmax::softmax(&selected).map_err(|e| ScorerError::Message(e.to_string()))?;
    let mass = (logsumexp_f64(&selected) - logsumexp_f32(vocabulary)).exp();

    let mut model_entries = match metadata {
        PyValue::Object(entries) => entries.clone(),
        _ => Vec::new(),
    };
    model_entries.push(("serving_config".into(), PyValue::Str(serving_config.into())));

    Ok(PyValue::Object(vec![
        ("id".into(), PyValue::Str(str_field(row, "id").into())),
        (
            "option_ids".into(),
            PyValue::Array(option_ids(row)),
        ),
        ("probabilities".into(), PyValue::Array(probabilities.into_iter().map(PyValue::Float).collect())),
        ("option_logits".into(), PyValue::Array(selected.into_iter().map(PyValue::Float).collect())),
        (
            "answer_token_ids".into(),
            PyValue::Array(slots.iter().map(|slot| PyValue::Int((*slot).to_string())).collect()),
        ),
        ("input_tokens".into(), PyValue::Int(ids.len().to_string())),
        ("allowed_token_mass".into(), PyValue::Float(mass)),
        ("full_vocab_argmax_id".into(), PyValue::Int(argmax(vocabulary).to_string())),
        ("prompt_sha256".into(), PyValue::Str(prompt_hash.into())),
        ("prompt_version".into(), PyValue::Str(semif_core::PROMPT_VERSION.into())),
        ("model".into(), PyValue::Object(model_entries)),
        ("readout".into(), PyValue::Str(readout.into())),
        (
            "probability_status".into(),
            PyValue::Str(
                "conditional option score over quantized weights; uncalibrated as decision confidence".into(),
            ),
        ),
    ]))
}

fn str_field<'a>(row: &'a PyValue, key: &str) -> &'a str {
    row.get(key).and_then(PyValue::as_str).unwrap_or_default()
}

fn option_ids(row: &PyValue) -> Vec<PyValue> {
    row.get("options")
        .and_then(PyValue::as_array)
        .map(|options| {
            options
                .iter()
                .map(|option| PyValue::Str(str_field(option, "id").into()))
                .collect()
        })
        .unwrap_or_default()
}

pub fn metadata(engine: &Engine, source: &str, revision: &str) -> PyValue {
    PyValue::Object(vec![
        ("source".into(), PyValue::Str(source.into())),
        ("revision".into(), PyValue::Str(revision.into())),
        ("backend".into(), PyValue::Str("llamacpp".into())),
        ("dtype".into(), PyValue::Str("gguf-quantized".into())),
        (
            "gguf".into(),
            PyValue::Object(vec![
                ("file".into(), PyValue::Str(engine.gguf_name.clone())),
                ("bytes".into(), PyValue::Int(engine.gguf_bytes.to_string())),
                ("sha256".into(), PyValue::Str(engine.gguf_sha256.clone())),
            ]),
        ),
        (
            "vocab_size".into(),
            PyValue::Int(engine.vocab_size.to_string()),
        ),
        ("threads".into(), PyValue::Int(engine.threads.to_string())),
        ("n_gpu_layers".into(), PyValue::Int("0".into())),
        (
            "max_prompt_tokens".into(),
            PyValue::Int(engine.max_prompt_tokens.to_string()),
        ),
        (
            "context_tokens".into(),
            PyValue::Int(engine.context_tokens.to_string()),
        ),
        ("decode_chunk".into(), PyValue::Int("512".into())),
        ("llama_cpp_python_version".into(), PyValue::Null),
        ("transformers_version".into(), PyValue::Null),
        (
            "engine".into(),
            PyValue::Str("semif-rs llamacpp (wheel libllama, dlopen)".into()),
        ),
    ])
}

/// `llamacpp_backend.score` (direct mode).
pub fn score_direct(
    engine: &Engine,
    tokenizer: &semif_core::tokenizer::ReferenceTokenizer,
    row: &PyValue,
    metadata_value: &PyValue,
    max_tokens: usize,
) -> Result<PyValue, ScorerError> {
    let started = std::time::Instant::now();
    let (ids, slots, prompt_hash) = engine.encode_verified(tokenizer, row, max_tokens)?;
    let forward_started = std::time::Instant::now();
    let vocabulary = engine.full_logits(&ids)?;
    let forward_seconds = forward_started.elapsed().as_secs_f64();
    let mut row_value = result_row(
        row,
        &ids,
        &slots,
        &prompt_hash,
        &vocabulary,
        metadata_value,
        "llamacpp-direct-v1",
        "quantized last-position logits restricted to declared answer slots; no generated tokens",
    )?;
    if let PyValue::Object(entries) = &mut row_value {
        entries.push(("forward_seconds".into(), PyValue::Float(forward_seconds)));
        entries.push((
            "total_seconds".into(),
            PyValue::Float(started.elapsed().as_secs_f64()),
        ));
    }
    Ok(row_value)
}

/// `llamacpp_backend.SerialPrefixScorer`: persistent branch-restore state.
#[derive(Default)]
pub struct SerialState {
    state_data: Option<Vec<u8>>,
    prefix: Option<Vec<LlamaToken>>,
}

pub fn score_serial(
    engine: &Engine,
    state: &mut SerialState,
    tokenizer: &semif_core::tokenizer::ReferenceTokenizer,
    row: &PyValue,
    metadata_value: &PyValue,
    max_tokens: usize,
) -> Result<PyValue, ScorerError> {
    let (ids, slots, prompt_hash) = engine.encode_verified(tokenizer, row, max_tokens)?;
    let prefix_u32 = semif_core::state_prefix(
        tokenizer,
        row.get("state").ok_or_else(|| {
            ScorerError::Message("State prefix does not match the full prompt".into())
        })?,
    )
    .map_err(|e| ScorerError::Message(e.to_string()))?;
    let prefix: Vec<LlamaToken> = prefix_u32.iter().map(|&t| t as LlamaToken).collect();
    let hit = state.state_data.is_some() && state.prefix.as_deref() == Some(prefix.as_slice());
    if prefix.is_empty() || ids.len() <= prefix.len() || ids[..prefix.len()] != prefix[..] {
        return Err(ScorerError::Message(
            "State prefix does not match the full prompt".into(),
        ));
    }
    let mut prefill_seconds = 0.0f64;
    if !hit {
        engine.clear();
        let started = std::time::Instant::now();
        engine.prefill(&prefix)?;
        prefill_seconds = started.elapsed().as_secs_f64();
        state.state_data = Some(engine.save_state()?);
        state.prefix = Some(prefix.clone());
    }
    let copy_started = std::time::Instant::now();
    let state_bytes = state
        .state_data
        .as_ref()
        .expect("hit or just prefilled")
        .len();
    engine.restore_state(state.state_data.as_ref().expect("hit or just prefilled"))?;
    let copy_seconds = copy_started.elapsed().as_secs_f64();

    let suffix_started = std::time::Instant::now();
    let vocabulary = engine.branch_logits(prefix.len(), &ids[prefix.len()..])?;
    let suffix_seconds = suffix_started.elapsed().as_secs_f64();

    let mut row_value = result_row(
        row,
        &ids,
        &slots,
        &prompt_hash,
        &vocabulary,
        metadata_value,
        "llamacpp-state-restore-v1",
        "quantized branch last-position logits over a restored prefix state",
    )?;
    let prefix_text = semif_core::pyjson::dumps_payload(&PyValue::Array(
        prefix_u32
            .iter()
            .map(|token| PyValue::Int((*token).to_string()))
            .collect(),
    ))
    .map_err(|e| ScorerError::Message(e.to_string()))?;
    let prefix_sha256 = digest(&prefix_text);
    if let PyValue::Object(entries) = &mut row_value {
        entries.push(("cache_hit".into(), PyValue::Bool(hit)));
        entries.push((
            "prefix_tokens".into(),
            PyValue::Int(prefix.len().to_string()),
        ));
        entries.push(("prefix_sha256".into(), PyValue::Str(prefix_sha256)));
        entries.push((
            "branch_state_bytes".into(),
            PyValue::Int(state_bytes.to_string()),
        ));
        entries.push(("prefill_seconds".into(), PyValue::Float(prefill_seconds)));
        entries.push(("copy_seconds".into(), PyValue::Float(copy_seconds)));
        entries.push((
            "suffix_forward_seconds".into(),
            PyValue::Float(suffix_seconds),
        ));
        entries.push((
            "forward_seconds".into(),
            PyValue::Float(prefill_seconds + suffix_seconds),
        ));
        entries.push(("total_seconds".into(), PyValue::Float(0.0)));
    }
    Ok(row_value)
}

/// `llamacpp_backend.score_shared` with its timing block.
pub fn score_shared(
    engine: &Engine,
    tokenizer: &semif_core::tokenizer::ReferenceTokenizer,
    rows: &[PyValue],
    metadata_value: &PyValue,
    max_tokens: usize,
) -> Result<(Vec<PyValue>, PyValue), ScorerError> {
    let started = std::time::Instant::now();
    let mut encoded = Vec::with_capacity(rows.len());
    for row in rows {
        encoded.push(engine.encode_verified(tokenizer, row, max_tokens)?);
    }
    let encode_seconds = started.elapsed().as_secs_f64();
    let state = rows
        .first()
        .and_then(|row| row.get("state"))
        .ok_or_else(|| {
            ScorerError::Message("Shared scoring requires one nonempty exact state".into())
        })?;
    let prefix_u32 = semif_core::state_prefix(tokenizer, state)
        .map_err(|e| ScorerError::Message(e.to_string()))?;
    let prefix: Vec<LlamaToken> = prefix_u32.iter().map(|&t| t as LlamaToken).collect();
    for (row, (ids, _, _)) in rows.iter().zip(&encoded) {
        if prefix.is_empty() || ids.len() <= prefix.len() || ids[..prefix.len()] != prefix[..] {
            return Err(ScorerError::Message(
                "The fixed state prefix does not match every full prompt".into(),
            ));
        }
        let _ = row;
    }
    let branch_started = std::time::Instant::now();
    engine.clear();
    engine.prefill(&prefix)?;
    let state_data = engine.save_state()?;
    let prefill_seconds = branch_started.elapsed().as_secs_f64();
    let mut copy_seconds = 0.0f64;
    let mut suffix_seconds = 0.0f64;

    let mut results = Vec::with_capacity(rows.len());
    for (row, (ids, slots, prompt_hash)) in rows.iter().zip(&encoded) {
        let copy_started = std::time::Instant::now();
        engine.restore_state(&state_data)?;
        copy_seconds += copy_started.elapsed().as_secs_f64();
        let suffix_started = std::time::Instant::now();
        let vocabulary = engine.branch_logits(prefix.len(), &ids[prefix.len()..])?;
        suffix_seconds += suffix_started.elapsed().as_secs_f64();
        results.push(result_row(
            row,
            ids,
            slots,
            prompt_hash,
            &vocabulary,
            metadata_value,
            "llamacpp-state-restore-shared-v1",
            "quantized branch last-position logits over a restored prefix state",
        )?);
    }
    let true_suffix = encoded
        .iter()
        .map(|(ids, _, _)| ids.len() - prefix.len())
        .sum::<usize>();
    let timing = PyValue::Object(vec![
        (
            "total_seconds".into(),
            PyValue::Float(started.elapsed().as_secs_f64()),
        ),
        ("encode_seconds".into(), PyValue::Float(encode_seconds)),
        (
            "prefix_tokens".into(),
            PyValue::Int(prefix.len().to_string()),
        ),
        ("prefill_seconds".into(), PyValue::Float(prefill_seconds)),
        ("replicate_seconds".into(), PyValue::Float(copy_seconds)),
        (
            "suffix_forward_seconds".into(),
            PyValue::Float(suffix_seconds),
        ),
        ("batch_size".into(), PyValue::Int(rows.len().to_string())),
        (
            "branch_state_bytes".into(),
            PyValue::Int(state_data.len().to_string()),
        ),
        (
            "true_suffix_tokens".into(),
            PyValue::Int(true_suffix.to_string()),
        ),
        (
            "padded_suffix_tokens".into(),
            PyValue::Int(true_suffix.to_string()),
        ),
    ]);
    Ok((results, timing))
}
