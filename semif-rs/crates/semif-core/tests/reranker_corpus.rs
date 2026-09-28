//! Stage 3 reranker fixture gates: the Python-authored pair prompts, token
//! sequences, answer contract, and refusal messages must reproduce exactly.
//!
//! Fixtures are `../../fixtures/reranker_*.{json,jsonl}`, authored by
//! `benchmarks/export_reranker_fixtures.py` against the pinned
//! Qwen3-Reranker-4B checkpoint. The tokenizer directory comes from
//! `reranker_manifest.json` rather than the direct-mode manifest.

use semif_core::parser::parse;
use semif_core::reranker::{answer_ids, build_pair_prompt, encode_pair};
use semif_core::tokenizer::ReferenceTokenizer;
use semif_types::PyValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");

fn fixtures() -> PathBuf {
    PathBuf::from(FIXTURES)
}

fn read_lines(name: &str) -> Vec<PyValue> {
    std::fs::read_to_string(fixtures().join(name))
        .expect("fixture file")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse(line).expect("fixture line"))
        .collect()
}

fn str_field<'a>(value: &'a PyValue, key: &str) -> &'a str {
    value.get(key).and_then(PyValue::as_str).unwrap_or_default()
}

fn u32_list(value: &PyValue) -> Vec<u32> {
    value
        .as_array()
        .unwrap_or_else(|| panic!("{value:?} is not a list"))
        .iter()
        .map(|item| match item {
            PyValue::Int(digits) => digits.parse().unwrap(),
            PyValue::Float(number) => *number as u32,
            other => panic!("not a number: {other:?}"),
        })
        .collect()
}

fn rows_by_id() -> HashMap<String, PyValue> {
    // The reranker corpus spans three sources plus two synthetic
    // instruction-branch clones, so it ships its own row file instead of
    // being reconstructed from `inputs.jsonl`.
    read_lines("reranker_rows.jsonl")
        .into_iter()
        .map(|row| (str_field(&row, "id").to_string(), row))
        .collect()
}

fn tokenizer() -> &'static ReferenceTokenizer {
    static TOKENIZER: OnceLock<ReferenceTokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let manifest = parse(
            &std::fs::read_to_string(fixtures().join("reranker_manifest.json")).expect("manifest"),
        )
        .expect("manifest");
        let source = str_field(&manifest, "tokenizer");
        ReferenceTokenizer::from_checkpoint_dir(Path::new(source)).expect("reranker tokenizer")
    })
}

#[test]
fn pair_prompt_text_and_hash_byte_equality() {
    let rows = rows_by_id();
    let prompts = read_lines("reranker_prompts.jsonl");
    assert!(!prompts.is_empty());
    for prompt in &prompts {
        let id = str_field(prompt, "id");
        let row = rows.get(id).unwrap_or_else(|| panic!("missing row {id}"));
        let options = row.get("options").and_then(PyValue::as_array).unwrap();
        let index: usize = match prompt.get("option_index") {
            Some(PyValue::Int(digits)) => digits.parse().unwrap(),
            other => panic!("bad option_index: {other:?}"),
        };
        let option = &options[index];
        assert_eq!(
            str_field(option, "id"),
            str_field(prompt, "option_id"),
            "{id}: option at index {index}"
        );
        let (text, hash) =
            build_pair_prompt(row, option).unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(text, str_field(prompt, "prompt"), "{id}: prompt text");
        assert_eq!(
            hash,
            str_field(prompt, "prompt_sha256"),
            "{id}: prompt hash"
        );
    }
}

#[test]
fn pair_token_ids_and_row_width_parity() {
    let rows = rows_by_id();
    let tokens = read_lines("reranker_tokens.jsonl");
    assert!(!tokens.is_empty());
    let tokenizer = tokenizer();
    for token in &tokens {
        let id = str_field(token, "id");
        let row = rows.get(id).unwrap_or_else(|| panic!("missing row {id}"));
        let options = row.get("options").and_then(PyValue::as_array).unwrap();
        let recorded = token.get("options").and_then(PyValue::as_array).unwrap();
        assert_eq!(recorded.len(), options.len(), "{id}: option count");

        let mut width = 0usize;
        for (index, entry) in recorded.iter().enumerate() {
            let option = &options[index];
            assert_eq!(
                str_field(option, "id"),
                str_field(entry, "option_id"),
                "{id}: option order"
            );
            // Python's default reranker budget, as the fixture recorded it.
            let (ids, _) = encode_pair(tokenizer, row, option, 4096)
                .unwrap_or_else(|error| panic!("{id}: {error}"));
            assert_eq!(ids, u32_list(entry.get("ids").unwrap()), "{id} ids");
            width = width.max(ids.len());
        }
        let recorded_width = match token.get("width") {
            Some(PyValue::Int(digits)) => digits.parse::<usize>().unwrap(),
            other => panic!("bad width: {other:?}"),
        };
        assert_eq!(width, recorded_width, "{id}: row width");
    }
}

#[test]
fn yes_no_answer_contract() {
    let manifest_text =
        std::fs::read_to_string(fixtures().join("reranker_answers.json")).expect("answers");
    let answers = parse(&manifest_text).expect("answers json");
    let (no, yes) = answer_ids(tokenizer()).expect("answer contract");
    let expected_no = u32_list(answers.get("no_ids").unwrap())[0];
    let expected_yes = u32_list(answers.get("yes_ids").unwrap())[0];
    assert_eq!(no, expected_no, "no token");
    assert_eq!(yes, expected_yes, "yes token");
    // `convert_tokens_to_ids` is a separate observation from the encode path.
    assert_eq!(
        tokenizer().token_to_id("no"),
        Some(expected_no),
        "vocab lookup for no"
    );
    assert_eq!(
        tokenizer().token_to_id("yes"),
        Some(expected_yes),
        "vocab lookup for yes"
    );
}

#[test]
fn over_budget_refusal_messages() {
    let rows = rows_by_id();
    let refusals = read_lines("reranker_refusals.jsonl");
    assert!(!refusals.is_empty());
    let tokenizer = tokenizer();
    for refusal in &refusals {
        let id = str_field(refusal, "id");
        let row = rows.get(id).unwrap_or_else(|| panic!("missing row {id}"));
        let option_id = str_field(refusal, "option_id");
        let options = row.get("options").and_then(PyValue::as_array).unwrap();
        let option = options
            .iter()
            .find(|option| str_field(option, "id") == option_id)
            .unwrap_or_else(|| panic!("{id}: missing option {option_id}"));
        let budget = match refusal.get("max_tokens") {
            Some(PyValue::Int(digits)) => digits.parse::<usize>().unwrap(),
            other => panic!("bad max_tokens: {other:?}"),
        };
        let error = encode_pair(tokenizer, row, option, budget)
            .expect_err("budget is deliberately one short of the prompt");
        assert_eq!(
            error.to_string(),
            str_field(refusal, "message"),
            "{id}/{option_id}: refusal message"
        );
    }
}
