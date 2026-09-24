//! Stage 1 corpus gates: every fixture must reproduce exactly.
//!
//! Fixtures live in `../../fixtures` (workspace `semif-rs/fixtures/`), authored
//! by the Python oracle (`benchmarks/export_fixtures.py` and
//! `benchmarks/export_logic_fixtures.py`). They are read with the crate's own
//! Python-compatible parser (they contain NaN literals where Python wrote
//! them). The tokenizer directory comes from the manifest's `loader_source`.

use semif_core::parser::parse;
use semif_core::prefix::state_prefix;
use semif_core::prompt::build_prompt;
use semif_core::pyfloat::py_repr;
use semif_core::pyjson::{dumps_output, dumps_payload};
use semif_core::softmax::softmax;
use semif_core::tokenizer::{ReferenceTokenizer, encode_prompt};
use semif_types::{PyValue, validate_row};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");

fn fixtures() -> PathBuf {
    PathBuf::from(FIXTURES)
}

fn read_text(name: &str) -> String {
    std::fs::read_to_string(fixtures().join(name)).expect("fixture file")
}

fn read_lines(name: &str) -> Vec<PyValue> {
    read_text(name)
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
        .unwrap()
        .iter()
        .map(|item| match item {
            PyValue::Int(digits) => digits.parse().unwrap(),
            PyValue::Float(f) => *f as u32,
            other => panic!("not a number: {other:?}"),
        })
        .collect()
}

fn f64_list(value: &PyValue) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| match item {
            PyValue::Int(digits) => digits.parse().unwrap(),
            PyValue::Float(f) => *f,
            other => panic!("not a number: {other:?}"),
        })
        .collect()
}

fn optional_str(value: &PyValue, key: &str) -> Option<String> {
    match value.get(key) {
        Some(PyValue::Str(text)) => Some(text.clone()),
        Some(PyValue::Null) | None => None,
        other => panic!("unexpected {key}: {other:?}"),
    }
}

fn inputs_by_id() -> HashMap<String, PyValue> {
    read_lines("inputs.jsonl")
        .into_iter()
        .map(|row| (str_field(&row, "id").to_string(), row))
        .collect()
}

fn tokenizer() -> &'static ReferenceTokenizer {
    static TOKENIZER: OnceLock<ReferenceTokenizer> = OnceLock::new();
    TOKENIZER.get_or_init(|| {
        let manifest = parse(&read_text("manifest.json")).expect("manifest");
        let loader = str_field(&manifest, "loader_source");
        ReferenceTokenizer::from_checkpoint_dir(Path::new(loader)).expect("tokenizer")
    })
}

#[test]
fn prompt_template_byte_equality_over_corpus() {
    let inputs = inputs_by_id();
    let prompts = read_lines("prompts.jsonl");
    assert!(!prompts.is_empty());
    for prompt in &prompts {
        let id = str_field(prompt, "id");
        let row = &inputs[id];
        let (text, hash) = build_prompt(row).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(text, str_field(prompt, "prompt"), "{id}: prompt text");
        assert_eq!(
            hash,
            str_field(prompt, "prompt_sha256"),
            "{id}: prompt hash"
        );
    }
}

#[test]
fn token_ids_slots_and_prefix_parity_over_corpus() {
    let inputs = inputs_by_id();
    let tokens = read_lines("tokens.jsonl");
    assert!(!tokens.is_empty());
    let tokenizer = tokenizer();
    for token in &tokens {
        let id = str_field(token, "id");
        let row = &inputs[id];
        let (ids, slots, _) =
            encode_prompt(tokenizer, row, 4096).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(ids, u32_list(token.get("ids").unwrap()), "{id}: token ids");
        assert_eq!(slots, u32_list(token.get("slots").unwrap()), "{id}: slots");

        let prefix = state_prefix(tokenizer, row.get("state").unwrap())
            .unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(
            prefix,
            u32_list(token.get("prefix_ids").unwrap()),
            "{id}: prefix ids"
        );
    }
}

#[test]
fn validation_and_encode_verdicts_over_corpus() {
    let inputs = inputs_by_id();
    let verdicts = read_lines("rows.jsonl");
    assert!(!verdicts.is_empty());
    let tokenizer = tokenizer();
    for verdict in &verdicts {
        let id = str_field(verdict, "id");
        let row = &inputs[id];
        let validation = validate_row(row).err().map(|e| e.to_string());
        assert_eq!(
            validation,
            optional_str(verdict, "validation_error"),
            "{id}: validation verdict"
        );
        if verdict
            .get("validation_error")
            .is_some_and(|v| !matches!(v, PyValue::Null))
        {
            continue;
        }
        let encode_error = match encode_prompt(tokenizer, row, 4096) {
            Ok(_) => None,
            Err(error) => Some(error.to_string()),
        };
        assert_eq!(
            encode_error,
            optional_str(verdict, "encode_error"),
            "{id}: encode verdict"
        );
    }
}

#[test]
fn python_float_repr_cases() {
    let logic = parse(&read_text("logic.json")).expect("logic.json");
    for case in logic.get("float_reprs").unwrap().as_array().unwrap() {
        let value: f64 = str_field(case, "input").parse().unwrap();
        assert_eq!(
            py_repr(value),
            str_field(case, "output"),
            "input {}",
            str_field(case, "input")
        );
    }
}

#[test]
fn python_dumps_profiles() {
    let logic = parse(&read_text("logic.json")).expect("logic.json");
    for case in logic.get("dumps").unwrap().as_array().unwrap() {
        let value = parse(str_field(case, "input")).unwrap();
        assert_eq!(
            dumps_payload(&value).unwrap(),
            str_field(case, "ensure_ascii_false"),
            "input {}",
            str_field(case, "input")
        );
        assert_eq!(
            dumps_output(&value).unwrap(),
            str_field(case, "ensure_ascii_true"),
            "input {}",
            str_field(case, "input")
        );
    }
}

#[test]
fn python_softmax_vectors() {
    let logic = parse(&read_text("logic.json")).expect("logic.json");
    for (name, case) in logic.get("softmax").unwrap().as_object().unwrap() {
        let input = f64_list(case.get("input").unwrap());
        match softmax(&input) {
            Ok(output) => {
                let expected: Vec<String> = case
                    .get("output")
                    .unwrap()
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|item| match item {
                        PyValue::Int(digits) => digits.clone(),
                        PyValue::Float(f) => py_repr(*f),
                        other => panic!("not a number: {other:?}"),
                    })
                    .collect();
                for (index, actual) in output.iter().enumerate() {
                    assert_eq!(py_repr(*actual), expected[index], "{name}: element {index}");
                }
            }
            Err(error) => {
                assert_eq!(error.to_string(), str_field(case, "error"), "{name}");
            }
        }
    }
}
