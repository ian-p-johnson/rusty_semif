//! Reranker pair prompts and the yes/no answer contract,
//! operation-for-operation with `semif_phase1.reranker`.
//!
//! Three surfaces, each byte-gated by `fixtures/reranker_*.jsonl`:
//! - the pair text is `PREFIX + body + SUFFIX` where the state is interpolated
//!   with Python's `str()` — strings pass through raw, dict/list states take
//!   their `repr()`, which is a different writer from the direct-mode payload;
//! - the budget refusal reuses Python's exact message string;
//! - `_answer_ids` verifies `convert_tokens_to_ids`, not just an encode
//!   round-trip, so vocab aliases cannot slip through.

use crate::digest::digest;
use crate::prompt::PromptError;
use crate::pyfloat::py_repr;
use crate::tokenizer::{ReferenceTokenizer, ScorerError};
use semif_types::{
    PyValue, RERANKER_DECISION_INSTRUCTION, RERANKER_PREFIX, RERANKER_RETRIEVAL_EXPERIMENTS,
    RERANKER_RETRIEVAL_INSTRUCTION, RERANKER_SUFFIX, validate_row,
};

/// Python `repr()` of a string: CPython picks `"` only when the text holds a
/// `'` and no `"`; everything else uses `'`.
///
/// Escapes cover `\`, the chosen quote, and non-printables. Rust's
/// `char::is_control` matches Python's `Cc` class; the remaining classes Python
/// treats as non-printable (`Cf`, `Zs` such as NBSP, …) are left raw here. No
/// pinned corpus contains one — a state that does would need a fixture first.
fn repr_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for character in text.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if c.is_control() => out.push_str(&escape_code(c)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// CPython's `\x` / `\u` / `\U` escape, chosen by code-point width.
fn escape_code(character: char) -> String {
    let code = character as u32;
    if code <= 0xFF {
        format!("\\x{code:02x}")
    } else if code <= 0xFFFF {
        format!("\\u{code:04x}")
    } else {
        format!("\\U{code:08x}")
    }
}

/// Python `repr()` of any JSON-shaped value, used inside containers.
fn repr_value(value: &PyValue) -> String {
    match value {
        PyValue::Null => "None".into(),
        PyValue::Bool(true) => "True".into(),
        PyValue::Bool(false) => "False".into(),
        PyValue::Int(digits) => digits.clone(),
        PyValue::Float(number) => py_repr(*number),
        PyValue::Str(text) => repr_str(text),
        PyValue::Array(items) => {
            let rendered: Vec<String> = items.iter().map(repr_value).collect();
            format!("[{}]", rendered.join(", "))
        }
        PyValue::Object(entries) => {
            let rendered: Vec<String> = entries
                .iter()
                .map(|(key, item)| format!("{}: {}", repr_str(key), repr_value(item)))
                .collect();
            format!("{{{}}}", rendered.join(", "))
        }
    }
}

/// Python `str(value)` as the reranker's f-string applies it: a string state is
/// spliced verbatim, every other shape renders as its `repr()`.
pub fn state_text(value: &PyValue) -> String {
    match value {
        PyValue::Str(text) => text.clone(),
        other => repr_value(other),
    }
}

/// Instruction branch, mirroring `reranker._encode`'s `provenance.experiment` test.
pub fn instruction(row: &PyValue) -> &'static str {
    let experiment = row
        .get("provenance")
        .and_then(|provenance| provenance.get("experiment"))
        .and_then(PyValue::as_str);
    match experiment {
        Some(name) if RERANKER_RETRIEVAL_EXPERIMENTS.contains(&name) => {
            RERANKER_RETRIEVAL_INSTRUCTION
        }
        _ => RERANKER_DECISION_INSTRUCTION,
    }
}

/// The exact pair prompt text: `PREFIX + body + SUFFIX`.
pub fn pair_text(row: &PyValue, option: &PyValue) -> String {
    let question = row
        .get("question")
        .and_then(PyValue::as_str)
        .unwrap_or_default();
    let description = option
        .get("description")
        .and_then(PyValue::as_str)
        .unwrap_or_default();
    let state = row.get("state").cloned().unwrap_or(PyValue::Null);
    format!(
        "{RERANKER_PREFIX}<Instruct>: {}\n<Query>: Question: {question}\n\
         Candidate answer: {description}\n<Document>: {}{RERANKER_SUFFIX}",
        instruction(row),
        state_text(&state),
    )
}

/// Validate the row and render one pair; returns (text, prompt_sha256).
pub fn build_pair_prompt(row: &PyValue, option: &PyValue) -> Result<(String, String), PromptError> {
    validate_row(row)?;
    let text = pair_text(row, option);
    let hash = digest(&text);
    Ok((text, hash))
}

/// `reranker._encode`: render, encode, enforce the no-truncation budget, hash.
pub fn encode_pair(
    tokenizer: &ReferenceTokenizer,
    row: &PyValue,
    option: &PyValue,
    max_tokens: usize,
) -> Result<(Vec<u32>, String), ScorerError> {
    let (text, prompt_hash) = build_pair_prompt(row, option)?;
    let ids = tokenizer.encode(&text)?;
    let row_id = row.get("id").and_then(PyValue::as_str).unwrap_or_default();
    let option_id = option
        .get("id")
        .and_then(PyValue::as_str)
        .unwrap_or_default();
    if ids.is_empty() || ids.len() > max_tokens {
        return Err(ScorerError::Message(format!(
            "Row {row_id} option {option_id}: {} tokens exceed limit {max_tokens}",
            ids.len()
        )));
    }
    Ok((ids, prompt_hash))
}

/// `reranker._answer_ids`: distinct single tokens **and** vocab conversion agreement.
pub fn answer_ids(tokenizer: &ReferenceTokenizer) -> Result<(u32, u32), ScorerError> {
    let no = tokenizer.encode("no")?;
    let yes = tokenizer.encode("yes")?;
    if no.len() != 1 || yes.len() != 1 || no == yes {
        return Err(ScorerError::Message(
            "Reranker yes/no answers must be distinct single tokens".into(),
        ));
    }
    let contract = || {
        ScorerError::Message(
            "Tokenizer conversion differs from the official yes/no token contract".into(),
        )
    };
    let convert_no = tokenizer.token_to_id("no").ok_or_else(contract)?;
    let convert_yes = tokenizer.token_to_id("yes").ok_or_else(contract)?;
    if convert_yes != yes[0] || convert_no != no[0] {
        return Err(contract());
    }
    Ok((no[0], yes[0]))
}

#[cfg(test)]
mod tests {
    use super::{pair_text, repr_str, state_text};
    use crate::parser::parse;
    use semif_types::PyValue;

    #[test]
    fn string_state_is_spliced_raw() {
        let value = parse("\"Evidence.\"").unwrap();
        assert_eq!(state_text(&value), "Evidence.");
    }

    #[test]
    fn container_state_takes_python_repr() {
        let value =
            parse("{\"policy\": \"Never request passwords\", \"limit\": 3, \"active\": true}")
                .unwrap();
        assert_eq!(
            state_text(&value),
            "{'policy': 'Never request passwords', 'limit': 3, 'active': True}"
        );
    }

    #[test]
    fn repr_quote_selection_follows_cpython() {
        assert_eq!(repr_str("it's"), "\"it's\"");
        assert_eq!(repr_str("say \"hi\""), "'say \"hi\"'");
        assert_eq!(repr_str("both ' and \""), "'both \\' and \"'");
        assert_eq!(repr_str("back\\slash"), "'back\\\\slash'");
        assert_eq!(repr_str("line\nbreak"), "'line\\nbreak'");
        assert_eq!(repr_str("uni \u{e9}\u{4e2d}"), "'uni \u{e9}\u{4e2d}'");
    }

    #[test]
    fn instruction_branch_follows_provenance() {
        let base = parse(
            r#"{"id": "x", "state": "s", "question": "q",
                "options": [{"id": "a", "description": "A"}, {"id": "b", "description": "B"}]}"#,
        )
        .unwrap();
        let option = base.get("options").unwrap().as_array().unwrap()[0].clone();
        assert!(
            pair_text(&base, &option)
                .contains("determine whether the evidence supports that answer")
        );

        let PyValue::Object(mut entries) = base.clone() else {
            panic!("row must be an object")
        };
        entries.push((
            "provenance".into(),
            parse(r#"{"experiment": "code-rag"}"#).unwrap(),
        ));
        let tagged = PyValue::Object(entries);
        assert!(pair_text(&tagged, &option).contains("determine whether the document is relevant"));
    }
}
