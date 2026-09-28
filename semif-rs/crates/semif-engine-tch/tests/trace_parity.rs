//! Stage 3 decisive gate for the reranker: Rust must execute the traced graph
//! **bit-exactly**.
//!
//! The exporter (`benchmarks/export_reranker_trace.py`) recorded a sha256 over
//! each check row's full 151,669-float vocabulary logits as it produced the
//! artifact. Because trace fusion legitimately changes kernels relative to the
//! eager Python run, that capture — not the eager run — is the numeric
//! reference: any difference here means the port feeds the graph different
//! inputs or reads a different position, not that CUDA is being CUDA.
//!
//! Ignored by default: it needs the traced artifact and one CUDA GPU. Run with
//! `cargo test -p semif-engine-tch -- --ignored`.

use semif_core::parser::parse;
use semif_core::reranker::encode_pair;
use semif_core::tokenizer::ReferenceTokenizer;
use semif_types::PyValue;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn workspace(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(relative)
}

fn str_field<'a>(value: &'a PyValue, key: &str) -> &'a str {
    value.get(key).and_then(PyValue::as_str).unwrap_or_default()
}

fn read_lines(path: &Path) -> Vec<PyValue> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("cannot read {path:?}: {error}"))
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| parse(line).expect("fixture line"))
        .collect()
}

/// sha256 over little-endian f32 vocabulary logits, matching the exporter's
/// `values.tobytes()` on a `<f4` numpy view.
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

#[test]
#[ignore = "needs the traced reranker artifact and one CUDA GPU"]
fn traced_graph_runs_bit_exactly() {
    let fixtures = workspace("fixtures");
    let artifacts = std::env::var("SEMIF_TCH_ARTIFACTS")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace("artifacts"));
    let check_path = artifacts.join("trace-check-reranker-cudabf16.json");
    let check = parse(
        &std::fs::read_to_string(&check_path)
            .unwrap_or_else(|error| panic!("cannot read {check_path:?}: {error}")),
    )
    .expect("trace-check json");
    let artifact_name = str_field(&check, "artifact");
    let artifact = artifacts.join(artifact_name);
    assert!(
        artifact.is_file(),
        "missing artifact {artifact:?}; run benchmarks/export_reranker_trace.py"
    );

    let manifest =
        parse(&std::fs::read_to_string(fixtures.join("reranker_manifest.json")).expect("manifest"))
            .expect("manifest json");
    let tokenizer =
        ReferenceTokenizer::from_checkpoint_dir(Path::new(str_field(&manifest, "tokenizer")))
            .expect("reranker tokenizer");

    let rows: HashMap<String, PyValue> = read_lines(&fixtures.join("reranker_rows.jsonl"))
        .into_iter()
        .map(|row| (str_field(&row, "id").to_string(), row))
        .collect();

    let engine = semif_engine_tch::TchEngine::new_reranker(
        &artifact,
        semif_engine_tch::device_for_context("cudabf16"),
        str_field(&manifest, "tokenizer"),
        "reranker",
        "bfloat16",
        "cudabf16",
    )
    .expect("load traced reranker");

    let checks = check
        .get("checks")
        .and_then(PyValue::as_array)
        .expect("checks array");
    assert!(!checks.is_empty(), "trace-check recorded no rows");

    let mut compared = 0usize;
    for row_check in checks {
        let id = str_field(row_check, "id");
        let row = rows
            .get(id)
            .unwrap_or_else(|| panic!("trace-check row {id} missing"));
        let options = row.get("options").and_then(PyValue::as_array).unwrap();
        let mut pairs = Vec::with_capacity(options.len());
        let mut width = 0usize;
        for option in options {
            let (ids, _) = encode_pair(&tokenizer, row, option, 4096).expect("encode pair");
            width = width.max(ids.len());
            pairs.push(ids);
        }
        let logits = engine
            .reranker_vocab(&pairs, width)
            .unwrap_or_else(|error| panic!("{id}: {error}"));
        assert_eq!(logits.len(), options.len(), "{id}: option count");

        let recorded = row_check
            .get("per_option")
            .and_then(PyValue::as_array)
            .expect("per_option array");
        assert_eq!(recorded.len(), options.len(), "{id}: recorded options");
        for (index, entry) in recorded.iter().enumerate() {
            assert_eq!(
                str_field(entry, "option_id"),
                str_field(&options[index], "id"),
                "{id}: option order"
            );
            let expected = str_field(entry, "traced_logits_sha256");
            let actual = sha256_f32_le(&logits[index]);
            assert_eq!(
                actual,
                expected,
                "{id}/{}: traced vocabulary logits are not bit-exact",
                str_field(&options[index], "id")
            );
            compared += 1;
        }
    }
    eprintln!("bit-exact vocabulary logits compared: {compared} option pairs");
}
