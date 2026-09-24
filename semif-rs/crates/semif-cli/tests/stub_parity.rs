//! Stage 1 wire gate: the Rust CLI's stub-engine output must be byte-identical
//! to the Python-authored stub envelope fixtures over the whole accepted corpus.

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_semif-cli");
const FIXTURES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures");

#[test]
fn stub_rows_byte_identical_over_accepted_corpus() {
    let fixtures = std::path::Path::new(FIXTURES);
    let manifest = std::fs::read_to_string(fixtures.join("manifest.json")).unwrap();
    let loader: String = semif_core_stub::loader_source(&manifest);
    let dir = std::env::temp_dir().join(format!("semif-stub-parity-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let output = dir.join("rust-stubs.jsonl");

    let result = Command::new(BIN)
        .args([
            "--mode",
            "direct",
            "--model",
            &loader,
            "--revision",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "--input",
            fixtures.join("inputs-accepted.jsonl").to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "CLI failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );

    let expected = std::fs::read_to_string(fixtures.join("stubs.jsonl")).unwrap();
    let actual = std::fs::read_to_string(&output).unwrap();
    let expected_rows: Vec<&str> = expected.lines().filter(|l| !l.trim().is_empty()).collect();
    let actual_rows: Vec<&str> = actual.lines().filter(|l| !l.trim().is_empty()).collect();
    assert_eq!(expected_rows.len(), actual_rows.len(), "row count");
    let mismatches: Vec<usize> = expected_rows
        .iter()
        .zip(&actual_rows)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, _)| index)
        .collect();
    assert!(
        mismatches.is_empty(),
        "byte mismatches at rows {mismatches:?}"
    );
}

mod semif_core_stub {
    pub fn loader_source(manifest: &str) -> String {
        // Manifest is flat JSON; find "loader_source": "..." without pulling serde in.
        let key = "\"loader_source\": \"";
        let start = manifest.find(key).expect("loader_source in manifest") + key.len();
        let end = manifest[start..].find('"').unwrap() + start;
        manifest[start..end].to_string()
    }
}
