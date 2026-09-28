//! CLI refusal contract, pinned by `fixtures/cli_errors.jsonl`:
//! exit codes, output-not-created behavior, and row-level ValueError messages.
//! Usage-message wrapper text diverges from argparse (accepted, von precedent).

use std::process::Command;

const BIN: &str = env!("CARGO_BIN_EXE_semif-cli");
const MODEL: &str = "unused-model";

fn last_stderr_line(output: &std::process::Output) -> String {
    let text = String::from_utf8_lossy(&output.stderr);
    text.lines()
        .rfind(|line| !line.trim().is_empty())
        .unwrap_or_default()
        .to_string()
}

fn write(path: &std::path::Path, content: &str) {
    std::fs::write(path, content).unwrap();
}

fn run_cli(
    dir: &std::path::Path,
    input_name: &str,
    extra: &[&str],
) -> (std::process::Output, std::path::PathBuf) {
    let input = dir.join(input_name);
    let output = dir.join("out.jsonl");
    let mut command_args = vec!["--model", MODEL, "--revision", "unused"];
    command_args.extend_from_slice(extra);
    command_args.extend_from_slice(&[
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
    ]);
    (
        Command::new(BIN).args(&command_args).output().unwrap(),
        output,
    )
}

const VALID_ROW: &str = r#"{"id": "x", "state": "s", "question": "q?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]}"#;

#[test]
fn row_valueerror_exits_1_without_output() {
    let dir = tempfile_dir("row-valueerror");
    write(
        &dir.join("in.jsonl"),
        r#"{"id": "x", "state": "Evidence.", "question": "Which?", "options": [{"id": "same", "description": "A."}, {"id": "same", "description": "B."}]}"#,
    );
    let (result, output) = run_cli(&dir, "in.jsonl", &["--mode", "direct"]);
    assert_eq!(result.status.code(), Some(1));
    assert_eq!(
        last_stderr_line(&result),
        "ValueError: Option IDs must be unique"
    );
    assert!(!output.exists());
}

#[test]
fn existing_output_exits_2_without_touching_it() {
    let dir = tempfile_dir("existing-output");
    write(&dir.join("in.jsonl"), VALID_ROW);
    let output_path = dir.join("out.jsonl");
    write(&output_path, "pre-existing\n");
    let (result, output) = run_cli(&dir, "in.jsonl", &["--mode", "direct"]);
    assert_eq!(result.status.code(), Some(2));
    assert!(
        last_stderr_line(&result).contains("Output must be new and max-tokens must be positive")
    );
    assert_eq!(std::fs::read_to_string(&output).unwrap(), "pre-existing\n");
}

#[test]
fn empty_input_exits_2() {
    let dir = tempfile_dir("empty-input");
    write(&dir.join("in.jsonl"), "\n \n");
    let (result, output) = run_cli(&dir, "in.jsonl", &["--mode", "direct"]);
    assert_eq!(result.status.code(), Some(2));
    assert!(last_stderr_line(&result).contains("Input is empty"));
    assert!(!output.exists());
}

#[test]
fn backend_flag_cross_validation_order() {
    let dir = tempfile_dir("backend-flags");
    write(&dir.join("in.jsonl"), VALID_ROW);

    let cases: &[(&[&str], &str)] = &[
        (
            &["--mode", "reranker", "--backend", "mlx"],
            "reranker requires torch",
        ),
        (
            &["--mode", "direct", "--mlx-bits", "4"],
            "requires --backend mlx",
        ),
        (
            &["--mode", "direct", "--mlx-cache-limit-mib", "0"],
            "requires --backend mlx",
        ),
        (
            &["--mode", "direct", "--gguf", "missing.gguf"],
            "requires --backend llamacpp",
        ),
        (
            &["--mode", "direct", "--llama-threads", "2"],
            "requires --backend llamacpp",
        ),
        (
            &["--mode", "reranker", "--backend", "llamacpp"],
            "reranker requires torch",
        ),
        (
            &["--mode", "direct", "--backend", "llamacpp"],
            "--backend llamacpp requires --gguf pointing at an existing GGUF file",
        ),
    ];
    for (extra, fragment) in cases {
        let (result, output) = run_cli(&dir, "in.jsonl", extra);
        assert_eq!(
            result.status.code(),
            Some(2),
            "case {extra:?}: {}",
            last_stderr_line(&result)
        );
        assert!(
            last_stderr_line(&result).contains(fragment),
            "case {extra:?}"
        );
        assert!(!output.exists(), "case {extra:?}");
    }
}

#[test]
fn negative_mlx_cache_limit_message() {
    let dir = tempfile_dir("negative-limit");
    write(&dir.join("in.jsonl"), VALID_ROW);
    let (result, output) = run_cli(
        &dir,
        "in.jsonl",
        &[
            "--mode",
            "direct",
            "--backend",
            "mlx",
            "--mlx-cache-limit-mib",
            "-1",
        ],
    );
    assert_eq!(result.status.code(), Some(2));
    assert!(last_stderr_line(&result).contains("--mlx-cache-limit-mib must be nonnegative"));
    assert!(!output.exists());
}

#[test]
fn negative_max_tokens_message() {
    let dir = tempfile_dir("negative-max-tokens");
    write(&dir.join("in.jsonl"), VALID_ROW);
    let (result, output) = run_cli(
        &dir,
        "in.jsonl",
        &["--mode", "direct", "--max-tokens", "-1"],
    );
    assert_eq!(result.status.code(), Some(2));
    assert!(
        last_stderr_line(&result).contains("Output must be new and max-tokens must be positive")
    );
    assert!(!output.exists());
}

/// cli.py forces reranker to CUDA *after* rows validate, so this refusal runs
/// before any tokenizer load and exits 2 without creating the output.
#[test]
fn reranker_requires_cuda() {
    let dir = tempfile_dir("reranker-cuda");
    write(&dir.join("in.jsonl"), VALID_ROW);
    let (result, output) = run_cli(&dir, "in.jsonl", &["--mode", "reranker", "--device", "cpu"]);
    assert_eq!(result.status.code(), Some(2));
    assert_eq!(
        last_stderr_line(&result),
        "error: Reranker mode requires CUDA; --device cpu is unsupported"
    );
    assert!(!output.exists());
}

/// The torch backend is trace-only, so reranker refuses rather than silently
/// falling through to the uniform stub (accepted divergence: Python would run
/// eagerly). Cleared explicitly so the assertion holds even when the caller's
/// shell has the flag exported.
#[test]
fn reranker_without_trace_engine_exits_2() {
    let dir = tempfile_dir("reranker-no-trace");
    write(&dir.join("in.jsonl"), VALID_ROW);
    let fixtures = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures"));
    let manifest = std::fs::read_to_string(fixtures.join("manifest.json")).unwrap();
    let loader = loader_source(&manifest);
    let output = dir.join("out.jsonl");
    let result = Command::new(BIN)
        .args([
            "--mode",
            "reranker",
            "--backend",
            "torch",
            "--device",
            "cuda",
            "--model",
            &loader,
            "--revision",
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "--input",
            dir.join("in.jsonl").to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .env_remove("SEMIF_TCH_TRACE")
        .output()
        .unwrap();
    assert_eq!(result.status.code(), Some(2));
    assert_eq!(
        last_stderr_line(&result),
        "error: reranker mode requires the traced tch engine; set SEMIF_TCH_TRACE=1"
    );
    assert!(!output.exists());
}

/// `manifest.json` loader path, read without a JSON dependency.
fn loader_source(manifest: &str) -> String {
    let key = "\"loader_source\": \"";
    let start = manifest.find(key).expect("loader_source") + key.len();
    let rest = &manifest[start..];
    rest[..rest.find('"').expect("closing quote")].to_string()
}

fn tempfile_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("semif-cli-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
