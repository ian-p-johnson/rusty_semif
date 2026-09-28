//! Create-only JSONL command line scorer (Rust port; llamacpp engine real,
//! torch/mlx backends remain stubs until Stages 3+).

mod args;

use args::{Args, validate};
use clap::Parser as _;
use semif_core::parse;
use semif_core::pyjson::dumps_output;
use semif_core::tokenizer::{ReferenceTokenizer, ScorerError};
use semif_engine::{ScoreContext, StubEngine};
use semif_types::PyValue;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn fail_usage(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}

fn fail_row(message: &str) -> ! {
    eprintln!("ValueError: {message}");
    std::process::exit(1);
}

/// Fixed trace width, mirroring the exporter's `--width`.
fn tch_width() -> usize {
    std::env::var("SEMIF_TCH_WIDTH")
        .ok()
        .and_then(|width| width.parse::<usize>().ok())
        .unwrap_or(4096)
}

/// `{artifacts}/{stem}-{context}-w{width}.pt`.
fn trace_artifact(stem: &str, context: &str, width: usize) -> PathBuf {
    Path::new(&std::env::var("SEMIF_TCH_ARTIFACTS").unwrap_or_else(|_| "semif-rs/artifacts".into()))
        .join(format!("{stem}-{context}-w{width}.pt"))
}

fn read_rows(path: &Path) -> Vec<PyValue> {
    let raw = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => fail_usage(&format!("cannot read {path:?}: {error}")),
    };
    let normalized = raw.replace("\r\n", "\n").replace('\r', "\n");
    let mut rows = Vec::new();
    for line in normalized.split('\n') {
        if line.trim().is_empty() {
            continue;
        }
        match parse(line) {
            Ok(value) => rows.push(value),
            Err(error) => fail_row(&error.to_string()),
        }
    }
    if rows.is_empty() {
        fail_usage("Input is empty");
    }
    for row in &rows {
        if let Err(error) = semif_types::validate_row(row) {
            fail_row(&error.to_string());
        }
    }
    rows
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if let Err(error) = validate(&args) {
        fail_usage(&error.message);
    }
    let rows = read_rows(&args.input);

    // Python forces reranker to CUDA *after* rows are validated (cli.py),
    // so a bad row still exits 1 before this usage error exits 2.
    if args.mode == "reranker" && (args.device == "mps" || args.device == "cpu") {
        fail_usage(&format!(
            "Reranker mode requires CUDA; --device {} is unsupported",
            args.device
        ));
    }

    let tokenizer = ReferenceTokenizer::from_checkpoint_dir(Path::new(&args.model))?;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut engine: Box<dyn semif_engine::Engine> = if args.backend == "llamacpp" {
        let gguf = args.gguf.clone().expect("validated");
        let library_path = semif_engine_llamacpp::LlamacppEngine::default_library_path()?;
        Box::new(semif_engine_llamacpp::LlamacppEngine::new(
            &library_path,
            &gguf,
            &tokenizer,
            threads,
            args.max_tokens as usize,
            &args.model,
            &args.revision,
        )?)
    } else if args.backend == "torch" && args.mode == "reranker" {
        // The torch backend is trace-only: there is no eager engine, so reranker
        // refuses rather than silently falling through to the stub (documented
        // divergence — Python would happily run eagerly).
        if std::env::var("SEMIF_TCH_TRACE").is_err() {
            fail_usage("reranker mode requires the traced tch engine; set SEMIF_TCH_TRACE=1");
        }
        if args.dtype != "bfloat16" {
            fail_usage(&format!(
                "no traced reranker artifact for dtype {}; available: cuda+bfloat16",
                args.dtype
            ));
        }
        semif_engine_tch::require_single_gpu()
            .unwrap_or_else(|error| fail_usage(&error.to_string()));
        // No run-level `max_tokens`/width guard here: the reranker trace is
        // 2048 wide while Python's budget default is 4096, so a guard would
        // reject runs whose rows all fit. Rows past the width are refused
        // individually instead (accepted divergence).
        let artifact = trace_artifact("qwen3reranker", "cudabf16", tch_width());
        Box::new(semif_engine_tch::TchEngine::new_reranker(
            &artifact,
            semif_engine_tch::device_for_context("cudabf16"),
            &args.model,
            &args.revision,
            &args.dtype,
            "cudabf16",
        )?)
    } else if args.backend == "torch" && std::env::var("SEMIF_TCH_TRACE").is_ok() {
        // Python's `auto` prefers CUDA; resolve it before naming the artifact
        // so the default device works the same way here.
        let device = match args.device.as_str() {
            "auto" => "cuda",
            other => other,
        };
        let context = match (device, args.dtype.as_str()) {
            ("cpu", "float32") => "cpufp32",
            ("cuda", "bfloat16") => "cudabf16",
            other => fail_usage(&format!(
                "no traced artifact for device/dtype {other:?}; available: cpu+float32, cuda+bfloat16"
            )),
        };
        if context == "cudabf16" {
            semif_engine_tch::require_single_gpu()
                .unwrap_or_else(|error| fail_usage(&error.to_string()));
        }
        // No run-level `max_tokens`/width guard: it would reject runs whose
        // rows all fit. Rows past the trace width are refused individually by
        // the engine instead (matches Python's per-row budget refusal).
        let artifact = trace_artifact("qwen35-direct", context, tch_width());
        Box::new(semif_engine_tch::TchEngine::new(
            &artifact,
            semif_engine_tch::device_for_context(context),
            &args.model,
            &args.revision,
            &args.dtype,
            context,
        )?)
    } else {
        Box::new(StubEngine)
    };
    let context = ScoreContext {
        tokenizer: &tokenizer,
        max_tokens: args.max_tokens as usize,
        source: args.model.clone(),
        revision: args.revision.clone(),
        dtype: args.dtype.clone(),
    };

    let destination = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut writer = std::io::BufWriter::new(destination);
    if args.mode == "shared" {
        let (results, timing) = engine.score_shared(&rows, &context)?;
        for result in results {
            let mut row = result;
            if let (Some(timing), PyValue::Object(entries)) = (&timing, &mut row) {
                entries.push(("shared_timing".into(), timing.clone()));
            }
            writer.write_all(dumps_output(&row)?.as_bytes())?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        return Ok(());
    }
    for row in &rows {
        let outcome = if args.mode == "serial" {
            engine.score_serial(row, &context)
        } else if args.mode == "reranker" {
            engine.score_reranker(row, &context)
        } else {
            engine.score_direct(row, &context)
        };
        match outcome {
            Ok(result) => {
                writer.write_all(dumps_output(&result)?.as_bytes())?;
                writer.write_all(b"\n")?;
                writer.flush()?;
            }
            Err(ScorerError::Message(message)) => fail_row(&message),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
