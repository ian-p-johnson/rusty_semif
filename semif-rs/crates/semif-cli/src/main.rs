//! Create-only JSONL command line scorer (Rust port, Stage 1: stub engine).

mod args;

use args::{Args, validate};
use clap::Parser as _;
use semif_core::parse;
use semif_core::pyjson::dumps_output;
use semif_core::tokenizer::{ReferenceTokenizer, ScorerError};
use semif_engine::{Engine, ScoreContext, StubEngine};
use semif_types::PyValue;
use std::io::Write;
use std::path::Path;
use std::process::ExitCode;

fn fail_usage(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(2);
}

fn fail_row(message: &str) -> ! {
    eprintln!("ValueError: {message}");
    std::process::exit(1);
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

    let tokenizer = ReferenceTokenizer::from_checkpoint_dir(Path::new(&args.model))?;
    let engine = StubEngine;
    let context = ScoreContext {
        tokenizer: &tokenizer,
        max_tokens: args.max_tokens.max(0) as usize,
        source: args.model.clone(),
        revision: args.revision.clone(),
        dtype: args.dtype.clone(),
    };

    let destination = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let mut writer = std::io::BufWriter::new(destination);
    for row in &rows {
        match engine.score(row, &context) {
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
