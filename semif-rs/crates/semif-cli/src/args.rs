//! Argument model and usage-validation order, mirroring `semif_phase1.cli`.
//!
//! Exit-code and behavior parity is pinned by `fixtures/cli_errors.jsonl`;
//! usage-message *text* matches Python's strings but clap's prefix differs
//! (accepted Stage 1 divergence, per the executed von precedent).

use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(
    name = "semif-cli",
    about = "Create-only JSONL command line scorer (Rust port)"
)]
pub struct Args {
    #[arg(long, required = true)]
    pub mode: String,

    #[arg(long, value_parser = ["torch", "mlx", "llamacpp"], default_value = "torch")]
    pub backend: String,

    #[arg(long)]
    pub mlx_bits: Option<i64>,

    #[arg(long, allow_negative_numbers = true)]
    pub mlx_cache_limit_mib: Option<i64>,

    #[arg(long)]
    pub gguf: Option<PathBuf>,

    #[arg(long, allow_negative_numbers = true)]
    pub llama_threads: Option<i64>,

    #[arg(long, required = true)]
    pub model: String,

    #[arg(long, required = true)]
    pub revision: String,

    #[arg(long, required = true)]
    pub input: PathBuf,

    #[arg(long, required = true)]
    pub output: PathBuf,

    #[arg(long, default_value = "4096", allow_negative_numbers = true)]
    pub max_tokens: i64,

    #[arg(long, value_parser = ["auto", "cuda", "mps", "cpu"], default_value = "auto")]
    pub device: String,

    #[arg(long, value_parser = ["bfloat16", "float16", "float32"], default_value = "bfloat16")]
    pub dtype: String,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct UsageError {
    pub message: String,
}

/// Validation order identical to `cli.py:main` before any model/backend work.
pub fn validate(args: &Args) -> Result<(), UsageError> {
    if args.output.exists() || args.max_tokens < 1 {
        return Err(UsageError {
            message: "Output must be new and max-tokens must be positive".into(),
        });
    }
    if args.mlx_bits.is_some() && args.backend != "mlx" {
        return Err(UsageError {
            message: "--mlx-bits requires --backend mlx".into(),
        });
    }
    if let Some(limit) = args.mlx_cache_limit_mib {
        if args.backend != "mlx" {
            return Err(UsageError {
                message: "--mlx-cache-limit-mib requires --backend mlx".into(),
            });
        }
        if limit < 0 {
            return Err(UsageError {
                message: "--mlx-cache-limit-mib must be nonnegative".into(),
            });
        }
    }
    if args.gguf.is_some() && args.backend != "llamacpp" {
        return Err(UsageError {
            message: "--gguf requires --backend llamacpp".into(),
        });
    }
    if let Some(threads) = args.llama_threads {
        if args.backend != "llamacpp" {
            return Err(UsageError {
                message: "--llama-threads requires --backend llamacpp".into(),
            });
        }
        if threads < 1 {
            return Err(UsageError {
                message: "--llama-threads must be positive".into(),
            });
        }
    }
    if args.backend == "mlx" && args.mode == "reranker" {
        return Err(UsageError {
            message: "MLX supports direct, serial, and shared modes; reranker requires torch"
                .into(),
        });
    }
    if args.backend == "llamacpp" {
        if args.mode == "reranker" {
            return Err(UsageError {
                message:
                    "llama.cpp supports direct, serial, and shared modes; reranker requires torch"
                        .into(),
            });
        }
        let gguf_ok = args.gguf.as_ref().is_some_and(|path| path.is_file());
        if !gguf_ok {
            return Err(UsageError {
                message: "--backend llamacpp requires --gguf pointing at an existing GGUF file"
                    .into(),
            });
        }
    }
    Ok(())
}
