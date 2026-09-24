//! Python-`json`-compatible infrastructure for the SemIf port.
//!
//! Parity surfaces (each pinned by `semif-rs/fixtures/`):
//! - `parser`:  `json.loads` semantics incl. NaN/Infinity literals and
//!   duplicate keys (last wins, position kept)
//! - `pyfloat`: `repr(float)` formatting incl. exponent thresholds
//! - `pyjson`:  the two writer profiles — prompt payload
//!   (`ensure_ascii=False`) and output rows (`ensure_ascii=True`,
//!   `allow_nan=False`)
//! - `softmax`, `digest`: pure-f64 numerics and sha256
//! - `template`: the pinned Qwen3.5 chat-template rendering
//! - `tokenizer`: reference tokenizer harness with truncation/padding detached
//! - `prefix`:  state-prefix extraction (`shared._state_prefix`)

pub mod digest;
pub mod parser;
pub mod prefix;
pub mod prompt;
pub mod pyfloat;
pub mod pyjson;
pub mod softmax;
pub mod tokenizer;

pub use digest::digest;
pub use parser::{ParseError, parse};
pub use prefix::state_prefix;
pub use prompt::{build_prompt, render_prompt};
pub use pyfloat::py_repr;
pub use pyjson::{JsonError, dumps_output, dumps_payload};
pub use softmax::softmax;
pub use tokenizer::{ReferenceTokenizer, ScorerError, encode_prompt};

/// `semif_phase1.direct.PROMPT_VERSION`.
pub const PROMPT_VERSION: &str = "direct-options-v1";
