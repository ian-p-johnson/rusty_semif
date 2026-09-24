//! Shared value model, row schema, and validation with Python-pinned error strings.

pub mod validate;
pub mod value;

pub use validate::{ValidationError, validate_row};
pub use value::PyValue;

/// Answer-slot letters, matching `semif_phase1.core.LETTERS`.
pub const LETTERS: &[char] = &[
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
];

/// System prompt, matching `semif_phase1.core.DIRECT_SYSTEM`.
pub const DIRECT_SYSTEM: &str = "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. Respond with only its uppercase letter, with no explanation or reasoning.";
