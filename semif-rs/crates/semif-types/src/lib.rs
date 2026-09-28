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

/// Fixed reranker preamble, matching `semif_phase1.reranker.PREFIX`.
/// The reranker never runs the chat template: this text is hard-coded.
pub const RERANKER_PREFIX: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";

/// Fixed reranker trailer, matching `semif_phase1.reranker.SUFFIX`.
pub const RERANKER_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// Default reranker instruction, matching `semif_phase1.reranker.DECISION_INSTRUCTION`.
pub const RERANKER_DECISION_INSTRUCTION: &str = "Given evidence and one possible answer to a question, determine whether the evidence supports that answer under the question's criterion. Use only the supplied evidence.";

/// Retrieval-branch reranker instruction, matching `semif_phase1.reranker.RETRIEVAL_INSTRUCTION`.
pub const RERANKER_RETRIEVAL_INSTRUCTION: &str = "Given a search query, determine whether the document is relevant and contains evidence that answers the query.";

/// Reranker row stamp, matching `semif_phase1.reranker.PROMPT_VERSION`.
pub const RERANKER_PROMPT_VERSION: &str = "qwen3-reranker-native-options-v1";

/// Experiments that switch the reranker to the retrieval instruction.
pub const RERANKER_RETRIEVAL_EXPERIMENTS: &[&str] = &["code-rag", "company-brain"];
