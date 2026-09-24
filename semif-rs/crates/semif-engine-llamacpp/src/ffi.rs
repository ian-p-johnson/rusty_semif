//! `dlopen` bindings to the llama.cpp shared library bundled with the pinned
//! Python wheel (llama-cpp-python 0.3.35).
//!
//! Loading the *same* native library the Python oracle links is what makes the
//! Stage 2 parity gate (≤1e-5 logits) measure port correctness instead of
//! llama.cpp version drift — the von-port load-dynamic precedent. Struct
//! layouts mirror the wheel's ctypes `_fields_` exactly; do not "modernize".

use libloading::Library;
use std::ffi::c_char;
use std::path::PathBuf;
use std::ptr;

pub type LlamaToken = i32;
pub type LlamaPos = i32;
pub type LlamaSeqId = i32;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut LlamaToken,
    pub embd: *mut f32,
    pub pos: *mut LlamaPos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut LlamaSeqId,
    pub logits: *mut i8,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LlamaModelParams {
    pub devices: *mut std::ffi::c_void,
    pub tensor_buft_overrides: *mut std::ffi::c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *mut f32,
    pub progress_callback: *mut std::ffi::c_void,
    pub progress_callback_user_data: *mut std::ffi::c_void,
    pub kv_overrides: *mut std::ffi::c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *mut std::ffi::c_void,
    pub cb_eval_user_data: *mut std::ffi::c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *mut std::ffi::c_void,
    pub abort_callback_data: *mut std::ffi::c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut std::ffi::c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut std::ffi::c_void,
}

macro_rules! symbols {
    ($library:expr, { $( $name:ident: fn($($arg:ty),*) $(-> $ret:ty)? ),* $(,)? }) => {
        pub struct Symbols { $( pub $name: unsafe extern "C" fn($($arg),*) $(-> $ret)? ),* }
        impl Symbols {
            /// # Safety
        /// `library` must be a validly loaded llama.cpp shared object.
        pub unsafe fn load(library: &Library) -> Result<Symbols, libloading::Error> {
                Ok(Symbols { $( $name: unsafe { *library.get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(concat!(stringify!($name), "\0").as_bytes())? } ),* })
            }
        }
    };
}

symbols!(library, {
    llama_backend_init: fn(),
    llama_log_set: fn(Option<unsafe extern "C" fn(i32, *const c_char, *mut std::ffi::c_void)>, *mut std::ffi::c_void),
    llama_model_default_params: fn() -> LlamaModelParams,
    llama_model_load_from_file: fn(*const c_char, LlamaModelParams) -> *mut std::ffi::c_void,
    llama_model_free: fn(*mut std::ffi::c_void),
    llama_context_default_params: fn() -> LlamaContextParams,
    llama_init_from_model: fn(*mut std::ffi::c_void, LlamaContextParams) -> *mut std::ffi::c_void,
    llama_free: fn(*mut std::ffi::c_void),
    llama_model_get_vocab: fn(*const std::ffi::c_void) -> *const std::ffi::c_void,
    llama_n_ctx: fn(*const std::ffi::c_void) -> i32,
    llama_n_vocab: fn(*const std::ffi::c_void) -> i32,
    llama_tokenize: fn(*const std::ffi::c_void, *const c_char, i32, *mut LlamaToken, i32, bool, bool) -> i32,
    llama_token_to_piece: fn(*const std::ffi::c_void, LlamaToken, *mut u8, i32, i32, bool) -> i32,
    llama_batch_init: fn(i32, i32, i32) -> LlamaBatch,
    llama_batch_free: fn(LlamaBatch),
    llama_decode: fn(*mut std::ffi::c_void, LlamaBatch) -> i32,
    llama_get_logits_ith: fn(*const std::ffi::c_void, i32) -> *mut f32,
    llama_get_memory: fn(*const std::ffi::c_void) -> *mut std::ffi::c_void,
    llama_memory_clear: fn(*mut std::ffi::c_void, bool),
    llama_memory_seq_rm: fn(*mut std::ffi::c_void, LlamaSeqId, LlamaPos, LlamaPos) -> bool,
    llama_state_seq_get_size: fn(*const std::ffi::c_void, LlamaSeqId) -> usize,
    llama_state_seq_get_data: fn(*const std::ffi::c_void, *mut u8, usize, LlamaSeqId) -> usize,
    llama_state_seq_set_data: fn(*mut std::ffi::c_void, *const u8, usize, LlamaSeqId) -> usize,
});

/// Load `libllama` and leak it so the symbol handles are valid for the process.
pub fn load_library(path: &std::path::Path) -> Result<&'static Library, String> {
    unsafe {
        let library = Library::new(path).map_err(|error| format!("{path:?}: {error}"))?;
        Ok(Box::leak(Box::new(library)))
    }
}

pub fn load_symbols(library: &'static Library) -> Result<Symbols, String> {
    unsafe { Symbols::load(library).map_err(|error| format!("symbol lookup failed: {error}")) }
}

/// Default search: env override, then venvs near the build tree and cwd.
pub fn default_library_path() -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("SEMIF_LLAMA_LIB") {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for base in [
        manifest.parent()?.parent()?.parent()?.join(".venv/lib"),
        PathBuf::from(".venv/lib"),
        PathBuf::from("../.venv/lib"),
    ] {
        let Ok(entries) = std::fs::read_dir(&base) else {
            continue;
        };
        for entry in entries.flatten() {
            let lib_dir = entry.path().join("site-packages/llama_cpp/lib");
            for name in ["libllama.so.0.1.0", "libllama.so.0", "libllama.so"] {
                let candidate = lib_dir.join(name);
                if candidate.exists() {
                    candidates.push(candidate);
                }
            }
        }
    }
    candidates.into_iter().next()
}

pub fn null() -> *mut std::ffi::c_void {
    ptr::null_mut()
}

/// Silence llama.cpp/ggml logging: a NULL callback resets to the default.
pub extern "C" fn noop_log(_level: i32, _message: *const c_char, _user: *mut std::ffi::c_void) {}
