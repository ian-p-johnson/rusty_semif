# Porting SemIf to Rust — Susceptibility Assessment & Staged Plan

*Assessed against the working tree at `semif-phase1` 0.1.0 (commit `23cf1f3`,
"Add explicit Torch CPU scoring"). Tree-integrity gates were executed during
assessment: `results/raw/SHA256SUMS` verifies clean and
`benchmarks/verify_published.py` confirms 69 summary claims against committed
row-level evidence. The pytest suite (55 tests) was **not** executed in the
assessment environment — the pinned `torch==2.10.0` stack is not installed
there — and running it is recorded as a Stage 0 precondition. No files outside
this document were touched.*

> **Companion documents.** `../rusty_von/PORTING_RUST.md` is the executed
> reference for this methodology (its Stage 2 is complete; the unmodified
> pytest suite already passes against its Rust engine). The von plan in turn
> carries **[proven in laya]** markers from `../rusty_laya`'s executed port.
> Where this document inherits a *measured* finding from either, it is marked
> **[proven in von]** or **[proven in laya]**.
>
> **Sympathy & independence.** Three ports now run side by side on one machine.
> This plan keeps the *methodology* sympathetic — oracle-and-gates philosophy,
> fixture-driven parity rings, one-command verification scripts, a
> CPU-verifiable engine before any GPU work — so the operator's mental model
> transfers — while keeping every *artifact* independent: a separate `semif-rs/`
> workspace and `semif-*` crate namespace, a separate venv, no shared code,
> fixtures, or build targets with either sibling. See §8.1 for coexistence
> rules. Nothing here requires or produces changes to the von or laya trees.

---

## 1. Verdict

**SemIf is susceptible to a Rust port, but it is the inverse of von: the
pure-logic surface is trivial and the model-execution surface is the hard
core.** A verified port is worthwhile, with two honest caveats that shape the
whole plan: the right acceptance gates are *decision-level tolerances*, not
von-style byte-exact parity (the repo's own committed evidence already
documents cross-path argmax flips), and the CUDA BF16 engine is a genuinely
harder build than anything von or laya attempted.

Four structural facts drive the verdict:

1. **The runtime is small, and most of it barely touches tensors.**
   `src/semif_phase1` is 1,413 lines across eight modules. Validation, prompt
   construction, softmax, digesting, row schemas, and CLI semantics are pure
   deterministic logic (~400 lines). The tensor-touching code is concentrated
   in forward-pass plumbing whose *semantics* are simple (prefill → branch →
   gather slot logits → softmax) even though its execution is not.

2. **The contract is a file format, not a server.** There is no HTTP surface.
   The product is a create-only JSONL batch scorer (`semif-score`) whose output
   rows join to committed evidence through `prompt_sha256`, `option_ids`, and
   pinned model revisions. That makes the differential harness simpler than
   von's (two CLIs over the same fixture, row-diffed) and makes the parity
   target unusually concrete: **the repo already ships row-level goldens** —
   the shape777 evidence is 777 decisions × 3 execution paths (2,331 rows
   with `option_logits`, `probabilities`, `allowed_token_mass`, and
   `prompt_sha256`), checksummed under `results/raw/` next to the
   authored144 / WANLI / typesafe102 / perturbations108 prediction files.

3. **The repo's own methodology already accepts cross-runtime divergence.**
   The README documents that BF16 execution changed 5–6 of 777 argmaxes
   between the repo's own torch paths, and tells llama.cpp users to "compare
   decisions or probabilities with a tolerance rather than raw logits bit for
   bit." `docs/REPRODUCE.md` says to treat model outputs as measurements, not
   byte-identical goldens. A Rust port inherits exactly this gate philosophy:
   *identity fields exact, numerics gated by tolerance, timings excluded.* The
   von-style byte-diff wire gate does not apply and should not be pretended.

4. **The hard part is genuinely hard, and it is the headline.** The pinned
   model is a 4B **hybrid-attention causal decoder** (Qwen3.5-4B — mixed
   full-attention and linear-attention memory per the `llamacpp_backend`
   docstring), not a 395M encoder. Von's Route A (ONNX export + `ort`) does
   not transfer: exporting a decoder graph with KV-cache prefill, branch
   replication (`reorder_cache` / cache deepcopy), and selective
   `logits_to_keep` is unsolved territory, and a direct-mode-only export would
   forfeit the prefix-reuse paths that *are* the project's speed claims. The
   credible engine route is a hand-written decoder over libtorch via `tch`
   (the laya-proven Route-B-with-torch-numerics), staged after a much cheaper
   CPU win: binding llama.cpp directly, which replicates the existing
   `llamacpp_backend.py` contract 1:1.

The plan below therefore front-loads a fully verified CPU/GGUF vertical slice
(Stages 0–2, ~3–5 weeks to a drop-in `semif-score` replacement for the
llama.cpp backend) before committing to the CUDA BF16 engine (Stage 3), which
is the long pole and the main schedule risk.

---

## 2. Inventory & Susceptibility by Component

### 2.1 Runtime (`src/semif_phase1`, the port target)

| Component | LOC | Nature | Rust susceptibility |
|---|---:|---|---|
| `core.py` | 141 | `validate_row`, `direct_messages` (JSON prompt payload), `softmax`, `digest`, device resolution, pinned model loading | **Validation/prompt/softmax: trivial.** serde + `preserve_order`, f64 softmax with left-to-right fold (bit-reproducible given identical logits). **Loading: does not port** — it is replaced by the engine strategy (§5). |
| `direct.py` | 77 | Chat-template rendering, no-truncation budget, single-token slot round-trip + prefix-stability checks, full-vocab last-position readout | **Easy *given* a template renderer + tokenizer** (§4.1–4.2). The `inspect.signature(model.forward)` capability probe becomes an explicit engine trait. |
| `serial.py` | 132 | One-state prefix cache + per-decision branch deepcopy, `allowed_token_mass` logsumexp | **Moderate.** Logic is simple; semantics live in cache copy/branch execution (§5). |
| `shared.py` | 185 | State-prefix extraction, padded suffix layout with `position_ids`, batched (`reorder_cache`) vs looped (MPS) execution, `logits_to_keep=tensor` gather | **Moderate–hard.** The position-shifted padded batch and per-row logit gather must be reproduced exactly; the looped/batched split is a `serving_config`-visible behavioral fork. |
| `reranker.py` | 126 | Hardcoded reranker prompt (no chat template), left-padded pair batching, yes/no single-token contract, per-option log-odds | **Easy engine-side, one trap:** `_answer_ids` must verify `convert_tokens_to_ids` parity, not just encode round-trip. |
| `cli.py` | 107 | argparse routing, backend/mode cross-validation, create-only output, flush-per-row, `allow_nan=False` dumps | **Easy.** clap; error-surface parity fixture-pinned (§4.5). |
| `llamacpp_backend.py` | 405 | CPU GGUF scoring via ctypes: reference-tokenizer verification, chunked decode, whole-sequence state save/restore (hybrid memory constraint), verified re-tokenization | **High.** This is the natural first engine: a thin, fully specified native-API contract that `llama-cpp-2` (or raw FFI) replicates exactly. Same C library ⇒ near-bit-exact logits parity is realistic. |
| `mlx_backend.py` | 237 | Apple-Silicon MLX-LM scoring, in-memory quantization, artifact hashing | **Out of scope.** Unverifiable on this machine (Linux + NVIDIA). Keep the Python backend; see §7 out-of-scope list. |

### 2.2 Everything else (mostly not the port target)

| Area | LOC | Disposition |
|---|---:|---|
| `tests/` (55 tests) | 978 | **Keep as the contract.** Largely mock-based (CLI routing, prefix layout, reranker contract, calibration math) and runs without model downloads; device/llamacpp/mlx tests gate on local resources. The mock-stub suites define pure-logic behavior that Rust tests can mirror fixture-for-fixture. |
| `benchmarks/` | ~2,700 | **Keep in Python, forever.** Builders with frozen manifests and hash verification, evaluators (`evaluate.py`, `evaluate_perturbations.py` — "every declared row counts"), `shape777.py`/`shape777_reranker.py` runners, calibration fitting (golden-section NLL search). These are the independent observer; porting them to the thing they observe would invalidate them (von §6 rule, adopted verbatim). Optionally add adapters so evaluators can consume Rust-produced prediction files — they already can, since the row schema is the contract. |
| `results/` + `phase1-summary.json` | — | **Untouchable** (AGENTS.md). Committed row-level evidence + SHA256SUMS are the numeric goldens for Stage 4. The Rust port never rewrites them; Rust outputs go to new create-only paths. |
| `exl3-bridge/` | ~330 | **Do not port.** A standalone quantized execution track with its own evidence; additive by design. |
| `webgpu-demo/` | — | **Unaffected.** Static, browser-only (wllama/WASM), no build step; AGENTS.md forbids disturbing it. A Rust port is invisible to it. |
| `docs/`, `manifests/`, `examples/`, `assets/`, `demo/` | — | Pinned references; consumed read-only by the port. |

### 2.3 Artifacts

- **Checkpoints** (pinned in `manifests/models.json`, never committed):
  `Qwen/Qwen3.5-4B @ 851bf6e8` (BF16 direct), `Qwen/Qwen3-Reranker-4B @ 22e6836`
  (reranker), plus browser-ladder GGUFs. A Rust engine consumes the same
  pinned revisions — HF cache is content-addressed and shared read-only with
  the sibling ports (§8.1).
- **GGUF quantizations** for the llama.cpp path (e.g.
  `bartowski/Qwen_Qwen3.5-4B-GGUF @ 4168f45a`): consumed directly by the
  Rust engine; every row already carries the GGUF sha256 (the Python backend
  computes it — the port must too).
- **Calibration temperatures** (`results/raw/calibration/`): fitted scalars the
  Rust scorer never applies (calibration is a separate labeled layer that
  reads committed logits — keep that layer Python).

---

## 3. What Makes This Port Different from von (and Harder)

Von's plan could promise byte-exact wire parity because von is a stateless
encoder exposed over HTTP. SemIf is neither stateless nor encoder-based, and
its repo culture is measurement-first. Five concrete consequences:

1. **No byte-diff gate.** Output rows contain `perf_counter` timings
   (`forward_seconds`, `prefill_seconds`, …) and runtime-version metadata
   (`torch_version`, `transformers_version`, `mlx_*`). The differential
   instrument is a **schema-aware row-diff** (§6), not `cmp`. Three field
   classes: *identity-exact* (ids, `prompt_sha256`, `input_tokens`,
   `option_ids`, `prompt_version`, `readout`, `serving_config`, revision &
   dtype metadata), *numeric-gated* (logits, probabilities,
   `allowed_token_mass`, `full_vocab_argmax_id`), *timing-excluded*.
2. **BF16 grid cuts both ways.** Committed `option_logits` (25.0, 25.25,
   28.375 …) sit visibly on the BF16 grid: a Rust engine producing logits
   within half a BF16 ulp snaps to *identical* stored values, making exact
   field matches common. But near-ties flip: the repo's own evidence records
   5–6/777 argmaxes changing across torch execution paths. Gates must be set
   *from that documented flip rate*, not from a bit-parity fantasy (§6).
3. **The cache is the product.** The speed headline (20.03 dec/s parallel
   suffixes vs 2.33 fresh) lives in cache replication and batched suffix
   execution. A port that only implements fresh direct scoring is a toy; the
   plan therefore gates serial and shared modes as first-class deliverables
   with their own parity instruments.
4. **The chat template is a new single point of failure.** Von rendered a
   fixed instruction string into a BERT tokenizer. SemIf's
   `apply_chat_template(..., enable_thinking=False)` executes the
   checkpoint's Jinja template with transformers' environment, and
   `prompt_sha256` — the join key to *all* committed evidence — is the hash
   of its exact output. Template divergence invalidates everything
   downstream while looking perfectly healthy locally (§4.1).
5. **Multi-backend matrix.** Von had one engine; SemIf has torch
   (CUDA/MPS/CPU × BF16/FP16/FP32), MLX, llama.cpp, plus out-of-tree exl3 and
   browser tracks. The port must scope honestly: CUDA-torch and llama.cpp are
   portable targets; MPS is best-effort; MLX is out of scope. Every output row
   already self-describes its backend, so mixed-backend evidence stays
   honest by construction.

---

## 4. Parity Traps (where a naive Rust port silently diverges)

### 4.1 Chat-template rendering — the crown jewel
`encode_prompt` renders `direct_messages(row)` through the checkpoint's Jinja
template (`add_generation_prompt=True`, `enable_thinking=False`), encodes with
`add_special_tokens=False`, and hashes the result. The Rust side needs a
renderer that produces **byte-identical prompt strings** for the pinned
checkpoints. Order of preference:
1. **Pinned-fixture renderer (Stage 1):** capture rendered prompt strings from
   the Python oracle for the whole golden corpus and implement the
   template's concrete output shape (headers, `<think>` block handling, JSON
   payload embedding) against those fixtures. The owned checkpoints are
   pinned revisions; their templates do not drift.
2. **General renderer (optional, later):** `minijinja` executing the
   checkpoint's own template with a transformers-compatible environment
   (`tojson` filter, whitespace control, `enable_thinking` flag). Only worth
   it if unpinned checkpoints become a goal; it must still pass fixture (1)
   byte-for-byte.
Either way the corpus must include: unicode states (the payload is dumped
`ensure_ascii=False` — core.py:55 — while CLI *output* dumps default to
`ensure_ascii=True`: an intentional asymmetry a byte-level comparator must
reproduce), structured dict/list states, 16-option rows, and long states.

### 4.2 Tokenizer
Fast tokenizers are already Rust (`tokenizers` crate) — structural parity
[proven in von]. SemIf-specific contract points that must be pinned, not
assumed:
- **Detached truncation/padding:** the shipped `tokenizer.json` can carry
  baked-in truncation that Python's bare `encode(..., add_special_tokens=False)`
  never applies [proven in von — Qwen3.5-class tokenizer configs did exactly
  this]. SemIf *refuses* truncation (`len(ids) > max_tokens` raises), so a
  silent 512-cut would produce wrong-but-plausible rows. Detach both settings
  at load; golden-test the ~1,900-token shape777 prompts.
- **Slot contract:** every letter `A`–`P` must encode to exactly one token
  whose decode round-trips (`_slot_ids`), **and** appending the letter to the
  prompt must not re-tokenize the boundary (`encode(prompt + letter) ==
  ids + [token]`). Reimplement both checks exactly — they are runtime
  guarantees users rely on, not internal asserts.
- **Reranker answer contract:** `no`/`yes` single tokens verified through
  `convert_tokens_to_ids`, not just encode (reranker.py:41) — a distinct
  predicate that catches vocab-alias divergences encode-round-trip misses.

### 4.3 JSON semantics at both edges
- **Input:** Python `json.loads` accepts `NaN`/`Infinity` literals and
  duplicate keys; `validate_row` then rejects non-finite state via a
  `allow_nan=False` re-dump ("state must be finite JSON-compatible data").
  `serde_json` rejects `NaN` at parse with a different error. Rejection
  surfaces (message text, stderr shape, exit code) must be fixture-pinned,
  not inherited from the JSON library.
- **Output:** Python `json.dumps(..., allow_nan=False)` emits shortest-*
  roundtrip f64; `serde_json` matches *given equal f64 values* [proven in
  von]. But it emits `null` for non-finites where Python raises, escapes
  non-ASCII only under a compat writer (Python's default `ensure_ascii=True`
  in cli.py:93/97/102), and needs `preserve_order` workspace-wide from day
  one [proven in von — a `BTreeMap` slip survived parsed-dict comparisons].
- **Prompt payload:** `ensure_ascii=False`, insertion-ordered keys — a second
  writer profile. Two profiles, both fixture-pinned.

### 4.4 Numeric semantics worth pinning even where easy
- `softmax` (core.py:59) is pure f64 with max-shift and a **left-to-right**
  `sum()`; the identical op sequence in Rust is bit-exact given identical
  inputs, so probability fields inherit engine logit fidelity directly.
  Do not "improve" it with pairwise summation.
- Slot logits are `float32` → f64 conversion — exact; but
  `allowed_token_mass` is `logsumexp` over the **full vocabulary** in f32
  (torch) vs numpy f64 in the llamacpp backend vs MLX's own — cross-backend
  values already differ by design. Gate per-backend, never cross-backend.
- `full_vocab_argmax_id`: torch argmax returns the **first** maximal index;
  write an explicit strict-`>` fold in Rust [proven in laya — Rust
  `max_by` keeps the last max and this genuinely fires].
- Reranker `odds = yes − no` and `softmax` over exactly two values in f32
  before f64 conversion — keep op order.

### 4.5 CLI & process semantics
- Create-only output (`open("x")`, refuse existing path) and
  flush-per-row streaming (crash-resumable at row granularity) are
  user-visible contracts — test them.
- Uncaught `ValueError` from `validate_row` currently exits as a traceback
  with code 1; `parser.error` paths exit 2 with usage text. A Rust port
  should reproduce the *contract* (which violations reject before any output
  file is created) while accepting clap-vs-argparse message divergence
  [proven in von: CLI help text divergence was accepted, wire behavior was
  not].
- `resolve_device` enforces **exactly one** visible CUDA GPU
  (`device_count() != 1` raises) — AGENTS.md's one-GPU-per-scorer rule is
  embedded in the loader. Keep it.
- `inspect.signature(model.forward)` capability probing (`logits_to_keep`
  availability) is a transformers-runtime smell; the Rust engine trait makes
  capabilities explicit per mode instead of introspected.

### 4.6 State-prefix extraction (serial/shared)
`_state_prefix` locates the evidence payload inside the rendered prompt via
string search, asserts it occurs exactly once, splices the prompt at the
payload start, and drops one trailing token ("appending JSON punctuation can
merge with the final boundary token"). Every step — the uniqueness assert,
the `json.dumps({"evidence": state})[:-1]` reconstruction, the `[:-1]`
token drop — is load-bearing for cache reuse and must be ported
operation-for-operation with fixture tests including dict/list states whose
JSON rendering embeds quotes and escapes.

---

## 5. Model Execution Strategy

The forward-pass shapes to support (per mode): fresh full-prompt →
full-vocab last-position logits (direct); prefix prefill → branch deepcopy →
suffix decode (serial); prefix prefill → cache replication → padded,
position-shifted batched suffix forward → per-row logit gather (shared,
CUDA path) or looped branch forwards (MPS path); left-padded batched pairs →
last-position logits (reranker). All modes read only last-position (or
selected-position) logits — **no generation loop exists anywhere**, which is
what makes the port tractable at all.

### Route A: ONNX export + `ort` — **not the backbone; optional direct-mode spike only**
Exporting Qwen3.5's hybrid full-attention/linear-attention decoder with
KV-cache prefill, cache replication, and tensor-`logits_to_keep` into a
static graph is unsolved in the von/laya recipe book; their proven
dynamo-exporter path targeted a stateless encoder [proven in laya, not
transferable]. At best a direct-mode-only export could serve the
"0 generated tokens" path — but it forfeits serial/shared, i.e. the project's
speed claims, and it would face the Blackwell `sm_120` runtime trap on this
very machine (fix known: `ort/load-dynamic` against the official wheel's
dylib [proven in von]). **Verdict: skip unless everything else has landed and
a one-graph direct readout is still interesting.**

### Route B: `tch` (libtorch bindings), hand-written Qwen3.5 — **recommended for the CUDA BF16 engine**
- Shares the *same* libtorch the oracle uses ⇒ kernel numerics inherit
  torch's CUDA paths rather than diverging by runtime (the laya-proven
  route reached 1.8e-5 corpus logit L∞ on a hand-written encoder).
- Work items: safetensors loading (strict, missing-key rejection like
  `load_causal_model`'s `output_loading_info` check), the hybrid layer stack
  (full-attention layers with SDPA + KV cache; linear-attention layers with
  their recurrent state ported op-for-op from transformers 5.17's modeling
  code), cache structures with explicit replicate/deepcopy and reorder
  (index-select) ops, position-shifted padded batching, selected-position
  logit gather, BF16 execution with f32 upcast at the readout.
- Numerics expectation: BF16-grid snapping makes many stored logits
  *exactly* reproducible; the residual risk is near-tie argmaxes, bounded by
  the repo's own documented 5–6/777 flip rate. Gate behavior, not bits (§6).
- Cross-reference instrument: per-layer forward-hook captures from the
  Python oracle — never a third-party reference implementation [proven in
  laya: candle's reference ModernBERT had opposite GeGLU gate order; assume
  nothing about Qwen3.5 reference ports either].
- **Known unknown to retire first (Stage 3 gate 0):** confirm transformers
  5.17's Qwen3.5 path executes with *public torch ops* (SDPA + linear-attention
  math) rather than private/custom kernels on this stack. If custom kernels
  dominate, fall back to capturing intermediate states and matching
  sub-module by sub-module — slower, same instruments.

### Route C: llama.cpp bindings (`llama-cpp-2` or raw FFI) — **the CPU engine, Stage 2**
`llamacpp_backend.py` is already a fully specified native-API contract:
backend init with GPU offload disabled, `n_seq_max=1` context, chunked
decode with last-position logits flags, whole-sequence state
save/restore for branch replication (the hybrid memory supports neither
sequence copies nor partial tail removal — the docstring says so and the
code complies), reference-tokenizer agreement verification before any
evaluation, GGUF checksum in every row. A Rust port replicates it 1:1
against the *same* llama.cpp the Python wheel links, so logit parity vs the
Python backend on one machine should be at noise level, and the parity
instrument is the backend itself. This is the cheapest full vertical slice
through all three modes and the CLI, and it is CPU-only ⇒ CI-runnable.

### Route D: `candle`/`burn` — **not recommended**
Qwen3.5's hybrid layers almost certainly lack reference implementations;
effort ≥ Route B with worse parity guarantees and no shared-libtorch
inheritance. Revisit only for edge/single-binary deployment goals after
Route B has produced a validated layer-by-layer spec.

### Tokenizer & template (all routes)
`tokenizers` crate with truncation/padding detached (§4.2) plus the §4.1
template strategy. Prompt construction stays on the pinned reference
tokenizer exactly as the llamacpp backend does it today — the
`prompt_sha256` row-for-row match across backends is a *property the port
must preserve*, and it is the cheapest regression tripwire in the repo:
any Rust row whose `prompt_sha256` differs from its Python twin is broken
before numerics are even discussed.

### Explicitly out of execution scope
- **MLX backend** — macOS/arm64-only; unverifiable here; the Python backend
  remains the Apple path (its committed evidence in `results/mlx/` stays
  authoritative).
- **MPS device** — best-effort only; gate as "runs, agrees at decision
  level," no timing claims.
- **`exl3-bridge/`, `webgpu-demo/`** — separate tracks by design; untouched.

---

## 6. Verification Architecture (the core of this plan)

**Guiding rule (von's, adopted):** the Python scorer is the oracle, forever
in-tree, and every stage proves equivalence against it — but the *gate
family* changes where the repo's own evidence philosophy demands it:
**identity fields exact; numerics tolerance-gated per backend; timings
excluded.** Pretending byte parity would make the gate either impossible
(timings) or dishonest (documented cross-path flips).

### Instruments the repo already ships (Stage 0 leans on them)
1. **Row-level prediction goldens:** `results/raw/*.predictions.jsonl`
   (shape777: 777 decisions × 3 execution paths; authored144, WANLI,
   typesafe102, perturbations108, every204) with `option_logits`,
   `probabilities`, `allowed_token_mass`, `prompt_sha256` — all checksummed
   via `SHA256SUMS`.
2. **`benchmarks/evaluate.py` and friends:** metric engines that accept any
   schema-conformant prediction file — a Rust-produced file is graded by the
   *unmodified* Python evaluator, which is exactly the independent-observer
   property we want.
3. **`verify_published.py`** — proves summary claims against row evidence;
   run it before and after any port stage that touches evidence-adjacent
   paths (it must never change).
4. **55-test pytest suite**, mostly mock-based; the pure-logic suites
   (core/serial/shared/reranker/calibrate/cli) double as behavioral specs
   for fixture generation.

### Instruments Stage 0 adds (Python-only, permanent)
5. **Prompt-rendering goldens** (`benchmarks/export_fixtures.py`): rendered
   prompt strings + `prompt_sha256` + token-ID sequences + slot IDs for a
   coverage corpus — owned examples, authored144, shape777 sample,
   perturbations108, plus authored edge rows (16 options; unicode;
   dict/list/nested states; near-limit token budget; refusal cases:
   over-budget row, duplicate option ids, non-finite state, missing fields).
6. **Logit probes** (`benchmarks/dump_logits.py`): pre-softmax full-vocab
   readouts (slot logits + mass + argmax) for every corpus row, captured
   **per device/dtype context**: CPU fp32 (the README's explicit reference
   path), CUDA BF16 (device-tagged — committed rows are RTX 3090 provenance,
   this machine's Blackwell laptop GPU gets its own capture), and GGUF
   via the llamacpp backend. These are the numerics stethoscope that lets a
   failure localize to engine vs prompt vs post-processing.
7. **Row-diff tool** (`scripts/diff_rows.py`): compares two JSONL scorer
   outputs row-by-row (joined by `id`): identity fields must be
   string-equal; numeric fields checked against per-field gates; timing and
   runtime-version metadata excluded but *presence-checked*. Emits a parity
   report (agreement %, max deltas, flip list). This is the workhorse for
   every later stage.
8. **Baseline record:** `pytest -q` green in the oracle venv (precondition);
   informational shape777 timing baseline on this machine (for Stage 4's
   honest speed comparison — never compared across machines to committed
   numbers).

### Gates (defined once, enforced at every stage)

| Gate | Threshold | Instrument |
|---|---|---|
| Template parity | rendered prompt strings byte-identical, 100% of corpus | Prompt goldens |
| Token parity | 100% identical IDs, slots, boundary checks; truncation/padding detached | Token goldens |
| Logit parity, CPU fp32 (tch vs torch CPU) | max abs Δ ≤ 1e-4 on slot logits | Logit probes |
| Logit parity, GGUF (Rust vs Python llamacpp, same lib) | max abs Δ ≤ 1e-5 target; decision agreement 100% | Logit probes |
| Logit parity, CUDA BF16 | BF16-grid-snapped equality ≥ 99% of slot logits; remainder ≤ 1 grid step | Logit probes |
| Decision agreement vs Python, same device | ≥ 99% argmax per fixture (repo's own cross-path flip evidence: 99.2–99.4%) — target 100%, escalate any miss | Row-diff |
| Decision agreement vs committed rows (3090-provenance) | informational, device-tagged; never a hard gate across GPU generations | Row-diff |
| Probability fields | exact f64 equality given equal logits (softmax is bit-reproducible); otherwise \|Δp\| ≤ 1e-3 with flagged flips | Row-diff |
| Identity fields | exact: `prompt_sha256`, `input_tokens`, `option_ids`, `prompt_version`, `readout`, `serving_config`, source/revision/dtype | Row-diff |
| Refusal contract | every corpus error case rejected *before output-file creation*, pinned message family | CLI fixtures |
| Accuracy fingerprint | authored144 / perturbations108 / shape777 agreement within the repo's documented cross-path envelope (±0.5pp balanced accuracy; shape777 ≥ 99% agreement) | `evaluate.py` on Rust-produced files |
| Evidence integrity | `SHA256SUMS` clean; `verify_published.py` 69/69 — unchanged | Repo gates |

---

## 7. Staged Plan

### Stage 0 — Oracle & instrumentation (Python-only; no Rust yet) — *2–3 days*

Deliverables:
1. Oracle venv brought up per `docs/REPRODUCE.md` (`pip install -r
   requirements.txt && pip install -e .`); **`pytest -q` green** recorded as
   the precondition the assessment environment could not run; HF cache
   primed for both pinned checkpoints (shared with sibling ports, read-only).
2. `benchmarks/export_fixtures.py` → `semif-rs/fixtures/`:
   `prompts.jsonl` (rendered text + sha256), `tokens.jsonl` (IDs, slots,
   boundary-check results), `rows.jsonl` (validation verdicts incl. refusal
   cases with exact error messages). Coverage matrix per §6.5.
3. `benchmarks/dump_logits.py` → `semif-rs/fixtures/logits-{cpufp32,gguf}.jsonl`
   + `logits-cuda-bf16.<gpu-tag>.jsonl`. CPU fp32 via the README's
   `--device cpu --dtype float32` reference path; GGUF via the llamacpp
   backend; CUDA via the standard BF16 path on this machine's GPU, tagged by
   device name (never merged with 3090-provenance committed evidence).
4. `scripts/diff_rows.py` with the three field classes and per-gate
   reporting; smoke-tested by diffing Python-vs-Python runs (must report
   perfect identity-exact agreement and zero numeric deltas — it is also a
   check on torch's own determinism on this machine).
5. Baseline file `semif-rs/baselines/python-this-machine.json` (informational
   timings + decision agreement of this GPU's BF16 run vs committed 3090
   rows — quantizes "normal" cross-device drift before any Rust exists).

Exit criteria: `pytest -q` green; goldens committed; diff tool green on
Python-vs-Python; integrity gates (`SHA256SUMS`, `verify_published.py`)
still clean.

### Stage 1 — Cargo workspace + pure-logic port — *1–2 weeks*

Deliverables (workspace `semif-rs/`, crates `semif-types`, `semif-core`,
`semif-engine` (trait + capability flags, stub), `semif-cli`):
- `semif-types`: row/option schemas (`preserve_order` workspace-wide from
  day one [proven in von]), validation with pinned refusal messages,
  metadata/result row envelope.
- `semif-core`: JSON writer profiles (prompt payload `ensure_ascii=False`
  insertion-ordered; output rows `ensure_ascii=True` + `allow_nan`
  rejection), f64 softmax with left-to-right fold, sha256 digesting,
  state-prefix extraction (§4.6), input parsing with Python-compatible
  acceptance of `NaN` literals then rejection at validation (§4.3).
- **Template renderer** for the pinned checkpoints against `prompts.jsonl`
  byte-equality fixtures (§4.1 stage-1 form).
- **Tokenizer harness**: `tokenizers` with truncation/padding detached;
  100% parity on `tokens.jsonl` including slot round-trip and boundary
  stability checks [proven in von: detach, don't trust the shipped config].
- `semif-cli`: clap wiring with backend/mode cross-validation, create-only
  output, flush-per-row, refusal-before-create contract; runs end-to-end
  against a stub engine (uniform-probability rows like von's Stage 1 stub).
- Fixture harness: Python script (re)generates all expected outputs;
  Rust tests consume the fixtures; the unmodified pytest pure-logic suites
  are mirrored test-by-test.

Verification: fixture gates 100% (template byte-equal, tokens identical,
refusals exact); `scripts/diff_rows.py` reports identity-exact for stub rows.

### Stage 2 — CPU engine via llama.cpp bindings (vertical slice) — *1–2 weeks*

Order of work (each sub-step gated):
1. `semif-engine-llamacpp`: model load with GPU offload disabled, context
   with `n_seq_max=1`, vocabulary-verification probe (the §4.2 checks
   against the GGUF tokenizer, including `_gguf_piece` letter round-trip),
   GGUF sha256 in metadata — all mirroring `llamacpp_backend.py` exactly.
2. Direct mode: chunked decode, last-position logits, slot gather, mass +
   argmax; gate **logit parity vs `dump_logits.py` GGUF capture ≤ 1e-5** and
   decision agreement 100% over the corpus.
3. Serial mode: prefill → whole-sequence state save/restore branches;
   gate on `cache_hit`, `prefix_sha256`, `branch_state_bytes` presence and
   decision parity vs Python on the same GGUF.
4. Shared mode: restore-per-branch loop (the Python llamacpp path is itself
   serial-restoring — no batched fork to argue with); timing-block field
   parity by schema.
5. CLI integration: `--backend llamacpp` equivalent runs the Rust engine;
   full row-diff vs Python GGUF runs over examples + authored144 + a
   shape777 slice.

Exit criteria: **a `semif-rs` binary scores every fixture the Python
llamacpp backend scores, with 100% decision agreement, identity-exact rows,
and logit deltas at native-library noise level — verified by the unmodified
`scripts/diff_rows.py`.** This is a complete, useful, CI-friendly
drop-in even if the project stopped here.

### Stage 3 — CUDA BF16 engine via `tch` (the long pole) — *4–8 weeks*

Order of work (each sub-step gated before the next):
1. **Gate 0 — feasibility spike:** confirm the pinned Qwen3.5 forward pass
   executes via public torch ops under `tch` on this machine (Blackwell
   laptop GPU), one corpus row, CPU first then CUDA; compare against the
   Python per-layer hook captures [proven in laya methodology]. If custom
   kernels block op-for-op porting, re-plan before writing the full layer
   stack.
2. Weight loading: safetensors, strict key/missing checks mirroring
   `load_causal_model`; BF16 parameter placement on one device; one-GPU
   enforcement (§4.5).
3. Direct mode: full-prompt forward, last-position full-vocab logits, f32
   upcast, slot gather; gates: CPU-fp32 logit ≤ 1e-4 (vs the CPU oracle
   capture), CUDA BF16 grid-snap ≥ 99% + decision ≥ 99% (vs this machine's
   Stage 0 CUDA capture, device-tagged).
4. Serial mode: KV/state cache prefill + explicit branch copy;
   `allowed_token_mass` and argmax via strict-`>` first-max fold; parity vs
   Python serial on the same device.
5. Shared mode: prefix cache replication (`reorder` equivalent), padded
   suffix layout with `position_ids`, selected-position logit gather
   (`logits_to_keep=tensor` semantics); decide the looped-vs-batched
   `serving_config` question explicitly — implement CUDA-batched first, keep
   looped as the debug path (mirroring the Python fork).
6. Reranker mode: hardcoded prompt (no template), left-padded pair batch,
   `convert_tokens_to_ids` yes/no contract, log-odds readout; CUDA-only like
   Python; parity vs Python reranker probes.
7. Dtype/device matrix: fp16/fp32 and CPU bf16 paths behind the same
   explicit capabilities; each new context re-gated against a fresh probe
   capture (GPU numerics are compared GPU-Rust vs GPU-Python on the *same*
   device, never across devices).

Exit criteria: row-diff green at the §6 gates for all four modes on CUDA
BF16 + CPU fp32; decision agreement ≥ 99% per fixture with any flip
individually inspected and explained (near-tie ledger, like the repo's own
5–6/777 accounting).

### Stage 4 — Evidence integration & fingerprints — *1–2 weeks*

Deliverables:
- Run the Rust scorer over the owned workloads into **new create-only
  paths** (`semif-rs/results/…`; never `results/`): authored144,
  perturbations108, shape777 (37×21, all three reuse modes), every-grid
  sample. Grade the Rust-produced files with the **unmodified Python
  evaluators** (`evaluate.py`, `evaluate_perturbations.py`) and record the
  fingerprint table vs the Python same-device baseline and (informationally)
  vs committed 3090 rows.
- Speed ledger: shape777 decisions/s for fresh/serial/shared, Rust vs
  Python on this machine, same exclusivity rules as sibling ports (§8.1).
  Tempered expectation from laya's honest end-state: Rust is *not*
  automatically faster; the wins are footprint, startup, concurrency, and
  a Python-free CPU path [proven in laya: ort/CUDA came out slower than
  torch before optimization].
- Documentation: `docs/RUST_SCORER.md` describing gates, provenance tags,
  and the flip ledger. **No changes to `results/`, `phase1-summary.json`,
  README headline claims** (AGENTS.md hard rule) — if the Rust engine ever
  earns a headline claim, that is a separate evidence commit per the repo's
  own policy, not a byproduct of this stage.

Exit criteria: fingerprints inside the §6 envelope; integrity gates clean;
side-by-side one-command check (Python run → Rust run → `diff_rows.py` +
evaluators) scripted as `scripts/verify_port.sh`.

### Stage 5 — Distribution & cutover (optional, product-shaped) — *1–2 weeks*

Independent deliverables, pick per goal:
- **PyO3 wheel**: `semif._rust` engine selectable from the Python CLI via
  env flag, defaulting off; gate = unmodified pytest + row-diff with the
  flag on. Keep maturin **crate-local** (`semif-rs/` own `pyproject.toml`)
  [proven in von: bare `Cargo.toml` maturin walks up to the repo root and
  can install over the oracle; with three ports on one machine that failure
  mode compounds].
- **Standalone binary**: the Rust CLI as the deployment artifact for
  CPU/GGUF scoring (the webgpu-demo's engine family, native); single
  static binary story.
- **CI rewiring**: oracle job (Python pytest + goldens regeneration on
  checkpoint bump) stays forever; Rust job (cargo test + fixture gates +
  CPU/GGUF parity; CUDA parity as a manual/gpu-tagged job).

### Explicitly out of scope (keep Python / untouched)
- `mlx_backend.py` (Apple-only verification impossible here; Python remains
  the Apple path).
- `benchmarks/` builders, evaluators, calibration fitting — the independent
  observer stays Python; the Rust port is one of the things it observes.
- `results/`, `phase1-summary.json`, README claims, `webgpu-demo/`,
  `exl3-bridge/`, `manifests/` — evidence and pinned references are
  read-only inputs.
- Third-party evaluation *sources* (TypeSafe/Every/WANLI fetch pipeline).

---

## 8. Effort & Risk Summary

**Total: ~10–17 weeks** of one experienced engineer for the full plan
(Stages 0–5). **Stages 0–2 alone (~3–5 weeks) deliver a verified,
useful CPU/GGUF `semif-score` drop-in** — the recommended commitment
boundary before deciding whether the CUDA BF16 engine (Stage 3, the dominant
cost and risk) is worth it for this project's actual usage pattern. If the
daily driver is the CUDA BF16 path, Stage 3 is the price; if GGUF/CPU or
evidence tooling is the pain point, stop after Stage 2.

**Performance expectations, tempered by laya's measurements:** do not assume
Rust is faster on GPU. At 4B BF16 the CUDA kernels dominate; a `tch` engine
shares torch's kernels and should land within noise of Python, with wins in
startup, memory footprint, and the elimination of the Python runtime on the
CPU/GGUF path. Measure and report (Stage 4 ledger); don't optimize.

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Qwen3.5 hybrid layers need private/custom torch paths (Gate 0 fails) | Medium | Stage 3 replan | Spike before building the stack; fallback is sub-module-by-sub-module hook matching; Route A/D remain rejected for good reasons — document whichever way it breaks |
| Near-tie argmax flips exceed the 1% gate | Medium | Stage 3 gate churn | Flip ledger with per-row logit deltas; compare against the repo's own 5–6/777 envelope; if systematically worse, it's an engine bug the probes localize |
| Template renderer drift on edge inputs (unicode, escapes, structured states) | Medium | Silent evidence invalidation (wrong `prompt_sha256`) | Byte-equality fixtures over the full coverage corpus in Stage 1, before any engine work; `prompt_sha256` mismatch fails loudly in every later diff |
| Tokenizer truncation config baked into `tokenizer.json` | High if untrusted (proven on Qwen3.5-class configs) | Silent 512-token cut, plausible wrong rows | Detach truncation+padding at load [proven in von]; long-prompt golden in Stage 1 |
| BF16 capture provenance confusion (3090 rows vs Blackwell captures) | Medium | Dishonest gates | Device-tag every capture; committed evidence never used as a same-device gate; Stage 0 quantifies normal cross-device drift first |
| `tch`/libtorch version coupling with the oracle venv | Medium | Build friction | Pin libtorch to the version shipped with the oracle's `torch==2.10.0` (shared-library reuse where practical, `LIBTORCH_USE_PYTORCH`-style) |
| Error-surface divergence (Python tracebacks vs Rust exits) | High if untested | Broken scripts expecting exit codes/messages | Refusal fixtures in Stage 1; document accepted divergences explicitly |
| Scope creep into "better Python" rewrite | Medium | Oracle contamination | §7 out-of-scope list is a hard rule; stage gates are the contract (von/laya defense, adopted) |
| Disk pressure (libtorch + ONNX/GGUF artifacts + cargo `target/`) | High (proven twice in laya; von carries the budget) | Blocked builds | ~5 GB budget for `semif-rs`; check free space before Stage 2/3; don't share `CARGO_TARGET_DIR` |
| GPU contention with sibling-port parity/benchmark runs | Medium | Invalid timings, flaky gates | §8.1 exclusivity rule; the loader's one-visible-GPU check helps but is not a scheduler |

### 8.1 Side-by-side coexistence with the von and laya ports

1. **Zero shared artifacts.** Workspace `semif-rs/`, crates `semif-*`;
   fixtures, scripts, and targets duplicated per project, never imported
   from `von-rs/` or any laya path. Parity instruments resemble the
   siblings' on purpose (methodology transfer) but are independent files.
2. **Separate environments.** SemIf's oracle venv lives in this repo
   (`.venv`, per `docs/REPRODUCE.md`); never install either port's PyO3
   cdylib into another project's venv; crate-local `pyproject.toml` from
   day one [proven in von].
3. **Disk budget, combined.** Three ports × ~5 GB each; check the partition
   before Stage 2/3; heavy artifacts on the roomy drive; no shared
   `CARGO_TARGET_DIR`.
4. **HF cache shared, read-only, fine.** All three projects resolve pinned
   revisions from `~/.cache/huggingface`; content-addressed and
   concurrent-safe; the Qwen3.5-4B checkpoint and its GGUF are new entries
   the siblings don't hold.
5. **GPU exclusivity for every timed or parity gate.** One Blackwell laptop
   GPU serves all three ports. SemIf's CUDA captures, shape777 runs, and
   speed ledger get the GPU to themselves — never concurrent with a von or
   laya parity/e2e run. (The loader's "exactly one visible GPU" rule is a
   correctness guard, not a scheduling one.)
6. **Divergent default ports are irrelevant here** — no server exists; the
   differential harness is CLI-vs-CLI over files, so there is nothing to
   collide with the siblings' :8001/:8002 habits.

---

## 9. Recommended Stack

| Concern | Choice |
|---|---|
| Tokenizer | `tokenizers` (same Rust impl HF Python calls), truncation/padding detached |
| Prompt template | Stage 1 pinned-fixture renderer; optional `minijinja` generalization later |
| CPU engine | `llama-cpp-2` (or raw FFI) against the same llama.cpp the Python wheel links |
| CUDA engine | `tch` pinned to the oracle's libtorch; hand-written Qwen3.5, BF16 params + f32 readout |
| Weights | safetensors direct (no conversion needed — HF checkpoints already are safetensors) |
| Schemas / JSON | serde + `serde_json` (`preserve_order`), two writer profiles (§4.3) |
| CLI | clap (derive), create-only + flush-per-row semantics preserved |
| PyO3 wheel (Stage 5, optional) | pyo3 + maturin, crate-local `pyproject.toml` |
| Testing | cargo test + fixture gates + `scripts/diff_rows.py` + unmodified Python evaluators as the observer |

---

## 10. Progress log

### Stage 0 — complete (2026-09-24)

Oracle instrumented; every exit criterion met. Two caveats recorded up front:
captures ran while two unrelated heavy tasks loaded the machine, so **all
timing fields are void** (numerics unaffected — same kernels, same math) and
the informational timing baseline is **deferred** to an idle-machine window;
and the CPU fp32 / GGUF probes cover a pinned 25-row subset (10 edge + 3 owned
+ 9 authored originals + 3 shape777 state-0 rows), with the full CPU corpus
deferred.

- **Oracle environment**: venv per `docs/REPRODUCE.md`; `pytest -q` **68 passed,
  2 skipped** (MLX and MPS — platform-impossible here; the llamacpp test runs
  green against the real GGUF via `SEMIF_LLAMACPP_GGUF`). Integrity gates clean
  at exit: `results/raw/SHA256SUMS` OK, `verify_published.py` 69/69.
- **Checkpoints**: the `hf` client stalled on this link (~3.5 KB/s via xet);
  replaced with resumable curl into a plain local dir
  (`/media/ianj/New Volume/hf-plain/`), every artifact sha256-verified against
  its LFS etag (both shards + the pinned Q4_K_M GGUF). HF cache relocated to
  the big drive (`New Volume/hf-cache`) — the root disk is 98% full; von/laya
  caches untouched.
- **Fixtures** (`semif-rs/fixtures/`, commit `00ffcc3`): 322 corpus rows → 307
  accepted + 15 refused. `prompts.jsonl`/`tokens.jsonl`/`rows.jsonl`/
  `inputs.jsonl`/`cli_errors.jsonl` + manifest. Quirk pinned: **empty option
  ids are accepted** by `validate_row` (`edge-description-empty`-adjacent row
  `refuse-empty-id` renders fine) — contract behavior, not a refusal. CLI
  process contract pinned: row ValueError → rc 1 + traceback line, no output
  file; existing-output/empty-input → rc 2 argparse messages.
- **Token parity vs committed evidence, for free**: the shape777 state-0 rows
  encode to **1,891 tokens — exactly the committed prediction rows'**
  `input_tokens`; slots 32–47 (A–P) match. The template/tokenizer contract is
  byte-stable across machines.
- **Probes** (`semif-rs/probes/`): CUDA BF16 **286 rows**; CPU fp32 and GGUF
  25-row subsets, row-aligned. Cross-profile: token identity holds everywhere;
  cpufp32-vs-cuda-bf16 argmax **100%** with logit |Δ| ≤ 0.196; gguf-vs-fp32
  argmax **96%** with |Δ| ≤ 1.63 and one substantive quantization flip
  (`8a4c3de2…`, fp32 [0.085, 0.173, 0.741] vs gguf [0.095, 0.514, 0.391] —
  "conditional on the quantized weights" made concrete).
- **Cross-device drift quantified before any Rust exists**
  (`compare-cuda-vs-committed-3090.json`): this machine's Blackwell BF16 vs
  committed 3090-provenance rows: **argmax agreement 99.31%** (1 flip in 144,
  `d6de731c…`, a knife-edge where local top-2 tie at 0.4045), bf16-snap-equal
  logits on only 20% of rows, mean prob delta 0.008. This is the honest scale
  of "normal" drift the §6 gates were designed around.
- **Determinism**: two identical CLI runs (48 authored rows, CUDA BF16) diff
  identity-exact, numerics clean, argmax 1.0 through `diff_rows.py` — the row
  diff instrument works on real scorer output.
- **Stage 3 Gate 0 effectively retired early**: transformers 5.17 warns that
  `chunk_gated_delta_rule` runs its **reference pure-torch implementation**
  (`flash-linear-attention` absent — and absent from `requirements.txt`, so
  the committed evidence used it too). The oracle's numerics are public torch
  ops ⇒ the `tch` route inherits torch's own kernels. Rule recorded in the
  baseline: **never install `flash-linear-attention` into the oracle**; it
  would silently change the parity target.
- **Config facts observed** for Stage 3: text config nested
  (`model_type qwen3_5` → `text_config`, `qwen3_5_text`), vocab **248,320**,
  32 layers, hidden 2560, `full_attention_interval: 4`, separate
  `chat_template.jinja` in the repo tree.

Rust work begins at Stage 1 against `semif-rs/fixtures/`.

### Stage 1 — complete (2026-09-24)

Cargo workspace landed in `semif-rs/` (four crates, edition 2024, rust 1.93,
`serde_json/preserve_order` workspace-wide, no async — SemIf is a batch CLI,
not a server). Every Stage 1 gate green; clippy 0 warnings; `cargo fmt` clean;
27 Rust tests passing. Committed in three deliverable commits.

- **Crates**: `semif-types` (Python-shaped value model + validation with the
  pinned check order and message strings), `semif-core` (Python-compatible
  parser, `repr(float)` formatter incl. exponent thresholds, the two
  `json.dumps` writer profiles, f64 softmax, sha256, the pinned Qwen3.5
  template renderer, the tokenizer harness with truncation/padding detached,
  state-prefix extraction), `semif-engine` (Engine trait + uniform-probability
  stub that still runs the real prompt/token/slot pipeline), `semif-cli`
  (clap; create-only output, flush-per-row, validation order mirroring
  `cli.py`).
- **Gates** (all over the Python-authored fixtures, in-tree as Rust tests):
  prompt byte-equality **307/307**; token+slot+prefix parity **307/307**;
  validation/encode verdicts **322/322** (incl. the accepted `refuse-empty-id`
  quirk); float repr 30, dumps profiles 9, softmax vectors 8; CLI refusal
  contract 6 cases (rc + output-not-created + exact row-level ValueError
  lines); and the headline wire gate — **stub rows byte-identical 307/307**
  (`fixtures/stubs.jsonl` vs the Rust CLI over `inputs-accepted.jsonl`).
- **New fixtures** (`benchmarks/export_logic_fixtures.py`, committed):
  `logic.json` (float reprs, both dumps profiles, softmax cases), plus
  `stubs.jsonl`/`inputs-accepted.jsonl` for the wire gate. The Python stub
  envelope mirrors `StubEngine` exactly (direct-shape row, 1/K probabilities,
  zero logits, no timing fields).
- **Findings**:
  1. The chat template's `render_content(...)|trim` filter is applied to both
     system and user content — a no-op for our payloads (they start `{`, end
     `}`), implemented anyway and pinned.
  2. Python's default `json.dumps` separators are `", "` / `": "` — the
     prompt payload and output rows are *spaced* JSON, not compact; the
     two writer profiles reproduce this and are byte-gated.
  3. Fixture files are themselves Python-flavored JSON (NaN literals in
     `inputs.jsonl`/`logic.json`), so the Rust tests read them with the
     crate's own parser — the parser is dogfooded by its own gate.
  4. Tokenizer parity confirmed cross-implementation: the `tokenizers` crate
     (0.23, truncation detached) reproduces Python's token IDs, slot IDs,
     boundary checks, and prefix extraction on all 307 rows — the von
     truncation trap was real here too (`tokenizer.json` ships baked
     truncation that Python never applies).
- **Accepted divergences (documented)**: usage-error wrapper text (clap vs
  argparse — messages bodies match, rc and behavior pinned); uncaught
  tracebacks render as a single `ValueError: …` line (rc 1, message pinned);
  lone UTF-16 surrogates in input JSON map to U+FFFD where Python would carry
  them to a later encode failure (same rc 1 envelope); `str.splitlines()`'s
  exotic separators beyond \r\n are not split boundaries.

Stage 2 next: the llama.cpp CPU engine behind the `Engine` trait, gated on
the `logits-cpufp32`/`logits-gguf` probe captures (≤1e-5 vs the Python GGUF
backend, 100% decision agreement on the probe corpus).

### Stage 2 — complete (2026-09-24)

The llama.cpp CPU engine is live in `semif-rs/crates/semif-engine-llamacpp`,
wired through `--backend llamacpp` for all three reuse modes, and **every
Stage 2 gate passed: 236 rows diffed Python-vs-Rust with zero findings and
bit-exact slot logits.**

- **Linkage (the load-dynamic decision)**: the engine `dlopen`s the *same*
  `libllama.so` bundled in the oracle venv's llama-cpp-python 0.3.35 wheel
  (`SEMIF_LLAMA_LIB` override; venv-relative search), with ~23 hand-written
  `extern "C"` symbols and ctypes-mirrored struct layouts. Same native
  library as the oracle ⇒ the parity gate measures the port, not llama.cpp
  version drift [von's load-dynamic precedent]. llama.cpp logging is
  silenced through a real no-op `llama_log_set` callback (passing NULL resets
  to default logging).
- **Gates, all via `scripts/diff_rows.py --profile gguf`**:
  | Comparison | Rows | Identity | Numerics | Argmax |
  |---|---:|---|---|---:|
  | Rust CLI vs Stage 0 GGUF probe capture | 25 | ✓ | 0 findings, **max slot-logit Δ = 0.0 (bit-exact)** | 100% |
  | direct, authored144, Python-vs-Rust | 144 | ✓ | 0 findings | 100% |
  | serial, shape777 state-0 (20 cache hits exercised) | 21 | ✓ | 0 findings | 100% |
  | shared, shape777 state-0 | 21 | ✓ | 0 findings | 100% |
  | direct, shape777 state-0 | 21 | ✓ | 0 findings | 100% |
- **Serial parity detail**: `cache_hit` true on exactly rows 2–21,
  `prefix_tokens: 1812`, whole-sequence state save/restore (112,103,612
  bytes/branch) matching the Python backend's hybrid-memory constraint.
- **Numeric subtlety pinned**: Python's `_logsumexp` runs over an f64 array
  for slot selections but an f32 array for the vocabulary; the Rust engine
  mirrors both paths. The one residual difference is numpy's pairwise-sum/
  vectorized-exp inside the f32 vocabulary sum — measured at ≤1.1e-4 relative
  on `allowed_token_mass` over 207 rows (1 row), so the gguf scalar gate is
  set to 2e-4 with that evidence recorded in `diff_rows.py`. Logits,
  probabilities, and argmax are bit-exact.
- **Port bugs caught by the gates this stage**: (1) Python's
  `if needed < 0: needed = -needed` in `_gguf_tokenize` — a clamp-to-zero
  mistranslation produced "The GGUF tokenizer rejected the prompt text";
  (2) `seq_id` must be set for *every* batch token, not just token 0
  (llama_decode fails on garbage sequence ids); (3) an 8 MiB stack buffer in
  GGUF hashing overflowed the main thread — heap it.
- **Hybrid architecture observed in the loaded GGUF**: 8 full-attention KV
  layers + 24 recurrent (gated delta net) layers, matching the
  `full_attention_interval: 4` config fact recorded in Stage 0.
- **Env contract**: `SEMIF_LLAMA_LIB` points at the wheel's `libllama`
  (auto-discovered next to the oracle venv); `--llama-threads` defaults to
  available parallelism like `os.cpu_count()`.

Stage 3 next: the CUDA BF16 engine via `tch` (Gate 0 feasibility spike
first), against the Stage 0 `logits-cuda-bf16-rtx5070ti-laptop.jsonl` and
`logits-cpufp32.jsonl` captures.
