# Rust scorer

An experimental Rust implementation of `semif-score` for CPU/GGUF and CUDA/BF16
scoring. It is a **port**, not a replacement: the Python scorer remains the
oracle and the shipped evaluators remain the independent observer. Nothing under
[`results/`](../results/) is produced or modified by it.

- Plan, gates and stage history: [`PORTING_RUST.md`](../PORTING_RUST.md)
- Workspace: [`semif-rs/`](../semif-rs/)
- One-command check: [`scripts/verify_port.sh`](../scripts/verify_port.sh)

## What it is

| Crate | Role |
|---|---|
| `semif-types` | row schema, validation with Python-pinned message strings |
| `semif-core` | Python-compatible JSON parsing/writing, `repr(float)`, f64 softmax, chat-template renderer, tokenizer harness, state-prefix extraction, reranker pair prompts |
| `semif-engine` | the `Engine` seam (`score_direct` / `score_serial` / `score_shared` / `score_reranker`) |
| `semif-engine-llamacpp` | CPU/GGUF engine — `dlopen`s the *same* `libllama` the oracle's wheel links |
| `semif-engine-tch` | CUDA/BF16 engine — executes TorchScript traces of the pinned checkpoints through `tch` |
| `semif-cli` | `semif-cli`, a drop-in `semif-score` with create-only, flush-per-row semantics |

Two engine families, deliberately not the same claim:

- **`--backend llamacpp`** reproduces `llamacpp_backend.py` against one native
  library, so its logits are **bit-exact** against the Python GGUF backend.
- **`--backend torch` with `SEMIF_TCH_TRACE=1`** runs a *fixed-width traced
  graph*, not an eager model. Trace fusion changes kernels, so the numeric
  reference for this route is the exporter's own trace capture — see below.

## Running it

```bash
# CPU / GGUF
SEMIF_LLAMA_LIB=.venv/lib/python3.12/site-packages/llama_cpp/libllama.so \
semif-rs/target/debug/semif-cli --mode serial --backend llamacpp \
  --gguf /path/to/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --model /path/to/Qwen_Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/authored144.jsonl --output out.jsonl

# CUDA / BF16 (traced). Nothing else may be using the GPU.
SEMIF_TCH_TRACE=1 SEMIF_TCH_WIDTH=2048 \
SEMIF_TCH_ARTIFACTS=$PWD/semif-rs/artifacts/w2048 \
CUDA_VISIBLE_DEVICES=0 \
semif-rs/target/debug/semif-cli --mode direct --backend torch \
  --device cuda --dtype bfloat16 \
  --model /path/to/Qwen_Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/authored144.jsonl --output out.jsonl
```

`semif-rs/artifacts/` is **regenerable and not committed** (~32 GB):

```bash
python benchmarks/export_trace_module.py --width 2048 --artifacts semif-rs/artifacts/w2048
python benchmarks/export_reranker_trace.py --width 2048 --artifacts semif-rs/artifacts/w2048
```

Both exporters are create-only and refuse to overwrite an artifact or its
trace-check record.

## Verification

```bash
scripts/verify_port.sh            # full: integrity, tests, scoring, diffs, fingerprints
scripts/verify_port.sh --quick    # gates only, on whatever is already scored
```

Three layers, in the order the script runs them:

1. **The repo's own gates, before and after.** `results/raw/SHA256SUMS` and
   `benchmarks/verify_published.py` must be identical on both sides of the port
   work — the port is read-only with respect to committed evidence.
2. **Fixture gates.** `cargo test --workspace` replays Python-authored fixtures:
   byte-equal prompt strings, identical token IDs and slot contracts, exact
   refusal messages, and the reranker's pair prompts / yes-no contract.
3. **Row-level parity and grading.** `scripts/diff_rows.py` compares Rust output
   against the same-device Python run, then `scripts/fingerprint_port.py` grades
   *both* files with the unmodified `benchmarks/evaluate.py`.

### Field classes

| Class | Fields |
|---|---|
| identity-exact | `id`, `option_ids`, `prompt_sha256`, `input_tokens`, `prompt_version`, `readout`, `probability_status`, `option_prompt_sha256`, `max_option_input_tokens`, `model.{source,revision,dtype,serving_config,backend,gguf}` |
| numeric-gated | `option_logits`, `probabilities`, `independent_binary_relevance`, `allowed_token_mass` |
| timing-excluded | `*_seconds`, `shared_timing`, `pair_batches` — reported, never value-compared |

Two identity fields are **documented route divergences** and excluded by name
(`--exclude`): the trace stamps `model.serving_config = tch-trace-*-v1`, and the
direct/serial/shared `readout` says `traced …` where Python says `native …`. The
reranker deliberately copies Python's `readout` verbatim, so it excludes only
`serving_config`.

## The two numeric references

The repo's own methodology says model outputs are measurements, not byte
goldens (README, `docs/REPRODUCE.md`), and it documents 5–6 of 777 argmaxes
changing between *its own* torch paths. The port inherits that framing:

- **GGUF**: same native library ⇒ the gate is bit-exactness. Measured over 236
  rows: identity-exact, bit-exact slot logits, 100% decision agreement.
- **Traced CUDA/BF16**: the graph is fixed-width, so kernels legitimately differ
  from the eager oracle. The gate is **bit-exactness against the exporter's own
  trace capture**, plus decision agreement against eager.

The reranker's port fidelity is therefore measured directly: Rust reproduces the
traced graph's full-vocabulary logits **bit-exactly (sha256 over all 151,669
floats, 15/15 option pairs)**, and over 144 authored rows the Python trace
probe versus the Rust run reports **identity-exact, zero numeric findings, and
100% decision agreement**.

## Decision agreement and the flip ledger

Any miss against the ≥99% decision gate is escalated rather than absorbed.
Measured on this machine (Rust traced vs Python eager, same device):

| case | agreement | flips | Python top-2 margin of the flipped rows |
|---|---:|---:|---|
| authored144 / direct | 99.31% | 1 | 0.000 (exact tie) |
| perturbations108 / direct | 97.22% | 3 | 0.051 – 0.062 |
| shape777 / direct | 99.36% | 5 | near-ties |
| shape777 / serial | 99.23% | 6 | near-ties |
| shape777 / shared | 99.36% | 5 | near-ties |
| authored144 / reranker | 95.83% | 6 | 0.000 – 0.128 |
| perturbations108 / reranker | 95.37% | 5 | 0.000 – 0.142 |

Two properties make this legible:

1. **Every flip is a near-tie.** The largest flipped row sits at a top-2 gap of
   0.142, against corpus medians of 0.94 (authored direct) and 0.26 (authored
   reranker). No comfortable row flips.
2. **The port is not the cause.** For the reranker this is measured, not
   argued: the Python trace probe and the Rust run agree on 144/144 rows with
   zero numeric findings, and Rust's vocabulary logits are bit-identical to the
   exporter's capture. The divergence is *eager vs fixed-width*, and it appears
   without any Rust in the loop: running the **eager weights at two different
   widths** already produces a max logit delta of 0.25 and 5 of the same 6
   flips. Narrowing the trace width does not help either (97.2% at width 256,
   96.5% at 512 and 2048).

Why the reranker fares worse than direct: its readout is a two-way yes/no
log-odds per option, which sits far closer to a tie than a three-way letter
choice. `perturbations108` is additionally a *stability* fixture built from
borderline decisions, so it concentrates exactly the rows this affects.

Reaching ≥99% for the reranker needs **dynamic shapes** — that is, the
hand-written eager layer stack, which the Stage 3 re-scope traded away in favour
of a traced graph. Under the approved route the honest number is the one above.

## Accuracy fingerprints

Both sides are graded by the **unmodified** `benchmarks/evaluate.py`, so the
port is scored by the independent observer rather than by its own tooling:

| case | Python acc | Rust acc | Δacc | Python bal-acc | Rust bal-acc | Δbal-acc |
|---|---:|---:|---:|---:|---:|---:|
| authored144 / direct | 0.8125 | 0.8056 | -0.0069 | 0.8225 | 0.8132 | -0.0093 |
| authored144 / reranker | 0.6389 | 0.6181 | -0.0208 | 0.6316 | 0.6157 | -0.0159 |
| perturbations108 / direct | 0.7685 | 0.7593 | -0.0093 | 0.7737 | 0.7640 | -0.0097 |
| perturbations108 / reranker | 0.4815 | 0.4722 | -0.0093 | 0.5739 | 0.5695 | -0.0044 |

Stability under the output-blind variants (`evaluate_perturbations.py`, run
once per side) moves by at most **0.009** in mean maximum probability
movement — the port perturbs the stability fingerprint no more than the
documented cross-path envelope.

Every delta above is the arithmetic of the flip ledger: on corpora of 108–144
rows a single flipped decision is worth ~0.7pp of accuracy, so the plan's
±0.5pp balanced-accuracy envelope is finer than one decision.

## Speed

`semif-rs/fingerprints/speed-ledger.json`, 777 rows, one machine, nothing else
on the card, each re-run diffed against the Stage 4 evidence file:

| side | mode | wall (s) | decisions/s | determinism |
|---|---|---:|---:|---|
| python | direct (fresh) | 596.5 | 1.303 | identity-exact |
| python | serial | 100.6 | 7.720 | identity-exact |
| python | shared | 126.0 | 6.168 | identity-exact |
| rust | direct | 1056.9 | 0.735 | identity-exact |
| rust | serial | 789.1 | 0.985 | identity-exact |
| rust | shared | 795.2 | 0.977 | identity-exact |

Read it with the two caveats the plan already states:

1. **Rust serial/shared are fresh recompute** — there is no prefix cache in the
   trace — so they are *not* a like-for-like comparison against Python's
   cache-reusing paths. The 6–8× gap is the cache, not the language.
2. **Rust direct is ~1.8× slower than Python direct**, and the cause is
   specific: the direct wrapper gathers at `length - 1`, which means it cannot
   pass `logits_to_keep=1`, so every row materialises the full
   `2048 × 248320` logits matrix before the gather. Python keeps a single
   position. Right-aligning the direct input the way the reranker wrapper does
   (which does use `logits_to_keep=1`) would remove this; it would also
   invalidate the recorded Stage 3 artifact and its gates, so it is recorded
   here as a follow-up rather than done mid-verification.

Six for six re-runs came back identity-exact, so both engines are deterministic
on this machine as well.

## Known limitations

- **Serial and shared run as fresh recompute.** The trace has no prefix cache,
  so `--mode serial` and `--mode shared` produce correct logits without the
  cache-reuse speedup, under `serving_config = tch-trace-direct-v1`. The cache
  is the project's speed headline; it is not claimed here.
- **Fixed trace width refuses long rows.** Rows past the width are refused with
  a message, never truncated. There is no run-level `--max-tokens` guard.
- **Reranker is CUDA-only**, matching Python. With the trace flag unset the CLI
  refuses (`exit 2`) instead of silently falling back to the stub.
- **MLX, MPS, `exl3-bridge/` and `webgpu-demo/` are untouched** — out of scope
  by design; Python remains the Apple path.
- **External corpora (WANLI, TypeSafe, Every) are not exercised**: the fetch
  pipeline is out of scope and no source snapshots are present locally.

## Provenance

Rust evidence lives under `semif-rs/` and never under `results/`:

| Path | Contents |
|---|---|
| `semif-rs/fixtures/` | Python-authored parity fixtures (create-only) |
| `semif-rs/probes/` | Python capture files used as numeric references |
| `semif-rs/runs/` | same-device Python baselines and diff reports |
| `semif-rs/results/` | **Rust-produced prediction files** (the graded artifacts) |
| `semif-rs/fingerprints/` | `evaluate.py` / `evaluate_perturbations.py` reports and the speed ledger |
| `semif-rs/artifacts/` | regenerable TorchScript traces (gitignored) |

Timings are measurements on this machine only. They are never compared against
the committed RTX 3090 numbers, and the GPU must be otherwise idle for any timed
or parity run.
