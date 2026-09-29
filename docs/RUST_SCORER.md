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

`semif-rs/fingerprints/speed-ledger.json` — 777 rows, release build, machine
quiet (1-minute load 1.61 against a `nproc/4 = 5` ceiling, recorded in the
file), GPU to itself, each re-run diffed against its Stage 4 evidence file:

| side | mode | wall (s) | decisions/s | determinism |
|---|---|---:|---:|---|
| python | direct (fresh) | 601.7 | 1.291 | identity-exact |
| python | serial | 101.3 | 7.667 | identity-exact |
| python | shared | 122.3 | 6.354 | identity-exact |
| rust | direct | 704.2 | 1.103 | identity-exact |
| rust | serial | 704.0 | 1.104 | identity-exact |
| rust | shared | 703.9 | 1.104 | identity-exact |

Three readings:

1. **On GPU the port is essentially speed-neutral on direct: 1.17x.** The
   forward alone accounts for 1.13x of that (850.5 vs 750.7 ms/row measured
   separately), so the Rust glue adds roughly 3% on top of the fixed-width
   forward. This is laya's tempered expectation confirmed: at 4B the kernels
   dominate, and there is no language win to be had.
2. **Rust serial/shared deliver ~14–17% of Python's rate**, because the trace
   has no prefix cache and they run as fresh recompute. That gap is the cache —
   Python's serial path is 6.9x faster than its own fresh path on the same
   hardware — not the language. It is the same limitation stamped in
   `serving_config = tch-trace-direct-v1`.
3. **Six for six re-runs came back identity-exact**, so both engines are
   deterministic on this machine.

### A voided attempt, kept for the record

`semif-rs/fingerprints/speed-ledger-void-debug-contended.json` is the first
attempt: 596–1057s with Rust at 0.74 d/s, implying a 1.77x gap. It is void on
two counts — it timed `target/debug/semif-cli` against optimised Python, and
the machine was shared during the run. Stage 0 set this exact precedent
(timings taken under load are void; the baseline waits for an idle window).
The re-take shows the debug binary alone cost the Rust path ~1.5x: 1056.9s
became 704.2s with no source change. `scripts/speed_ledger.sh` now refuses to
start above `nproc/4` (`SEMIF_MAX_LOAD_1MIN` overrides), times the release
profile, and records the load and profile it ran under.

### One claim measured and rejected

A Stage 4 follow-up proposed re-exporting the direct trace with
`logits_to_keep=1`, on the theory that the wrapper's length-gather forces a
full `W x 248320` logits matrix every row and that this explained ~1.8x.
`benchmarks/measure_readout_layout.py` tested it on a quiet machine,
interleaved round-robin so a load spike drifts all configurations together:

| config | what it is | ms/row | vs A |
|---|---|---:|---:|
| A | eager, unpadded, `logits_to_keep=1` — the Python oracle | 750.7 | 1.000 |
| B | eager, right-padded, full logits + gather — the current trace | 864.6 | 1.152 |
| C | eager, left-padded, `logits_to_keep=1` — the proposal | 832.4 | 1.109 |
| D | the existing traced artifact | 850.5 | 1.133 (0.984 vs B) |

- **`logits_to_keep` saves 3.9%** (B vs C = 1.039), not 1.8x. This checkpoint's
  forward is dominated by the reference `chunk_gated_delta_rule` and
  `causal_conv1d` fallbacks — both warn at import — so the lm_head share is
  small.
- **Trace fusion costs nothing**: D runs at 0.984x the same shape eagerly.
- The proposed layout also measures `max|Δlogit| = 3.125e-01` against the
  oracle, the same padding-invariance class as the reranker export gate.

**Recommendation: do not change the wrapper.** Re-exporting invalidates every
recorded direct-mode parity report and the Stage 4 direct evidence to buy 3.9%
that sits inside the noise of a shared box. The policy is to measure and
report, not optimise; the instrument is committed so the question can be
re-answered under better conditions than these.

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
