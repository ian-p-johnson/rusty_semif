#!/usr/bin/env bash
# Stage 4 speed ledger (PORTING_RUST.md §7 Stage 4).
#
# Times the owned 37x21 fixture for fresh / serial / shared on both engines,
# same machine, same exclusivity rules as the sibling ports (§8.1) — nothing
# else may be on the card. Each re-run is also diffed against the Stage 4
# evidence file, so the ledger doubles as a determinism check: identical
# inputs must still produce identity-exact rows.
#
# Report: semif-rs/fingerprints/speed-ledger.json (create-only).
# Timings are measurements on this machine only and are never compared against
# the committed RTX 3090 numbers.
set -uo pipefail

cd "$(dirname "$0")/.."
REPO="$PWD"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export SEMIF_TCH_TRACE=1
export SEMIF_TCH_WIDTH=2048
export SEMIF_TCH_ARTIFACTS="$REPO/semif-rs/artifacts/w2048"
TORCHLIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="$TORCHLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CAUSAL="/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B"
CAUSAL_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
INPUT="$REPO/benchmarks/data/shape777.jsonl"
RUNS="$REPO/semif-rs/runs"
RESULTS="$REPO/semif-rs/results"
TMP="$REPO/semif-rs/runs/timing"
FP="$REPO/semif-rs/fingerprints"
LEDGER="$FP/speed-ledger.json"
mkdir -p "$TMP" "$FP"
if [[ -f "$LEDGER" ]]; then
  echo "ledger already exists: $LEDGER" >&2
  echo "remove it first (create-only evidence)" >&2
  exit 2
fi

# This machine is shared. Stage 0 voided every timing field it took under load,
# and so will this: refuse rather than record a number that cannot be trusted.
# Threshold is one quarter of the cores (5 on a 20-core box); override with
# SEMIF_MAX_LOAD_1MIN if you know the box is otherwise idle.
CORES="$(nproc)"
MAX_LOAD="${SEMIF_MAX_LOAD_1MIN:-$(( CORES / 4 ))}"
READ1="$(cut -d' ' -f1 /proc/loadavg)"
LOAD_OK="$(awk "BEGIN {print ($READ1 <= $MAX_LOAD) ? 1 : 0}")"
if [[ "$LOAD_OK" != "1" ]]; then
  echo "refusing: 1-minute load average $READ1 exceeds $MAX_LOAD" >&2
  echo "         (nproc=$CORES; run when the box is quiet, or set SEMIF_MAX_LOAD_1MIN)" >&2
  exit 3
fi
echo "load average at start: $READ1 (max $MAX_LOAD, nproc=$CORES)"

ROWS=777
entries=()

# timed <side> <mode> <reference-file>
timed() {
  local side="$1" mode="$2" reference="$3"
  local out="$TMP/$side-$mode.jsonl"
  rm -f "$out"
  local start end seconds
  start="$(date +%s.%N)"
  if [[ "$side" == "python" ]]; then
    if [[ "$mode" == "shared" ]]; then
      python benchmarks/score_shared_chunked.py \
        --model "$CAUSAL" --revision "$CAUSAL_REV" --input "$INPUT" \
        --output "$out" > /dev/null 2>&1
    else
      semif-score --mode "$mode" --model "$CAUSAL" --revision "$CAUSAL_REV" \
        --input "$INPUT" --output "$out" > /dev/null 2>&1
    fi
  else
    # Release profile: comparing an unoptimised Rust glue path against
    # optimised Python measures the compiler, not the port.
    semif-rs/target/release/semif-cli --mode "$mode" --backend torch \
      --device cuda --dtype bfloat16 --model "$CAUSAL" --revision "$CAUSAL_REV" \
      --input "$INPUT" --output "$out" > /dev/null 2>&1
  fi
  local rc=$?
  end="$(date +%s.%N)"
  seconds="$(awk "BEGIN {printf \"%.3f\", $end - $start}")"
  if [[ $rc -ne 0 || ! -f "$out" || "$(wc -l < "$out")" -ne "$ROWS" ]]; then
    echo "  FAIL $side/$mode rc=$rc"
    rm -f "$out"
    entries+=("$(python -c "
import json; print(json.dumps({'side':'$side','mode':'$mode','status':'failed','returncode':$rc}))")")
    return $rc
  fi
  local determinism="not-compared"
  if [[ -f "$reference" ]]; then
    if python "$REPO/scripts/diff_rows.py" "$reference" "$out" --profile cuda-bf16 \
        --exclude model.serving_config > /dev/null 2>&1; then
      determinism="identity-exact"
    else
      # timings differ by construction; the row diff already excludes them, so a
      # failure here is a real content change.
      determinism="differs"
    fi
  fi
  local rate
  rate="$(awk "BEGIN {printf \"%.3f\", $ROWS / $seconds}")"
  echo "  $side/$mode: ${seconds}s ($rate decisions/s, determinism=$determinism)"
  entries+=("$(python -c "
import json; print(json.dumps({'side':'$side','mode':'$mode','status':'ok',
  'wall_seconds':$seconds,'decisions_per_second':float('$rate'),
  'rows':$ROWS,'determinism':'$determinism'}))")")
}

echo "== speed ledger (one engine at a time, GPU to itself) =="
for mode in direct serial shared; do
  timed python "$mode" "$RUNS/py-torch-cuda-$mode-shape777.jsonl"
done
for mode in direct serial shared; do
  reference=""
  case "$mode" in
    direct) reference="$RESULTS/shape777-direct.jsonl" ;;
    serial) reference="$RESULTS/shape777-serial.jsonl" ;;
    shared) reference="$RESULTS/shape777-shared.jsonl" ;;
  esac
  timed rust "$mode" "$reference"
done

python - "$LEDGER" "$READ1" "${entries[@]}" <<'PY'
import json, platform, sys
out, load, *entries = sys.argv[1:]
record = {
    "schema": "semif-rs-speed-ledger-v1",
    "fixture": "benchmarks/data/shape777.jsonl",
    "rows": 777,
    "scope": (
        "Wall time on this machine for one warm-to-cold pass including prompt "
        "construction, tokenization, transfers, forward passes and readout. "
        "Compared only against itself: the committed RTX 3090 numbers are a "
        "different device and are never a target."
    ),
    "note": (
        "Rust serial/shared run as documented fresh-recompute under the trace "
        "route (no prefix-cache reuse yet), so their numbers are not a like-for-"
        "like comparison against Python's cache-reusing paths."
    ),
    "host": platform.platform(),
    "load_average_at_start": float(load),
    "rust_profile": "release",
    "entries": [json.loads(entry) for entry in entries],
}
with open(out, "x") as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(f"ledger: {out}")
PY
