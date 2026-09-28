#!/usr/bin/env bash
# Stage 4 scoring driver (PORTING_RUST.md §7 Stage 4).
#
# Produces the Rust scorer's evidence files under semif-rs/results/ and the
# same-device Python baselines under semif-rs/runs/. Every output is
# create-only; this script is restartable and idempotent, re-running any file
# that is missing or was truncated by an interrupted session.
#
#   scripts/run_stage4_scores.sh            # python baselines then rust outputs
#   scripts/run_stage4_scores.sh rust       # one side only
#
# GPU exclusivity (§8.1): run this with nothing else on the card.
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

RERANKER="/media/ianj/New Volume/hf-plain/models/Qwen3-Reranker-4B"
RERANKER_REV=22e683669bc0f0bd69640a1354a6d0aebcfeede5
CAUSAL="/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B"
CAUSAL_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

RUST="$REPO/semif-rs/target/debug/semif-cli"
RUNS="$REPO/semif-rs/runs"
RESULTS="$REPO/semif-rs/results"
DATA="$REPO/benchmarks/data"
mkdir -p "$RUNS" "$RESULTS"

# drop <file> unless it already has exactly <rows> lines
want() {
  local file="$1" rows="$2"
  if [[ -f "$file" ]]; then
    local have
    have="$(wc -l < "$file")"
    if [[ "$have" -eq "$rows" ]]; then
      return 1          # complete: caller skips
    fi
    echo "  partial ($have/$rows): $(basename "$file") — removing"
    rm -f "$file"
  fi
  return 0
}

python_side() {
  local mode="$1" model="$2" rev="$3" input="$4" out="$5" rows="$6"
  want "$out" "$rows" || { echo "  keep $(basename "$out") ($rows rows)"; return; }
  if [[ "$mode" == "shared" ]]; then
    python benchmarks/score_shared_chunked.py \
      --model "$model" --revision "$rev" --input "$input" --output "$out" \
      > >(grep -E '^[0-9]+/' >&2) 2>&1
  else
    semif-score --mode "$mode" --model "$model" --revision "$rev" \
      --input "$input" --output "$out" > /dev/null 2>&1
  fi
  local rc=$?
  if [[ $rc -eq 0 && "$(wc -l < "$out")" -eq "$rows" ]]; then
    echo "  ok   $(basename "$out") ($rows rows)"
  else
    echo "  FAIL $(basename "$out") rc=$rc"
    rm -f "$out"
    return $rc
  fi
}

rust_side() {
  local mode="$1" model="$2" rev="$3" input="$4" out="$5" rows="$6"
  want "$out" "$rows" || { echo "  keep $(basename "$out") ($rows rows)"; return; }
  "$RUST" --mode "$mode" --backend torch --device cuda --dtype bfloat16 \
    --model "$model" --revision "$rev" --input "$input" --output "$out" \
    > /dev/null 2>&1
  local rc=$?
  if [[ $rc -eq 0 && -f "$out" && "$(wc -l < "$out")" -eq "$rows" ]]; then
    echo "  ok   $(basename "$out") ($rows rows)"
  else
    echo "  FAIL $(basename "$out") rc=$rc"
    rm -f "$out"
    return $rc
  fi
}

run_python() {
  echo "== python baselines =="
  python_side direct   "$CAUSAL"   "$CAUSAL_REV" "$DATA/authored144.jsonl"     "$RUNS/py-torch-cuda-direct-authored144.jsonl"     144
  python_side reranker "$RERANKER" "$RERANKER_REV" "$DATA/authored144.jsonl"   "$RUNS/py-torch-cuda-reranker-authored144.jsonl"   144
  python_side direct   "$CAUSAL"   "$CAUSAL_REV" "$DATA/perturbations108.jsonl" "$RUNS/py-torch-cuda-direct-perturbations108.jsonl" 108
  python_side reranker "$RERANKER" "$RERANKER_REV" "$DATA/perturbations108.jsonl" "$RUNS/py-torch-cuda-reranker-perturbations108.jsonl" 108
  python_side direct   "$CAUSAL"   "$CAUSAL_REV" "$DATA/shape777.jsonl"        "$RUNS/py-torch-cuda-direct-shape777.jsonl"        777
  python_side serial   "$CAUSAL"   "$CAUSAL_REV" "$DATA/shape777.jsonl"        "$RUNS/py-torch-cuda-serial-shape777.jsonl"        777
  python_side shared   "$CAUSAL"   "$CAUSAL_REV" "$DATA/shape777.jsonl"        "$RUNS/py-torch-cuda-shared-shape777.jsonl"        777
}

run_rust() {
  echo "== rust outputs =="
  rust_side direct   "$CAUSAL"   "$CAUSAL_REV" "$DATA/authored144.jsonl"     "$RESULTS/direct-authored144.jsonl"     144
  rust_side reranker "$RERANKER" "$RERANKER_REV" "$DATA/authored144.jsonl"   "$RESULTS/reranker-authored144.jsonl"   144
  rust_side direct   "$CAUSAL"   "$CAUSAL_REV" "$DATA/perturbations108.jsonl" "$RESULTS/direct-perturbations108.jsonl" 108
  rust_side reranker "$RERANKER" "$RERANKER_REV" "$DATA/perturbations108.jsonl" "$RESULTS/reranker-perturbations108.jsonl" 108
  rust_side direct   "$CAUSAL"   "$CAUSAL_REV" "$DATA/shape777.jsonl"        "$RESULTS/shape777-direct.jsonl"        777
  rust_side serial   "$CAUSAL"   "$CAUSAL_REV" "$DATA/shape777.jsonl"        "$RESULTS/shape777-serial.jsonl"        777
  rust_side shared   "$CAUSAL"   "$CAUSAL_REV" "$DATA/shape777.jsonl"        "$RESULTS/shape777-shared.jsonl"        777
}

case "${1:-all}" in
  python) run_python ;;
  rust)   run_rust ;;
  all)    run_python; run_rust ;;
  *) echo "usage: $0 [python|rust|all]" >&2; exit 2 ;;
esac
echo "stage4 scoring done"
