"""Probe the traced reranker graph in Python (PORTING_RUST.md §6, Stage 3).

The row-diff gate compares a Rust reranker run against the *eager* Python
oracle. Two things move in that comparison: the port, and the trace artifact
itself (fixed-width padding plus trace fusion, which the exporter already
measured at up to 0.51 on full-vocabulary logits). This probe holds the port
constant by running the **same traced module from Python**, so:

  eager vs traced  = the route's divergence from the oracle
  traced vs Rust   = the port's divergence from the route

If the second is zero, any decision miss in the first belongs to the route and
the port is innocent of it.

Rows carry the full logical envelope (no timings, no model block) so
`scripts/diff_rows.py --intersection` can compare them against either side.
Create-only output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from semif_phase1 import reranker
from semif_phase1.core import softmax

READOUT = "native yes/no log-odds per option, normalized only for relative comparison"
PROBABILITY_STATUS = (
    "relative option compatibility; uncalibrated as categorical probability"
)


def layout(ids: list[int], width: int, row_width: int, pad: int):
    """The Rust engine's input layout: real block right-aligned in the fixed
    width, masked, positioned at `index - (width - row_width)` so a row of
    width `w` lands on the same RoPE offsets Python's width-`w` batch uses."""
    left = width - len(ids)
    offset = width - row_width
    return (
        [pad] * left + list(ids),
        [0] * left + [1] * len(ids),
        [1 if index < offset else index - offset for index in range(width)],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/media/ianj/New Volume/hf-plain/models/Qwen3-Reranker-4B")
    parser.add_argument("--revision", default="22e683669bc0f0bd69640a1354a6d0aebcfeede5")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("semif-rs/artifacts/w2048"))
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")

    artifact = args.artifacts / f"qwen3reranker-cudabf16-w{args.width}.pt"
    check = json.loads(
        (args.artifacts / f"trace-check-reranker-cudabf16.json").read_text()
    )
    if artifact.name != check["artifact"]:
        parser.error("trace-check does not describe this artifact")
    pad, no_id, yes_id = check["pad_id"], check["no_id"], check["yes_id"]

    # Only the tokenizer and the traced graph are needed: loading the eager
    # weights alongside a 7.7 GiB artifact does not fit this 12 GiB card. The
    # eager reference for the route-divergence column is the already-recorded
    # `py-torch-cuda-reranker-*.jsonl` run.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    traced = torch.jit.load(str(artifact), map_location="cuda")
    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for row in rows:
            reranker.validate_row(row)
            encoded = [
                reranker._encode(tokenizer, row, option, args.max_tokens)
                for option in row["options"]
            ]
            row_width = max(len(ids) for ids, _ in encoded)
            if row_width > args.width:
                raise SystemExit(f"{row['id']}: {row_width} tokens exceed trace width")

            layouts = [
                layout(ids, args.width, row_width, pad) for ids, _ in encoded
            ]
            tensors = [
                torch.tensor([part[index] for part in layouts], dtype=torch.long, device="cuda")
                for index in range(3)
            ]
            with torch.inference_mode():
                logits = traced(*tensors).float()
            no = logits[:, no_id]
            yes = logits[:, yes_id]
            log_odds = (yes - no).tolist()
            relevance = (
                torch.stack([no, yes], dim=-1).softmax(-1)[:, 1].tolist()
            )
            stream.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "option_ids": [option["id"] for option in row["options"]],
                        "probabilities": softmax(log_odds),
                        "option_logits": log_odds,
                        "independent_binary_relevance": relevance,
                        "input_tokens": sum(len(ids) for ids, _ in encoded),
                        "max_option_input_tokens": row_width,
                        "option_prompt_sha256": [text for _, text in encoded],
                        "prompt_version": reranker.PROMPT_VERSION,
                        "readout": READOUT,
                        "probability_status": PROBABILITY_STATUS,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            print(f"{row['id']}: width={row_width} traced", flush=True)


if __name__ == "__main__":
    main()
