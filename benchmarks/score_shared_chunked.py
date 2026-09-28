"""Score a corpus in `shared` mode with fixed-size batches.

`semif-score --mode shared` forwards every input row at once, which OOMs the
12 GB card on the committed 37x21 fixture (the Python side has never run
batch-21 cache replication on this machine). Chunking *within* each group is
the established workaround — Stage 3's Python shared runs are batch-7 — and it
is behaviour-preserving: every chunk shares one state prefix, exactly like the
un-chunked call, and `shared_timing.batch_size` records what actually happened.

Rows are emitted in input order, flushed per chunk (crash-resumable at chunk
granularity). Create-only output.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

from semif_phase1.core import load_causal_model
from semif_phase1.shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")
    if args.batch_size < 1:
        parser.error("Batch size must be positive")

    rows = [
        json.loads(line)
        for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)

    model, tokenizer, metadata = load_causal_model(
        args.model, args.revision, "cuda"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored = 0
    with args.output.open("x") as destination:
        for group in groups.values():
            for offset in range(0, len(group), args.batch_size):
                chunk = group[offset : offset + args.batch_size]
                results, _timing = score_shared(
                    model, tokenizer, chunk, metadata, args.max_tokens
                )
                for result in results:
                    destination.write(json.dumps(result, allow_nan=False) + "\n")
                    scored += 1
                destination.flush()
                print(f"{scored}/{len(rows)} rows (batch {len(chunk)})", flush=True)
    if scored != len(rows):
        raise SystemExit(f"scored {scored} of {len(rows)} rows")


if __name__ == "__main__":
    main()
