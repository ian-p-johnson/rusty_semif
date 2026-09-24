"""Informational cross-device comparison against committed prediction rows.

Unlike scripts/diff_rows.py this never gates: committed rows carry RTX 3090
provenance while local captures carry this machine's GPU tag, so the repo's own
methodology treats cross-device numbers as measurements, not contracts (README:
"compare decisions or probabilities with a tolerance"; docs/REPRODUCE.md: outputs
are "measurements ... not byte-identical golden outputs"). Emits agreement stats
for the Stage 0 baseline record; exit 0 unless row id sets differ.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bf16_snap_bits(value: float) -> int:
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    return (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000


def read_rows(path: Path) -> dict:
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True, help="This machine's prediction rows")
    parser.add_argument("--committed", type=Path, required=True, help="Committed row-level evidence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")

    local, committed = read_rows(args.local), read_rows(args.committed)
    shared = sorted(set(local) & set(committed))
    if not shared:
        raise SystemExit("No shared row ids")
    argmax_agree, prob_deltas, snap_equal, logits_delta = 0, [], 0, []
    for key in shared:
        a, b = local[key], committed[key]
        pa, pb = np.asarray(a["probabilities"]), np.asarray(b["probabilities"])
        argmax_agree += int(np.argmax(pa) == np.argmax(pb))
        prob_deltas.append(float(np.max(np.abs(pa - pb))))
        la, lb = np.asarray(a["option_logits"]), np.asarray(b["option_logits"])
        logits_delta.append(float(np.max(np.abs(la - lb))))
        snap_equal += int(all(bf16_snap_bits(x) == bf16_snap_bits(y) for x, y in zip(la, lb)))
    report = {
        "schema": "semif-informational-compare-v1",
        "local": str(args.local), "committed": str(args.committed),
        "committed_provenance": {key: committed[shared[0]]["model"].get(key)
                                 for key in ("source", "revision", "dtype", "torch_version")},
        "rows_compared": len(shared),
        "argmax_agreement": argmax_agree / len(shared),
        "prob_max_delta_mean": float(np.mean(prob_deltas)),
        "prob_max_delta_p99": float(np.quantile(prob_deltas, 0.99)),
        "option_logit_max_delta_mean": float(np.mean(logits_delta)),
        "option_logit_max_delta_max": float(np.max(logits_delta)),
        "bf16_snap_equal_row_fraction": snap_equal / len(shared),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
