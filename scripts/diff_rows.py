"""Row-level differential verifier for scorer outputs (PORTING_RUST.md §6).

Compares two semif-score JSONL outputs joined by row id under three field classes:
  identity-exact    must be equal (ids, prompt_sha256, option_ids, serving_config, ...)
  numeric-gated     compared under a profile's tolerances (logits, probabilities, mass)
  timing-excluded   *_seconds / shared_timing blocks; presence-checked only

Profiles:
  cpufp32    torch-CPU fp32 oracle vs engine: slot-logit |delta| <= 1e-4
  gguf       Rust-vs-Python over the same native llama.cpp: |delta| <= 1e-5
  cuda-bf16  option logits compared as bf16-grid snap equality; probabilities 1e-3

Decision agreement (argmax of probabilities, full_vocab_argmax_id) is reported for
every profile; any flip is listed as a finding. Exit 0 only when identity and all
numeric gates pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

IDENTITY_FIELDS = ("id", "option_ids", "prompt_sha256", "prompt_version", "readout",
                   "probability_status", "input_tokens", "answer_token_ids", "prefix_tokens",
                   "prefix_sha256", "cache_hit", "batch_size")
IDENTITY_MODEL_FIELDS = ("source", "revision", "dtype", "serving_config", "backend", "gguf")
PRESENCE_MODEL_FIELDS = ("torch_version", "transformers_version", "llama_cpp_python_version")
NUMERIC_VECTOR_FIELDS = ("option_logits", "probabilities")
NUMERIC_SCALAR_FIELDS = ("allowed_token_mass",)
PROFILE_GATES = {
    "cpufp32": {"logit_abs": 1e-4, "prob_abs": 1e-6, "scalar_rel": 1e-4},
    # scalar_rel 2e-4: measured worst case over 207 rows vs the Python GGUF
    # backend — allowed_token_mass differs up to 1.1e-4 relative from numpy's
    # pairwise-sum/vectorized-exp f32 path (logits themselves are bit-exact).
    "gguf": {"logit_abs": 1e-5, "prob_abs": 1e-6, "scalar_rel": 2e-4},
    "cuda-bf16": {"logit_abs": None, "prob_abs": 1e-3, "scalar_rel": 2e-3, "bf16_snap": True},
}


def read_rows(path: Path) -> dict:
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"Duplicate id in {path}: {row['id']}")
        rows[row["id"]] = row
    return rows


def bf16_snap_bits(value: float) -> int:
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    return (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000


def compare_vector(field, a, b, gates, findings, bf16: bool) -> bool:
    if len(a) != len(b):
        findings.append({"field": field, "issue": "length_mismatch", "a": len(a), "b": len(b)})
        return False
    tolerance = gates["prob_abs"] if field == "probabilities" else gates["logit_abs"]
    okay = True
    for index, (x, y) in enumerate(zip(a, b)):
        if bf16:
            matches = bf16_snap_bits(x) == bf16_snap_bits(y)
        else:
            matches = abs(float(x) - float(y)) <= tolerance
        if not matches:
            okay = False
            findings.append({"field": field, "index": index, "issue": "gate_exceeded",
                             "a": x, "b": y, "abs_delta": abs(float(x) - float(y))})
    return okay


def compare(row_a: dict, row_b: dict, gates: dict, intersection: bool = False) -> dict:
    findings = []
    if intersection:
        identity_bad = [field for field in IDENTITY_FIELDS
                        if field in row_a and field in row_b and row_a[field] != row_b[field]]
    else:
        identity_bad = [field for field in IDENTITY_FIELDS
                        if (field in row_a or field in row_b) and row_a.get(field) != row_b.get(field)]
    model_a, model_b = row_a.get("model", {}), row_b.get("model", {})
    model_present = "model" in row_a and "model" in row_b
    if not intersection or model_present:
        identity_bad += [f"model.{field}" for field in IDENTITY_MODEL_FIELDS
                         if (field in model_a or field in model_b) and model_a.get(field) != model_b.get(field)]
    for field in identity_bad:
        findings.append({"field": field, "issue": "identity_mismatch",
                         "a": row_a.get(field.split(".")[-1] if "." in field else field),
                         "b": row_b.get(field.split(".")[-1] if "." in field else field)})

    numeric_ok = True
    for field in NUMERIC_VECTOR_FIELDS:
        if field in row_a and field in row_b:
            if not compare_vector(field, row_a[field], row_b[field], gates, findings,
                                  bf16=bool(gates.get("bf16_snap")) and field == "option_logits"):
                numeric_ok = False
        elif field in ("probabilities",) and (field in row_a) != (field in row_b):
            numeric_ok = False
            findings.append({"field": field, "issue": "missing_in_one_side"})
    for field in NUMERIC_SCALAR_FIELDS:
        if field in row_a and field in row_b:
            a, b = float(row_a[field]), float(row_b[field])
            if abs(a - b) > gates["scalar_rel"] * max(1.0, abs(a)):
                numeric_ok = False
                findings.append({"field": field, "issue": "gate_exceeded", "a": a, "b": b})
    if "logits_sha256" in row_a and "logits_sha256" in row_b:
        if row_a["logits_sha256"] != row_b["logits_sha256"]:
            findings.append({"field": "logits_sha256", "issue": "vocab_bits_differ"})

    argmax_a = int(np.argmax(row_a["probabilities"])) if "probabilities" in row_a else None
    argmax_b = int(np.argmax(row_b["probabilities"])) if "probabilities" in row_b else None
    decision = {
        "argmax_agrees": None if argmax_a is None or argmax_b is None else argmax_a == argmax_b,
        "full_vocab_argmax_agrees": (None if "full_vocab_argmax_id" not in row_a and "full_vocab_argmax_id" not in row_b
                                     else row_a.get("full_vocab_argmax_id") == row_b.get("full_vocab_argmax_id")),
    }
    if decision["argmax_agrees"] is False or decision["full_vocab_argmax_agrees"] is False:
        findings.append({"field": "decision", "issue": "argmax_flip", "decision": decision})

    timing_keys = sorted({key for row in (row_a, row_b) for key in row
                          if key.endswith("_seconds") or key == "shared_timing"})
    return {"identity_ok": not identity_bad, "numeric_ok": numeric_ok,
            "decision": decision, "findings": findings, "timing_keys_present": timing_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_a", type=Path)
    parser.add_argument("file_b", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILE_GATES), required=True)
    parser.add_argument("--report", type=Path, default=None, help="Also write the JSON report here")
    parser.add_argument("--intersection", action="store_true",
                        help="Compare identity fields only when present on both sides (for probe-vs-run diffs)")
    parser.add_argument("--max-findings", type=int, default=20)
    args = parser.parse_args()

    rows_a, rows_b = read_rows(args.file_a), read_rows(args.file_b)
    if set(rows_a) != set(rows_b):
        only_a, only_b = sorted(set(rows_a) - set(rows_b)), sorted(set(rows_b) - set(rows_a))
        raise SystemExit(f"Row id sets differ: only-in-A={only_a[:5]} only-in-B={only_b[:5]}")

    gates = PROFILE_GATES[args.profile]
    results = [compare(rows_a[key], rows_b[key], gates, args.intersection) for key in sorted(rows_a)]
    argmax = [r["decision"]["argmax_agrees"] for r in results if r["decision"]["argmax_agrees"] is not None]
    vocab = [r["decision"]["full_vocab_argmax_agrees"] for r in results if r["decision"]["full_vocab_argmax_agrees"] is not None]
    report = {
        "schema": "semif-diff-rows-v1", "profile": args.profile,
        "file_a": str(args.file_a), "file_b": str(args.file_b),
        "rows_compared": len(results),
        "identity_ok": all(r["identity_ok"] for r in results),
        "numeric_ok": all(r["numeric_ok"] for r in results),
        "argmax_agreement": sum(argmax) / len(argmax) if argmax else None,
        "full_vocab_argmax_agreement": sum(vocab) / len(vocab) if vocab else None,
        "finding_count": sum(len(r["findings"]) for r in results),
        "findings": [f for r in results for f in r["findings"]][:args.max_findings],
    }
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + "\n")
    print(text)
    sys.exit(0 if report["identity_ok"] and report["numeric_ok"] else 1)


if __name__ == "__main__":
    main()
