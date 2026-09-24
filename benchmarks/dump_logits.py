"""Dump full-vocabulary readout probes for Rust-parity captures (PORTING_RUST.md §6).

For every accepted fixture row, captures: slot logits, probabilities, allowed token
mass, full-vocabulary argmax, and a sha256 of the bit-cast f32 vocabulary (a strong
bit-parity witness without shipping 150k floats per row). Captures are device-tagged
and never compared across devices bit-wise.

Profiles:
  cpufp32    torch CPU float32 (the README reference path)
  cuda-bf16  torch CUDA bfloat16 on this machine's GPU
  gguf       llama.cpp backend over the pinned Q4_K_M GGUF (CPU)

Outputs are create-only. No generation is performed; single forward per row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from semif_phase1.core import digest, load_causal_model, softmax, synchronize
from semif_phase1.direct import _forward, encode_prompt


def accepted_ids(fixtures: Path) -> set[str]:
    return {json.loads(line)["id"] for line in (fixtures / "prompts.jsonl").read_text().splitlines() if line.strip()}


def load_rows(fixtures: Path, shape_states: set[int], ids_file: Path | None = None) -> list[dict]:
    rows = [json.loads(line) for line in (fixtures / "inputs.jsonl").read_text().splitlines() if line.strip()]
    keep = accepted_ids(fixtures)
    if ids_file:
        wanted = {line.strip() for line in ids_file.read_text().splitlines() if line.strip()}
        keep &= wanted
    selected = []
    for row in rows:
        if row["id"] not in keep:
            continue
        provenance = row.get("provenance") or {}
        if provenance.get("kind") == "project_owned_systems_fixture" and provenance.get("state_index") not in shape_states:
            continue
        selected.append(row)
    return selected


def logsumexp(values: np.ndarray) -> float:
    peak = float(values.max())
    return peak + float(np.log(np.exp(values - peak).sum()))


def derived(vocabulary: np.ndarray, slots: list[int]) -> dict:
    selected = vocabulary[slots]
    return {
        "option_logits": [float(value) for value in selected],
        "probabilities": softmax([float(value) for value in selected]),
        "allowed_token_mass": float(np.exp(logsumexp(selected) - logsumexp(vocabulary))),
        "full_vocab_argmax_id": int(vocabulary.argmax()),
        "logits_sha256": hashlib.sha256(vocabulary.astype("<f4").tobytes()).hexdigest(),
        "vocab_size": int(vocabulary.shape[0]),
    }


def torch_probe(rows: list[dict], source: str, revision: str, device: str, dtype: str, profile: str, output: Path) -> list[dict]:
    import torch

    model, tokenizer, metadata = load_causal_model(source, revision, device=device, dtype=dtype)
    target = next(model.parameters()).device
    device_tag = torch.cuda.get_device_name(0) if target.type == "cuda" else target.type
    captures = []
    with output.open("x") as stream:
        for row in rows:
            started = time.perf_counter()
            ids, slots, prompt_hash = encode_prompt(tokenizer, row, 4096)
            inputs = {"input_ids": torch.tensor([ids], dtype=torch.long, device=target),
                      "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=target)}
            synchronize(target)
            mark = time.perf_counter()
            with torch.inference_mode():
                vocabulary = _forward(model, inputs)[0].float().cpu().numpy()
            synchronize(target)
            capture = {"id": row["id"], "option_ids": [option["id"] for option in row["options"]],
                       "prompt_sha256": prompt_hash, "input_tokens": len(ids),
                       "forward_seconds": time.perf_counter() - mark,
                       "total_seconds": time.perf_counter() - started,
                       **derived(vocabulary, slots),
                       "capture": {**{key: metadata[key] for key in ("source", "revision", "dtype", "device")},
                                   "profile": profile, "device_tag": device_tag,
                                   "torch_version": metadata["torch_version"],
                                   "transformers_version": metadata["transformers_version"]}}
            stream.write(json.dumps(capture) + "\n")
            stream.flush()
            captures.append(capture)
            print(f"{row['id']}: {capture['forward_seconds']:.3f}s forward, {len(ids)} tokens", flush=True)
    return captures


def gguf_probe(rows: list[dict], source: str, revision: str, gguf: Path, threads: int | None, output: Path) -> list[dict]:
    from semif_phase1 import llamacpp_backend

    model, tokenizer, metadata = llamacpp_backend.load_model(source, revision, gguf, threads=threads, context_tokens=4096)
    captures = []
    with output.open("x") as stream:
        for row in rows:
            started = time.perf_counter()
            encoded = model.encode_verified(row, 4096)
            ids, slots, prompt_hash = encoded
            mark = time.perf_counter()
            vocabulary = model.engine.full_logits(ids)
            capture = {"id": row["id"], "option_ids": [option["id"] for option in row["options"]],
                       "prompt_sha256": prompt_hash, "input_tokens": len(ids),
                       "forward_seconds": time.perf_counter() - mark,
                       "total_seconds": time.perf_counter() - started,
                       **derived(vocabulary, slots),
                       "capture": {**{key: metadata[key] for key in ("source", "revision", "gguf", "vocab_size")},
                                   "profile": "gguf", "device_tag": "cpu",
                                   "threads": metadata["threads"], "llama_cpp_python_version": metadata["llama_cpp_python_version"]}}
            stream.write(json.dumps(capture) + "\n")
            stream.flush()
            captures.append(capture)
            print(f"{row['id']}: {capture['forward_seconds']:.3f}s forward, {len(ids)} tokens", flush=True)
    model.close()
    return captures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("semif-rs/fixtures"))
    parser.add_argument("--profile", choices=("cpufp32", "cuda-bf16", "gguf"), required=True)
    parser.add_argument("--source", default="/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--gguf", type=Path, default=Path("/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B-Q4_K_M.gguf"))
    parser.add_argument("--llama-threads", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape-states", type=int, nargs="+", default=(0,),
                        help="shape777 state_index values to include (long prompts)")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="Optional file of row ids; restricts the probe to those rows")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be new")

    rows = load_rows(args.fixtures, set(args.shape_states), args.ids_file)
    print(f"probing {len(rows)} accepted rows, profile={args.profile}", flush=True)
    if args.profile == "gguf":
        captures = gguf_probe(rows, args.source, args.revision, args.gguf, args.llama_threads, args.output)
    else:
        captures = torch_probe(rows, args.source, args.revision,
                               "cpu" if args.profile == "cpufp32" else "cuda",
                               "float32" if args.profile == "cpufp32" else "bfloat16", args.profile, args.output)
    print(f"wrote {len(captures)} probes to {args.output}")


if __name__ == "__main__":
    main()
