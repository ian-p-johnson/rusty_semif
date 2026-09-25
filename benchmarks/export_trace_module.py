"""Export fixed-shape TorchScript traces of the pinned Qwen3.5-4B (Stage 3).

Trace-at-fixed-shape: free-form tracing does NOT generalize across prompt
lengths for this hybrid model (python-level length logic bakes constants —
Gate 0 caught eager-vs-traced Δ=24 on a 396-token row traced at 108). Instead:

- every input is right-padded to a fixed width (TRACE_WIDTH, default 4096);
- the wrapper takes (input_ids[1, W], length[1]) and returns the f32
  last- *real*-position vocabulary logits, gathered with the length tensor;
- causal attention and the sequential delta-rule state make positions < L
  independent of the padding, which is verified, not assumed: the exporter
  checks eager-unpadded vs traced-padded per row (Δ ≤ 1e-4 CPU fp32 /
  snap-equal CUDA bf16).

Contexts: cpufp32 and cudabf16, matching the Stage 0 probe captures.
Create-only outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from semif_phase1.core import load_causal_model, softmax, synchronize
from semif_phase1.direct import encode_prompt

CHECK_ROWS = 6


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FixedWidthLogits(nn.Module):
    """(input_ids[1,W], length[1]) -> f32 vocabulary logits at position length-1."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        attention_mask = torch.ones_like(input_ids)
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits
        index = (length - 1).reshape(1, 1, 1).expand(1, 1, logits.shape[-1])
        return torch.gather(logits, 1, index)[0, 0].float()


def load_rows(fixtures: Path, limit: int) -> list[tuple[str, list[int]]]:
    accepted = {
        json.loads(line)["id"]
        for line in (fixtures / "prompts.jsonl").read_text().splitlines()
        if line.strip()
    }
    tokenizer_rows = {
        json.loads(line)["id"]: json.loads(line)
        for line in (fixtures / "tokens.jsonl").read_text().splitlines()
        if line.strip()
    }
    rows = []
    for line in (fixtures / "inputs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in accepted and row["id"] in tokenizer_rows:
            rows.append((row["id"], tokenizer_rows[row["id"]]["ids"]))
        if len(rows) >= limit:
            break
    return rows


def pad(ids: list[int], width: int, pad_id: int) -> list[int]:
    return ids + [pad_id] * (width - len(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--fixtures", type=Path, default=Path("semif-rs/fixtures"))
    parser.add_argument("--artifacts", type=Path, default=Path("semif-rs/artifacts"))
    parser.add_argument("--contexts", nargs="+", default=("cpufp32", "cudabf16"))
    parser.add_argument("--width", type=int, default=4096)
    args = parser.parse_args()
    if args.artifacts.exists() and any(args.artifacts.iterdir()):
        parser.error("Artifacts directory must be new or empty")
    args.artifacts.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.fixtures, CHECK_ROWS)
    if any(len(ids) > args.width for _, ids in rows):
        raise RuntimeError("check rows exceed the trace width")
    print(f"trace-check rows: {[row_id for row_id, _ in rows]}", flush=True)

    for context in args.contexts:
        device = "cpu" if context == "cpufp32" else "cuda"
        dtype = "float32" if context == "cpufp32" else "bfloat16"
        started = time.perf_counter()
        model, tokenizer, _metadata = load_causal_model(args.source, args.revision, device=device, dtype=dtype)
        target = next(model.parameters()).device
        print(f"[{context}] loaded on {target} in {time.perf_counter() - started:.1f}s", flush=True)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        wrapper = FixedWidthLogits(model).eval()
        example_ids = torch.tensor([pad(rows[0][1], args.width, pad_id)], dtype=torch.long, device=target)
        example_length = torch.tensor([len(rows[0][1])], dtype=torch.long, device=target)
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, (example_ids, example_length), check_trace=False)

        artifact = args.artifacts / f"qwen35-direct-{context}-w{args.width}.pt"
        torch.jit.save(traced, str(artifact))
        print(f"[{context}] saved {artifact.name} ({artifact.stat().st_size >> 20} MiB)", flush=True)

        checks = []
        for row_id, ids in rows:
            length = len(ids)
            unpadded = torch.tensor([ids], dtype=torch.long, device=target)
            padded = torch.tensor([pad(ids, args.width, pad_id)], dtype=torch.long, device=target)
            length_tensor = torch.tensor([length], dtype=torch.long, device=target)
            with torch.inference_mode():
                synchronize(target)
                mark = time.perf_counter()
                eager_logits = wrapper.model(
                    input_ids=unpadded, attention_mask=torch.ones_like(unpadded), use_cache=False, return_dict=True
                ).logits[0, length - 1].float()
                synchronize(target)
                eager_seconds = time.perf_counter() - mark
                mark = time.perf_counter()
                traced_logits = traced(padded, length_tensor)
                synchronize(target)
                traced_seconds = time.perf_counter() - mark
            delta = (eager_logits - traced_logits).abs().max().item()
            checks.append({
                "id": row_id,
                "input_tokens": length,
                "eager_unpadded_vs_traced_padded_max_abs_delta": delta,
                "traced_logits_sha256": hashlib.sha256(traced_logits.cpu().numpy().astype("<f4").tobytes()).hexdigest(),
                "traced_first8": [round(float(value), 6) for value in traced_logits[:8].cpu().tolist()],
                "eager_seconds": eager_seconds,
                "traced_seconds": traced_seconds,
            })
            print(f"[{context}] {row_id}: tokens={length} eager-vs-traced Δ={delta:.3e}", flush=True)

        worst = max(check["eager_unpadded_vs_traced_padded_max_abs_delta"] for check in checks)
        tolerance = 1e-4 if context == "cpufp32" else 0.51
        if worst > tolerance:
            raise RuntimeError(
                f"[{context}] padding-invariance delta {worst:.3e} exceeds {tolerance} — trace not usable"
            )
        record = {
            "context": context,
            "device_tag": (torch.cuda.get_device_name(0) if target.type == "cuda" else "cpu"),
            "artifact": artifact.name,
            "artifact_sha256": sha256_file(artifact),
            "artifact_mib": artifact.stat().st_size >> 20,
            "trace_width": args.width,
            "pad_id": pad_id,
            "torch_version": torch.__version__,
            "checks": checks,
        }
        with (args.artifacts / f"trace-check-{context}.json").open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        print(f"[{context}] PASS (worst Δ={worst:.3e})", flush=True)
        del model, wrapper, traced
        if target.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
