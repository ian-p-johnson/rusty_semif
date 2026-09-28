"""Export a fixed-shape TorchScript trace of the pinned Qwen3-Reranker-4B (Stage 3).

The reranker is scored as independent yes/no pairs. Python's
`score_pair_batch` left-pads every option of one row to the row's widest pair
and lets the model default to `position_ids = arange(width)`, so a shorter
option is evaluated at *shifted* RoPE positions — a property the published
batch-size sweep exercises and this port must reproduce, not "fix".

Trace layout (batch-flexible; the example batch is B=1):

  input_ids      [B, W]  = [pad] * (W - L) + ids        (real block ends at W-1)
  attention_mask [B, W]  = 0 * (W - L) + 1 * L
  position_ids   [B, W]  = index - (W - row_width) on real tokens, 1 elsewhere
  output         [B, V]  = f32 vocabulary logits at position W-1

Because the real block is right-aligned, `logits[:, -1, :]` is the last real
token, and the position offset makes a row of width `w` sit exactly where
Python's width-`w` batch puts it. The exporter verifies this against the eager
Python path per row (padding-invariance is checked, never assumed) and also
proves the trace runs at B = the row's option count.

Context: cudabf16 only — Python refuses reranker mode without CUDA.
Create-only outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from torch import nn

from semif_phase1 import reranker
from semif_phase1.core import load_causal_model

CHECK_ROWS = 6
CONTEXT = "cudabf16"
TOLERANCE = 0.51


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class FixedWidthRerankerLogits(nn.Module):
    """(input_ids[B,W], attention_mask[B,W], position_ids[B,W]) -> f32 [B,V]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        ).logits
        return logits[:, -1, :].float()


def layout(ids: list[int], width: int, row_width: int, pad: int) -> tuple[list[int], list[int], list[int]]:
    left = width - len(ids)
    offset = width - row_width
    array = [pad] * left + list(ids)
    mask = [0] * left + [1] * len(ids)
    position = [1 if index < offset else index - offset for index in range(width)]
    return array, mask, position


def eager_logits(model, tokenizer, batch: list[list[int]], width: int) -> torch.Tensor:
    """The forward `reranker.score_pair_batch` performs for one batch."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise RuntimeError("Tokenizer requires padding or EOS token")
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [[pad] * (width - len(ids)) + list(ids) for ids in batch], dtype=torch.long, device=device
    )
    attention_mask = torch.tensor(
        [[0] * (width - len(ids)) + [1] * len(ids) for ids in batch],
        dtype=torch.long,
        device=device,
    )
    kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True
    )
    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs["logits_to_keep"] = 1
    with torch.inference_mode():
        return model(**kwargs).logits[:, -1, :].float()


def load_rows(fixtures: Path) -> list[dict]:
    """Span the corpus width range: padding-invariance at W=2048 for a 100-token
    row is a different regime from a 1755-token row nearly filling the width."""
    tokens = [
        json.loads(line)
        for line in (fixtures / "reranker_tokens.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ordered = sorted(tokens, key=lambda row: (row["width"], row["id"]))
    marks = (0, 1, len(ordered) // 4, len(ordered) // 2, (3 * len(ordered)) // 4,
             len(ordered) - 2, len(ordered) - 1)
    picks, seen = [], set()
    for mark in marks:
        row = ordered[max(0, min(len(ordered) - 1, mark))]
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        picks.append(row)
    return picks[:CHECK_ROWS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/media/ianj/New Volume/hf-plain/models/Qwen3-Reranker-4B")
    parser.add_argument("--revision", default="22e683669bc0f0bd69640a1354a6d0aebcfeede5")
    parser.add_argument("--fixtures", type=Path, default=Path("semif-rs/fixtures"))
    parser.add_argument("--artifacts", type=Path, default=Path("semif-rs/artifacts"))
    parser.add_argument("--width", type=int, default=2048)
    args = parser.parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    artifact = args.artifacts / f"qwen3reranker-{CONTEXT}-w{args.width}.pt"
    check_path = args.artifacts / f"trace-check-reranker-{CONTEXT}.json"
    for target in (artifact, check_path):
        if target.exists():
            parser.error(f"{target} already exists")

    rows = load_rows(args.fixtures)
    longest = max(
        len(ids) for row in rows for ids in (option["ids"] for option in row["options"])
    )
    if longest > args.width:
        raise RuntimeError(f"check rows reach {longest} tokens, trace width is {args.width}")
    print(f"trace-check rows: {[row['id'] for row in rows]}", flush=True)

    started = time.perf_counter()
    model, tokenizer, _metadata = load_causal_model(
        args.source, args.revision, device="cuda", dtype="bfloat16"
    )
    target = next(model.parameters()).device
    print(f"[{CONTEXT}] loaded on {target} in {time.perf_counter() - started:.1f}s", flush=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise RuntimeError("Tokenizer requires padding or EOS token")
    no_id, yes_id = reranker._answer_ids(tokenizer)

    wrapper = FixedWidthRerankerLogits(model).eval()
    example = layout(rows[0]["options"][0]["ids"], args.width, rows[0]["width"], pad)
    example_ids = torch.tensor([example[0]], dtype=torch.long, device=target)
    example_mask = torch.tensor([example[1]], dtype=torch.long, device=target)
    example_pos = torch.tensor([example[2]], dtype=torch.long, device=target)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (example_ids, example_mask, example_pos), check_trace=False)
    torch.jit.save(traced, str(artifact))
    print(f"[{CONTEXT}] saved {artifact.name} ({artifact.stat().st_size >> 20} MiB)", flush=True)

    def run_traced(option_batch: list[dict], row_width: int) -> torch.Tensor:
        layouts = [
            layout(option["ids"], args.width, row_width, pad) for option in option_batch
        ]
        tensors = [
            torch.tensor([part[index] for part in layouts], dtype=torch.long, device=target)
            for index in range(3)
        ]
        with torch.inference_mode():
            return traced(*tensors)

    checks = []
    worst = 0.0
    worst_batch1 = 0.0
    for row in rows:
        options = row["options"]
        row_width = row["width"]
        with torch.inference_mode():
            eager = eager_logits(
                model, tokenizer, [option["ids"] for option in options], row_width
            )
        traced_logits = run_traced(options, row_width)
        delta = (eager - traced_logits).abs().max().item()
        worst = max(worst, delta)

        batch1 = []
        for option in options:
            one = run_traced([option], row_width)
            single = eager_logits(model, tokenizer, [option["ids"]], len(option["ids"]))
            batch1.append((single - one).abs().max().item())
        worst_batch1 = max(worst_batch1, max(batch1))

        per_option = []
        for option, logits in zip(options, traced_logits):
            values = logits.cpu().numpy().astype("<f4")
            odds = float(values[yes_id]) - float(values[no_id])
            per_option.append(
                {
                    "option_id": option["option_id"],
                    "input_tokens": len(option["ids"]),
                    "traced_logits_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "traced_no_logit": round(float(values[no_id]), 6),
                    "traced_yes_logit": round(float(values[yes_id]), 6),
                    "traced_log_odds": round(odds, 6),
                    "traced_binary_relevance": round(
                        float(
                            torch.softmax(logits[[no_id, yes_id]], dim=-1)[1]
                        ),
                        6,
                    ),
                }
            )
        checks.append(
            {
                "id": row["id"],
                "row_width": row_width,
                "options": len(options),
                "eager_vs_traced_max_abs_delta": delta,
                "batch1_eager_vs_traced_max_abs_delta": max(batch1),
                "per_option": per_option,
            }
        )
        print(
            f"[{CONTEXT}] {row['id']}: width={row_width} eager-vs-traced Δ={delta:.3e} "
            f"batch1 Δ={max(batch1):.3e}",
            flush=True,
        )

    if worst > TOLERANCE or worst_batch1 > TOLERANCE:
        raise RuntimeError(
            f"[{CONTEXT}] padding-invariance delta batch={worst:.3e} "
            f"batch1={worst_batch1:.3e} exceeds {TOLERANCE} — trace not usable"
        )
    record = {
        "context": CONTEXT,
        "model_kind": "reranker",
        "device_tag": torch.cuda.get_device_name(0),
        "artifact": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "artifact_mib": artifact.stat().st_size >> 20,
        "trace_width": args.width,
        "pad_id": pad,
        "no_id": no_id,
        "yes_id": yes_id,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "worst_eager_vs_traced_delta": worst,
        "worst_batch1_eager_vs_traced_delta": worst_batch1,
        "tolerance": TOLERANCE,
        "checks": checks,
    }
    with check_path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(f"[{CONTEXT}] PASS (worst Δ={worst:.3e}, batch1 Δ={worst_batch1:.3e})", flush=True)


if __name__ == "__main__":
    main()
