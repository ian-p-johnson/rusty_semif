"""Measure the direct-mode readout layout: cost and numerics, per configuration.

Built because a Stage 4 follow-up claimed the traced direct wrapper was ~1.8x
slower than the Python oracle because it cannot pass `logits_to_keep=1`. That
claim was measured sequentially on a shared machine and did not survive: the
saving was ~3%, because this checkpoint's forward is dominated by the reference
`chunk_gated_delta_rule` / `causal_conv1d` fallbacks, not by the lm_head.

Configurations, all over the same rows:

  A  eager, unpadded,          logits_to_keep=1   what the Python oracle does
  B  eager, right-padded to W, full logits+gather  what the current trace does
  C  eager, left-padded to W,  logits_to_keep=1    the proposed wrapper
  D  the existing traced artifact                  fusion cost on top of B

A/B/C are timed **interleaved** round-robin and summarised by median, so a load
spike drifts all three together instead of favouring whichever ran first. D needs
the 8 GiB artifact alongside the 8 GiB weights, so it runs in a second phase
after the model is released.

Reports the load average it ran under: on this shared box a timing taken under
contention is void, per Stage 0's own precedent.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def load_average() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--input", type=Path, default=Path("benchmarks/data/shape777.jsonl"))
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument(
        "--artifacts", type=Path, default=Path("semif-rs/artifacts/w2048")
    )
    parser.add_argument(
        "--skip-trace",
        action="store_true",
        help="Phase 1 only (avoids loading the artifact alongside the weights)",
    )
    args = parser.parse_args()

    import inspect

    from semif_phase1.core import load_causal_model
    from semif_phase1.direct import encode_prompt

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = rows[: args.rows]
    load_start = load_average()

    model, tokenizer, _meta = load_causal_model(
        args.model, args.revision, device="cuda", dtype="bfloat16"
    )
    device = next(model.parameters()).device
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    has_keep = "logits_to_keep" in inspect.signature(model.forward).parameters
    encoded = [encode_prompt(tokenizer, row, 100_000) for row in rows]
    ids_only = [ids for ids, _, _ in encoded]
    width = args.width

    def config_a() -> None:
        for ids in ids_only:
            tensor = torch.tensor([ids], dtype=torch.long, device=device)
            kwargs = dict(
                input_ids=tensor,
                attention_mask=torch.ones_like(tensor),
                use_cache=False,
                return_dict=True,
            )
            if has_keep:
                kwargs["logits_to_keep"] = 1
            with torch.inference_mode():
                model(**kwargs).logits[:, -1, :].float()

    def config_b() -> None:
        for ids in ids_only:
            left = width - len(ids)
            padded = torch.tensor(
                [[pad] * left + list(ids)], dtype=torch.long, device=device
            )
            kwargs = dict(
                input_ids=padded,
                attention_mask=torch.ones_like(padded),
                use_cache=False,
                return_dict=True,
            )
            with torch.inference_mode():
                logits = model(**kwargs).logits
            index = torch.tensor([[len(ids) - 1]], device=device).reshape(1, 1, 1).expand(
                1, 1, logits.shape[-1]
            )
            torch.gather(logits, 1, index)[:, 0, :].float()

    def config_c() -> None:
        for ids in ids_only:
            left = width - len(ids)
            padded = torch.tensor(
                [[pad] * left + list(ids)], dtype=torch.long, device=device
            )
            mask = torch.tensor(
                [[0] * left + [1] * len(ids)], dtype=torch.long, device=device
            )
            position = torch.tensor(
                [[1 if k < left else k - left for k in range(width)]],
                dtype=torch.long,
                device=device,
            )
            kwargs = dict(
                input_ids=padded,
                attention_mask=mask,
                position_ids=position,
                use_cache=False,
                return_dict=True,
            )
            if has_keep:
                kwargs["logits_to_keep"] = 1
            with torch.inference_mode():
                model(**kwargs).logits[:, -1, :].float()

    def time_pass(function) -> float:
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / len(ids_only)

    # Warm every path once before anything is recorded.
    for function in (config_a, config_b, config_c):
        function()

    samples: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    functions = (("A", config_a), ("B", config_b), ("C", config_c))
    for _ in range(args.rounds):
        # Interleaved: any drift in machine load hits all three each round.
        for name, function in functions:
            samples[name].append(time_pass(function))

    print(f"load average at start: {load_start}")
    print(f"rows={len(ids_only)} tokens={[len(i) for i in ids_only]} rounds={args.rounds}")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    base = medians["A"]
    for name in ("A", "B", "C"):
        print(
            f"  {name}: median {medians[name]*1000:7.1f} ms/row   "
            f"x{medians[name]/base:.3f} vs A   raw="
            f"{[round(v * 1000, 1) for v in samples[name]]}"
        )
    print(f"  B vs C (the claimed win): x{medians['B']/medians['C']:.3f}")

    # Numeric equivalence of the proposed layout against the oracle path.
    with torch.inference_mode():
        worst = 0.0
        for ids in ids_only:
            plain = torch.tensor([ids], dtype=torch.long, device=device)
            keep_kwargs = dict(
                input_ids=plain,
                attention_mask=torch.ones_like(plain),
                use_cache=False,
                return_dict=True,
            )
            if has_keep:
                keep_kwargs["logits_to_keep"] = 1
            eager = model(**keep_kwargs).logits[:, -1, :].float()

            left = width - len(ids)
            padded = torch.tensor(
                [[pad] * left + list(ids)], dtype=torch.long, device=device
            )
            mask = torch.tensor(
                [[0] * left + [1] * len(ids)], dtype=torch.long, device=device
            )
            position = torch.tensor(
                [[1 if k < left else k - left for k in range(width)]],
                dtype=torch.long,
                device=device,
            )
            kwargs = dict(
                input_ids=padded,
                attention_mask=mask,
                position_ids=position,
                use_cache=False,
                return_dict=True,
            )
            if has_keep:
                kwargs["logits_to_keep"] = 1
            proposed = model(**kwargs).logits[:, -1, :].float()
            worst = max(worst, (eager - proposed).abs().max().item())
    print(f"  A vs C max|logit delta| = {worst:.3e} (padding-invariance of the proposal)")

    if not args.skip_trace:
        del model
        torch.cuda.empty_cache()
        artifact = args.artifacts / f"qwen35-direct-cudabf16-w{args.width}.pt"
        traced = torch.jit.load(str(artifact), map_location="cuda")
        print(f"loaded {artifact.name}")

        def config_d() -> None:
            for ids in ids_only:
                left = width - len(ids)
                padded = torch.tensor(
                    [[pad] * left + list(ids)], dtype=torch.long, device=device
                )
                length = torch.tensor([len(ids)], dtype=torch.long, device=device)
                with torch.inference_mode():
                    traced(padded, length)

        config_d()
        samples_d = [time_pass(config_d) for _ in range(args.rounds)]
        median_d = statistics.median(samples_d)
        print(
            f"  D: median {median_d*1000:7.1f} ms/row   "
            f"x{median_d/base:.3f} vs A, x{median_d/medians['B']:.3f} vs B   "
            f"raw={[round(v * 1000, 1) for v in samples_d]}"
        )

    print(
        json.dumps(
            {
                "load_average_start": load_start,
                "load_average_end": load_average(),
                "rows": len(ids_only),
                "rounds": args.rounds,
                "median_ms_per_row": {
                    name: round(value * 1000, 2) for name, value in medians.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
