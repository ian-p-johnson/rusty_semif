"""Export reranker prompt/token fixtures for the Rust port (PORTING_RUST.md §7 Stage 3).

`reranker._encode` builds the pair prompt as PREFIX + body + SUFFIX where the
state is interpolated with Python's `str()` (not `json.dumps`) — dict/list
states therefore render in repr form, which is a distinct writer surface from
the direct-mode payload. Every rendered prompt is hashed exactly as the runtime
does (`digest(text)`), so the Rust side is gated byte-for-byte before any model
work.

Outputs (create-only):
  fixtures/reranker_rows.jsonl      the exact corpus the other files cover
  fixtures/reranker_prompts.jsonl   per (row, option) rendered prompt + sha256
  fixtures/reranker_tokens.jsonl    per-row option id sequences + row width
  fixtures/reranker_answers.json    the yes/no single-token contract
  fixtures/reranker_refusals.jsonl  over-budget messages at a pinned budget
  fixtures/reranker_manifest.json   corpus provenance and counts
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from semif_phase1 import reranker

SHAPE_GROUPS = 3


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def corpus(fixtures: Path, data: Path) -> list[dict]:
    rows = read_jsonl(fixtures / "inputs-accepted.jsonl")
    seen = {row["id"] for row in rows}
    shape = read_jsonl(data / "shape777.jsonl")
    groups = sorted({row["group_id"] for row in shape})
    keep = set(groups[:SHAPE_GROUPS])
    added = [row for row in shape if row["group_id"] in keep and row["id"] not in seen]
    rows.extend(added)
    # Pin both reranker instruction branches: the published corpora carry no
    # `provenance.experiment`, so clone one owned row per branch.
    base = next(row for row in rows if isinstance(row["state"], str))
    for experiment in ("code-rag", "company-brain"):
        clone = json.loads(json.dumps(base))
        clone["id"] = f"{base['id']}-reranker-{experiment}"
        clone.setdefault("provenance", {})["experiment"] = experiment
        rows.append(clone)
    return rows


def pair_text(row: dict, option: dict) -> tuple[str, str]:
    experiment = (row.get("provenance") or {}).get("experiment")
    instruction = (
        reranker.RETRIEVAL_INSTRUCTION
        if experiment in {"code-rag", "company-brain"}
        else reranker.DECISION_INSTRUCTION
    )
    body = (
        f"<Instruct>: {instruction}\n"
        f"<Query>: Question: {row['question']}\n"
        f"Candidate answer: {option['description']}\n"
        f"<Document>: {row['state']}"
    )
    return reranker.PREFIX + body + reranker.SUFFIX, instruction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("semif-rs/fixtures"))
    parser.add_argument("--data", type=Path, default=Path("benchmarks/data"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("/media/ianj/New Volume/hf-plain/models/Qwen3-Reranker-4B"),
    )
    args = parser.parse_args()

    for name in (
        "reranker_rows.jsonl",
        "reranker_prompts.jsonl",
        "reranker_tokens.jsonl",
        "reranker_answers.json",
        "reranker_refusals.jsonl",
        "reranker_manifest.json",
    ):
        target = args.fixtures / name
        if target.exists():
            parser.error(f"{target} already exists")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    rows = corpus(args.fixtures, args.data)

    prompts, tokens, refusal_candidates = [], [], []
    corpus_digest = hashlib.sha256()
    widths = []
    for row in rows:
        corpus_digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
        option_tokens = []
        width = 0
        for index, option in enumerate(row["options"]):
            text, instruction = pair_text(row, option)
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                raise SystemExit(f"empty encoding for {row['id']}/{option['id']}")
            digest = reranker.digest(text)
            prompts.append(
                {
                    "id": row["id"],
                    "option_index": index,
                    "option_id": option["id"],
                    "instruction": instruction,
                    "prompt_sha256": digest,
                    "prompt": text,
                }
            )
            option_tokens.append({"option_id": option["id"], "ids": ids})
            width = max(width, len(ids))
            refusal_candidates.append(
                {
                    "id": row["id"],
                    "option_id": option["id"],
                    "input_tokens": len(ids),
                }
            )
        tokens.append(
            {"id": row["id"], "width": width, "options": option_tokens}
        )
        widths.append(width)

    # A budget of `len(ids) - 1` forces the runtime's over-budget refusal, so
    # the message is pinned without needing a model run.
    picked = refusal_candidates[:6]
    picked += [
        min(refusal_candidates, key=lambda item: item["input_tokens"]),
        max(refusal_candidates, key=lambda item: item["input_tokens"]),
    ]
    refusals = []
    for item in picked:
        budget = item["input_tokens"] - 1
        refusals.append(
            {
                **item,
                "max_tokens": budget,
                "message": (
                    f"Row {item['id']} option {item['option_id']}: "
                    f"{item['input_tokens']} tokens exceed limit {budget}"
                ),
            }
        )

    no_ids = tokenizer.encode("no", add_special_tokens=False)
    yes_ids = tokenizer.encode("yes", add_special_tokens=False)
    if len(no_ids) != 1 or len(yes_ids) != 1 or no_ids == yes_ids:
        raise SystemExit("Reranker yes/no answers must be distinct single tokens")
    convert_no = tokenizer.convert_tokens_to_ids("no")
    convert_yes = tokenizer.convert_tokens_to_ids("yes")
    if convert_no != no_ids[0] or convert_yes != yes_ids[0]:
        raise SystemExit("Tokenizer conversion differs from the official yes/no token contract")
    answers = {
        "no_ids": no_ids,
        "yes_ids": yes_ids,
        "convert_tokens_to_ids_no": convert_no,
        "convert_tokens_to_ids_yes": convert_yes,
        "prompt_version": reranker.PROMPT_VERSION,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    def write_jsonl(path: Path, records: list[dict]) -> None:
        with path.open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    # The corpus spans three sources plus two synthetic instruction-branch
    # clones, so it is shipped alongside the prompts rather than reconstructed
    # from `inputs.jsonl`.
    write_jsonl(args.fixtures / "reranker_rows.jsonl", rows)
    write_jsonl(args.fixtures / "reranker_prompts.jsonl", prompts)
    write_jsonl(args.fixtures / "reranker_tokens.jsonl", tokens)
    write_jsonl(args.fixtures / "reranker_refusals.jsonl", refusals)
    with (args.fixtures / "reranker_answers.json").open("x") as stream:
        json.dump(answers, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    all_ids = [ids for token_row in tokens for ids in (e["ids"] for e in token_row["options"])]
    manifest = {
        "rows": len(rows),
        "pairs": len(prompts),
        "shape_groups": SHAPE_GROUPS,
        "max_pair_tokens": max(len(ids) for ids in all_ids),
        "row_width_min": min(widths),
        "row_width_max": max(widths),
        "corpus_sha256": corpus_digest.hexdigest(),
        "tokenizer": str(args.tokenizer),
    }
    with (args.fixtures / "reranker_manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
