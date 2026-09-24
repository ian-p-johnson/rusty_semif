"""Export Rust-parity fixtures: rendered prompts, token encodings, validation verdicts, CLI refusals.

Stage 0 of PORTING_RUST.md. Model-independent except for the pinned reference
tokenizer: prompt rendering executes the checkpoint's chat-template Jinja, so the
captured prompt text and prompt_sha256 are the parity contract for Stage 1.
Outputs are create-only under --output-dir; no model weights are loaded.

Files written:
  inputs.jsonl       every corpus row exactly as consumed (Python-flavored JSON:
                     NaN/Infinity literals preserved where present)
  prompts.jsonl      rendered prompt text + prompt_sha256 for accepted rows
  tokens.jsonl       token IDs, answer-slot IDs, state-prefix IDs for accepted rows
  rows.jsonl         validation and encode verdicts, including exact error strings
  cli_errors.jsonl   subprocess CLI outcomes for contract refusals (exit code, stderr,
                     output-not-created check)
  manifest.json      pinned source revisions and library versions (no timestamps)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from semif_phase1.core import LETTERS, digest, direct_messages
from semif_phase1.direct import encode_prompt
from semif_phase1.shared import _state_prefix

MAX_TOKENS = 4096
SHAPE_STATES = {0, 1}


def edge_rows() -> list[dict]:
    return [
        {"id": "edge-options-min2", "state": "A single invoice line: 12.50 EUR, settled.", "question": "Is the invoice settled?",
         "options": [{"id": "yes", "description": "Settled."}, {"id": "no", "description": "Not settled."}]},
        {"id": "edge-options-max16", "state": "Ticket triage pool.", "question": "Which team owns the ticket?",
         "options": [{"id": f"team-{index:02d}", "description": f"Team {LETTERS[index]}: owned by unit {index}."} for index in range(16)]},
        {"id": "edge-unicode-cjk", "state": "客户 reports the 支付 failed twice today. Ledger shows pending.", "question": "该交易是否已退款?",
         "options": [{"id": "refunded-是", "description": "退款已完成。"}, {"id": "pending", "description": "Still pending."}]},
        {"id": "edge-unicode-emoji", "state": "Merchant “Café Renée” 🔥 flags the terminal outage 🚀; £40 charge pending.", "question": "Does the outage explain the failed charge?",
         "options": [{"id": "yes", "description": "Outage explains it."}, {"id": "no", "description": "Unrelated."}]},
        {"id": "edge-state-dict-flat", "state": {"policy": "Never request passwords", "limit": 3, "active": True}, "question": "Does the request comply?",
         "options": [{"id": "yes", "description": "Complies."}, {"id": "no", "description": "Violates."}]},
        {"id": "edge-state-dict-nested", "state": {"txn": {"amount": 1250.5, "currency": "EUR", "tags": ["sepa", "instant"], "memo": None},
                                                   "limits": {"daily": 5000.0, "per_contact": [100.0, 250.0]}, "reviewed": False, "note": "/customer requested"},
         "question": "Is the transaction within daily limits?",
         "options": [{"id": "within", "description": "Within limits."}, {"id": "over", "description": "Exceeds limits."}]},
        {"id": "edge-state-list-mixed", "state": ["chargeback", 7, 99.5, False, None, "GBP"], "question": "Does the record indicate a chargeback?",
         "options": [{"id": "yes", "description": "Chargeback present."}, {"id": "no", "description": "No chargeback."}]},
        {"id": "edge-description-empty", "state": "Ambiguous ledger entry.", "question": "Which option applies?",
         "options": [{"id": "a", "description": ""}, {"id": "b", "description": "Normal path."}]},
        {"id": "edge-escapes", "state": 'Memo says "reversed \"for sure\"" \\\\ and\na new line\tstays.', "question": "Was the memo quoted verbatim?",
         "options": [{"id": "yes", "description": "Verbatim \"quote\" preserved."}, {"id": "no", "description": "Altered."}]},
    ]


def refusal_rows() -> list[dict]:
    return [
        {"id": "refuse-missing-question", "state": "Evidence only.", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-state-empty-string", "state": "", "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-state-empty-dict", "state": {}, "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-state-empty-list", "state": [], "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-state-int", "state": 42, "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-state-null", "state": None, "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-duplicate-option-ids", "state": "Evidence.", "question": "Which?",
         "options": [{"id": "same", "description": "A."}, {"id": "same", "description": "B."}]},
        {"id": "refuse-one-option", "state": "Evidence.", "question": "Which?", "options": [{"id": "only", "description": "Only one."}]},
        {"id": "refuse-17-options", "state": "Evidence.", "question": "Which?",
         "options": [{"id": f"o{index}", "description": "D."} for index in range(17)]},
        {"id": "refuse-option-missing-description", "state": "Evidence.", "question": "Which?", "options": [{"id": "yes", "description": "Y."}, {"id": "no"}]},
        {"id": "refuse-option-nonstring-id", "state": "Evidence.", "question": "Which?", "options": [{"id": "yes", "description": "Y."}, {"id": 7, "description": "N."}]},
        {"id": "refuse-empty-id", "state": "Evidence.", "question": "Which?", "options": [{"id": "", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-empty-question", "state": "Evidence.", "question": "", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
    ]


def nonfinite_rows() -> list[dict]:
    return [
        {"id": "refuse-state-nan", "state": {"score": float("nan")}, "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
        {"id": "refuse-state-inf", "state": {"limit": float("inf")}, "question": "Any evidence?", "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]},
    ]


def over_budget_row(tokenizer) -> dict:
    unit = ("Payment of 1250.00 EUR to merchant ACME GmbH was authorized at 2026-09-24T10:15:00Z "
            "and settled in the same batch; the ledger line and the customer statement agree. ")
    state = unit
    while len(tokenizer.encode(state, add_special_tokens=False)) < MAX_TOKENS - 50:
        state += unit
    return {"id": "refuse-over-budget", "state": state, "question": "Was the payment settled?",
            "options": [{"id": "yes", "description": "Settled."}, {"id": "no", "description": "Not settled."}]}


def load_corpus() -> list[dict]:
    corpus = []
    for source in (Path("examples/decisions.jsonl"), Path("benchmarks/data/authored144.jsonl"),
                   Path("benchmarks/data/perturbations108.jsonl")):
        corpus.extend(json.loads(line) for line in source.read_text().splitlines() if line.strip())
    corpus.extend(row for row in (json.loads(line) for line in Path("benchmarks/data/shape777.jsonl").read_text().splitlines() if line.strip())
                  if row["provenance"]["state_index"] in SHAPE_STATES)
    return corpus


def main() -> None:
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B", help="Recorded canonical model identity")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", help="Recorded pinned revision")
    parser.add_argument("--source", default=None, help="Loader path; defaults to --model (HF repo id)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--with-cli-probes", action="store_true", help="Run subprocess CLI refusal probes")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory must be new or empty")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.source or args.model, revision=None if args.source else args.revision,
        local_files_only=bool(args.source), trust_remote_code=False)

    inputs = edge_rows() + refusal_rows() + nonfinite_rows()
    inputs.append(over_budget_row(tokenizer))
    inputs.extend(load_corpus())

    out = args.output_dir
    out.mkdir(parents=True)
    with (out / "inputs.jsonl").open("x") as stream:
        for row in inputs:
            stream.write(json.dumps(row) + "\n")

    prompts, tokens, verdicts = [], [], []
    for row in inputs:
        verdict = {"id": row["id"], "valid": True, "validation_error": None, "encode_error": None}
        try:
            direct_messages(row)
        except (ValueError, TypeError, KeyError) as error:
            verdict.update(valid=False, validation_error=str(error))
            verdicts.append(verdict)
            continue
        try:
            prompt = tokenizer.apply_chat_template(
                direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids, slots, prompt_hash = encode_prompt(tokenizer, row, MAX_TOKENS)
        except (ValueError, TypeError) as error:
            verdict["encode_error"] = str(error)
            verdicts.append(verdict)
            continue
        if digest(prompt) != prompt_hash:
            raise RuntimeError(f"{row['id']}: rendered prompt digest disagrees with encode_prompt")
        prefix = _state_prefix(tokenizer, row["state"])
        prompts.append({"id": row["id"], "prompt": prompt, "prompt_sha256": prompt_hash, "prompt_chars": len(prompt)})
        tokens.append({"id": row["id"], "ids": ids, "slots": slots, "n_tokens": len(ids),
                       "prefix_ids": prefix, "prefix_sha256": digest(json.dumps(prefix))})
        verdicts.append(verdict)

    for name, records in (("prompts.jsonl", prompts), ("tokens.jsonl", tokens), ("rows.jsonl", verdicts)):
        with (out / name).open("x") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
    print(f"inputs={len(inputs)} accepted={len(prompts)} refused={sum(1 for v in verdicts if not v['valid'] or v['encode_error'])}")

    if args.with_cli_probes:
        cli_probes(out, inputs, args.model, args.revision)

    manifest = {
        "schema": "semif-stage0-fixtures-v1",
        "model_source": args.model,
        "model_revision": args.revision,
        "loader_source": args.source or args.model,
        "max_tokens": MAX_TOKENS,
        "transformers_version": transformers.__version__,
        "tokenizers_version": __import__("tokenizers").__version__,
        "python_version": sys.version.split()[0],
    }
    with (out / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def cli_probes(out: Path, inputs: list[dict], model: str, revision: str) -> None:
    import tempfile

    probes = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = ["--mode", "direct", "--model", model, "--revision", revision, "--device", "cpu"]
        cases = {
            "cli-row-valueerror": next(row for row in inputs if row["id"] == "refuse-duplicate-option-ids"),
            "cli-existing-output": None,
            "cli-empty-input": None,
        }
        for case, row in cases.items():
            source, destination = tmp / f"{case}.jsonl", tmp / f"{case}.out.jsonl"
            if row is not None:
                source.write_text(json.dumps(row) + "\n")
            elif case == "cli-empty-input":
                source.write_text("\n \n")
            else:
                source.write_text(json.dumps({"id": "x", "state": "s", "question": "q?",
                                              "options": [{"id": "yes", "description": "Y."}, {"id": "no", "description": "N."}]}) + "\n")
                destination.write_text("pre-existing\n")
            command = [sys.executable, "-m", "semif_phase1.cli", *base, "--input", str(source), "--output", str(destination)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            stderr_lines = [line for line in completed.stderr.splitlines() if line.strip()]
            probes.append({"case": case, "returncode": completed.returncode,
                           "stderr_last_line": stderr_lines[-1] if stderr_lines else None,
                           "output_created": destination.exists() and destination.read_text() != "pre-existing\n"})
    with (out / "cli_errors.jsonl").open("x") as stream:
        for probe in probes:
            stream.write(json.dumps(probe) + "\n")


if __name__ == "__main__":
    main()
