"""Export pure-logic and stub-envelope fixtures for Rust Stage 1 gates.

Complements export_fixtures.py (Stage 0): adds `logic.json` (float repr, both
json.dumps profiles, softmax vectors) and the stub-envelope pair
`stubs.jsonl` + `inputs-accepted.jsonl` used for the CLI byte-parity gate.
The stub envelope mirrors semif_engine::StubEngine exactly: direct-shape row,
uniform 1/K probabilities, zero logits, no timing fields. Create-only outputs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from semif_phase1.core import softmax as core_softmax
from semif_phase1.direct import encode_prompt


def float_cases() -> list[tuple[str, str]]:
    values = [
        0.0, -0.0, 1.0, -1.0, 0.5, 1.5, 2.5, 0.1, 0.2, 3.0, 150.0, 100.0,
        1234567890123456800.0, 1e15, 1e16, 1e-4, 1e-5, 1.5e-05, 5e-324,
        1.7976931348623157e308, 2.2250738585072014e-308, 3.141592653589793,
        123456789012345.67, 9999999999999998.0, 1e22, 9.999999999999999e21,
        -1250.5, 5000.0, 1e100, 6.02214076e23,
    ]
    return [(repr(value), repr(value)) for value in values]


def dumps_cases() -> list[tuple[str, str, str]]:
    raw_cases = [
        '{"a": 1, "b": [true, false, null], "c": "x y"}',
        '{"k": "中文"}',
        '"🔥"',
        '{"deep": {"deeper": [1.5, -0.0, 1e16, 1e-5], "t": "tab\\tnewline\\nquote\\"slash\\\\"}}',
        '"\\u0041\\u00e9\\u4e2d"',
        '{"z": 1, "a": 2}',
        '[1e400]'.replace("1e400", "1e2"),
        '{"empty": {}, "list": []}',
        '{"mixed": ["s", 7, 99.5, false, null]}',
    ]
    cases = []
    for text in raw_cases:
        value = json.loads(text)
        cases.append((text, json.dumps(value, ensure_ascii=False), json.dumps(value, ensure_ascii=True)))
    return cases


def softmax_cases() -> dict:
    suite = {
        "spread": [1000.0, 999.0, -1000.0],
        "ties": [0.0, 0.0],
        "committed_pair": [25.0, 25.25],
        "small": [0.1, 0.2, 0.3],
        "negative": [-1.5, -2.5],
        "near_equal": [123.456, 123.4560001],
    }
    cases = {}
    for name, values in suite.items():
        try:
            cases[name] = {"input": values, "output": core_softmax(values)}
        except ValueError as error:
            cases[name] = {"input": values, "error": str(error)}
    cases["single"] = {"input": [1.0], "error": "Need at least two finite scores"}
    cases["nan"] = {"input": [1.0, float("nan")], "error": "Need at least two finite scores"}
    return cases


def main() -> None:
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("semif-rs/fixtures"))
    parser.add_argument("--source", default="/media/ianj/New Volume/hf-plain/models/Qwen_Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--model-record", default=None,
                        help="Recorded model.source for stub rows (defaults to --source)")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    for name in ("logic.json", "stubs.jsonl", "inputs-accepted.jsonl"):
        if (args.fixtures / name).exists():
            parser.error(f"{name} already exists")

    logic = {
        "schema": "semif-stage1-logic-v1",
        "float_reprs": [{"input": text, "output": output} for text, output in float_cases()],
        "dumps": [{"input": a, "ensure_ascii_false": b, "ensure_ascii_true": c}
                  for a, b, c in dumps_cases()],
        "softmax": softmax_cases(),
    }
    with (args.fixtures / "logic.json").open("x") as stream:
        json.dump(logic, stream, indent=1)
        stream.write("\n")

    accepted = {json.loads(line)["id"] for line in (args.fixtures / "prompts.jsonl").read_text().splitlines() if line.strip()}
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.source, local_files_only=True, trust_remote_code=False)
    model_record = args.model_record or args.source
    stubs, accepted_lines = [], []
    for line in (args.fixtures / "inputs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] not in accepted:
            continue
        accepted_lines.append(line)
        ids, _slots, prompt_hash = encode_prompt(tokenizer, row, args.max_tokens)
        k = len(row["options"])
        stubs.append({
            "id": row["id"],
            "option_ids": [option["id"] for option in row["options"]],
            "probabilities": [1 / k] * k,
            "option_logits": [0.0] * k,
            "input_tokens": len(ids),
            "prompt_sha256": prompt_hash,
            "prompt_version": "direct-options-v1",
            "model": {"source": model_record, "revision": args.revision,
                      "dtype": args.dtype, "serving_config": "stub-uniform-v1"},
            "readout": "stub-uniform-v1",
            "probability_status": "stub; not a model score",
        })
    with (args.fixtures / "inputs-accepted.jsonl").open("x") as stream:
        for line in accepted_lines:
            stream.write(line + "\n")
    with (args.fixtures / "stubs.jsonl").open("x") as stream:
        for stub in stubs:
            stream.write(json.dumps(stub) + "\n")
    print(f"logic cases: {len(logic['float_reprs'])} floats, {len(logic['dumps'])} dumps, {len(logic['softmax'])} softmax; stub rows: {len(stubs)}")


if __name__ == "__main__":
    main()
