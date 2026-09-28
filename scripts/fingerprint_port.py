"""Grade Rust-produced prediction files with the unmodified Python evaluators
and print the fingerprint table (PORTING_RUST.md §7 Stage 4).

For each (gold, python-baseline, rust) triple the gold is scored twice — once
against the same-device Python run, once against the Rust run — using
`benchmarks/evaluate.py` exactly as shipped, so the port is graded by the
independent observer rather than by its own tooling. The table reports both
fingerprints side by side plus the head-to-head decision agreement.

Every `evaluate.py` report is written to a create-only path under
`semif-rs/fingerprints/` so the numbers behind the table are inspectable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "benchmarks"))

import evaluate  # noqa: E402  (the shipped, unmodified observer)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def argmax(row: dict):
    values = row.get("probabilities")
    if not values:
        return row.get("prediction_id")
    return row["option_ids"][max(range(len(values)), key=values.__getitem__)]


def decision_agreement(a: Path, b: Path) -> tuple[int, int, list[str]]:
    left = {row["id"]: row for row in read_jsonl(a)}
    right = {row["id"]: row for row in read_jsonl(b)}
    shared = sorted(set(left) & set(right))
    flips = [key for key in shared if argmax(left[key]) != argmax(right[key])]
    return len(shared) - len(flips), len(shared), flips


def grade(gold: Path, predictions: Path, report_path: Path) -> dict:
    """Run the shipped evaluator; the report file is create-only."""
    result = evaluate.evaluate(read_jsonl(gold), read_jsonl(predictions))
    if not report_path.exists():
        report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return {
        "accuracy": result["family_results"] and _weighted_accuracy(result),
        "mean_family_balanced_accuracy": result["mean_family_balanced_accuracy"],
        "mean_family_macro_f1": result["mean_family_macro_f1"],
        "coverage": result["coverage"],
        "invalid_or_missing": result["invalid"] + result["missing"],
    }


def _weighted_accuracy(result: dict) -> float:
    families = result["family_results"].values()
    total = sum(f["n"] for f in families)
    return sum(f["accuracy"] * f["n"] for f in families) / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("LABEL", "GOLD", "PYTHON", "RUST"),
        required=True,
        help="repeatable: a label plus the gold, same-device Python, and Rust files",
    )
    parser.add_argument("--out-dir", type=Path, default=REPO / "semif-rs" / "fingerprints")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    header = (
        "| case | rows | python acc | rust acc | Δacc | python bal-acc | rust bal-acc | "
        "Δbal-acc | decisions agree |"
    )
    print(header)
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = []
    for label, gold, python_file, rust_file in args.case:
        gold_path, py_path, rs_path = Path(gold), Path(python_file), Path(rust_file)
        for path in (gold_path, py_path, rs_path):
            if not path.is_file():
                raise SystemExit(f"{label}: missing {path}")
        safe = label.replace("/", "-")
        py_report = grade(gold_path, py_path, args.out_dir / f"{safe}-python.json")
        rs_report = grade(gold_path, rs_path, args.out_dir / f"{safe}-rust.json")
        agree, total, flips = decision_agreement(py_path, rs_path)
        rows = total
        print(
            f"| {label} | {rows} "
            f"| {py_report['accuracy']:.4f} | {rs_report['accuracy']:.4f} "
            f"| {rs_report['accuracy'] - py_report['accuracy']:+.4f} "
            f"| {py_report['mean_family_balanced_accuracy']:.4f} "
            f"| {rs_report['mean_family_balanced_accuracy']:.4f} "
            f"| {rs_report['mean_family_balanced_accuracy'] - py_report['mean_family_balanced_accuracy']:+.4f} "
            f"| {agree}/{total} ({agree / total:.4f}) |"
        )
        summary.append(
            {
                "case": label,
                "rows": rows,
                "python": py_report,
                "rust": rs_report,
                "decision_agreement": agree / total,
                "decision_flips": flips,
            }
        )
    (args.out_dir / "fingerprint.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"\nreports: {args.out_dir}")


if __name__ == "__main__":
    main()
