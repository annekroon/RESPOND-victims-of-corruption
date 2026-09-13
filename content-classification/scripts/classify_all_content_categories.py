"""Run all LLM content-category coders on one validation/sample file."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from content_prompts import CONTENT_CLASSIFIERS

CLASSIFIER_SCRIPT = Path(__file__).resolve().with_name("classify_content.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all zero-shot LLM content coders on a CSV/CSV.GZ sample."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start fresh outputs for all four variables.",
    )
    parser.add_argument(
        "--keep-non-political",
        action="store_true",
        help="Pass through non-political rows if present. Usually not needed for codebook samples.",
    )
    return parser.parse_args()


def output_name(input_path: Path, classifier_name: str) -> str:
    stem = input_path.name
    if stem.endswith(".csv.gz"):
        stem = stem[:-7]
    elif stem.endswith(".csv"):
        stem = stem[:-4]
    return f"{stem}_{classifier_name}_gpt_labels.csv.gz"


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for variable, specification in CONTENT_CLASSIFIERS.items():
        output_path = args.output_dir / output_name(
            args.input,
            specification.name,
        )
        command = [
            sys.executable,
            str(CLASSIFIER_SCRIPT),
            "--variable",
            variable,
            "--source",
            "csv",
            "--input",
            str(args.input),
            "--output",
            str(output_path),
            "--max-chars",
            str(args.max_chars),
            "--save-every",
            str(args.save_every),
            "--sleep",
            str(args.sleep),
        ]
        if args.model:
            command.extend(["--model", args.model])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.retry_errors:
            command.append("--retry-errors")
        if args.overwrite:
            command.append("--overwrite")
        if args.keep_non_political:
            command.append("--keep-non-political")

        print("\n" + "=" * 80, flush=True)
        print(f"Running {variable}", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)

    print("\nDone. All content category coders finished.", flush=True)


if __name__ == "__main__":
    main()
