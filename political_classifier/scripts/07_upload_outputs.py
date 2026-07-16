"""Upload manuscript tables and attention outputs to Research Drive."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload political-classifier manuscript outputs.")
    parser.add_argument("--skip-manuscript-tables", action="store_true")
    parser.add_argument("--skip-attention-outputs", action="store_true")
    return parser.parse_args()


def run_script(script_name: str) -> None:
    command = [sys.executable, str(SCRIPT_DIR / script_name)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if not args.skip_manuscript_tables:
        run_script("upload_manuscript_tables.py")
    if not args.skip_attention_outputs:
        run_script("upload_attention_outputs.py")


if __name__ == "__main__":
    main()
