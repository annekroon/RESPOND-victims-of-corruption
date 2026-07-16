"""Execute the attention notebook to rebuild attention CSVs, figures, and LaTeX.

The notebook remains the interactive inspection layer, but this script provides
a reproducible command-line entry point.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = PROJECT_ROOT / "political_classifier" / "notebooks" / "03_analyze_political_corruption_attention.ipynb"
DEFAULT_EXECUTED_NOTEBOOK = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline/executed_notebooks/"
    "03_analyze_political_corruption_attention_executed.ipynb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the political-corruption attention notebook.")
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXECUTED_NOTEBOOK,
        help="Executed notebook output path. Defaults to the pipeline output folder, not the tracked notebook.",
    )
    parser.add_argument("--timeout", type=int, default=-1, help="Notebook execution timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(args.notebook),
        "--output",
        str(output.name),
        "--output-dir",
        str(output.parent),
        "--ExecutePreprocessor.timeout",
        str(args.timeout),
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
