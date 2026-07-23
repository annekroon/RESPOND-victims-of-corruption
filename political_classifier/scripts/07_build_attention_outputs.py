"""Rebuild attention outputs and all political-classifier manuscript artifacts.

The tracked notebook remains an optional inspection layer. This numbered script
executes a clean copy outside the repository and then regenerates classifier
tables, the corpus-construction table, and Figure 1 from saved pipeline outputs.
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
DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild political-corruption attention and manuscript outputs."
    )
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXECUTED_NOTEBOOK,
        help="Executed notebook output path. Defaults to the pipeline output folder, not the tracked notebook.",
    )
    parser.add_argument("--timeout", type=int, default=-1, help="Notebook execution timeout in seconds.")
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Pipeline output directory read by the manuscript-output generator.",
    )
    parser.add_argument(
        "--skip-notebook",
        action="store_true",
        help="Regenerate manuscript tables/Figure 1 from existing attention CSVs without rerunning the notebook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_notebook:
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

    manuscript_command = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "political_classifier"
            / "scripts"
            / "_impl"
            / "build_manuscript_outputs.py"
        ),
        "--pipeline-dir",
        str(args.pipeline_dir),
    ]
    print("Running:", " ".join(manuscript_command), flush=True)
    subprocess.run(manuscript_command, check=True)


if __name__ == "__main__":
    main()
