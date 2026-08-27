"""Rebuild attention outputs and political-classifier manuscript artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild political-corruption attention and manuscript outputs."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Pipeline output directory read by the manuscript-output generator.",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Regenerate manuscript artifacts from existing attention CSVs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_manifest = (
        args.pipeline_dir
        / "manuscript_tables"
        / "manuscript_output_manifest.json"
    )
    build_manifest.unlink(missing_ok=True)
    if not args.skip_analysis:
        for generated_dir in [
            args.pipeline_dir / "attention_tables",
            args.pipeline_dir / "attention_figures",
        ]:
            if generated_dir.exists():
                shutil.rmtree(generated_dir)
                print(f"Removed stale generated directory: {generated_dir}", flush=True)
        command = [
            sys.executable,
            str(
                PROJECT_ROOT
                / "political_classifier"
                / "scripts"
                / "_impl"
                / "build_attention_analysis.py"
            ),
            "--pipeline-dir",
            str(args.pipeline_dir),
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
