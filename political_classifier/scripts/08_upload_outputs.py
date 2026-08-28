"""Upload manuscript tables and attention outputs to Research Drive."""

from __future__ import annotations

import argparse
import json
import posixpath
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMPL_DIR = SCRIPT_DIR / "_impl"
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_RD_OUTPUT_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "output",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload political-classifier manuscript outputs.")
    parser.add_argument("--skip-manuscript-tables", action="store_true")
    parser.add_argument("--skip-attention-outputs", action="store_true")
    parser.add_argument("--skip-manuscript-fragments", action="store_true")
    return parser.parse_args()


def run_script(script_name: str) -> None:
    command = [sys.executable, str(IMPL_DIR / script_name)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def upload_latest_navigation() -> None:
    from rd_utils import webdav_mkdirs, webdav_upload_bytes

    manifest = (
        DEFAULT_PIPELINE_DIR
        / "manuscript_tables"
        / "manuscript_output_manifest.json"
    )
    latest = (
        DEFAULT_PIPELINE_DIR
        / "manuscript_tables"
        / "00_LATEST_MANUSCRIPT_BUILD.txt"
    )
    if not manifest.exists() or not latest.exists():
        raise FileNotFoundError(
            "Current manuscript manifest/index is missing. Run step 07 first."
        )
    json.loads(manifest.read_text(encoding="utf-8"))

    uploads = {
        PROJECT_ROOT / "docs" / "method.tex": "manuscript/method.tex",
        PROJECT_ROOT / "docs" / "results_political_corruption_attention.tex": (
            "manuscript/results_political_corruption_attention.tex"
        ),
        PROJECT_ROOT / "docs" / "appendix_political_corruption.tex": (
            "manuscript/appendix_political_corruption.tex"
        ),
        latest: latest.name,
        manifest: manifest.name,
    }
    for local_path, relative_remote in uploads.items():
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        remote_path = posixpath.join(DEFAULT_RD_OUTPUT_DIR, relative_remote)
        webdav_mkdirs(posixpath.dirname(remote_path))
        content_type = (
            "application/json"
            if local_path.suffix == ".json"
            else "text/plain; charset=utf-8"
        )
        webdav_upload_bytes(remote_path, local_path.read_bytes(), content_type)
        print(f"Uploaded {local_path.name} -> {remote_path}", flush=True)


def main() -> None:
    args = parse_args()
    if not args.skip_manuscript_tables:
        run_script("upload_manuscript_tables.py")
    if not args.skip_attention_outputs:
        run_script("upload_attention_outputs.py")
    if not args.skip_manuscript_fragments:
        upload_latest_navigation()


if __name__ == "__main__":
    main()
