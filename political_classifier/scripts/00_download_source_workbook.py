"""Download the reviewed source-inclusion workbook from Research Drive.

This avoids relying on the mounted WebDAV folder, which can raise I/O errors on
large remote drives. The final source filter keeps only outlets where the
workbook column ``conventional_journalism`` is ``Yes``.
"""

from __future__ import annotations

import argparse
import posixpath
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR
from political_classifier.source_filter import DEFAULT_SOURCE_DECISION_FILE


DEFAULT_RD_WORKBOOK_PATH = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "political_corruption_all_sources_classified.xlsx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download political_corruption_all_sources_classified.xlsx from Research Drive."
    )
    parser.add_argument(
        "--rd-path",
        default=DEFAULT_RD_WORKBOOK_PATH,
        help="Research Drive path to the reviewed workbook.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SOURCE_DECISION_FILE,
        help="Local destination path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the local workbook if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from rd_utils import webdav_download_bytes

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.output.exists() and not args.overwrite:
        print(f"Already exists: {args.output}", flush=True)
        print("Use --overwrite to download again.", flush=True)
        return

    print(f"Downloading: {args.rd_path}", flush=True)
    data = webdav_download_bytes(args.rd_path)
    args.output.write_bytes(data)

    print(f"Saved: {args.output}", flush=True)
    print(f"Size:  {args.output.stat().st_size / 1024 / 1024:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
