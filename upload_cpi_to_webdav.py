"""Upload extracted CPI country-year scores to Research Drive/WebDAV.

Run this after `extract_cpi_from_pdfs.py` has created the tidy CPI CSV and
extraction log.

Example
-------
    python3 upload_cpi_to_webdav.py
"""

from __future__ import annotations

import argparse
import mimetypes
import posixpath
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR


DEFAULT_OUTPUT = Path("output/cpi_country_year_scores.csv")
DEFAULT_LOG = Path("output/cpi_country_year_scores_extraction_log.csv")
DEFAULT_RD_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "cpi",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload extracted CPI CSV outputs to Research Drive/WebDAV."
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Local CPI country-year scores CSV.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Local CPI extraction log CSV.",
    )
    parser.add_argument(
        "--rd-dir",
        default=DEFAULT_RD_DIR,
        help="Research Drive destination directory.",
    )
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="Upload only the scores CSV and skip the extraction log.",
    )
    return parser.parse_args()


def rd_join(*parts: str) -> str:
    clean = [part.strip("/ ") for part in parts if part]
    return posixpath.join(*clean)


def rd_parent(path: str) -> str:
    return posixpath.dirname(path.rstrip("/"))


def upload_file(local_path: Path, rd_path: str) -> None:
    from rd_utils import webdav_mkdirs, webdav_upload_bytes

    if not local_path.exists():
        raise FileNotFoundError(local_path)

    content_type = mimetypes.guess_type(local_path.name)[0] or "text/csv"
    if local_path.suffix.lower() == ".csv":
        content_type = "text/csv; charset=utf-8"

    webdav_mkdirs(rd_parent(rd_path))
    webdav_upload_bytes(rd_path, local_path.read_bytes(), content_type)
    print(f"Uploaded {local_path} -> {rd_path}", flush=True)


def main() -> None:
    args = parse_args()

    files = [args.scores]
    if not args.scores_only:
        files.append(args.log)

    print(f"Research Drive target: {args.rd_dir}", flush=True)
    for local_path in files:
        upload_file(local_path, rd_join(args.rd_dir, local_path.name))

    print(f"Done. Uploaded {len(files)} file(s).", flush=True)


if __name__ == "__main__":
    main()
