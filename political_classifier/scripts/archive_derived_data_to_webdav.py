"""Archive political-corruption derived data to Research Drive/WebDAV.

This script uploads the expensive-to-recreate derived outputs from Part 1 of
the RESPOND victims-of-corruption workflow. It is meant for reproducibility:
GitHub keeps the code and notebooks; Research Drive keeps the generated data.

Example:
    python3 political_classifier/scripts/archive_derived_data_to_webdav.py

Default Research Drive target:
    ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
      victims-of-corruption-paper/derived_data/political_classifier/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import platform
import posixpath
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import RD_BASE_DIR


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_RD_ARCHIVE_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "political_classifier",
)


@dataclass(frozen=True)
class ArchiveGroup:
    name: str
    source_dir: Path
    patterns: tuple[str, ...]
    destination_subdir: str


ARCHIVE_GROUPS = [
    ArchiveGroup(
        name="cleaned_deduped",
        source_dir=DEFAULT_PIPELINE_DIR,
        patterns=(
            "*_cleaned_deduped.csv.gz",
            "all_countries_cleaned_deduped_minimal.csv.gz",
            "denominator_country_year.csv",
            "denominator_country_month.csv",
            "denominator_country_week.csv",
            "denominator_country_year_source.csv",
        ),
        destination_subdir="cleaned_deduped",
    ),
    ArchiveGroup(
        name="silver_training_data",
        source_dir=DEFAULT_PIPELINE_DIR / "active_learning",
        patterns=(
            "*with_llm_suggestions.csv",
            "*for_annotation.csv",
        ),
        destination_subdir="silver_training_data",
    ),
    ArchiveGroup(
        name="validation_data",
        source_dir=DEFAULT_PIPELINE_DIR / "active_learning",
        patterns=(
            "uk_human_validation_reviewed.csv",
            "uk_human_validation_review_batch.csv",
        ),
        destination_subdir="validation_data",
    ),
    ArchiveGroup(
        name="classifier_outputs",
        source_dir=DEFAULT_PIPELINE_DIR / "silver_classifier",
        patterns=(
            "*.csv",
            "*.txt",
            "*.joblib",
            "classified_country_files/*.csv.gz",
        ),
        destination_subdir="classifier_outputs",
    ),
    ArchiveGroup(
        name="classifier_comparison",
        source_dir=DEFAULT_PIPELINE_DIR / "classifier_comparison",
        patterns=("*.csv",),
        destination_subdir="classifier_comparison",
    ),
    ArchiveGroup(
        name="attention_outputs",
        source_dir=DEFAULT_PIPELINE_DIR,
        patterns=(
            "attention_tables/*.csv",
            "attention_tables/latex/*.tex",
            "attention_figures/*.png",
            "attention_figures/*.pdf",
            "attention_figures/*.svg",
            "manuscript_tables/table_pc_classifier*.tex",
        ),
        destination_subdir="attention_outputs",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Archive political-corruption derived data to Research Drive/WebDAV."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Local political_corruption_pipeline directory.",
    )
    parser.add_argument(
        "--rd-archive-dir",
        default=DEFAULT_RD_ARCHIVE_DIR,
        help="Research Drive archive destination directory.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=[group.name for group in ARCHIVE_GROUPS],
        choices=[group.name for group in ARCHIVE_GROUPS],
        help="Archive groups to upload.",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Skip SHA-256 checksums in the manifest for a faster run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be uploaded without uploading them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files on Research Drive. PUT overwrites by default; this flag is documented for clarity.",
    )
    return parser.parse_args()


def rd_join(*parts: str) -> str:
    clean = [p.strip("/ ") for p in parts if p]
    return posixpath.join(*clean)


def enc_path(rel_path: str) -> str:
    parts = [p for p in rel_path.strip("/").split("/") if p]
    return "/".join(quote(p, safe="") for p in parts)


def get_webdav_session():
    import requests

    app_password = getattr(config, "APP_PASSWORD", None)
    user = getattr(config, "USER", None)
    base_url = getattr(config, "BASE_URL", None)

    missing = [
        name
        for name, value in {
            "BASE_URL or RD_BASE_URL": base_url,
            "USER or RD_USER": user,
            "APP_PASSWORD or RD_PASS": app_password,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Research Drive WebDAV setting(s): "
            + ", ".join(missing)
            + ". Add them to config_local.py or set environment variables."
        )

    session = requests.Session()
    session.auth = (user, app_password)
    base = f"{base_url.rstrip('/')}/{quote(user, safe='')}"
    return session, base


def collect_files(group: ArchiveGroup, pipeline_dir: Path) -> list[tuple[Path, Path]]:
    source_dir = pipeline_dir / group.source_dir.relative_to(DEFAULT_PIPELINE_DIR)
    if not source_dir.exists():
        print(f"Skipping missing source directory for {group.name}: {source_dir}", flush=True)
        return []

    files: list[tuple[Path, Path]] = []
    for pattern in group.patterns:
        for path in source_dir.glob(pattern):
            if path.is_file():
                files.append((path, path.relative_to(source_dir)))
    return sorted(set(files), key=lambda item: item[1].as_posix())


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def upload_file_stream(session, base_url: str, local_path: Path, rd_path: str) -> None:
    from rd_utils import webdav_mkdirs

    webdav_mkdirs(posixpath.dirname(rd_path.rstrip("/")))
    url = f"{base_url}/{enc_path(rd_path)}"
    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    with local_path.open("rb") as handle:
        response = session.put(url, data=handle, headers={"Content-Type": content_type})
    response.raise_for_status()


def manifest_header(args: argparse.Namespace) -> dict:
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project_root": str(PROJECT_ROOT),
        "pipeline_dir": str(args.pipeline_dir),
        "rd_archive_dir": args.rd_archive_dir,
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "checksum_algorithm": None if args.no_checksum else "sha256",
        "groups": args.groups,
    }


def main() -> None:
    args = parse_args()
    selected_groups = [group for group in ARCHIVE_GROUPS if group.name in args.groups]

    manifest = manifest_header(args)
    manifest["files"] = []

    planned: list[tuple[ArchiveGroup, Path, Path, str]] = []
    for group in selected_groups:
        for local_path, relative_path in collect_files(group, args.pipeline_dir):
            rd_path = rd_join(args.rd_archive_dir, group.destination_subdir, relative_path.as_posix())
            planned.append((group, local_path, relative_path, rd_path))

    if not planned:
        raise FileNotFoundError("No archive files found for the selected groups.")

    print(f"Local pipeline directory: {args.pipeline_dir}", flush=True)
    print(f"Research Drive archive:  {args.rd_archive_dir}", flush=True)
    print(f"Files selected:          {len(planned):,}", flush=True)
    if args.dry_run:
        print("\nDry run; no files will be uploaded.", flush=True)

    session = None
    base_url: str | None = None
    if not args.dry_run:
        session, base_url = get_webdav_session()

    for group, local_path, relative_path, rd_path in planned:
        size = local_path.stat().st_size
        checksum = None if args.no_checksum else sha256_file(local_path)
        record = {
            "group": group.name,
            "local_path": str(local_path),
            "relative_path": relative_path.as_posix(),
            "rd_path": rd_path,
            "size_bytes": size,
            "sha256": checksum,
        }
        manifest["files"].append(record)

        print(f"{group.name}: {relative_path} ({size / 1024 / 1024:.1f} MB)", flush=True)
        if not args.dry_run:
            assert session is not None and base_url is not None
            upload_file_stream(session, base_url, local_path, rd_path)
            print(f"  uploaded -> {rd_path}", flush=True)

    manifest_name = "derived_data_manifest.json"
    manifest_path = rd_join(args.rd_archive_dir, manifest_name)
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    local_manifest = args.pipeline_dir / manifest_name
    local_manifest.write_bytes(manifest_json)
    print(f"\nSaved local manifest: {local_manifest}", flush=True)

    if not args.dry_run:
        assert session is not None and base_url is not None
        temp_manifest_path = args.pipeline_dir / manifest_name
        upload_file_stream(session, base_url, temp_manifest_path, manifest_path)
        print(f"Uploaded manifest -> {manifest_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
