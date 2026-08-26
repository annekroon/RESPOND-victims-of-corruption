"""Restore archived political-corruption derived data from Research Drive/WebDAV.

Use this on a fresh machine, or after cleaning local outputs, to recreate the
expected local political_corruption_pipeline folder from the reproducibility
archive made by archive_derived_data_to_webdav.py.

Example:
    python3 political_classifier/scripts/restore_derived_data_from_webdav.py \
      --groups classifier_outputs attention_outputs

Default Research Drive source:
    ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
      victims-of-corruption-paper/derived_data/political_classifier/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import sys
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR
from rd_utils import (
    DEFAULT_TIMEOUT,
    _get_session,
    webdav_download_to_path,
    webdav_list,
)


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
ARCHIVE_GROUPS = [
    "cleaned_deduped",
    "silver_training_data",
    "validation_data",
    "source_inclusion",
    "classifier_outputs",
    "classifier_comparison",
    "attention_outputs",
]
GROUP_LOCAL_BASES = {
    "cleaned_deduped": Path("."),
    "silver_training_data": Path("active_learning"),
    "validation_data": Path("active_learning"),
    "source_inclusion": Path("source_inclusion"),
    "classifier_outputs": Path("silver_classifier"),
    "classifier_comparison": Path("classifier_comparison"),
    "attention_outputs": Path("."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore archived political-corruption derived data from Research Drive/WebDAV."
    )
    parser.add_argument(
        "--pipeline-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR,
        help="Local political_corruption_pipeline directory to restore into.",
    )
    parser.add_argument(
        "--rd-archive-dir",
        default=DEFAULT_RD_ARCHIVE_DIR,
        help="Research Drive archive source directory.",
    )
    parser.add_argument(
        "--archive-version",
        default=None,
        help=(
            "Snapshot folder below runs/. Omit to restore the lexicographically "
            "latest timestamped snapshot."
        ),
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=ARCHIVE_GROUPS,
        choices=ARCHIVE_GROUPS,
        help="Archive groups to restore.",
    )
    parser.add_argument(
        "--manifest-name",
        default="derived_data_manifest.json",
        help="Manifest file name in rd-archive-dir.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not download files that already exist locally.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification even if checksums are present in the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be downloaded without downloading them.",
    )
    return parser.parse_args()


def enc_path(rel_path: str) -> str:
    parts = [p for p in rel_path.strip("/").split("/") if p]
    return "/".join(quote(p, safe="") for p in parts)


def get_webdav_session():
    return _get_session()


def webdav_get_bytes(session, base_url: str, rd_path: str) -> bytes:
    url = f"{base_url}/{enc_path(rd_path)}"
    response = session.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    return response.content


def download_file_stream(session, base_url: str, rd_path: str, local_path: Path) -> None:
    del session, base_url
    webdav_download_to_path(rd_path, local_path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 8) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_target_path(pipeline_dir: Path, record: dict) -> Path:
    group = record["group"]
    relative_path = Path(record["relative_path"])
    local_base = GROUP_LOCAL_BASES[group]
    return pipeline_dir / local_base / relative_path


def main() -> None:
    args = parse_args()
    if args.archive_version is None:
        runs_dir = posixpath.join(args.rd_archive_dir, "runs")
        candidates = sorted(name for name in webdav_list(runs_dir) if name and not name.startswith("."))
        if not candidates:
            raise FileNotFoundError(f"No archived runs found in {runs_dir}.")
        args.archive_version = candidates[-1]
    archive_root = posixpath.join(args.rd_archive_dir, "runs", args.archive_version)
    session, base_url = get_webdav_session()
    manifest_rd_path = posixpath.join(archive_root, args.manifest_name)
    manifest = json.loads(webdav_get_bytes(session, base_url, manifest_rd_path).decode("utf-8"))

    records = [
        record
        for record in manifest.get("files", [])
        if record.get("group") in set(args.groups)
    ]
    if not records:
        raise FileNotFoundError(f"No manifest records found for groups: {args.groups}")

    print(f"Research Drive archive: {archive_root}", flush=True)
    print(f"Local pipeline target:  {args.pipeline_dir}", flush=True)
    print(f"Files selected:         {len(records):,}", flush=True)
    if args.dry_run:
        print("\nDry run; no files will be downloaded.", flush=True)

    for record in records:
        local_path = local_target_path(args.pipeline_dir, record)
        if args.skip_existing and local_path.exists():
            print(f"skip existing: {local_path}", flush=True)
            continue

        print(f"{record['group']}: {record['relative_path']} -> {local_path}", flush=True)
        if not args.dry_run:
            download_file_stream(session, base_url, record["rd_path"], local_path)
            expected_sha = record.get("sha256")
            if expected_sha and not args.no_verify:
                actual_sha = sha256_file(local_path)
                if actual_sha != expected_sha:
                    local_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"Checksum mismatch for {local_path}: "
                        f"expected {expected_sha}, got {actual_sha}"
                    )

    manifest_local_path = args.pipeline_dir / args.manifest_name
    if not args.dry_run:
        manifest_local_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_local_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"\nSaved local manifest: {manifest_local_path}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
