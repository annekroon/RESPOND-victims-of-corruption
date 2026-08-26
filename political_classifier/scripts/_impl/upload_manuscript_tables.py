"""Upload generated manuscript LaTeX tables to Research Drive/WebDAV.

Run numbered step 07 first so the local .tex files are current.

Example:
    python3 political_classifier/scripts/08_upload_outputs.py --skip-attention-outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import sys
from pathlib import Path

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR


DEFAULT_LOCAL_TABLE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline/manuscript_tables"
)
DEFAULT_BUILD_MANIFEST = DEFAULT_LOCAL_TABLE_DIR / "manuscript_output_manifest.json"
DEFAULT_RD_TABLE_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "output",
    "tables",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload generated manuscript LaTeX tables to Research Drive/WebDAV."
    )
    parser.add_argument(
        "--local-table-dir",
        type=Path,
        default=DEFAULT_LOCAL_TABLE_DIR,
        help="Local directory containing generated .tex tables.",
    )
    parser.add_argument(
        "--rd-table-dir",
        default=DEFAULT_RD_TABLE_DIR,
        help="Research Drive destination directory.",
    )
    parser.add_argument(
        "--pattern",
        default="table_pc_classifier*.tex",
        help=(
            "Glob pattern of files to upload from local-table-dir. The default "
            "uploads only political-corruption classifier tables and avoids older "
            "generic table_classifier*.tex outputs."
        ),
    )
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=DEFAULT_BUILD_MANIFEST,
        help="Fresh step-07 build manifest required before upload.",
    )
    return parser.parse_args()


def rd_join(*parts: str) -> str:
    clean = [p.strip("/ ") for p in parts if p]
    return posixpath.join(*clean)


def rd_parent(path: str) -> str:
    return posixpath.dirname(path.rstrip("/"))


def upload_bytes(rd_path: str, data: bytes, content_type: str) -> None:
    from rd_utils import webdav_mkdirs, webdav_upload_bytes

    webdav_mkdirs(rd_parent(rd_path))
    webdav_upload_bytes(rd_path, data, content_type)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()

    if not args.build_manifest.exists():
        raise FileNotFoundError(
            f"Build manifest not found: {args.build_manifest}. "
            "Refusing to upload possibly stale tables. Run "
            "political_classifier/scripts/07_build_attention_outputs.py first."
        )
    manifest = json.loads(args.build_manifest.read_text(encoding="utf-8"))
    artifact_hashes = {
        record["relative_path"]: record["sha256"]
        for record in manifest.get("artifacts", [])
    }
    print(
        "Validated step-07 build manifest: "
        f"N={manifest['final_political_corruption_articles']:,}, "
        f"threshold={manifest['selected_threshold']:.2f}, "
        f"built={manifest['built_at_utc']}",
        flush=True,
    )

    if not args.local_table_dir.exists():
        raise FileNotFoundError(
            f"Local table directory does not exist: {args.local_table_dir}. "
            "Run political_classifier/scripts/07_build_attention_outputs.py first."
        )

    table_paths = sorted(args.local_table_dir.glob(args.pattern))
    if not table_paths:
        raise FileNotFoundError(
            f"No files matching {args.pattern!r} found in {args.local_table_dir}."
        )

    print(f"Local table directory: {args.local_table_dir}", flush=True)
    print(f"Research Drive target: {args.rd_table_dir}", flush=True)

    for path in table_paths:
        artifact_key = f"manuscript_tables/{path.name}"
        if artifact_hashes.get(artifact_key) != sha256_file(path):
            raise ValueError(
                f"Table differs from the step-07 build manifest: {path}. "
                "Rerun 07_build_attention_outputs.py before upload."
            )
        rd_path = rd_join(args.rd_table_dir, path.name)
        upload_bytes(rd_path, path.read_bytes(), "text/plain; charset=utf-8")
        print(f"Uploaded {path.name} -> {rd_path}", flush=True)

    manifest_rd_path = rd_join(args.rd_table_dir, args.build_manifest.name)
    upload_bytes(
        manifest_rd_path,
        args.build_manifest.read_bytes(),
        "application/json",
    )
    print(
        f"Uploaded {args.build_manifest.name} -> {manifest_rd_path}",
        flush=True,
    )

    print(f"Done. Uploaded {len(table_paths) + 1} file(s).", flush=True)


if __name__ == "__main__":
    main()
