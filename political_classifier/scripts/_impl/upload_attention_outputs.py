"""Upload generated political-corruption attention outputs to Research Drive.

Run numbered step 07 first so the local attention figures and tables are
current.

Example:
    python3 political_classifier/scripts/08_upload_outputs.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import posixpath
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_LOCAL_FIGURE_DIR = DEFAULT_PIPELINE_DIR / "attention_figures"
DEFAULT_LOCAL_TABLE_DIR = DEFAULT_PIPELINE_DIR / "attention_tables"
DEFAULT_BUILD_MANIFEST = (
    DEFAULT_PIPELINE_DIR
    / "manuscript_tables"
    / "manuscript_output_manifest.json"
)
DEFAULT_RD_OUTPUT_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "output",
)
DEFAULT_FIGURE_PATTERNS = ["*.png", "*.pdf", "*.svg"]
DEFAULT_TABLE_PATTERNS = ["*.csv", "latex/*.tex"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload generated attention figures and tables to Research Drive/WebDAV."
    )
    parser.add_argument(
        "--local-figure-dir",
        type=Path,
        default=DEFAULT_LOCAL_FIGURE_DIR,
        help="Local directory containing generated attention figures.",
    )
    parser.add_argument(
        "--local-table-dir",
        type=Path,
        default=DEFAULT_LOCAL_TABLE_DIR,
        help="Local directory containing generated attention tables.",
    )
    parser.add_argument(
        "--rd-output-dir",
        default=DEFAULT_RD_OUTPUT_DIR,
        help="Research Drive base output directory.",
    )
    parser.add_argument(
        "--build-manifest",
        type=Path,
        default=DEFAULT_BUILD_MANIFEST,
        help="Fresh step-07 build manifest required before upload.",
    )
    parser.add_argument(
        "--rd-figure-subdir",
        default="figures/attention",
        help="Subdirectory under rd-output-dir for attention figures.",
    )
    parser.add_argument(
        "--rd-table-subdir",
        default="tables/attention",
        help="Subdirectory under rd-output-dir for attention tables.",
    )
    parser.add_argument(
        "--figure-patterns",
        nargs="+",
        default=DEFAULT_FIGURE_PATTERNS,
        help="Glob patterns to upload from local-figure-dir.",
    )
    parser.add_argument(
        "--table-patterns",
        nargs="+",
        default=DEFAULT_TABLE_PATTERNS,
        help="Glob patterns to upload from local-table-dir.",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Upload figures only.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Upload tables only.",
    )
    return parser.parse_args()


def rd_join(*parts: str) -> str:
    clean = [p.strip("/ ") for p in parts if p]
    return posixpath.join(*clean)


def rd_parent(path: str) -> str:
    return posixpath.dirname(path.rstrip("/"))


def collect_paths(directory: Path, patterns: list[str]) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Local directory does not exist: {directory}")

    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(directory.glob(pattern))
    return sorted({path for path in paths if path.is_file()})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_hashes(
    paths: list[Path], local_base_dir: Path, prefix: str, manifest: dict
) -> None:
    expected = {
        record["relative_path"]: record["sha256"]
        for record in manifest.get("artifacts", [])
    }
    for path in paths:
        key = f"{prefix}/{path.relative_to(local_base_dir).as_posix()}"
        if expected.get(key) != sha256_file(path):
            raise ValueError(
                f"Artifact differs from the step-07 build manifest: {path}. "
                "Rerun 07_build_attention_outputs.py before upload."
            )


def remote_relative_path(path: Path, local_base_dir: Path) -> PurePosixPath:
    """Flatten the internal latex/ directory in the published output tree."""
    relative = PurePosixPath(path.relative_to(local_base_dir).as_posix())
    if relative.parts and relative.parts[0] == "latex":
        relative = PurePosixPath(*relative.parts[1:])
    return relative


def upload_file(path: Path, local_base_dir: Path, rd_dir: str) -> str:
    from rd_utils import webdav_mkdirs, webdav_upload_bytes

    relative_path = remote_relative_path(path, local_base_dir)
    rd_path = rd_join(rd_dir, relative_path.as_posix())
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    webdav_mkdirs(rd_parent(rd_path))
    webdav_upload_bytes(rd_path, path.read_bytes(), content_type)
    return rd_path


def upload_group(label: str, paths: list[Path], local_base_dir: Path, rd_dir: str) -> int:
    if not paths:
        print(f"No {label} found to upload.", flush=True)
        return 0

    print(f"\nUploading {len(paths)} {label} to: {rd_dir}", flush=True)
    for path in paths:
        rd_path = upload_file(path, local_base_dir, rd_dir)
        print(f"Uploaded {path.name} -> {rd_path}", flush=True)
    return len(paths)


def validate_current_build(
    manifest_path: Path,
    local_table_dir: Path,
) -> dict[str, object]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Build manifest not found: {manifest_path}. "
            "Refusing to upload possibly stale attention outputs. Run "
            "political_classifier/scripts/07_build_attention_outputs.py first."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_n = int(manifest["final_political_corruption_articles"])
    threshold = float(manifest["selected_threshold"])
    tikz_path = (
        local_table_dir
        / "latex"
        / "figure_political_corruption_data_pipeline_tikz.tex"
    )
    if not tikz_path.exists():
        raise FileNotFoundError(
            f"Figure 1 TikZ output not found: {tikz_path}. "
            "Run political_classifier/scripts/07_build_attention_outputs.py first."
        )

    tikz = tikz_path.read_text(encoding="utf-8")
    required_fragments = [
        "Cleaning and exact deduplication",
        "Source exclusions removed",
        "Explicit outlet exclusions:",
        f"final $N = {final_n:,}$",
        f"$p \\geq {threshold:.2f}$",
    ]
    missing = [fragment for fragment in required_fragments if fragment not in tikz]
    if missing:
        raise ValueError(
            "Figure 1 does not match the current step-07 build manifest. "
            f"Missing expected content: {missing}. Rerun "
            "political_classifier/scripts/07_build_attention_outputs.py."
        )

    print(
        "Validated current attention build: "
        f"N={final_n:,}, threshold={threshold:.2f}, "
        f"built={manifest['built_at_utc']}",
        flush=True,
    )
    return manifest


def main() -> None:
    args = parse_args()

    if args.skip_tables and args.skip_figures:
        raise ValueError("Both --skip-tables and --skip-figures were set; nothing to upload.")

    manifest = validate_current_build(args.build_manifest, args.local_table_dir)

    total = 0

    if not args.skip_figures:
        figure_paths = collect_paths(args.local_figure_dir, args.figure_patterns)
        validate_artifact_hashes(
            figure_paths,
            args.local_figure_dir,
            "attention_figures",
            manifest,
        )
        figure_rd_dir = rd_join(args.rd_output_dir, args.rd_figure_subdir)
        total += upload_group(
            "attention figure(s)",
            figure_paths,
            args.local_figure_dir,
            figure_rd_dir,
        )

    if not args.skip_tables:
        table_paths = collect_paths(args.local_table_dir, args.table_patterns)
        validate_artifact_hashes(
            table_paths,
            args.local_table_dir,
            "attention_tables",
            manifest,
        )
        table_rd_dir = rd_join(args.rd_output_dir, args.rd_table_subdir)
        total += upload_group(
            "attention table(s)",
            table_paths,
            args.local_table_dir,
            table_rd_dir,
        )

    print(f"\nDone. Uploaded {total} file(s).", flush=True)


if __name__ == "__main__":
    main()
