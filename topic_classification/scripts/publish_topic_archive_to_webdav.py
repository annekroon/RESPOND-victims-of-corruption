"""Package topic-model outputs for paper archiving and upload to Research Drive.

This follows the same Research Drive/WebDAV convention as the political
classifier scripts: destination paths are relative to the Research Drive root,
and credentials are read from `config_local.py` or `RD_USER`/`RD_PASS`.

Default archive target:
    ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
      victims-of-corruption-paper/derived_data/topic_classification/

Default LaTeX table target:
    ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
      victims-of-corruption-paper/output/tables/topic models/
"""

from __future__ import annotations

import argparse
import mimetypes
import posixpath
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import RD_BASE_DIR


DEFAULT_RD_ARCHIVE_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "topic_classification",
)
DEFAULT_RD_TABLE_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "output",
    "tables",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and optionally upload a topic-output archive to Research Drive/WebDAV."
    )
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=None, help="Exact stratified sample CSV used for BERTopic.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--archive-name", default=None)
    parser.add_argument("--no-model", action="store_true", help="Exclude the saved BERTopic model directory.")
    parser.add_argument("--upload", action="store_true", help="Upload the archive to Research Drive/WebDAV.")
    parser.add_argument(
        "--upload-latex-tables",
        action="store_true",
        help="Upload generated LaTeX appendix tables to a Research Drive output/tables subfolder.",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only upload LaTeX tables; do not create the full publication archive.",
    )
    parser.add_argument(
        "--rd-archive-dir",
        default=DEFAULT_RD_ARCHIVE_DIR,
        help="Research Drive destination directory for the full archive.",
    )
    parser.add_argument(
        "--rd-table-dir",
        default=DEFAULT_RD_TABLE_DIR,
        help="Research Drive output/tables directory.",
    )
    parser.add_argument(
        "--tables-folder-name",
        default="topic models",
        help="Subfolder to create under rd-table-dir.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if src.exists() and src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)


def stage_archive_contents(args: argparse.Namespace, staging_dir: Path) -> None:
    bertopic_dir = args.bertopic_dir.resolve()
    if not bertopic_dir.exists():
        raise FileNotFoundError(bertopic_dir)

    # Code and documentation needed to understand/recreate the workflow.
    copy_file(PROJECT_ROOT / "topic_classification" / "README.md", staging_dir / "code" / "topic_classification" / "README.md")
    copy_file(
        PROJECT_ROOT / "topic_classification" / "requirements-topic.txt",
        staging_dir / "code" / "topic_classification" / "requirements-topic.txt",
    )
    copy_tree(
        PROJECT_ROOT / "topic_classification" / "scripts",
        staging_dir / "code" / "topic_classification" / "scripts",
    )
    copy_tree(
        PROJECT_ROOT / "topic_classification" / "notebooks",
        staging_dir / "code" / "topic_classification" / "notebooks",
    )

    if args.sample is not None:
        copy_file(args.sample.resolve(), staging_dir / "inputs" / args.sample.name)
        strata_path = args.sample.with_name(args.sample.name.replace(".csv.gz", "_strata.csv"))
        manifest_path = args.sample.with_name(args.sample.name.replace(".csv.gz", "_run_manifest.json"))
        copy_file(strata_path, staging_dir / "inputs" / strata_path.name)
        copy_file(manifest_path, staging_dir / "inputs" / manifest_path.name)

    output_files = [
        "topic_info.csv",
        "document_topics.csv.gz",
        "topic_labels_llm.csv",
        "topic_labels_llm_audit.jsonl",
        "topic_groups_llm.csv",
        "topic_groups_llm_audit.json",
        "topic_group_summaries_llm.csv",
        "run_manifest.json",
        "topic_labels_run_manifest.json",
        "topic_groups_run_manifest.json",
    ]
    for name in output_files:
        copy_file(bertopic_dir / name, staging_dir / "outputs" / name)

    copy_tree(bertopic_dir / "inspection_tables", staging_dir / "outputs" / "inspection_tables")
    copy_tree(bertopic_dir / "visualizations", staging_dir / "outputs" / "visualizations")
    if not args.no_model:
        copy_tree(bertopic_dir / "topic_model", staging_dir / "outputs" / "topic_model")

    readme = staging_dir / "README_ARCHIVE.txt"
    readme.write_text(
        "\n".join(
            [
                "RESPOND victims-of-corruption paper topic archive",
                f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
                f"BERTopic directory: {bertopic_dir}",
                "",
                "Key manuscript tables:",
                "- outputs/inspection_tables/latex/table_topic_coverage_frame_summary.tex",
                "- outputs/inspection_tables/latex/table_all_topics_llm_coverage_frames.tex",
                "",
                "The CSV/JSON outputs are the archival source of truth for published topic labels and LLM group assignments.",
                "The LLM audit files contain prompts and raw responses where available.",
            ]
        )
        + "\n"
    )


def make_tarball(staging_dir: Path, archive_path: Path) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(staging_dir, arcname=staging_dir.name)
    return archive_path


def rd_join(*parts: str) -> str:
    clean = [p.strip("/ ") for p in parts if p]
    return posixpath.join(*clean)


def rd_parent(path: str) -> str:
    return posixpath.dirname(path.rstrip("/"))


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


def upload_file_stream(session, base_url: str, local_path: Path, rd_path: str) -> None:
    from rd_utils import webdav_mkdirs

    webdav_mkdirs(rd_parent(rd_path))
    url = f"{base_url}/{enc_path(rd_path)}"
    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    with local_path.open("rb") as handle:
        response = session.put(url, data=handle, headers={"Content-Type": content_type})
    response.raise_for_status()


def upload_bytes(rd_path: str, data: bytes, content_type: str) -> None:
    from rd_utils import webdav_mkdirs, webdav_upload_bytes

    webdav_mkdirs(rd_parent(rd_path))
    webdav_upload_bytes(rd_path, data, content_type)


def latex_table_paths(bertopic_dir: Path) -> list[Path]:
    latex_dir = bertopic_dir / "inspection_tables" / "latex"
    paths = [
        latex_dir / "table_topic_coverage_frame_summary.tex",
        latex_dir / "table_all_topics_llm_coverage_frames.tex",
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing LaTeX table(s): " + ", ".join(str(path) for path in missing))
    return paths


def main() -> None:
    args = parse_args()

    if args.tables_only and not args.upload_latex_tables:
        raise RuntimeError("--tables-only requires --upload-latex-tables.")

    if not args.tables_only:
        output_root = args.output_root or (args.bertopic_dir / "publication_archive")
        archive_name = args.archive_name or f"topic_archive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.tar.gz"
        archive_path = output_root / archive_name

        with tempfile.TemporaryDirectory(prefix="topic_archive_") as tmpdir:
            staging_dir = Path(tmpdir) / archive_name.replace(".tar.gz", "")
            staging_dir.mkdir(parents=True)
            stage_archive_contents(args, staging_dir)
            make_tarball(staging_dir, archive_path)

        print(f"Created archive: {archive_path}", flush=True)

        if args.upload:
            session, base_url = get_webdav_session()
            rd_path = rd_join(args.rd_archive_dir, archive_path.name)
            upload_file_stream(session, base_url, archive_path, rd_path)
            print(f"Uploaded archive -> {rd_path}", flush=True)

    if args.upload_latex_tables:
        rd_table_dir = rd_join(args.rd_table_dir, args.tables_folder_name)
        for table_path in latex_table_paths(args.bertopic_dir):
            rd_path = rd_join(rd_table_dir, table_path.name)
            upload_bytes(rd_path, table_path.read_bytes(), "text/plain; charset=utf-8")
            print(f"Uploaded LaTeX table {table_path.name} -> {rd_path}", flush=True)


if __name__ == "__main__":
    main()
