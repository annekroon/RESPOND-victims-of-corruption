"""Package topic-model outputs for paper archiving and optionally upload to SURF.

The SURF web URL is a browser interface. For command-line upload, use the
corresponding WebDAV collection URL, for example:

    https://uva.data.surf.nl/remote.php/dav/files/<username>/ASCOR-FMG-5580-RESPOND-news-data%20%28Projectfolder%29/victims-of-corruption-paper

Set `SURF_WEBDAV_URL`, `SURF_USERNAME`, and `SURF_PASSWORD` to upload the
archive after it is created.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import mimetypes
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and optionally upload a topic-output archive for SURF.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=None, help="Exact stratified sample CSV used for BERTopic.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--archive-name", default=None)
    parser.add_argument("--no-model", action="store_true", help="Exclude the saved BERTopic model directory.")
    parser.add_argument("--upload", action="store_true", help="Upload the archive to SURF WebDAV.")
    parser.add_argument("--webdav-url", default=None, help="Destination SURF WebDAV collection URL.")
    parser.add_argument("--username", default=None, help="SURF username. Defaults to SURF_USERNAME.")
    parser.add_argument("--password", default=None, help="SURF app password/password. Defaults to SURF_PASSWORD.")
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


def upload_webdav(file_path: Path, webdav_url: str, username: str, password: str) -> None:
    parsed = urlparse(webdav_url.rstrip("/") + "/" + quote(file_path.name))
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS WebDAV URLs are supported.")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    connection = http.client.HTTPSConnection(parsed.netloc)
    headers = {
        "Authorization": f"Basic {token}",
        "Content-Type": mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
        "Content-Length": str(file_path.stat().st_size),
    }
    path = parsed.path
    if parsed.query:
        path += f"?{parsed.query}"
    connection.putrequest("PUT", path)
    for key, value in headers.items():
        connection.putheader(key, value)
    connection.endheaders()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            connection.send(chunk)
    response = connection.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status not in {200, 201, 204}:
        raise RuntimeError(f"Unexpected WebDAV upload status: {response.status} {response.reason}: {body[:500]}")


def main() -> None:
    args = parse_args()
    import os

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
        webdav_url = args.webdav_url or os.environ.get("SURF_WEBDAV_URL", "")
        username = args.username or os.environ.get("SURF_USERNAME", "")
        password = args.password or os.environ.get("SURF_PASSWORD", "")
        if not webdav_url or not username or not password:
            raise RuntimeError("Upload requires --webdav-url/--username/--password or SURF_WEBDAV_URL/SURF_USERNAME/SURF_PASSWORD.")
        upload_webdav(archive_path, webdav_url, username, password)
        print(f"Uploaded archive to: {webdav_url.rstrip('/')}/{archive_path.name}", flush=True)


if __name__ == "__main__":
    main()
