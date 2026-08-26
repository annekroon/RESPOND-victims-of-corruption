"""Small provenance helpers shared by production classifier scripts."""

from __future__ import annotations

import hashlib
import subprocess
from importlib import metadata
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path) -> dict[str, object]:
    path = Path(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def git_commit(project_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "sentence-transformers",
        "transformers",
        "torch",
    ]
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def embedding_revision(embedder) -> str | None:
    """Return the Hugging Face commit hash recorded in a loaded model, if any."""
    for module in getattr(embedder, "_modules", {}).values():
        config = getattr(getattr(module, "auto_model", None), "config", None)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            return str(revision)
    return None


def frame_fingerprint(frame, columns: list[str]) -> str:
    """Hash selected columns in their current row order."""
    available = [column for column in columns if column in frame.columns]
    payload = frame[available].fillna("").astype(str).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
