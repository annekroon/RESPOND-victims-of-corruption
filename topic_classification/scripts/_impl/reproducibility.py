"""Small helpers for writing reproducibility metadata.

The topic workflow is partly stochastic and partly dependent on external
models. These helpers do not make those dependencies immutable, but they do
record enough context to audit a run and to recreate the software/input state
as closely as possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    try:
        from importlib.metadata import distributions
    except Exception:
        return packages

    for distribution in distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name:
            packages[name.lower()] = version
    return dict(sorted(packages.items()))


def args_to_dict(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, (list, tuple)):
            out[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            out[key] = value
    return out


def write_run_manifest(
    output_dir: Path,
    *,
    script_name: str,
    args: argparse.Namespace | dict[str, Any],
    inputs: dict[str, Path] | None = None,
    outputs: dict[str, Path] | None = None,
    extra: dict[str, Any] | None = None,
    manifest_name: str = "run_manifest.json",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]

    input_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in (inputs or {}).items()
    }
    output_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in (outputs or {}).items()
    }

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": script_name,
        "command": " ".join(sys.argv),
        "args": args_to_dict(args) if isinstance(args, argparse.Namespace) else args,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], cwd=project_root),
        "git_status_short": command_output(["git", "status", "--short"], cwd=project_root),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "environment": {
            key: os.environ.get(key, "")
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "TMPDIR",
            ]
        },
        "installed_packages": installed_packages(),
        "inputs": input_hashes,
        "outputs": output_hashes,
        "extra": extra or {},
    }

    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path
