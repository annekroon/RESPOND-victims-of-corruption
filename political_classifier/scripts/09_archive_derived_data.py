"""Numbered pipeline wrapper for archiving derived data to Research Drive."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("_impl") / "archive_derived_data_to_webdav.py"),
        run_name="__main__",
    )
