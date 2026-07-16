"""Numbered pipeline wrapper for creating the source-filtered silver seed."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("create_source_filtered_silver_seed.py")), run_name="__main__")
