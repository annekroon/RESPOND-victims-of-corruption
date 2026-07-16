"""Numbered pipeline wrapper for training/scoring the final classifier."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("_impl") / "train_final_classifier.py"),
        run_name="__main__",
    )
