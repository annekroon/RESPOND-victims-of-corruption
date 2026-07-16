"""Compatibility wrapper for the renamed classifier-training sample script.

Prefer:
    python3 political_classifier/scripts/02_prepare_classifier_training_sample.py
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "_impl" / "prepare_classifier_training_sample.py"),
        run_name="__main__",
    )
