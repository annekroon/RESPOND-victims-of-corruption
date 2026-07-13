"""Classify individualized/systemic corruption frames."""

from __future__ import annotations

from content_classifier_common import run_classifier
from content_prompts import CORRUPTION_FRAME


if __name__ == "__main__":
    run_classifier(CORRUPTION_FRAME)
