"""Classify whether political-corruption cases are domestic or abroad."""

from __future__ import annotations

from content_classifier_common import run_classifier
from content_prompts import ABROAD_CASE


if __name__ == "__main__":
    run_classifier(ABROAD_CASE)
