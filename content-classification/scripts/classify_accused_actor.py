"""Classify accused-actor visibility in political-corruption articles."""

from __future__ import annotations

from content_classifier_common import run_classifier
from content_prompts import ACCUSED_ACTOR


if __name__ == "__main__":
    run_classifier(ACCUSED_ACTOR)
