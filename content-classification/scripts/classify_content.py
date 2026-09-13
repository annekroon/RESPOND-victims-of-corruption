"""Run one of the four canonical corruption-content classifiers."""

from __future__ import annotations

import argparse

from content_classifier_common import run_classifier
from content_prompts import CONTENT_CLASSIFIERS


def parse_variable(argv: list[str] | None = None) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--variable",
        choices=list(CONTENT_CLASSIFIERS),
        required=True,
        help="Substantive content variable to classify.",
    )
    selected, remaining = parser.parse_known_args(argv)
    return selected.variable, remaining


def main(argv: list[str] | None = None) -> None:
    variable, classifier_args = parse_variable(argv)
    run_classifier(CONTENT_CLASSIFIERS[variable], classifier_args)


if __name__ == "__main__":
    main()
