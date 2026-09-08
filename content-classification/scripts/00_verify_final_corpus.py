"""Verify and summarize the final corpus used for content classification."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALL_COUNTRIES
from political_classifier.final_corpus import (
    load_verified_classifier_run,
    verify_classified_country_file,
)


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the source-filtered political-corruption corpus."
    )
    parser.add_argument(
        "--classifier-output-dir",
        type=Path,
        default=DEFAULT_PIPELINE_DIR / "silver_classifier",
    )
    parser.add_argument("--classified-dir", type=Path, default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Check manifests and totals without streaming the country files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classified_dir = (
        args.classified_dir
        or args.classifier_output_dir / "classified_country_files"
    )
    run = load_verified_classifier_run(
        args.classifier_output_dir,
        list(ALL_COUNTRIES),
        classified_dir=classified_dir,
        require_local_country_files=not args.manifest_only,
    )

    selected = run.summary[
        run.summary["country"].astype(str).isin(args.countries)
    ].copy()
    missing = set(args.countries) - set(selected["country"].astype(str))
    if missing:
        raise ValueError("Unknown countries: " + ", ".join(sorted(missing)))

    if not args.manifest_only:
        for country in args.countries:
            path = classified_dir / f"{country}_classified.csv.gz"
            verify_classified_country_file(
                path,
                country=country,
                classifier_run=run,
                chunksize=args.chunksize,
            )
            print(f"Verified {country}: {path}", flush=True)

    print("\nFinal content-classification input", flush=True)
    print(f"Threshold: {run.threshold:.2f}", flush=True)
    print(f"Source policy: {run.source_filter_manifest['source_filter_policy']}", flush=True)
    print(f"Source-filtered articles: {run.total_articles:,}", flush=True)
    print(f"Political-corruption articles: {run.political_articles:,}", flush=True)
    print("\nCountry counts:", flush=True)
    print(
        selected[
            [
                "country",
                "total_articles",
                "predicted_political_corruption",
                "predicted_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nClassifier manifest SHA-256: {run.manifest_sha256}", flush=True)
    print("SUCCESS: final corpus and provenance are internally consistent.", flush=True)


if __name__ == "__main__":
    main()
