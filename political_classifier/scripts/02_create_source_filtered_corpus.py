"""Create the full cleaned/deduplicated/source-filtered corpus.

This script starts from the existing ``*_cleaned_deduped.csv.gz`` files,
retains only country/source pairs where the reviewed workbook has
``conventional_journalism == Yes``, and writes a full source-filtered corpus.

It does not sample rows. Sampling for classifier training happens in
``03_prepare_classifier_training_sample.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.source_filter import (
    DEFAULT_PIPELINE_DIR,
    DEFAULT_SOURCE_DECISION_FILE,
    apply_source_inclusion_filter,
    load_source_decisions,
    source_filter_summary,
)


DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "cleaned_deduped_source_filtered"
DEFAULT_SUMMARY_DIR = DEFAULT_PIPELINE_DIR / "source_inclusion"
DEFAULT_COUNTRIES = [
    "Bulgaria",
    "France",
    "Hungary",
    "Italy",
    "Netherlands",
    "Serbia",
    "Sweden",
    "Ukraine",
    "United_Kingdom",
]
MINIMAL_COLUMNS = [
    "uri",
    "country",
    "dateTime",
    "date_parsed",
    "year",
    "month",
    "week",
    "source_uri",
    "word_count",
    "text_hash",
    "near_dup_hash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full cleaned/deduplicated/source-filtered country files."
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--source-decision-file", type=Path, default=DEFAULT_SOURCE_DECISION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument(
        "--write-combined-minimal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write all_countries_cleaned_deduped_source_filtered_minimal.csv.gz.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing source-filtered country files.",
    )
    return parser.parse_args()


def cleaned_country_path(pipeline_dir: Path, country: str) -> Path:
    return pipeline_dir / f"{country}_cleaned_deduped.csv.gz"


def output_country_path(output_dir: Path, country: str) -> Path:
    return output_dir / f"{country}_cleaned_deduped_source_filtered.csv.gz"


def main() -> None:
    args = parse_args()

    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    decisions = load_source_decisions(args.source_decision_file)

    output_rows = []
    filter_summaries = []
    combined_minimal_parts = []

    for country in args.countries:
        input_path = cleaned_country_path(args.pipeline_dir, country)
        output_path = output_country_path(args.output_dir, country)

        if not input_path.exists():
            print(f"Skipping missing cleaned file for {country}: {input_path}", flush=True)
            continue
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

        print(f"\nFiltering {country}: {input_path}", flush=True)
        data = pd.read_csv(input_path)
        data["country"] = country
        if "source_uri" not in data.columns and "source.uri" in data.columns:
            data["source_uri"] = data["source.uri"]

        filtered, merged = apply_source_inclusion_filter(data, decisions, country_column="country")
        filtered.to_csv(output_path, index=False, compression="gzip")

        summary = source_filter_summary(merged, group_columns=["country"])
        summary["input_file"] = input_path.name
        summary["output_file"] = output_path.name
        filter_summaries.append(summary)

        output_rows.append(
            {
                "country": country,
                "input_rows": len(data),
                "output_rows": len(filtered),
                "excluded_rows": len(data) - len(filtered),
                "included_share_pct": len(filtered) / len(data) * 100 if len(data) else 0,
                "output_path": str(output_path),
            }
        )
        print(f"{country}: retained {len(filtered):,} / {len(data):,} rows -> {output_path}", flush=True)

        if args.write_combined_minimal:
            minimal_cols = [column for column in MINIMAL_COLUMNS if column in filtered.columns]
            combined_minimal_parts.append(filtered[minimal_cols].copy())

        del data, filtered, merged

    if not output_rows:
        raise RuntimeError("No source-filtered country files were written.")

    output_summary = pd.DataFrame(output_rows)
    output_summary_path = args.summary_dir / "cleaned_source_filter_output_summary.csv"
    output_summary.to_csv(output_summary_path, index=False)

    decision_summary_path = args.summary_dir / "cleaned_source_filter_decision_summary_by_country.csv"
    pd.concat(filter_summaries, ignore_index=True).to_csv(decision_summary_path, index=False)

    if args.write_combined_minimal and combined_minimal_parts:
        combined_output = args.output_dir / "all_countries_cleaned_deduped_source_filtered_minimal.csv.gz"
        combined_minimal = pd.concat(combined_minimal_parts, ignore_index=True)
        combined_minimal.to_csv(combined_output, index=False, compression="gzip")
        print(f"\nSaved minimal combined source-filtered file: {combined_output}", flush=True)

    print(f"Saved output summary:   {output_summary_path}", flush=True)
    print(f"Saved decision summary: {decision_summary_path}", flush=True)
    print("\nBy country:", flush=True)
    print(output_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
