"""Apply the source-inclusion scheme to already classified country files.

This avoids rerunning the expensive embedding classifier when the only change is
the source inclusion/exclusion scheme. The filter is strict: keep only
country/source pairs whose ``conventional_journalism`` annotation says Yes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.source_filter import (
    DEFAULT_PIPELINE_DIR,
    DEFAULT_SOURCE_DECISION_FILE,
    apply_source_inclusion_filter,
    load_source_decisions,
    source_filter_summary,
)

DEFAULT_INPUT_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files"
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files_source_filtered"
DEFAULT_SUMMARY_DIR = DEFAULT_PIPELINE_DIR / "source_inclusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter classified political-corruption outputs to conventional-journalism sources only."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--source-decision-file", type=Path, default=DEFAULT_SOURCE_DECISION_FILE)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Write only rows classified as political corruption.",
    )
    return parser.parse_args()


def main() -> None:
    import pandas as pd

    args = parse_args()
    decisions = load_source_decisions(args.source_decision_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)

    summary_frames = []
    output_summaries = []

    for path in sorted(args.input_dir.glob("*_classified.csv.gz")):
        country = path.name.replace("_classified.csv.gz", "")
        print(f"Filtering {country}: {path}", flush=True)
        data = pd.read_csv(path)
        if args.positive_only and "pred_political_corruption" in data.columns:
            data = data[data["pred_political_corruption"].eq(1)].copy()

        filtered, merged = apply_source_inclusion_filter(data, decisions, country_column="country")
        report = source_filter_summary(merged, group_columns=["country"])
        report["input_file"] = path.name
        summary_frames.append(report)

        output_path = args.output_dir / path.name
        filtered.to_csv(output_path, index=False, compression="gzip")
        output_summaries.append(
            {
                "country": country,
                "input_rows": len(data),
                "output_rows": len(filtered),
                "excluded_rows": len(data) - len(filtered),
                "output_path": str(output_path),
            }
        )
        print(f"  retained {len(filtered):,} / {len(data):,} rows -> {output_path}", flush=True)

    if not output_summaries:
        raise FileNotFoundError(f"No classified files found in {args.input_dir}")

    pd.DataFrame(output_summaries).to_csv(
        args.summary_dir / "classified_source_filter_output_summary.csv",
        index=False,
    )
    pd.concat(summary_frames, ignore_index=True).to_csv(
        args.summary_dir / "classified_source_filter_decision_summary_by_country.csv",
        index=False,
    )
    print(f"Saved summaries to: {args.summary_dir}", flush=True)


if __name__ == "__main__":
    main()
