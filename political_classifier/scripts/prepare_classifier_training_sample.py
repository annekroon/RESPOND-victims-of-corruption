"""Prepare a fresh source-filtered classifier training sample.

This is the restart path for the political-corruption classifier. It does not
use previous silver labels or a provisional classifier. Instead, it samples from
the cleaned corruption-query corpus after applying the source-inclusion scheme.

The output is intended for political_classifier/scripts/03_label_silver_batch.py.
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


DEFAULT_OUTPUT_PATH = (
    DEFAULT_PIPELINE_DIR
    / "active_learning"
    / "silver_training_source_filtered_for_annotation.csv"
)
DEFAULT_COUNTRY_TARGETS = {
    "Bulgaria": 500,
    "France": 500,
    "Hungary": 500,
    "Italy": 500,
    "Netherlands": 500,
    "Serbia": 500,
    "Sweden": 500,
    "Ukraine": 500,
    "United_Kingdom": 500,
}
OUTPUT_COLUMNS = [
    "al_bucket",
    "uri",
    "country",
    "date_parsed",
    "year",
    "month",
    "week",
    "source_uri",
    "word_count",
    "human_final_label",
    "human_notes",
    "article_text",
]


def parse_country_targets(text: str | None) -> dict[str, int]:
    if not text:
        return DEFAULT_COUNTRY_TARGETS.copy()

    targets = {}
    for item in text.split(","):
        country, count = item.split(":", 1)
        targets[country.strip()] = int(count.strip())
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fresh source-filtered classifier training sample."
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--source-decision-file", type=Path, default=DEFAULT_SOURCE_DECISION_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--country-targets",
        default=None,
        help="Comma-separated targets, e.g. Sweden:600,United_Kingdom:600.",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=60,
        help="Maximum sampled rows per country/source before top-up.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def cleaned_country_path(pipeline_dir: Path, country: str) -> Path:
    return pipeline_dir / f"{country}_cleaned_deduped.csv.gz"


def infer_year(data):
    import pandas as pd

    if "year" in data.columns:
        return pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    if "date_parsed" in data.columns:
        return pd.to_datetime(data["date_parsed"], errors="coerce", utc=True).dt.year.astype("Int64")
    if "dateTime" in data.columns:
        return pd.to_datetime(data["dateTime"], errors="coerce", utc=True).dt.year.astype("Int64")
    return pd.Series(pd.NA, index=data.index, dtype="Int64")


def sample_country(data, country: str, target_n: int, max_per_source: int, random_state: int):
    import pandas as pd

    data = data.copy()
    data["year"] = infer_year(data)
    data = data[data["article_text"].fillna("").astype(str).str.strip().ne("")].copy()
    if data.empty:
        return data

    sampled_parts = []
    selected_idx = set()

    # First pass: sample across year/source cells with a per-source cap so the
    # silver set is not dominated by very large outlets.
    for source, source_df in data.groupby("source_uri", dropna=False):
        take_n = min(max_per_source, len(source_df))
        if take_n <= 0:
            continue
        sampled = (
            source_df.groupby("year", dropna=False, group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(len(group), max(1, round(take_n * len(group) / len(source_df)))),
                    random_state=random_state,
                )
            )
            .head(take_n)
        )
        sampled_parts.append(sampled)
        selected_idx.update(sampled.index)

    sampled_pool = pd.concat(sampled_parts) if sampled_parts else data.head(0)
    sampled_pool = sampled_pool.drop_duplicates(subset=["uri"] if "uri" in sampled_pool.columns else None)

    if len(sampled_pool) >= target_n:
        result = sampled_pool.sample(n=target_n, random_state=random_state).copy()
    else:
        remaining = data.loc[~data.index.isin(selected_idx)].copy()
        topup_n = min(target_n - len(sampled_pool), len(remaining))
        topup = remaining.sample(n=topup_n, random_state=random_state) if topup_n else remaining.head(0)
        result = pd.concat([sampled_pool, topup], ignore_index=False).head(target_n).copy()

    result["country"] = country
    result["al_bucket"] = "source_filtered_seed"
    result["human_final_label"] = ""
    result["human_notes"] = ""
    return result.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Pass --overwrite to replace it.")

    import pandas as pd

    country_targets = parse_country_targets(args.country_targets)
    decisions = load_source_decisions(args.source_decision_file)

    columns = [
        "uri",
        "country",
        "date_parsed",
        "dateTime",
        "year",
        "month",
        "week",
        "source_uri",
        "source.uri",
        "word_count",
        "article_text",
    ]

    batches = []
    filter_summaries = []
    for country, target_n in country_targets.items():
        path = cleaned_country_path(args.pipeline_dir, country)
        if not path.exists():
            print(f"Skipping missing cleaned file for {country}: {path}", flush=True)
            continue

        print(f"\nLoading {country}: {path}", flush=True)
        data = pd.read_csv(path, usecols=lambda column: column in columns)
        data["country"] = country
        if "source_uri" not in data.columns and "source.uri" in data.columns:
            data["source_uri"] = data["source.uri"]

        filtered, merged = apply_source_inclusion_filter(data, decisions, country_column="country")
        summary = source_filter_summary(merged, group_columns=["country"])
        filter_summaries.append(summary)
        print(summary, flush=True)

        sampled = sample_country(
            filtered,
            country=country,
            target_n=target_n,
            max_per_source=args.max_per_source,
            random_state=args.random_state,
        )
        print(f"{country}: sampled {len(sampled):,} / target {target_n:,}", flush=True)
        batches.append(sampled)

    if not batches:
        raise RuntimeError("No silver-label seed rows were sampled.")

    batch = pd.concat(batches, ignore_index=True)
    output_columns = [column for column in OUTPUT_COLUMNS if column in batch.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch[output_columns].to_csv(args.output, index=False)

    summary_path = args.output.with_name(args.output.stem + "_source_filter_summary.csv")
    pd.concat(filter_summaries, ignore_index=True).to_csv(summary_path, index=False)

    print(f"\nSaved source-filtered classifier training sample: {args.output}", flush=True)
    print(f"Rows: {len(batch):,}", flush=True)
    print(f"Saved source-filter summary: {summary_path}", flush=True)
    print("\nBy country:", flush=True)
    print(batch["country"].value_counts(), flush=True)
    print("\nTop sampled sources:", flush=True)
    print(batch["source_uri"].value_counts().head(30), flush=True)


if __name__ == "__main__":
    main()
