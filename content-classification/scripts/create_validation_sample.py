"""Create a stratified validation sample from political-corruption articles.

The default sample is 500 articles allocated as evenly as possible across
non-empty country-year strata. The output is intended for human validation of
the article-level content variables and can also be passed to the GPT content
classifiers with `--source csv --input`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import ALL_COUNTRIES
from content_classifier_common import DEFAULT_OUTPUT_DIR, load_input, write_csv_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a country-year stratified validation sample for content coding."
    )
    parser.add_argument(
        "--source",
        choices=["classified", "classified-webdav", "csv"],
        default="classified",
        help="Read local classified country files, archived WebDAV files, or one CSV file.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input CSV/CSV.GZ when --source csv.")
    parser.add_argument("--classified-dir", type=Path, default=None)
    parser.add_argument("--classified-rd-dir", default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default="content_validation_sample_500.csv.gz")
    parser.add_argument("--total-sample", type=int, default=500)
    parser.add_argument(
        "--per-country",
        type=int,
        default=None,
        help=(
            "Sample this many articles per country, stratified across years "
            "within country. If set, overrides --total-sample."
        ),
    )
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--keep-non-political",
        action="store_true",
        help="Do not filter to pred_political_corruption == 1.",
    )
    return parser.parse_args()


def sample_total(data, total_sample: int, random_state: int):
    import pandas as pd

    strata = list(data.groupby(["country", "year"], dropna=False).groups)
    if not strata:
        raise ValueError("No non-empty country-year strata found.")

    base_n = total_sample // len(strata)
    remainder = total_sample % len(strata)
    sampled = []

    for index, (_, group) in enumerate(data.groupby(["country", "year"], dropna=False)):
        target = base_n + int(index < remainder)
        if target <= 0:
            continue
        sampled.append(group.sample(n=min(len(group), target), random_state=random_state + index))

    if not sampled:
        raise ValueError("No rows were sampled. Increase --total-sample or check filters.")
    return pd.concat(sampled, ignore_index=True)


def sample_per_country(data, per_country: int, random_state: int):
    import pandas as pd

    sampled = []
    for country_index, (country, country_data) in enumerate(data.groupby("country", dropna=False)):
        strata = list(country_data.groupby("year", dropna=False).groups)
        if not strata:
            continue

        base_n = per_country // len(strata)
        remainder = per_country % len(strata)
        country_sampled = []
        for year_index, (_, group) in enumerate(country_data.groupby("year", dropna=False)):
            target = base_n + int(year_index < remainder)
            if target <= 0:
                continue
            country_sampled.append(
                group.sample(
                    n=min(len(group), target),
                    random_state=random_state + country_index * 1000 + year_index,
                )
            )

        if country_sampled:
            sampled.append(pd.concat(country_sampled, ignore_index=True))

    if not sampled:
        raise ValueError("No rows were sampled. Check --per-country and input filters.")
    return pd.concat(sampled, ignore_index=True)


def add_stratum_weights(data, sample):
    stratum_sizes = (
        data.groupby(["country", "year"], dropna=False)
        .size()
        .rename("stratum_total_rows")
        .reset_index()
    )
    sampled_sizes = (
        sample.groupby(["country", "year"], dropna=False)
        .size()
        .rename("stratum_sample_rows")
        .reset_index()
    )
    sample = sample.merge(stratum_sizes, on=["country", "year"], how="left")
    sample = sample.merge(sampled_sizes, on=["country", "year"], how="left")
    sample["validation_weight"] = sample["stratum_total_rows"] / sample["stratum_sample_rows"]
    return sample


def add_human_validation_columns(sample):
    sample = sample.copy()
    for column in [
        "human_victim_visibility",
        "human_corruption_frame",
        "human_case_location",
        "human_abroad_case",
        "human_accused_actor_visibility",
        "human_accused_actor_visible",
        "human_notes",
    ]:
        sample[column] = ""
    return sample


def main() -> None:
    args = parse_args()

    if args.classified_dir is None:
        from content_classifier_common import DEFAULT_CLASSIFIED_DIR

        args.classified_dir = DEFAULT_CLASSIFIED_DIR
    if args.classified_rd_dir is None:
        from content_classifier_common import DEFAULT_RD_CLASSIFIED_DIR

        args.classified_rd_dir = DEFAULT_RD_CLASSIFIED_DIR

    data = load_input(args)
    print(f"Eligible political-corruption rows: {len(data):,}", flush=True)

    if args.per_country is not None:
        sample = sample_per_country(data, args.per_country, args.random_state)
        if args.output_name == "content_validation_sample_500.csv.gz":
            args.output_name = f"content_validation_sample_{args.per_country}_per_country.csv.gz"
    else:
        sample = sample_total(data, args.total_sample, args.random_state)
    sample = add_stratum_weights(data, sample)
    sample = add_human_validation_columns(sample)

    preferred_columns = [
        "article_id",
        "uri",
        "country",
        "year",
        "month",
        "week",
        "source_uri",
        "source.uri",
        "word_count",
        "prob_political_corruption",
        "pred_political_corruption",
        "stratum_total_rows",
        "stratum_sample_rows",
        "validation_weight",
        "article_text",
        "human_victim_visibility",
        "human_corruption_frame",
        "human_case_location",
        "human_abroad_case",
        "human_accused_actor_visibility",
        "human_accused_actor_visible",
        "human_notes",
    ]
    columns = [column for column in preferred_columns if column in sample.columns]
    sample = sample[columns].sort_values(["country", "year", "article_id"]).reset_index(drop=True)

    output_path = args.output_dir / args.output_name
    write_csv_atomic(sample, output_path)

    strata = (
        sample.groupby(["country", "year"], dropna=False)
        .agg(
            sampled_articles=("article_id", "size"),
            stratum_total_rows=("stratum_total_rows", "first"),
            validation_weight=("validation_weight", "first"),
        )
        .reset_index()
        .sort_values(["country", "year"])
    )
    strata_path = output_path.with_name(output_path.stem.replace(".csv", "") + "_strata.csv")
    strata.to_csv(strata_path, index=False)

    print(f"Saved validation sample: {output_path} ({len(sample):,} rows)", flush=True)
    print(f"Saved stratum diagnostics: {strata_path}", flush=True)
    print("Rows by country:", flush=True)
    print(sample["country"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
