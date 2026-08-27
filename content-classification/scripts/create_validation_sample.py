"""Create a stratified sample from political-corruption articles.

The default sample is 500 articles allocated as evenly as possible across
non-empty country-year strata. The output is intended for human validation of
the article-level content variables and can also be passed to the GPT content
classifiers with `--source csv --input`.

Use `--sample-purpose codebook_development` for small development samples that
are read while refining the codebook or prompts. Keep those samples separate
from final held-out validation data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

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
        "--sample-purpose",
        default="final_validation",
        help=(
            "Stored in the sample_purpose column. Use codebook_development for "
            "samples used to refine definitions/prompts; use final_validation "
            "for held-out validation samples."
        ),
    )
    parser.add_argument(
        "--per-country",
        type=int,
        default=None,
        help=(
            "Sample this many articles per country, stratified across years "
            "within country. If set, overrides --total-sample."
        ),
    )
    parser.add_argument(
        "--stratify-confidence",
        action="store_true",
        help=(
            "Also stratify by classifier-confidence band. This is useful for "
            "codebook development; final validation normally uses country-year "
            "strata only."
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


def probability_band(probability: float) -> str:
    if probability < 0.60:
        return "near_threshold"
    if probability < 0.75:
        return "medium"
    return "high"


def sampling_columns(stratify_confidence: bool) -> list[str]:
    columns = ["country", "year"]
    if stratify_confidence:
        columns.append("probability_band")
    return columns


def allocate_sample(data, group_columns: list[str], target: int, random_state: int):
    """Sample exactly ``target`` rows while representing every feasible stratum."""
    if target < 1:
        raise ValueError("Sample size must be positive.")
    if data.empty:
        raise ValueError("No eligible rows found after filtering.")
    target = min(target, len(data))
    grouped = list(data.groupby(group_columns, dropna=False, sort=True))
    if not grouped:
        raise ValueError(f"No non-empty strata found for {group_columns}.")

    sizes = pd.Series(
        {key if isinstance(key, tuple) else (key,): len(group) for key, group in grouped},
        dtype="int64",
    )
    if target >= len(sizes):
        base_n = target // len(sizes)
        allocation = sizes.clip(upper=base_n).astype("int64")
    else:
        allocation = pd.Series(0, index=sizes.index, dtype="int64")
        selected = sizes.sample(
            n=target,
            weights=sizes,
            random_state=random_state,
            replace=False,
        ).index
        allocation.loc[selected] = 1

    cursor = 0
    while int(allocation.sum()) < target:
        eligible = [
            key for key in sizes.index if allocation.loc[key] < sizes.loc[key]
        ]
        if not eligible:
            break
        selected = eligible[cursor % len(eligible)]
        allocation.loc[selected] += 1
        cursor += 1

    pieces = []
    for group_index, (key, group) in enumerate(grouped):
        normalized_key = key if isinstance(key, tuple) else (key,)
        take_n = int(allocation.loc[normalized_key])
        if take_n:
            pieces.append(
                group.sample(n=take_n, random_state=random_state + group_index)
            )
    if not pieces:
        raise ValueError("No rows were sampled. Check the requested sample size.")
    return pd.concat(pieces, ignore_index=True)


def sample_total(
    data,
    total_sample: int,
    random_state: int,
    group_columns: list[str],
):

    return allocate_sample(data, group_columns, total_sample, random_state)


def sample_per_country(
    data,
    per_country: int,
    random_state: int,
    group_columns: list[str],
):
    sampled = []
    for country_index, (country, country_data) in enumerate(data.groupby("country", dropna=False)):
        within_country_columns = [column for column in group_columns if column != "country"]
        sampled.append(
            allocate_sample(
                country_data,
                within_country_columns,
                per_country,
                random_state + country_index * 1000,
            )
        )

    if not sampled:
        raise ValueError("No rows were sampled. Check --per-country and input filters.")
    return pd.concat(sampled, ignore_index=True)


def add_stratum_weights(data, sample, group_columns: list[str]):
    stratum_sizes = (
        data.groupby(group_columns, dropna=False)
        .size()
        .rename("stratum_total_rows")
        .reset_index()
    )
    sampled_sizes = (
        sample.groupby(group_columns, dropna=False)
        .size()
        .rename("stratum_sample_rows")
        .reset_index()
    )
    sample = sample.merge(stratum_sizes, on=group_columns, how="left")
    sample = sample.merge(sampled_sizes, on=group_columns, how="left")
    sample["validation_weight"] = sample["stratum_total_rows"] / sample["stratum_sample_rows"]
    return sample


def add_human_validation_columns(sample, sample_purpose: str):
    sample = sample.copy()
    sample["sample_purpose"] = sample_purpose
    for column in [
        "human_victim_visibility",
        "human_corruption_frame",
        "human_case_location",
        "human_abroad_case",
        "human_accused_actor_visibility",
        "human_accused_actor_visible",
        "human_notes",
        "human_coder_id",
        "human_coder_first_name",
        "human_code_session_id",
        "human_coded_at",
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

    if args.stratify_confidence:
        if "prob_political_corruption" not in data.columns:
            raise ValueError(
                "--stratify-confidence requires prob_political_corruption."
            )
        data["probability_band"] = data["prob_political_corruption"].map(
            probability_band
        )
    group_columns = sampling_columns(args.stratify_confidence)

    if args.per_country is not None:
        sample = sample_per_country(
            data,
            args.per_country,
            args.random_state,
            group_columns,
        )
        if args.output_name == "content_validation_sample_500.csv.gz":
            args.output_name = f"content_validation_sample_{args.per_country}_per_country.csv.gz"
    else:
        sample = sample_total(
            data,
            args.total_sample,
            args.random_state,
            group_columns,
        )
    sample = add_stratum_weights(data, sample, group_columns)
    sample = add_human_validation_columns(sample, args.sample_purpose)
    sample = sample.sort_values(["country", "year", "article_id"]).reset_index(drop=True)
    sample.insert(
        0,
        "content_sample_id",
        [f"CCV_{index:04d}" for index in range(1, len(sample) + 1)],
    )

    preferred_columns = [
        "content_sample_id",
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
        "probability_band",
        "stratum_total_rows",
        "stratum_sample_rows",
        "validation_weight",
        "sample_purpose",
        "article_text",
        "human_victim_visibility",
        "human_corruption_frame",
        "human_case_location",
        "human_abroad_case",
        "human_accused_actor_visibility",
        "human_accused_actor_visible",
        "human_notes",
        "human_coder_id",
        "human_coder_first_name",
        "human_code_session_id",
        "human_coded_at",
    ]
    columns = [column for column in preferred_columns if column in sample.columns]
    sample = sample[columns]

    output_path = args.output_dir / args.output_name
    write_csv_atomic(sample, output_path)

    strata = (
        sample.groupby(group_columns, dropna=False)
        .agg(
            sampled_articles=("article_id", "size"),
            stratum_total_rows=("stratum_total_rows", "first"),
            validation_weight=("validation_weight", "first"),
        )
        .reset_index()
        .sort_values(group_columns)
    )
    strata_path = output_path.with_name(output_path.stem.replace(".csv", "") + "_strata.csv")
    strata.to_csv(strata_path, index=False)

    print(f"Saved validation sample: {output_path} ({len(sample):,} rows)", flush=True)
    print(f"Saved stratum diagnostics: {strata_path}", flush=True)
    print("Rows by country:", flush=True)
    print(sample["country"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
