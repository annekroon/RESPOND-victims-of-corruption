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
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import ALL_COUNTRIES
from content_classifier_common import (
    DEFAULT_CLASSIFIED_DIR,
    DEFAULT_CLASSIFIER_OUTPUT_DIR,
    DEFAULT_CONTENT_ROOT,
    content_sample_manifest_path,
    iter_input_frames,
    verify_production_input,
    write_csv_atomic,
)
from political_classifier.reproducibility import file_record, git_commit


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
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument(
        "--classifier-output-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_OUTPUT_DIR,
    )
    parser.add_argument("--classified-rd-dir", default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument(
        "--input-chunksize",
        type=int,
        default=25_000,
        help="Rows per chunk while verifying and reading classified country files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CONTENT_ROOT / "validation_final",
    )
    parser.add_argument("--output-name", default="content_validation_final_n500.csv.gz")
    parser.add_argument("--total-sample", type=int, default=500)
    parser.add_argument(
        "--sample-purpose",
        choices=["codebook_development", "final_validation"],
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
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--keep-non-political",
        action="store_true",
        help="Do not filter to pred_political_corruption == 1.",
    )
    parser.add_argument(
        "--exclude-sample",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Previously inspected or coded sample files to exclude by country/URI. "
            "At least one is required for sample-purpose final_validation."
        ),
    )
    parser.add_argument(
        "--minimum-excluded-articles",
        type=int,
        default=108,
        help=(
            "Minimum number of distinct previously inspected articles supplied for "
            "a final-validation draw. The project codebook-development set has 108."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sample, strata table, and sample manifest.",
    )
    parser.set_defaults(random_sample=None)
    return parser.parse_args()


def probability_band(probability: float, threshold: float) -> str:
    if probability < threshold + 0.10:
        return "near_threshold"
    if probability < threshold + 0.25:
        return "medium"
    return "high"


def exclusion_keys(paths: list[Path]) -> tuple[set[str], list[dict]]:
    keys: set[str] = set()
    records = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(
            path,
            compression="gzip" if path.name.endswith(".gz") else "infer",
            usecols=lambda column: column in {"article_id", "country", "uri"},
        )
        if {"country", "uri"}.issubset(frame.columns):
            uri_values = frame["uri"].fillna("").astype(str)
            uri_keys = (
                frame["country"].fillna("").astype(str)
                + "::"
                + uri_values
            )
            if "article_id" in frame.columns:
                article_keys = frame["article_id"].fillna("").astype(str)
                frame_keys = uri_keys.where(uri_values.str.strip().ne(""), article_keys)
            else:
                frame_keys = uri_keys
        elif "article_id" in frame.columns:
            frame_keys = frame["article_id"].fillna("").astype(str)
        else:
            raise ValueError(
                f"Exclusion sample lacks article_id or country plus uri: {path}"
            )
        keys.update(frame_keys[frame_keys.str.strip().ne("")])
        records.append(file_record(path))
    return keys, records


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
        "human_victim_evidence",
        "human_corruption_frame",
        "human_corruption_frame_evidence",
        "human_case_location",
        "human_abroad_case",
        "human_accused_actor_visibility",
        "human_accused_actor_visible",
        "human_accused_individual_evidence",
        "human_accused_organization_evidence",
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
    if args.input_chunksize < 1:
        raise ValueError("--input-chunksize must be positive.")
    if args.classified_rd_dir is None:
        from content_classifier_common import DEFAULT_RD_CLASSIFIED_DIR

        args.classified_rd_dir = DEFAULT_RD_CLASSIFIED_DIR

    if args.sample_purpose == "final_validation" and not args.exclude_sample:
        raise ValueError(
            "Final validation must exclude every codebook-development sample. "
            "Pass the inspected/coded files with --exclude-sample."
        )
    if args.sample_purpose == "final_validation":
        if set(args.countries) != set(ALL_COUNTRIES):
            raise ValueError(
                "Final validation must include all nine publication countries."
            )
        if args.keep_non_political:
            raise ValueError(
                "Final validation must contain political-corruption articles only."
            )
        if args.min_words != 1:
            raise ValueError(
                "Final validation must use --min-words 1 to preserve the final corpus."
            )
    classifier_run = verify_production_input(args)
    if args.sample_purpose == "final_validation" and classifier_run is None:
        raise ValueError(
            "Final validation must be sampled directly from the verified classified "
            "corpus, not from an arbitrary CSV."
        )

    frames = [frame for _, frame in iter_input_frames(args) if not frame.empty]
    if not frames:
        raise ValueError("No eligible political-corruption rows found.")
    data = pd.concat(frames, ignore_index=True)
    print(f"Eligible political-corruption rows: {len(data):,}", flush=True)

    excluded_keys, exclusion_records = exclusion_keys(args.exclude_sample)
    if (
        args.sample_purpose == "final_validation"
        and len(excluded_keys) < args.minimum_excluded_articles
    ):
        raise ValueError(
            "Final validation must exclude all codebook-development articles: "
            f"found {len(excluded_keys):,} distinct exclusions, expected at least "
            f"{args.minimum_excluded_articles:,}."
        )
    if excluded_keys:
        uri_values = data["uri"].fillna("").astype(str)
        uri_keys = (
            data["country"].fillna("").astype(str)
            + "::"
            + uri_values
        )
        data_keys = uri_keys.where(
            uri_values.str.strip().ne(""),
            data["article_id"].fillna("").astype(str),
        )
        excluded = data_keys.isin(excluded_keys)
        data = data.loc[~excluded].copy()
        print(
            f"Excluded {int(excluded.sum()):,} previously inspected article(s)",
            flush=True,
        )

    if args.stratify_confidence:
        if "prob_political_corruption" not in data.columns:
            raise ValueError(
                "--stratify-confidence requires prob_political_corruption."
            )
        probabilities = pd.to_numeric(
            data["prob_political_corruption"], errors="raise"
        )
        if probabilities.isna().any():
            raise ValueError(
                "Classifier probabilities contain missing values; cannot stratify."
            )
        threshold = classifier_run.threshold if classifier_run else 0.60
        data["probability_band"] = probabilities.map(
            lambda value: probability_band(float(value), threshold)
        )
    group_columns = sampling_columns(args.stratify_confidence)

    if args.per_country is not None:
        sample = sample_per_country(
            data,
            args.per_country,
            args.random_state,
            group_columns,
        )
        if args.output_name == "content_validation_final_n500.csv.gz":
            args.output_name = (
                f"content_validation_{args.sample_purpose}_"
                f"{args.per_country}_per_country.csv.gz"
            )
    else:
        sample = sample_total(
            data,
            args.total_sample,
            args.random_state,
            group_columns,
        )
    sample = add_stratum_weights(data, sample, group_columns)
    sample = add_human_validation_columns(sample, args.sample_purpose)
    if classifier_run is not None:
        upstream = classifier_run.provenance_record()
        sample["classifier_threshold"] = classifier_run.threshold
        sample["classifier_manifest_sha256"] = upstream[
            "classifier_manifest_sha256"
        ]
        sample["source_filter_policy"] = upstream["source_filter_policy"]
    sample = sample.sort_values(["country", "year", "article_id"]).reset_index(drop=True)
    prefix = "CCD" if args.sample_purpose == "codebook_development" else "CFV"
    sample.insert(
        0,
        "content_sample_id",
        [
            f"{prefix}_{hashlib.sha256(f'{args.sample_purpose}|{args.random_state}|{article_id}'.encode()).hexdigest()[:12]}"
            for article_id in sample["article_id"].astype(str)
        ],
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
        "classifier_threshold",
        "classifier_manifest_sha256",
        "source_filter_policy",
        "probability_band",
        "stratum_total_rows",
        "stratum_sample_rows",
        "validation_weight",
        "sample_purpose",
        "article_text",
        "human_victim_visibility",
        "human_victim_evidence",
        "human_corruption_frame",
        "human_corruption_frame_evidence",
        "human_case_location",
        "human_abroad_case",
        "human_accused_actor_visibility",
        "human_accused_actor_visible",
        "human_accused_individual_evidence",
        "human_accused_organization_evidence",
        "human_notes",
        "human_coder_id",
        "human_coder_first_name",
        "human_code_session_id",
        "human_coded_at",
    ]
    columns = [column for column in preferred_columns if column in sample.columns]
    sample = sample[columns]

    output_path = args.output_dir / args.output_name
    strata_path = output_path.with_name(
        output_path.stem.replace(".csv", "") + "_strata.csv"
    )
    manifest_path = content_sample_manifest_path(output_path)
    existing_paths = [output_path, strata_path, manifest_path]
    if not args.overwrite and any(path.exists() for path in existing_paths):
        raise FileExistsError(
            "Refusing to overwrite an existing content sample. Pass --overwrite "
            "only when intentionally drawing a fresh sample: "
            + ", ".join(str(path) for path in existing_paths if path.exists())
        )
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
    strata.to_csv(strata_path, index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "sample_purpose": args.sample_purpose,
        "political_only": not args.keep_non_political,
        "countries": list(args.countries),
        "random_state": args.random_state,
        "sampling": {
            "total_sample": args.total_sample,
            "per_country": args.per_country,
            "stratify_confidence": args.stratify_confidence,
            "group_columns": group_columns,
            "min_words": args.min_words,
        },
        "eligible_rows_after_exclusions": len(data),
        "sample_rows": len(sample),
        "development_exclusions": exclusion_records,
        "upstream_classifier": (
            classifier_run.provenance_record() if classifier_run else None
        ),
        "outputs": {
            "sample": file_record(output_path),
            "strata": file_record(strata_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Saved validation sample: {output_path} ({len(sample):,} rows)", flush=True)
    print(f"Saved stratum diagnostics: {strata_path}", flush=True)
    print(f"Saved sample manifest: {manifest_path}", flush=True)
    print("Rows by country:", flush=True)
    print(sample["country"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
