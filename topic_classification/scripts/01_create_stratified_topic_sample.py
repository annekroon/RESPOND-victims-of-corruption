"""Create country-year stratified samples for topic modeling.

Examples:
    python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
      --source classified --political-only

    python3 topic_classification/scripts/01_create_stratified_topic_sample.py \
      --source source-filtered \
      --output-name source_filtered_country_year_sample.csv.gz
"""

from __future__ import annotations

import argparse
import io
import posixpath
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALL_COUNTRIES, RD_BASE_DIR
from topic_classification.provenance import (
    load_verified_classifier_run,
    verify_classified_country_frame,
)
from topic_classification.scripts._impl.reproducibility import write_run_manifest


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_SOURCE_FILTERED_DIR = DEFAULT_PIPELINE_DIR / "cleaned_deduped_source_filtered"
DEFAULT_CLASSIFIED_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files"
DEFAULT_CLASSIFIER_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier"
DEFAULT_OUTPUT_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "topic_classification"
)
DEFAULT_RD_CLASSIFIED_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "political_classifier",
    "classifier_outputs",
    "classified_country_files",
)
DEFAULT_RD_CLEANED_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "political_classifier",
    "cleaned_deduped_source_filtered",
)

KEEP_COLUMNS = [
    "uri",
    "country",
    "date_parsed",
    "dateTime",
    "date",
    "published_at",
    "year",
    "month",
    "week",
    "source_uri",
    "source.uri",
    "word_count",
    "prob_political_corruption",
    "pred_political_corruption",
    "article_text",
    "combined_text",
    "title",
    "body",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible country-year stratified sample for topic modeling."
    )
    parser.add_argument(
        "--source",
        choices=[
            "source-filtered",
            "classified",
            "source-filtered-webdav",
            "classified-webdav",
        ],
        default="classified",
        help=(
            "Use local source-filtered cleaned files or classified files, or archived "
            "Research Drive/WebDAV equivalents. Topic modelling of political-corruption "
            "articles should normally use --source classified --political-only."
        ),
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--source-filtered-dir", type=Path, default=DEFAULT_SOURCE_FILTERED_DIR)
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument(
        "--classifier-output-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_OUTPUT_DIR,
        help=(
            "Directory containing classifier_run_manifest.json, "
            "classified_country_summary.csv, and selected_threshold.txt."
        ),
    )
    parser.add_argument(
        "--source-filtered-rd-dir",
        "--cleaned-rd-dir",
        dest="source_filtered_rd_dir",
        default=DEFAULT_RD_CLEANED_DIR,
        help="Research Drive directory containing archived cleaned/deduped/source-filtered country files.",
    )
    parser.add_argument(
        "--classified-rd-dir",
        default=DEFAULT_RD_CLASSIFIED_DIR,
        help="Research Drive directory containing archived classified country files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument(
        "--per-country-year",
        type=int,
        default=100,
        help="Maximum number of rows to sample from each country-year stratum.",
    )
    parser.add_argument(
        "--total-sample",
        type=int,
        default=None,
        help=(
            "Optional total sample size. If set, rows are allocated as evenly "
            "as possible across non-empty country-year strata."
        ),
    )
    parser.add_argument(
        "--political-only",
        action="store_true",
        help="Keep only pred_political_corruption == 1. Requires classified source.",
    )
    parser.add_argument(
        "--allow-unverified-classifier",
        action="store_true",
        help="Permit legacy/exploratory classified inputs without final-run verification.",
    )
    parser.add_argument("--min-words", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing sample and its diagnostics deliberately.",
    )
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def choose_text(data):
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]
    if {"title", "body"}.issubset(data.columns):
        return data["title"].fillna("").astype(str) + "\n" + data["body"].fillna("").astype(str)
    raise ValueError("Input data must contain article_text or title/body columns.")


def country_path(args: argparse.Namespace, country: str) -> Path:
    if args.source == "source-filtered":
        return args.source_filtered_dir / f"{country}_cleaned_deduped_source_filtered.csv.gz"
    return args.classified_dir / f"{country}_classified.csv.gz"


def country_rd_path(args: argparse.Namespace, country: str) -> str:
    if args.source == "source-filtered-webdav":
        return posixpath.join(args.source_filtered_rd_dir, f"{country}_cleaned_deduped_source_filtered.csv.gz")
    if args.source == "classified-webdav":
        return posixpath.join(args.classified_rd_dir, f"{country}_classified.csv.gz")
    raise ValueError(f"Source is not a WebDAV source: {args.source}")


def read_country_csv(args: argparse.Namespace, country: str):
    import pandas as pd

    if args.source.endswith("-webdav"):
        from rd_utils import webdav_download_bytes

        rd_path = country_rd_path(args, country)
        try:
            data = webdav_download_bytes(rd_path)
        except Exception as exc:
            print(f"Skipping missing/unreadable WebDAV file for {country}: {rd_path} ({exc!r})", flush=True)
            return None, rd_path
        frame = pd.read_csv(
            io.BytesIO(data),
            compression="gzip" if rd_path.endswith(".gz") else "infer",
            usecols=lambda column: column in set(KEEP_COLUMNS),
        )
        return frame, rd_path

    path = country_path(args, country)
    if not path.exists():
        print(f"Skipping missing file for {country}: {path}", flush=True)
        return None, str(path)
    frame = pd.read_csv(path, usecols=lambda column: column in set(KEEP_COLUMNS))
    return frame, str(path)


def load_country_file(args: argparse.Namespace, country: str, classifier_run=None):
    import pandas as pd

    data, source_path = read_country_csv(args, country)
    if data is None:
        return None
    if classifier_run is not None:
        verify_classified_country_frame(
            data,
            country=country,
            classifier_run=classifier_run,
        )
    data["country"] = country
    data["article_text"] = choose_text(data).map(normalize_text)
    data = data[data["article_text"].str.split().str.len().fillna(0).ge(args.min_words)].copy()

    if args.political_only:
        if "pred_political_corruption" not in data.columns:
            raise ValueError("--political-only requires pred_political_corruption in classified files.")
        data = data[data["pred_political_corruption"].eq(1)].copy()

    date_column = next(
        (column for column in ["date_parsed", "dateTime", "date", "published_at"] if column in data.columns),
        None,
    )
    if date_column is None and "year" not in data.columns:
        raise ValueError(f"No usable date/year column found in {source_path}.")

    if "year" not in data.columns:
        data["date_parsed"] = pd.to_datetime(data[date_column], errors="coerce", utc=True)
        data["year"] = data["date_parsed"].dt.year
    else:
        data["year"] = pd.to_numeric(data["year"], errors="coerce")

    data = data[data["year"].notna()].copy()
    data["year"] = data["year"].astype(int)
    return data


def sample_fixed_per_stratum(data, per_country_year: int, random_state: int):
    return (
        data.groupby(["country", "year"], group_keys=False, dropna=False)
        .apply(lambda group: group.sample(n=min(len(group), per_country_year), random_state=random_state))
        .reset_index(drop=True)
    )


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
        raise ValueError("No rows were sampled. Increase --total-sample or check input filters.")
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
    weights = stratum_sizes.merge(sampled_sizes, on=["country", "year"], how="inner")
    weights["analysis_weight"] = weights["stratum_total_rows"] / weights["stratum_sample_rows"]
    return sample.merge(weights, on=["country", "year"], how="left")


def default_output_name(args: argparse.Namespace) -> str:
    prefix = "political_corruption" if args.political_only else args.source
    return f"{prefix}_country_year_sample.csv.gz"


def main() -> None:
    args = parse_args()
    if args.political_only and args.source not in {"classified", "classified-webdav"}:
        raise ValueError("--political-only can only be used with --source classified or classified-webdav.")

    import pandas as pd

    output_name = args.output_name or default_output_name(args)
    output_path = args.output_dir / output_name
    diagnostics_path = output_path.with_name(
        output_path.name.replace(".csv.gz", "_strata.csv")
    )
    manifest_path = output_path.with_name(
        output_path.name.replace(".csv.gz", "_run_manifest.json")
    )
    existing_outputs = [
        path for path in [output_path, diagnostics_path, manifest_path] if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        raise FileExistsError(
            "Topic sample outputs already exist. Pass --overwrite for a deliberate "
            "rebuild: "
            + ", ".join(str(path) for path in existing_outputs)
        )

    classifier_run = None
    if args.political_only and not args.allow_unverified_classifier:
        classifier_run = load_verified_classifier_run(
            args.classifier_output_dir,
            list(args.countries),
            classified_dir=args.classified_dir,
            require_local_country_files=not args.source.endswith("-webdav"),
        )
        print(
            "Verified final classifier run: "
            f"threshold={classifier_run.threshold:.2f}, "
            f"source-filtered N={classifier_run.total_articles:,}, "
            f"political-corruption N={classifier_run.political_articles:,}",
            flush=True,
        )

    frames = []
    for country in args.countries:
        frame = load_country_file(args, country, classifier_run=classifier_run)
        if frame is not None and not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No input rows were loaded. Check source paths and country names.")

    data = pd.concat(frames, ignore_index=True)
    print(f"Loaded eligible rows: {len(data):,}", flush=True)
    if classifier_run is not None and len(data) != classifier_run.political_articles:
        raise ValueError(
            "Topic eligibility filters changed the final political-corruption corpus: "
            f"{len(data):,} eligible rows versus "
            f"{classifier_run.political_articles:,} classifier positives. "
            "Resolve missing text/date fields instead of silently changing the universe."
        )

    if args.total_sample is not None:
        sample = sample_total(data, args.total_sample, args.random_state)
    else:
        sample = sample_fixed_per_stratum(data, args.per_country_year, args.random_state)

    sample = add_stratum_weights(data, sample)
    sample = sample.sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    sample["topic_sample_source"] = args.source
    sample["topic_sample_political_only"] = args.political_only
    if classifier_run is not None:
        sample["upstream_classifier_threshold"] = classifier_run.threshold
        sample["upstream_classifier_git_commit"] = classifier_run.manifest.get(
            "git_commit", ""
        )
        sample["upstream_classifier_manifest_sha256"] = (
            classifier_run.manifest_sha256
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False, compression="gzip")

    diagnostics = (
        sample.groupby(["country", "year"], dropna=False)
        .size()
        .rename("sampled_rows")
        .reset_index()
        .sort_values(["country", "year"])
    )
    diagnostics.to_csv(diagnostics_path, index=False)

    manifest_inputs = {}
    upstream_classifier = {"verified_final_classifier": False}
    if classifier_run is not None:
        manifest_inputs = {
            "classifier_manifest": classifier_run.manifest_path,
            "classified_country_summary": classifier_run.summary_path,
            "selected_threshold": classifier_run.threshold_path,
            "source_filter_manifest": classifier_run.source_filter_manifest_path,
        }
        upstream_classifier = classifier_run.provenance_record()

    write_run_manifest(
        args.output_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs=manifest_inputs,
        outputs={
            "sample": output_path,
            "strata_diagnostics": diagnostics_path,
        },
        extra={
            "loaded_eligible_rows": int(len(data)),
            "saved_sample_rows": int(len(sample)),
            "nonempty_country_year_strata": int(diagnostics.shape[0]),
            "political_only": bool(args.political_only),
            "upstream_classifier": upstream_classifier,
        },
        manifest_name=manifest_path.name,
    )

    print(f"Saved sample rows: {len(sample):,}", flush=True)
    print(f"Sample: {output_path}", flush=True)
    print(f"Strata diagnostics: {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()
