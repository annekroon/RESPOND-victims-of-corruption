"""Prepare a fresh source-filtered classifier training sample.

This is the restart path for the political-corruption classifier. It does not
use previous silver labels or a provisional classifier. Instead, it samples from
the full cleaned/deduplicated/source-filtered corpus created by
``02_create_source_filtered_corpus.py``.

The output is intended for political_classifier/scripts/04_label_silver_batch.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.source_filter import DEFAULT_PIPELINE_DIR
from political_classifier.split_integrity import remove_validation_overlap
from political_classifier.reproducibility import (
    file_record,
    frame_fingerprint,
    git_commit,
)


DEFAULT_OUTPUT_PATH = (
    DEFAULT_PIPELINE_DIR
    / "active_learning"
    / "silver_training_source_filtered_for_annotation.csv"
)
DEFAULT_INPUT_DIR = DEFAULT_PIPELINE_DIR / "cleaned_deduped_source_filtered"
DEFAULT_SOURCE_FILTER_MANIFEST = (
    DEFAULT_PIPELINE_DIR / "source_inclusion" / "source_filter_run_manifest.json"
)
DEFAULT_EXTRA_HUMAN_VALIDATION = (
    DEFAULT_PIPELINE_DIR / "active_learning" / "uk_human_validation_reviewed.csv"
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
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing *_cleaned_deduped_source_filtered.csv.gz files.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--source-filter-manifest",
        type=Path,
        default=DEFAULT_SOURCE_FILTER_MANIFEST,
        help="Completed step-02 manifest for the source-filtered corpus.",
    )
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
        "--extra-human-validation",
        type=Path,
        nargs="+",
        default=[DEFAULT_EXTRA_HUMAN_VALIDATION],
        help="Additional human benchmark files whose URI/text must be excluded.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def choose_text_series(data):
    if "translated_text" in data.columns:
        return data["translated_text"]
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]
    title = data["title"].fillna("").astype(str) if "title" in data.columns else ""
    body = data["body"].fillna("").astype(str) if "body" in data.columns else ""
    return title + "\n" + body


def choose_integrity_text_series(data):
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]
    return choose_text_series(data)


def load_human_benchmark(extra_paths):
    import pandas as pd
    from dataloader import load_human_annotated_for_translation_webdav

    frames = [load_human_annotated_for_translation_webdav()]
    for path in extra_paths or []:
        if not path.exists():
            raise FileNotFoundError(
                f"Required extra human benchmark file not found: {path}"
            )
        frames.append(pd.read_csv(path))
    benchmark = pd.concat(frames, ignore_index=True, sort=False)
    benchmark["model_text"] = choose_text_series(benchmark).fillna("").astype(str)
    benchmark["integrity_text"] = (
        choose_integrity_text_series(benchmark).fillna("").astype(str)
    )
    return benchmark


def source_filtered_country_path(input_dir: Path, country: str) -> Path:
    return input_dir / f"{country}_cleaned_deduped_source_filtered.csv.gz"


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
    if not args.source_filter_manifest.exists():
        raise FileNotFoundError(
            f"Completed source-filter manifest not found: {args.source_filter_manifest}. "
            "Run 02_create_source_filtered_corpus.py first."
        )

    import pandas as pd

    country_targets = parse_country_targets(args.country_targets)
    invalid_targets = {
        country: target for country, target in country_targets.items() if target < 1
    }
    if invalid_targets:
        raise ValueError(f"Country targets must be positive: {invalid_targets}")
    if args.max_per_source < 1:
        raise ValueError("--max-per-source must be positive.")
    input_paths = {
        country: source_filtered_country_path(args.input_dir, country)
        for country in country_targets
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot draw a complete silver sample; missing source-filtered files: "
            + ", ".join(missing_inputs)
        )
    human_benchmark = load_human_benchmark(args.extra_human_validation)
    print(f"Human benchmark exclusions loaded: {len(human_benchmark):,}", flush=True)
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
    sample_summaries = []
    for country, target_n in country_targets.items():
        path = source_filtered_country_path(args.input_dir, country)
        print(f"\nLoading {country}: {path}", flush=True)
        data = pd.read_csv(path, usecols=lambda column: column in columns)
        data["country"] = country
        if "source_uri" not in data.columns and "source.uri" in data.columns:
            data["source_uri"] = data["source.uri"]

        data["model_text"] = data["article_text"].fillna("").astype(str)
        data["integrity_text"] = data["model_text"]
        overlap_audit = args.output.with_name(
            f"{args.output.stem}_{country}_excluded_human_overlap.csv"
        )
        data, overlap = remove_validation_overlap(
            data,
            human_benchmark,
            text_column="integrity_text",
            audit_path=overlap_audit,
        )
        print(
            f"{country}: excluded {len(overlap):,} human-benchmark overlap row(s)",
            flush=True,
        )

        sampled = sample_country(
            data,
            country=country,
            target_n=target_n,
            max_per_source=args.max_per_source,
            random_state=args.random_state,
        )
        print(f"{country}: sampled {len(sampled):,} / target {target_n:,}", flush=True)
        if len(sampled) != target_n:
            raise ValueError(
                f"{country}: sampled {len(sampled):,} rows but target was "
                f"{target_n:,}. Reduce the target only after inspecting eligibility."
            )
        batches.append(sampled)
        sample_summaries.append(
            {
                "country": country,
                "source_filtered_rows": len(data),
                "sampled_rows": len(sampled),
                "target_rows": target_n,
                "input_file": path.name,
            }
        )

    if not batches:
        raise RuntimeError("No silver-label seed rows were sampled.")

    batch = pd.concat(batches, ignore_index=True)
    output_columns = [column for column in OUTPUT_COLUMNS if column in batch.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch[output_columns].to_csv(args.output, index=False)

    summary_path = args.output.with_name(args.output.stem + "_sample_summary.csv")
    pd.DataFrame(sample_summaries).to_csv(summary_path, index=False)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "input_dir": str(args.input_dir),
        "source_filter_manifest": file_record(args.source_filter_manifest),
        "country_targets": country_targets,
        "max_per_source": args.max_per_source,
        "random_state": args.random_state,
        "human_benchmark_rows": len(human_benchmark),
        "human_benchmark_sha256": frame_fingerprint(
            human_benchmark,
            ["country", "uri", "integrity_text"],
        ),
        "sample": file_record(args.output),
        "sample_summary": file_record(summary_path),
    }
    manifest_path = args.output.with_name(args.output.name + ".run.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\nSaved source-filtered classifier training sample: {args.output}", flush=True)
    print(f"Rows: {len(batch):,}", flush=True)
    print(f"Saved sample summary: {summary_path}", flush=True)
    print(f"Saved run manifest: {manifest_path}", flush=True)
    print("\nBy country:", flush=True)
    print(batch["country"].value_counts(), flush=True)
    print("\nTop sampled sources:", flush=True)
    print(batch["source_uri"].value_counts().head(30), flush=True)


if __name__ == "__main__":
    main()
