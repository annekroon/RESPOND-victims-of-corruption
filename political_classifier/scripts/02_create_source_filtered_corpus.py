"""Create the full cleaned/deduplicated/source-filtered corpus.

This script starts from the existing ``*_cleaned_deduped.csv.gz`` files,
retains only country/source pairs where the reviewed workbook has
``conventional_journalism == Yes``, and writes a full source-filtered corpus.

It does not sample rows. Sampling for classifier training happens in
``03_prepare_classifier_training_sample.py``.
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

from political_classifier.source_filter import (
    DEFAULT_PIPELINE_DIR,
    DEFAULT_SOURCE_DECISION_FILE,
    apply_source_inclusion_filter,
    load_source_decisions,
)
from political_classifier.reproducibility import file_record, git_commit


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
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument(
        "--max-missing-source-share",
        type=float,
        default=0.02,
        help="Fail when more than this share lacks a country/source decision.",
    )
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
    (args.summary_dir / "source_filter_run_manifest.json").unlink(missing_ok=True)

    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive.")
    if not 0 <= args.max_missing_source_share <= 1:
        raise ValueError("--max-missing-source-share must be between 0 and 1.")

    clean_manifest_path = args.pipeline_dir / "clean_dedupe_run_manifest.json"
    clean_audit_path = args.pipeline_dir / "clean_dedupe_audit.csv"
    if not clean_manifest_path.exists() or not clean_audit_path.exists():
        raise FileNotFoundError(
            "Completed cleaning manifest/audit not found. Run "
            "01_clean_dedupe_data.py successfully before source filtering."
        )
    clean_manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    missing_manifest_countries = sorted(
        set(args.countries) - set(clean_manifest.get("countries", []))
    )
    if missing_manifest_countries:
        raise ValueError(
            "Cleaning manifest does not cover requested countries: "
            + ", ".join(missing_manifest_countries)
        )
    clean_audit = pd.read_csv(clean_audit_path)
    expected_clean_rows = dict(
        zip(
            clean_audit["country"].astype(str),
            pd.to_numeric(
                clean_audit["cleaned_deduplicated_rows"], errors="raise"
            ).astype(int),
        )
    )

    input_paths = {
        country: cleaned_country_path(args.pipeline_dir, country)
        for country in args.countries
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot start a complete source-filter run; missing country files: "
            + ", ".join(missing_inputs)
        )
    existing_outputs = [
        output_country_path(args.output_dir, country)
        for country in args.countries
        if output_country_path(args.output_dir, country).exists()
    ]
    combined_output = args.output_dir / "all_countries_cleaned_deduped_source_filtered_minimal.csv.gz"
    if (
        existing_outputs
        or (args.write_combined_minimal and combined_output.exists())
    ) and not args.overwrite:
        raise FileExistsError(
            "Source-filtered outputs already exist. Use --overwrite for a "
            "deliberate complete rebuild."
        )

    decisions = load_source_decisions(args.source_decision_file)

    output_rows = []
    filter_summaries = []
    combined_temporary = combined_output.with_name(combined_output.name + ".part")
    if combined_temporary.exists():
        combined_temporary.unlink()
    combined_first = True

    for country in args.countries:
        input_path = cleaned_country_path(args.pipeline_dir, country)
        output_path = output_country_path(args.output_dir, country)

        print(f"\nFiltering {country}: {input_path}", flush=True)
        temporary = output_path.with_name(output_path.name + ".part")
        if temporary.exists():
            temporary.unlink()
        first_chunk = True
        input_n = output_n = missing_n = 0
        decision_counts = {}
        for chunk_number, data in enumerate(
            pd.read_csv(input_path, chunksize=args.chunksize, low_memory=False), start=1
        ):
            data["country"] = country
            if "source_uri" not in data.columns and "source.uri" in data.columns:
                data["source_uri"] = data["source.uri"]

            filtered, merged = apply_source_inclusion_filter(
                data, decisions, country_column="country"
            )
            filtered.to_csv(
                temporary,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
                compression="gzip",
            )
            input_n += len(data)
            output_n += len(filtered)
            chunk_counts = merged["source_filter_decision"].value_counts(dropna=False)
            for decision, count in chunk_counts.items():
                decision_counts[str(decision)] = decision_counts.get(str(decision), 0) + int(count)
            missing_n += int(merged["source_filter_decision"].eq("MISSING").sum())

            if args.write_combined_minimal and not filtered.empty:
                minimal_cols = [column for column in MINIMAL_COLUMNS if column in filtered.columns]
                filtered[minimal_cols].to_csv(
                    combined_temporary,
                    mode="w" if combined_first else "a",
                    header=combined_first,
                    index=False,
                    compression="gzip",
                )
                combined_first = False
            first_chunk = False
            print(
                f"{country} chunk {chunk_number}: retained {output_n:,} / {input_n:,}",
                flush=True,
            )

        if first_chunk:
            raise RuntimeError(f"No rows found in {input_path}.")
        expected_rows = expected_clean_rows.get(country)
        if expected_rows is None or input_n != expected_rows:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"{country}: cleaned file has {input_n:,} rows but the completed "
                f"cleaning audit records {expected_rows!r}. Rerun step 01."
            )
        missing_share = missing_n / input_n if input_n else 0.0
        if missing_share > args.max_missing_source_share:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"{country}: {missing_n:,}/{input_n:,} rows ({missing_share:.2%}) "
                "lack a source decision, exceeding --max-missing-source-share. "
                "Update the workbook or explicitly relax the threshold."
            )
        temporary.replace(output_path)

        summary_row = {
            "country": country,
            "rows_before": input_n,
            "rows_after": output_n,
            "rows_excluded": input_n - output_n,
            "included_share_pct": output_n / input_n * 100 if input_n else 0,
            "missing_source_decision_rows": missing_n,
            "missing_source_decision_share": missing_share,
            "input_file": input_path.name,
            "output_file": output_path.name,
            **decision_counts,
        }
        filter_summaries.append(pd.DataFrame([summary_row]))

        output_rows.append(
            {
                "country": country,
                "input_rows": input_n,
                "output_rows": output_n,
                "excluded_rows": input_n - output_n,
                "included_share_pct": output_n / input_n * 100 if input_n else 0,
                "missing_source_decision_rows": missing_n,
                "missing_source_decision_share": missing_share,
                "output_path": str(output_path),
            }
        )
        print(f"{country}: retained {output_n:,} / {input_n:,} rows -> {output_path}", flush=True)

    if not output_rows:
        raise RuntimeError("No source-filtered country files were written.")
    if args.write_combined_minimal and combined_first:
        raise RuntimeError("No source-eligible rows were written to the combined corpus.")

    output_summary = pd.DataFrame(output_rows)
    output_summary_path = args.summary_dir / "cleaned_source_filter_output_summary.csv"
    output_summary.to_csv(output_summary_path, index=False)

    decision_summary_path = args.summary_dir / "cleaned_source_filter_decision_summary_by_country.csv"
    pd.concat(filter_summaries, ignore_index=True).to_csv(decision_summary_path, index=False)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "source_decision_file": file_record(args.source_decision_file),
        "clean_dedupe_manifest": file_record(clean_manifest_path),
        "clean_dedupe_audit": file_record(clean_audit_path),
        "countries": args.countries,
        "chunksize": args.chunksize,
        "max_missing_source_share": args.max_missing_source_share,
        "input_rows": int(output_summary["input_rows"].sum()),
        "output_rows": int(output_summary["output_rows"].sum()),
        "missing_source_decision_rows": int(
            output_summary["missing_source_decision_rows"].sum()
        ),
        "output_summary": file_record(output_summary_path),
        "decision_summary": file_record(decision_summary_path),
    }
    manifest_path = args.summary_dir / "source_filter_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.write_combined_minimal and not combined_first:
        combined_temporary.replace(combined_output)
        print(f"\nSaved minimal combined source-filtered file: {combined_output}", flush=True)

    print(f"Saved output summary:   {output_summary_path}", flush=True)
    print(f"Saved decision summary: {decision_summary_path}", flush=True)
    print(f"Saved run manifest:     {manifest_path}", flush=True)
    print("\nBy country:", flush=True)
    print(output_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
