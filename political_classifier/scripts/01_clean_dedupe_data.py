"""Clean and deduplicate raw corruption-query country files.

The rebuild is deliberately country-scoped and memory-bounded. Raw files are
streamed from Research Drive to a local cache, read in chunks, deduplicated
within country, and written incrementally. Exact URI and normalized-text
duplicates are removed by default. Prefix-based near-duplicate removal is
available only as an explicit diagnostic option because shared article leads
can otherwise remove substantively different stories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR
from rd_io import rd_join, rd_list_dir
from rd_utils import webdav_download_to_path
from political_classifier.reproducibility import file_record, git_commit


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_DOWNLOAD_DIR = DEFAULT_PIPELINE_DIR / "raw_news_downloads"
RAW_SUFFIX = "_news.csv"

TEXT_COLUMN_CANDIDATES = [
    "translated_text",
    "combined_text",
    "body",
    "text",
    "article_text",
    "content",
]
KEEP_CLEANED_COLUMNS = [
    "uri",
    "country",
    "dateTime",
    "dateTimePub",
    "date",
    "date_parsed",
    "year",
    "month",
    "week",
    "source_uri",
    "article_text",
    "word_count",
    "text_hash",
    "near_dup_hash",
]
MINIMAL_COMBINED_COLUMNS = [
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
        description="Clean and deduplicate raw RESPOND corruption-query news files."
    )
    parser.add_argument(
        "--data-dir",
        default=RD_BASE_DIR,
        help="Research Drive folder containing *_news.csv files.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Countries to process. Omit to discover all *_news.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--local-input-dir", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument(
        "--near-duplicate-prefix-tokens",
        type=int,
        default=0,
        help=(
            "Opt-in prefix fingerprint length. The default 0 disables approximate "
            "deduplication and removes exact URI/text duplicates only."
        ),
    )
    parser.add_argument(
        "--refresh-downloads",
        action="store_true",
        help="Redownload raw Research Drive CSV files even when cached locally.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing cleaned outputs.",
    )
    return parser.parse_args()


def choose_text_column(dataframe) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column
    raise ValueError(
        "No usable text column found. Expected one of: "
        + ", ".join(TEXT_COLUMN_CANDIDATES)
    )


def fix_mojibake(text) -> str:
    if not isinstance(text, str):
        return ""
    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def badness(value: str) -> int:
        return sum(value.count(marker) for marker in ["Ð", "Ñ", "Ã", "Â", "Ä", "Å", "�"])

    return min(candidates, key=badness)


def normalize_text(text) -> str:
    text = fix_mojibake(text)
    return re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()


def normalize_for_hash(text) -> str:
    text = normalize_text(text).casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def text_hash(text) -> str:
    return stable_hash(normalize_for_hash(text))


def near_duplicate_fingerprint(text, n_tokens: int) -> str:
    normalized = normalize_for_hash(text)
    return stable_hash(" ".join(normalized.split()[:n_tokens]))


def add_article_text(dataframe):
    dataframe = dataframe.copy()
    if "combined_text" not in dataframe.columns and {"title", "body"}.issubset(dataframe.columns):
        title = dataframe["title"].fillna("").astype(str).str.strip()
        body = dataframe["body"].fillna("").astype(str).str.strip()
        dataframe["combined_text"] = title + "\n" + body
    text_column = choose_text_column(dataframe)
    dataframe["article_text"] = dataframe[text_column].map(normalize_text)
    return dataframe


def prepare_chunk(dataframe, country: str, min_words: int, near_tokens: int):
    import pandas as pd

    data = add_article_text(dataframe)
    data["country"] = country

    if "isDuplicate" in data.columns:
        duplicate = data["isDuplicate"].astype(str).str.casefold().isin({"true", "1", "yes"})
        data = data[~duplicate].copy()

    data["word_count"] = data["article_text"].str.split().str.len().fillna(0).astype(int)
    data = data[data["article_text"].ne("") & data["word_count"].ge(min_words)].copy()

    date_column = next(
        (column for column in ("dateTime", "dateTimePub", "date") if column in data.columns),
        None,
    )
    date_source = data[date_column] if date_column else pd.Series(pd.NaT, index=data.index)
    data["date_parsed"] = pd.to_datetime(date_source, errors="coerce", utc=True)
    date_naive = data["date_parsed"].dt.tz_convert(None)
    data["year"] = date_naive.dt.year.astype("Int64")
    data["month"] = date_naive.dt.to_period("M").astype(str)
    data["week"] = date_naive.dt.to_period("W-SUN").dt.start_time

    if "source_uri" not in data.columns and "source.uri" in data.columns:
        data["source_uri"] = data["source.uri"]
    if "source_uri" not in data.columns:
        data["source_uri"] = "unknown"
    data["source_uri"] = data["source_uri"].fillna("").astype(str).str.strip().str.casefold()

    if "uri" not in data.columns:
        data["uri"] = ""
    data["uri"] = data["uri"].fillna("").astype(str).str.strip()
    data["text_hash"] = data["article_text"].map(text_hash)
    data["near_dup_hash"] = (
        data["article_text"].map(lambda value: near_duplicate_fingerprint(value, near_tokens))
        if near_tokens > 0
        else ""
    )
    return data


def remove_seen(data, seen_uris: set[str], seen_texts: set[str], seen_near: set[str]):
    uri_duplicate = data["uri"].ne("") & data["uri"].isin(seen_uris)
    data = data[~uri_duplicate].copy()
    duplicate_uri_in_chunk = data["uri"].ne("") & data["uri"].duplicated(keep="first")
    data = data[~duplicate_uri_in_chunk].copy()
    current_uris = set(data.loc[data["uri"].ne(""), "uri"])

    text_duplicate = data["text_hash"].isin(seen_texts)
    data = data[~text_duplicate].drop_duplicates(subset=["text_hash"], keep="first").copy()

    if data["near_dup_hash"].ne("").any():
        near_duplicate = data["near_dup_hash"].isin(seen_near)
        data = data[~near_duplicate].drop_duplicates(subset=["near_dup_hash"], keep="first").copy()

    seen_uris.update(current_uris)
    seen_texts.update(data["text_hash"])
    seen_near.update(data.loc[data["near_dup_hash"].ne(""), "near_dup_hash"])
    return data


def append_csv(data, path: Path, columns: list[str], first_chunk: bool) -> None:
    selected = data.reindex(columns=columns)
    selected.to_csv(
        path,
        mode="w" if first_chunk else "a",
        header=first_chunk,
        index=False,
        compression="gzip",
    )


def update_counter(counter: Counter, frame, columns: list[str]) -> None:
    grouped = frame.groupby(columns, dropna=False).size()
    for key, count in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        counter[key] += int(count)


def counter_frame(counter: Counter, columns: list[str]):
    import pandas as pd

    rows = [dict(zip(columns, key), total_articles=value) for key, value in counter.items()]
    return pd.DataFrame(rows, columns=[*columns, "total_articles"])


def discover_countries(data_dir: str) -> list[str]:
    return sorted(
        filename[: -len(RAW_SUFFIX)]
        for filename in rd_list_dir(data_dir)
        if filename.endswith(RAW_SUFFIX)
    )


def discover_local_countries(directory: Path) -> list[str]:
    return sorted(
        path.name[: -len(RAW_SUFFIX)]
        for path in directory.glob(f"*{RAW_SUFFIX}")
        if path.is_file()
    )


def local_raw_path(args: argparse.Namespace, country: str) -> Path:
    filename = f"{country}{RAW_SUFFIX}"
    if args.local_input_dir is not None:
        path = args.local_input_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    path = args.download_dir / filename
    if args.refresh_downloads or not path.exists():
        remote = rd_join(args.data_dir, filename)
        print(f"Downloading {remote} -> {path}", flush=True)
        webdav_download_to_path(remote, path)
    else:
        print(f"Reusing cached raw file: {path}", flush=True)
    return path


def assert_can_write_outputs(output_dir: Path, overwrite: bool) -> None:
    outputs = list(output_dir.glob("*_cleaned_deduped.csv.gz"))
    outputs.extend(output_dir.glob("denominator_country_*.csv"))
    if outputs and not overwrite:
        raise FileExistsError(
            f"Found {len(outputs)} existing output file(s) in {output_dir}. "
            "Use --overwrite to replace them."
        )


def main() -> None:
    args = parse_args()
    import pandas as pd

    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive.")
    if args.near_duplicate_prefix_tokens not in {0} and args.near_duplicate_prefix_tokens < 200:
        raise ValueError(
            "Approximate prefix deduplication is risky. Use 0 to disable it or at least 200 tokens."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)
    assert_can_write_outputs(args.output_dir, args.overwrite)
    (args.output_dir / "clean_dedupe_run_manifest.json").unlink(missing_ok=True)

    if args.countries:
        countries = args.countries
    elif args.local_input_dir is not None:
        countries = discover_local_countries(args.local_input_dir)
    else:
        countries = discover_countries(args.data_dir)
    if not countries:
        raise RuntimeError(f"No {RAW_SUFFIX} files found in {args.data_dir}.")

    combined_output = args.output_dir / "all_countries_cleaned_deduped_minimal.csv.gz"
    temporary_combined = combined_output.with_name(combined_output.name + ".part")
    if temporary_combined.exists():
        temporary_combined.unlink()

    counters = {
        "country": Counter(),
        "country_year": Counter(),
        "country_month": Counter(),
        "country_week": Counter(),
        "country_year_source": Counter(),
    }
    audit_rows = []
    combined_first = True

    for country in countries:
        raw_path = local_raw_path(args, country)
        print(f"Hashing raw input for provenance: {raw_path}", flush=True)
        raw_record = file_record(raw_path)
        output_path = args.output_dir / f"{country}_cleaned_deduped.csv.gz"
        temporary_output = output_path.with_name(output_path.name + ".part")
        if temporary_output.exists():
            temporary_output.unlink()

        seen_uris: set[str] = set()
        seen_texts: set[str] = set()
        seen_near: set[str] = set()
        input_rows = eligible_rows = retained_rows = 0
        first_country_chunk = True

        print(f"\nProcessing {country}: {raw_path}", flush=True)
        reader = pd.read_csv(raw_path, chunksize=args.chunksize, low_memory=False)
        for chunk_number, chunk in enumerate(reader, start=1):
            input_rows += len(chunk)
            prepared = prepare_chunk(
                chunk,
                country=country,
                min_words=args.min_words,
                near_tokens=args.near_duplicate_prefix_tokens,
            )
            eligible_rows += len(prepared)
            cleaned = remove_seen(prepared, seen_uris, seen_texts, seen_near)
            retained_rows += len(cleaned)

            if not cleaned.empty:
                append_csv(cleaned, temporary_output, KEEP_CLEANED_COLUMNS, first_country_chunk)
                append_csv(cleaned, temporary_combined, MINIMAL_COMBINED_COLUMNS, combined_first)
                first_country_chunk = False
                combined_first = False
                update_counter(counters["country"], cleaned, ["country"])
                update_counter(counters["country_year"], cleaned, ["country", "year"])
                update_counter(counters["country_month"], cleaned, ["country", "month"])
                update_counter(counters["country_week"], cleaned, ["country", "week"])
                update_counter(
                    counters["country_year_source"],
                    cleaned,
                    ["country", "year", "source_uri"],
                )

            print(
                f"{country} chunk {chunk_number}: input={input_rows:,}, retained={retained_rows:,}",
                flush=True,
            )

        if first_country_chunk:
            raise RuntimeError(f"No eligible rows remained for {country}.")
        temporary_output.replace(output_path)
        audit_rows.append(
            {
                "country": country,
                "raw_rows": input_rows,
                "eligible_after_flags_text_length": eligible_rows,
                "cleaned_deduplicated_rows": retained_rows,
                "removed_as_exact_or_optional_near_duplicates": eligible_rows - retained_rows,
                "dedupe_scope": "within_country",
                "near_duplicate_prefix_tokens": args.near_duplicate_prefix_tokens,
                "raw_file": str(raw_path),
                "raw_file_size_bytes": raw_record["size_bytes"],
                "raw_file_sha256": raw_record["sha256"],
                "output_file": str(output_path),
            }
        )
        print(f"Saved {retained_rows:,} rows: {output_path}", flush=True)

    if combined_first:
        raise RuntimeError("No cleaned rows were written.")
    temporary_combined.replace(combined_output)

    output_specs = [
        ("country", ["country"], "denominator_country_total.csv"),
        ("country_year", ["country", "year"], "denominator_country_year.csv"),
        ("country_month", ["country", "month"], "denominator_country_month.csv"),
        ("country_week", ["country", "week"], "denominator_country_week.csv"),
        (
            "country_year_source",
            ["country", "year", "source_uri"],
            "denominator_country_year_source.csv",
        ),
    ]
    for counter_name, columns, filename in output_specs:
        table = counter_frame(counters[counter_name], columns).sort_values(columns)
        table.to_csv(args.output_dir / filename, index=False)

    audit = pd.DataFrame(audit_rows)
    audit_path = args.output_dir / "clean_dedupe_audit.csv"
    audit.to_csv(audit_path, index=False)
    total = int(audit["cleaned_deduplicated_rows"].sum())
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "data_dir": args.data_dir,
        "countries": countries,
        "chunksize": args.chunksize,
        "min_words": args.min_words,
        "dedupe_scope": "within_country",
        "dedupe_keys": ["nonblank_uri", "normalized_text_sha256"],
        "near_duplicate_prefix_tokens": args.near_duplicate_prefix_tokens,
        "total_cleaned_deduplicated_rows": total,
        "audit": file_record(audit_path),
    }
    manifest_path = args.output_dir / "clean_dedupe_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("\nCountry totals:", flush=True)
    print(audit[["country", "raw_rows", "cleaned_deduplicated_rows"]].to_string(index=False), flush=True)
    print(f"Total cleaned/deduplicated rows: {total:,}", flush=True)
    print(f"Saved combined minimal file: {combined_output}", flush=True)
    print(f"Saved audit: {audit_path}", flush=True)
    print(f"Saved run manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
