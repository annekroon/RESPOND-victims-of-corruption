"""Clean and deduplicate the raw corruption-query country files.

This is the reproducible entry point for the clean/deduplicate step. It loads
``*_news.csv`` files from
Research Drive through the WebDAV API, normalizes article text, removes source
duplicates and exact/near-exact duplicate text, and writes cleaned country files
plus denominator tables.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)

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
    parser.add_argument("--data-dir", default=RD_BASE_DIR, help="Research Drive folder with *_news.csv files.")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Countries to load. Omit to discover all *_news.csv files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting cleaned output files.",
    )
    return parser.parse_args()


def choose_text_column(dataframe) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column
    raise ValueError("No usable text column found. Expected one of: " + ", ".join(TEXT_COLUMN_CANDIDATES))


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
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_hash(text) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def text_hash(text) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def cheap_near_duplicate_fingerprint(text, n_tokens: int = 80) -> str:
    normalized = normalize_for_hash(text)
    fingerprint = " ".join(normalized.split()[:n_tokens])
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()


def clean_and_dedupe(dataframe, min_words: int):
    import pandas as pd

    df_clean = dataframe.copy()
    print(f"Starting rows: {len(df_clean):,}", flush=True)

    text_col = choose_text_column(df_clean)
    print(f"Using text column: {text_col}", flush=True)
    df_clean["article_text"] = df_clean[text_col].map(normalize_text)

    if "isDuplicate" in df_clean.columns:
        before = len(df_clean)
        is_duplicate = df_clean["isDuplicate"].astype(str).str.lower().isin(["true", "1", "yes"])
        df_clean = df_clean[~is_duplicate].copy()
        print(f"After dropping isDuplicate=True rows: {len(df_clean):,} (-{before - len(df_clean):,})", flush=True)

    before = len(df_clean)
    df_clean["word_count"] = df_clean["article_text"].str.split().str.len().fillna(0).astype(int)
    df_clean = df_clean[df_clean["article_text"].ne("")].copy()
    print(f"After removing missing/empty text: {len(df_clean):,} (-{before - len(df_clean):,})", flush=True)

    before = len(df_clean)
    df_clean = df_clean[df_clean["word_count"] >= min_words].copy()
    print(f"After removing articles with word_count < {min_words}: {len(df_clean):,} (-{before - len(df_clean):,})", flush=True)

    if "dateTime" in df_clean.columns:
        date_source = df_clean["dateTime"]
    elif "dateTimePub" in df_clean.columns:
        date_source = df_clean["dateTimePub"]
    elif "date" in df_clean.columns:
        date_source = df_clean["date"]
    else:
        date_source = pd.Series(pd.NaT, index=df_clean.index)

    df_clean["date_parsed"] = pd.to_datetime(date_source, errors="coerce", utc=True)
    df_clean["year"] = df_clean["date_parsed"].dt.year.astype("Int64")
    df_clean["month"] = df_clean["date_parsed"].dt.to_period("M").astype(str)
    df_clean["week"] = df_clean["date_parsed"].dt.to_period("W").apply(lambda period: period.start_time if pd.notna(period) else pd.NaT)

    if "source.uri" in df_clean.columns and "source_uri" not in df_clean.columns:
        df_clean["source_uri"] = df_clean["source.uri"].fillna("").astype(str).str.strip().str.lower()
    elif "source_uri" in df_clean.columns:
        df_clean["source_uri"] = df_clean["source_uri"].fillna("").astype(str).str.strip().str.lower()

    if "uri" in df_clean.columns:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["uri"], keep="first").copy()
        print(f"After URI dedupe: {len(df_clean):,} (-{before - len(df_clean):,})", flush=True)

    df_clean["text_hash"] = df_clean["article_text"].map(text_hash)
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["text_hash"], keep="first").copy()
    print(f"After exact text dedupe: {len(df_clean):,} (-{before - len(df_clean):,})", flush=True)

    df_clean["near_dup_hash"] = df_clean["article_text"].map(cheap_near_duplicate_fingerprint)
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["near_dup_hash"], keep="first").copy()
    print(f"After cheap near-duplicate dedupe: {len(df_clean):,} (-{before - len(df_clean):,})", flush=True)

    return df_clean


def assert_can_write_outputs(output_dir: Path, overwrite: bool) -> None:
    if overwrite:
        return
    existing = sorted(output_dir.glob("*_cleaned_deduped.csv.gz"))
    if existing:
        raise FileExistsError(
            f"Found {len(existing)} existing cleaned output file(s) in {output_dir}. "
            "Use --overwrite if you want to replace them."
        )


def main() -> None:
    args = parse_args()

    import pandas as pd
    from dataloader import load_country_news_files_webdav

    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert_can_write_outputs(args.output_dir, args.overwrite)

    print(f"Research Drive input directory: {args.data_dir}", flush=True)
    print(f"Output directory:              {args.output_dir}", flush=True)

    df = load_country_news_files_webdav(data_dir=args.data_dir, countries=args.countries)
    print(f"Loaded {len(df):,} rows and {len(df.columns):,} columns.", flush=True)

    df_clean = clean_and_dedupe(df, args.min_words)

    keep_columns = [column for column in KEEP_CLEANED_COLUMNS if column in df_clean.columns]
    minimal_columns = [column for column in MINIMAL_COMBINED_COLUMNS if column in df_clean.columns]

    combined_output = args.output_dir / "all_countries_cleaned_deduped_minimal.csv.gz"
    df_clean[minimal_columns].to_csv(combined_output, index=False, compression="gzip")
    print(f"Saved minimal combined cleaned file: {combined_output}", flush=True)

    for country, country_df in df_clean.groupby("country", dropna=False):
        safe_country = str(country).replace("/", "_")
        country_output = args.output_dir / f"{safe_country}_cleaned_deduped.csv.gz"
        country_df[keep_columns].to_csv(country_output, index=False, compression="gzip")
        print(f"Saved {len(country_df):,} rows: {country_output}", flush=True)

    denom_country_total = (
        df_clean.groupby("country", dropna=False)
        .size()
        .reset_index(name="total_articles")
        .sort_values("total_articles", ascending=False)
    )
    denom_country_year = df_clean.groupby(["country", "year"], dropna=False).size().reset_index(name="total_articles")
    denom_country_month = df_clean.groupby(["country", "month"], dropna=False).size().reset_index(name="total_articles")
    denom_country_week = df_clean.groupby(["country", "week"], dropna=False).size().reset_index(name="total_articles")

    denom_country_total.to_csv(args.output_dir / "denominator_country_total.csv", index=False)
    denom_country_year.to_csv(args.output_dir / "denominator_country_year.csv", index=False)
    denom_country_month.to_csv(args.output_dir / "denominator_country_month.csv", index=False)
    denom_country_week.to_csv(args.output_dir / "denominator_country_week.csv", index=False)

    if "source_uri" in df_clean.columns:
        denom_country_year_source = (
            df_clean.groupby(["country", "year", "source_uri"], dropna=False)
            .size()
            .reset_index(name="total_articles")
        )
        denom_country_year_source.to_csv(args.output_dir / "denominator_country_year_source.csv", index=False)

    print("Country totals:", flush=True)
    print(denom_country_total.to_string(index=False), flush=True)
    print(f"Total cleaned/deduped rows: {len(df_clean):,}", flush=True)


if __name__ == "__main__":
    main()
