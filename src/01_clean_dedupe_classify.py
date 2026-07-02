"""Clean, deduplicate, and classify RESPOND political-corruption news.

This script processes country CSV files one at a time, saves cleaned/deduped
country files, classifies with checkpointing, and writes a final subset of
articles labeled Yes, Maybe, or Unclear.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Iterable

pd = None
tqdm = None


PROJECT_DIR = Path("~/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)").expanduser()
OUTPUT_DIR = PROJECT_DIR / "output" / "political_corruption_pipeline"

COUNTRIES = [
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

TEXT_COLUMN_CANDIDATES = [
    "translated_text",
    "combined_text",
    "body",
    "text",
    "article_text",
    "content",
]

USE_KEYWORD_PREFILTER = False
MAX_CHARS = 6000

# This English keyword prefilter is safest when article_text is translated
# English text. If article_text is original-language text, use multilingual
# keyword lists before relying on this filter.
CORRUPTION_KEYWORDS = [
    "corruption",
    "corrupt",
    "bribery",
    "bribe",
    "embezzlement",
    "fraud",
    "kickback",
    "nepotism",
    "cronyism",
    "clientelism",
    "conflict of interest",
    "abuse of office",
    "misuse of funds",
    "public contract",
    "procurement",
    "tender",
    "lobbying",
    "money laundering",
    "campaign finance",
    "vote buying",
    "state capture",
]

KEEP_CLEANED_COLUMNS = [
    "uri",
    "country",
    "dateTime",
    "date_parsed",
    "year",
    "source.uri",
    "article_text",
    "word_count",
    "text_hash",
    "near_dup_hash",
]

CLASSIFIED_COLUMNS = [
    "uri",
    "country",
    "date_parsed",
    "year",
    "source_uri",
    "article_text",
    "word_count",
    "text_hash",
    "near_dup_hash",
    "pc_label",
    "pc_confidence",
    "pc_rationale",
    "pc_evidence",
    "classification_error",
]


def _ensure_runtime_deps() -> None:
    global pd, tqdm
    if pd is None:
        import pandas as _pd

        pd = _pd
    if tqdm is None:
        from tqdm import tqdm as _tqdm

        tqdm = _tqdm


def build_input_files(project_dir: Path) -> dict[str, Path]:
    return {country: project_dir / f"{country}_news.csv" for country in COUNTRIES}


def choose_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "No usable text column found. Expected one of: "
        + ", ".join(TEXT_COLUMN_CANDIDATES)
    )


def normalize_text(text) -> str:
    if not isinstance(text, str):
        return ""
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
    tokens = normalized.split()[:n_tokens]
    fingerprint = " ".join(tokens)
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()


def passes_keyword_prefilter(text) -> bool:
    normalized = normalize_text(text).lower()
    return any(keyword in normalized for keyword in CORRUPTION_KEYWORDS)


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "dateTime" in df.columns:
        source = df["dateTime"]
    elif "date" in df.columns:
        source = df["date"]
    else:
        source = pd.Series(pd.NaT, index=df.index)

    df["date_parsed"] = pd.to_datetime(source, errors="coerce", utc=True)
    df["year"] = df["date_parsed"].dt.year.astype("Int64")
    return df


def _print_step(country: str, message: str, rows: int) -> None:
    print(f"[{country}] {message}: {rows:,} rows")


def clean_country_file(
    country: str,
    input_file: Path,
    output_dir: Path,
    min_words: int = 80,
) -> Path:
    _ensure_runtime_deps()
    print("\n" + "=" * 80)
    print(f"Country: {country}")
    print(f"File:    {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found for {country}: {input_file}")

    df = pd.read_csv(input_file, low_memory=False)
    _print_step(country, "loaded", len(df))

    text_col = choose_text_column(df)
    print(f"[{country}] using text column: {text_col}")
    df["article_text"] = df[text_col].map(normalize_text)

    df["word_count"] = df["article_text"].str.split().str.len().fillna(0).astype(int)
    df = df[df["article_text"].ne("")].copy()
    _print_step(country, "after removing missing/empty text", len(df))

    df = df[df["word_count"] >= min_words].copy()
    _print_step(country, f"after removing articles with word_count < {min_words}", len(df))

    df = normalize_dates(df)
    if "country" not in df.columns:
        df["country"] = country
    else:
        df["country"] = df["country"].fillna(country)

    if "uri" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["uri"], keep="first").copy()
        print(f"[{country}] dropped duplicate uri rows: {before - len(df):,}")
        _print_step(country, "after uri dedupe", len(df))

    df["text_hash"] = df["article_text"].map(text_hash)
    before = len(df)
    df = df.drop_duplicates(subset=["text_hash"], keep="first").copy()
    print(f"[{country}] dropped exact normalized text duplicates: {before - len(df):,}")
    _print_step(country, "after text_hash dedupe", len(df))

    df["near_dup_hash"] = df["article_text"].map(cheap_near_duplicate_fingerprint)
    before = len(df)
    df = df.drop_duplicates(subset=["near_dup_hash"], keep="first").copy()
    print(f"[{country}] dropped cheap near-duplicates: {before - len(df):,}")
    _print_step(country, "after near-duplicate dedupe", len(df))

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_file = output_dir / f"{country}_cleaned_deduped.csv"
    keep_columns = [column for column in KEEP_CLEANED_COLUMNS if column in df.columns]
    df[keep_columns].to_csv(cleaned_file, index=False)
    print(f"[{country}] saved cleaned file: {cleaned_file}")
    return cleaned_file


def _load_classifier():
    try:
        from utils.classifier import classify_article
    except ImportError as exc:
        raise ImportError(
            "Could not import classify_article from utils.classifier. Run this "
            "script from the project/environment where that package is available."
        ) from exc
    return classify_article


def _source_uri(row: pd.Series):
    return row.get("source.uri", row.get("source_uri", ""))


def _base_classification_row(row: pd.Series) -> dict:
    return {
        "uri": row.get("uri", ""),
        "country": row.get("country", ""),
        "date_parsed": row.get("date_parsed", ""),
        "year": row.get("year", ""),
        "source_uri": _source_uri(row),
        "article_text": row.get("article_text", ""),
        "word_count": row.get("word_count", ""),
        "text_hash": row.get("text_hash", ""),
        "near_dup_hash": row.get("near_dup_hash", ""),
        "classification_error": "",
    }


def _save_classification_checkpoint(existing: pd.DataFrame, new_rows: list[dict], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows, columns=CLASSIFIED_COLUMNS)
    if existing.empty:
        combined = new_df
    else:
        combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(output_file, index=False)


def classify_cleaned_file(
    input_file: Path,
    output_file: Path,
    text_col: str = "article_text",
    batch_save_every: int = 250,
    use_keyword_prefilter: bool = USE_KEYWORD_PREFILTER,
    max_chars: int = MAX_CHARS,
) -> Path:
    _ensure_runtime_deps()
    classify_article = _load_classifier()

    df = pd.read_csv(input_file, low_memory=False)
    existing = pd.DataFrame(columns=CLASSIFIED_COLUMNS)
    completed_hashes: set[str] = set()

    if output_file.exists():
        existing = pd.read_csv(output_file, low_memory=False)
        if "text_hash" in existing.columns:
            completed_hashes = set(existing["text_hash"].dropna().astype(str))
        print(f"Resuming {output_file}: {len(completed_hashes):,} rows already classified")

    if "text_hash" not in df.columns:
        df["text_hash"] = df[text_col].map(text_hash)

    todo = df[~df["text_hash"].astype(str).isin(completed_hashes)].copy()
    print(f"Classifying {len(todo):,} remaining rows from {input_file.name}")

    new_rows: list[dict] = []
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc=f"Classifying {input_file.stem}"):
        article_text = normalize_text(row.get(text_col, ""))
        out_row = _base_classification_row(row)

        if not article_text:
            out_row.update(
                {
                    "pc_label": "No",
                    "pc_confidence": "",
                    "pc_rationale": "No content",
                    "pc_evidence": "",
                }
            )
        elif use_keyword_prefilter and not passes_keyword_prefilter(article_text):
            out_row.update(
                {
                    "pc_label": "No",
                    "pc_confidence": "keyword_prefilter",
                    "pc_rationale": "No corruption-related keyword found",
                    "pc_evidence": "",
                }
            )
        else:
            try:
                output = classify_article(article_text[:max_chars])
                out_row.update(
                    {
                        "pc_label": output.get("tentative_label", ""),
                        "pc_confidence": output.get("confidence", ""),
                        "pc_rationale": output.get("rationale", ""),
                        "pc_evidence": "; ".join(output.get("highlights", [])),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - keep long-running jobs alive.
                out_row.update(
                    {
                        "pc_label": "",
                        "pc_confidence": "",
                        "pc_rationale": "",
                        "pc_evidence": "",
                        "classification_error": repr(exc),
                    }
                )

        new_rows.append(out_row)

        if len(new_rows) % batch_save_every == 0:
            _save_classification_checkpoint(existing, new_rows, output_file)
            print(f"Saved checkpoint: {output_file} ({len(new_rows):,} new rows)")

    _save_classification_checkpoint(existing, new_rows, output_file)
    print(f"Saved classified file: {output_file}")
    return output_file


def classify_country_files(
    countries: Iterable[str],
    output_dir: Path,
    batch_save_every: int = 250,
    use_keyword_prefilter: bool = USE_KEYWORD_PREFILTER,
) -> list[Path]:
    classified_files = []
    for country in countries:
        input_file = output_dir / f"{country}_cleaned_deduped.csv"
        output_file = output_dir / f"{country}_political_corruption_classified.csv"
        classified_files.append(
            classify_cleaned_file(
                input_file=input_file,
                output_file=output_file,
                batch_save_every=batch_save_every,
                use_keyword_prefilter=use_keyword_prefilter,
            )
        )
    return classified_files


def create_final_subset(output_dir: Path, countries: Iterable[str]) -> Path:
    _ensure_runtime_deps()
    frames = []
    for country in countries:
        classified_file = output_dir / f"{country}_political_corruption_classified.csv"
        if not classified_file.exists():
            print(f"Skipping missing classified file: {classified_file}")
            continue
        frames.append(pd.read_csv(classified_file, low_memory=False))

    if not frames:
        raise FileNotFoundError(f"No classified country files found in {output_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["pc_label_clean"] = df["pc_label"].astype(str).str.strip()
    subset = df[df["pc_label_clean"].isin(["Yes", "Maybe", "Unclear"])].copy()

    output_file = output_dir / "political_corruption_subset_yes_maybe_unclear.csv"
    subset.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("Final political corruption subset")
    print(f"Total rows classified: {len(df):,}")
    print(f"Total rows kept:       {len(subset):,}")

    if not subset.empty:
        print("\nRows kept by country:")
        print(subset.groupby("country").size().sort_values(ascending=False))

        print("\nRows kept by country and pc_label_clean:")
        print(subset.groupby(["country", "pc_label_clean"]).size())

        print("\nRows kept by country and year:")
        print(subset.groupby(["country", "year"]).size())

    print(f"\nSaved final subset: {output_file}")
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean, deduplicate, classify, and subset RESPOND news for political corruption."
    )
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--countries", nargs="+", default=COUNTRIES)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--batch-save-every", type=int, default=250)
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-classify", action="store_true")
    parser.add_argument("--skip-final-subset", action="store_true")
    parser.add_argument("--use-keyword-prefilter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.expanduser()
    output_dir = args.output_dir.expanduser() if args.output_dir else project_dir / "output" / "political_corruption_pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)

    countries = args.countries
    input_files = build_input_files(project_dir)

    if not args.skip_clean:
        for country in countries:
            clean_country_file(
                country=country,
                input_file=input_files[country],
                output_dir=output_dir,
                min_words=args.min_words,
            )

    if not args.skip_classify:
        classify_country_files(
            countries=countries,
            output_dir=output_dir,
            batch_save_every=args.batch_save_every,
            use_keyword_prefilter=args.use_keyword_prefilter,
        )

    if not args.skip_final_subset:
        create_final_subset(output_dir=output_dir, countries=countries)


if __name__ == "__main__":
    main()
