"""Data loading helpers for RESPOND corruption annotation analysis."""

import os
from typing import Dict, List

import pandas as pd

from config import (
    ANNOTATION_ENCODING,
    ANNOTATION_FILE,
    ANNOTATION_PATH,
    NEWS_FOLDER,
    SELECTED_COUNTRIES,
    VALID_CORRUPTION_LABELS,
)


def load_selected_outlets(outlet_dir: str, countries: List[str]) -> Dict[str, List[str]]:
    selected_outlets = {}
    outlet_dir = os.path.expanduser(outlet_dir)

    for country in countries:
        outlet_file = os.path.join(outlet_dir, f"{country}_outlets_selection.txt")
        if os.path.exists(outlet_file):
            with open(outlet_file, "r", encoding="utf-8") as f:
                outlets = [line.strip().lower() for line in f if line.strip()]
                selected_outlets[country] = outlets
        else:
            print(f"No outlet file for {country}: {outlet_file}")
            selected_outlets[country] = []

    return selected_outlets


def load_and_prepare_data(news_folder: str, countries: List[str], outlet_dir: str) -> pd.DataFrame:
    news_folder = os.path.expanduser(news_folder)
    outlet_dir = os.path.expanduser(outlet_dir)
    all_data = []

    selected_outlets = load_selected_outlets(outlet_dir, countries)

    for country in countries:
        file_path = os.path.join(news_folder, f"{country}_news.csv")
        print(f"Looking for: {file_path}")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        print(f"Found: {file_path}")
        df = pd.read_csv(file_path)
        df = df.drop_duplicates(subset=["title", "body"])

        df["source.uri"] = df["source.uri"].astype(str).str.strip().str.lower()
        allowed_outlets = selected_outlets.get(country, [])
        if allowed_outlets:
            df = df[df["source.uri"].isin(allowed_outlets)]
            print(f"{country}: {len(df)} articles after outlet filtering")
        else:
            print(f"No selected outlets for {country}, skipping all rows")
            df = df.iloc[0:0]

        if not df.empty:
            df["country"] = country
            df["combined_text"] = df["title"].astype(str).str.strip() + "\n" + df["body"].astype(str).str.strip()
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No valid data after outlet filtering for countries: {countries}")

    return pd.concat(all_data, ignore_index=True)


def load_country_news_files(
    news_folder: str = NEWS_FOLDER,
    countries: List[str] = SELECTED_COUNTRIES,
    add_combined_text: bool = True,
) -> pd.DataFrame:
    """Load raw country news CSV files such as Bulgaria_news.csv."""
    news_folder = os.path.expanduser(news_folder)
    all_data = []

    for country in countries:
        file_path = os.path.join(news_folder, f"{country}_news.csv")
        print(f"Looking for: {file_path}")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)
        df["country"] = country

        if add_combined_text and {"title", "body"}.issubset(df.columns):
            title = df["title"].fillna("").astype(str).str.strip()
            body = df["body"].fillna("").astype(str).str.strip()
            df["combined_text"] = title + "\n" + body

        all_data.append(df)
        print(f"Loaded {len(df):,} rows from {os.path.basename(file_path)}")

    if not all_data:
        raise FileNotFoundError(f"No country news files found in {news_folder} for countries: {countries}")

    return pd.concat(all_data, ignore_index=True)


def balanced_sample(df: pd.DataFrame, total_samples: int, countries: List[str]) -> pd.DataFrame:
    df = df.copy()
    df["dateTime"] = pd.to_datetime(df["dateTime"], errors="coerce")
    df["year"] = df["dateTime"].dt.year
    df["month"] = df["dateTime"].dt.month

    per_country = total_samples // len(countries)
    sampled_dfs = []

    for country in countries:
        country_df = df[df["country"] == country]
        groups = country_df.groupby(["year", "month"])
        group_sizes = groups.size()

        if group_sizes.empty:
            continue

        group_proportions = group_sizes / group_sizes.sum()
        group_samples = (group_proportions * per_country).round().astype(int)

        sampled_groups = []
        for group_key, n in group_samples.items():
            group_data = groups.get_group(group_key)
            n = min(n, len(group_data))
            if n > 0:
                sampled_groups.append(group_data.sample(n, random_state=42))

        if sampled_groups:
            sampled_dfs.append(pd.concat(sampled_groups))

    if not sampled_dfs:
        raise ValueError("No rows were sampled. Check countries, dates, and total_samples.")

    result = pd.concat(sampled_dfs).reset_index(drop=True)
    return result.drop(columns=["year", "month"], errors="ignore")


def load_human_annotated_for_translation() -> pd.DataFrame:
    filepath = os.path.expanduser(os.path.join(ANNOTATION_PATH, ANNOTATION_FILE))
    df = pd.read_csv(filepath, encoding=ANNOTATION_ENCODING, encoding_errors="replace")

    df = df[df["country"].isin(SELECTED_COUNTRIES)]

    valid_labels = [label.lower() for label in VALID_CORRUPTION_LABELS]
    df["corruption_label_m"] = df["corruption_label_m"].str.strip().str.lower()
    df = df[df["corruption_label_m"].isin(valid_labels)]

    df = df[df["title"].notna() & df["body"].notna()]
    df = df[df["title"].str.strip().ne("") & df["body"].str.strip().ne("")]

    df["combined_text"] = df["title"].str.strip() + "\n" + df["body"].str.strip()

    if "country" in df.columns:
        df["country"] = df["country"].str.strip()
    else:
        df["country"] = "manual_annotated"

    return df
