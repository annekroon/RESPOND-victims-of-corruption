"""Build attention CSVs, figures, and descriptive LaTeX tables.

This is the production implementation behind numbered step 07. The former
notebook code is kept out of the execution path so a clean run cannot depend on
stale notebook state or cell order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.source_filter import (
    apply_source_inclusion_filter,
    load_source_decisions,
    source_filter_summary,
)

DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)

def build_attention_analysis(pipeline_dir: Path) -> None:
    display = print
    from pathlib import Path

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    import pandas as pd

    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 100)

    PIPELINE_DIR = pipeline_dir
    CLASSIFIER_DIR = PIPELINE_DIR / "silver_classifier"
    CLASSIFIED_DIR = CLASSIFIER_DIR / "classified_country_files"
    FIGURE_DIR = PIPELINE_DIR / "attention_figures"
    TABLE_DIR = PIPELINE_DIR / "attention_tables"
    SOURCE_INCLUSION_DIR = PIPELINE_DIR / "source_inclusion"
    SOURCE_DECISION_FILE = SOURCE_INCLUSION_DIR / "political_corruption_all_sources_classified.xlsx"
    # Canonical classified files were already source-filtered in step 02.
    # Set True only for an explicit diagnostic recheck, never for production output.
    APPLY_SOURCE_INCLUSION_FILTER = False

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_INCLUSION_DIR.mkdir(parents=True, exist_ok=True)

    # Local fallback paths. If these do not exist, use the WebDAV fallbacks below.
    TOTAL_WEEK_PATH = PIPELINE_DIR / "total_news_coverage_week.csv"
    TOTAL_MONTH_PATH = PIPELINE_DIR / "total_news_coverage_month.csv"

    # Mounted WebDAV source for total-news denominator files.
    # This folder contains one weekly count CSV per country, e.g. Bulgaria_weekly_count.csv.
    TOTAL_COVERAGE_MOUNT_DIR = Path(
        "/home/akroon/webdav/ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/"
        "weekly_counts_total_coverage"
    )

    # Research Drive/WebDAV API fallback for the same folder.
    TOTAL_COVERAGE_RD_DIR = (
        "ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/"
        "weekly_counts_total_coverage"
    )

    COUNTRY_ORDER = [
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

    COUNTRY_LABELS = {
        "United_Kingdom": "United Kingdom",
    }

    # Monthly is usually clearer for manuscript figures; weekly is also computed if a weekly denominator exists.
    PLOT_LEVEL = "month"  # "month" or "week"

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#333333",
        "grid.color": "#dddddd",
        "grid.linewidth": 0.7,
    })

    print(f"Pipeline dir: {PIPELINE_DIR}")
    print(f"Classified dir: {CLASSIFIED_DIR}")
    print(f"Source decision file: {SOURCE_DECISION_FILE}")
    print(f"Apply source inclusion filter: {APPLY_SOURCE_INCLUSION_FILTER}")
    print(f"Local total-news weekly denominator path:  {TOTAL_WEEK_PATH}")
    print(f"Local total-news monthly denominator path: {TOTAL_MONTH_PATH}")
    print(f"Mounted total-news denominator folder:    {TOTAL_COVERAGE_MOUNT_DIR}")
    print(f"WebDAV total-news denominator folder:      {TOTAL_COVERAGE_RD_DIR}")

    COUNT_COLUMN_CANDIDATES = [
        "total_news_articles",
        "total_articles",
        "total_coverage",
        "n_articles",
        "count",
        "n",
        "weekly_count",
    ]

    PERIOD_COLUMN_CANDIDATES = {
        "week": [
            "week",
            "week_start",
            "week_start_date",
            "week_starting",
            "date_week",
            "period",
            "date",
            "start_date",
            "start",
            "from",
        ],
        "month": ["month", "month_start", "date_month", "period", "date", "start_date"],
    }

    FILENAME_COUNTRY_MAP = {
        "UK": "United_Kingdom",
        "United Kingdom": "United_Kingdom",
        "United_Kingdom": "United_Kingdom",
    }


    def country_from_filename(filename):
        stem = Path(filename).stem
        stem = stem.replace("_weekly_count", "").replace("_week_count", "")
        stem = stem.replace("_weekly_counts", "").replace("_week_counts", "")
        return FILENAME_COUNTRY_MAP.get(stem, stem)


    def normalize_column_name(column):
        return str(column).strip().lower().replace(" ", "_").replace("-", "_")


    def infer_period_column(data, period_column, source_name=""):
        normalized_to_original = {normalize_column_name(column): column for column in data.columns}

        for candidate in PERIOD_COLUMN_CANDIDATES[period_column]:
            normalized_candidate = normalize_column_name(candidate)
            if normalized_candidate in normalized_to_original:
                return normalized_to_original[normalized_candidate]

        fuzzy_tokens = [period_column, "date", "start", "period"]
        for normalized, original in normalized_to_original.items():
            if any(token in normalized for token in fuzzy_tokens):
                parsed = pd.to_datetime(data[original], errors="coerce")
                if parsed.notna().mean() >= 0.75:
                    print(f"Inferred {period_column!r} column {original!r} for {source_name}")
                    return original

        # Last resort: choose the first column that parses mostly as dates.
        for original in data.columns:
            parsed = pd.to_datetime(data[original], errors="coerce")
            if parsed.notna().mean() >= 0.90:
                print(f"Inferred {period_column!r} date column {original!r} for {source_name}")
                return original

        raise ValueError(
            f"Could not find a {period_column!r} column in {source_name}. "
            f"Expected one of: {PERIOD_COLUMN_CANDIDATES[period_column]}. "
            f"Available columns: {list(data.columns)}"
        )


    def infer_count_column(data, period_column, source_name=""):
        normalized_to_original = {normalize_column_name(column): column for column in data.columns}

        for candidate in COUNT_COLUMN_CANDIDATES:
            normalized_candidate = normalize_column_name(candidate)
            if normalized_candidate in normalized_to_original:
                return normalized_to_original[normalized_candidate]

        numeric_candidates = []
        for column in data.columns:
            if column == period_column or normalize_column_name(column) == "country":
                continue
            numeric = pd.to_numeric(data[column], errors="coerce")
            if numeric.notna().mean() >= 0.90:
                numeric_candidates.append(column)

        if len(numeric_candidates) == 1:
            print(f"Using numeric count column {numeric_candidates[0]!r} for {source_name}")
            return numeric_candidates[0]

        preferred_tokens = ["count", "total", "articles", "coverage", "n"]
        for column in numeric_candidates:
            normalized = normalize_column_name(column)
            if any(token in normalized for token in preferred_tokens):
                print(f"Using inferred count column {column!r} for {source_name}")
                return column

        raise ValueError(
            f"Could not find a count column in {source_name}. Expected one of: {COUNT_COLUMN_CANDIDATES}. "
            f"Numeric candidates: {numeric_candidates}. Available columns: {list(data.columns)}"
        )


    def standardize_total_coverage(data, period_column, source_name="", country=None):
        data = data.copy()
        print(f"Columns in {source_name}: {list(data.columns)}")

        original_period_column = infer_period_column(data, period_column, source_name)
        if original_period_column != period_column:
            data = data.rename(columns={original_period_column: period_column})

        count_column = infer_count_column(data, period_column, source_name)

        if "country" not in data.columns:
            if country is None:
                raise ValueError(
                    f"Missing country column in {source_name}; pass country=... or include a country column."
                )
            data["country"] = country

        data = data.rename(columns={count_column: "total_news_articles"})
        data[period_column] = pd.to_datetime(data[period_column], errors="coerce")
        data["country"] = data["country"].astype(str).replace(FILENAME_COUNTRY_MAP)
        data = data[["country", period_column, "total_news_articles"]].copy()
        data["total_news_articles"] = pd.to_numeric(data["total_news_articles"], errors="coerce")
        data = data[data[period_column].notna() & data["total_news_articles"].notna()].copy()
        data["total_news_articles"] = data["total_news_articles"].astype(int)
        data["country"] = pd.Categorical(data["country"], COUNTRY_ORDER, ordered=True)
        return data


    def path_exists_safely(path):
        try:
            return path.exists()
        except OSError as exc:
            print(f"Could not access {path}: {exc}")
            return False


    def load_total_coverage_local(path, period_column):
        data = pd.read_csv(path)
        return standardize_total_coverage(data, period_column, source_name=str(path))


    def load_weekly_total_coverage_from_mount(folder):
        files = sorted(folder.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in mounted folder: {folder}")

        frames = []
        for path in files:
            country = country_from_filename(path.name)
            print(f"Reading: {path}")
            data = pd.read_csv(path)
            data = standardize_total_coverage(data, "week", source_name=str(path), country=country)
            data["source_file"] = path.name
            frames.append(data)
            print(f"Loaded {country}: {len(data):,} rows")

        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.groupby(["country", "week"], observed=True)["total_news_articles"]
            .sum()
            .reset_index()
        )
        return combined


    def list_rd_csvs(rd_dir):
        from rd_io import rd_list_dir

        files = rd_list_dir(rd_dir)
        return sorted(file for file in files if file.lower().endswith(".csv"))


    def read_rd_csv(rd_dir, filename):
        from rd_io import rd_join, rd_read_csv_df

        return rd_read_csv_df(rd_join(rd_dir, filename))


    def load_weekly_total_coverage_from_webdav(rd_dir):
        files = list_rd_csvs(rd_dir)
        if not files:
            raise FileNotFoundError(f"No CSV files found in WebDAV folder: {rd_dir}")

        print("WebDAV total-coverage CSV files found:")
        for filename in files:
            print(f"- {filename}")

        weekly_files = [filename for filename in files if "week" in filename.lower()]
        selected_files = weekly_files or files

        frames = []
        for filename in selected_files:
            country = country_from_filename(filename)
            data = read_rd_csv(rd_dir, filename)
            data = standardize_total_coverage(data, "week", source_name=filename, country=country)
            data["source_file"] = filename
            frames.append(data)

        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.groupby(["country", "week"], observed=True)["total_news_articles"]
            .sum()
            .reset_index()
        )
        return combined


    # Load weekly total-news denominator.
    if path_exists_safely(TOTAL_WEEK_PATH):
        total_week = load_total_coverage_local(TOTAL_WEEK_PATH, "week")
        print(f"Loaded local cached weekly total-news denominator: {TOTAL_WEEK_PATH}")
    else:
        total_week = None

    if total_week is None and path_exists_safely(TOTAL_COVERAGE_MOUNT_DIR):
        try:
            print(f"Loading weekly total-news denominator from mounted WebDAV: {TOTAL_COVERAGE_MOUNT_DIR}")
            total_week = load_weekly_total_coverage_from_mount(TOTAL_COVERAGE_MOUNT_DIR)
            total_week.to_csv(TOTAL_WEEK_PATH, index=False)
            print(f"Cached mounted weekly denominator locally: {TOTAL_WEEK_PATH}")
        except OSError as exc:
            print(f"Mounted WebDAV read failed: {exc}")
            total_week = None

    if total_week is None:
        print(f"Local weekly total-news denominator missing or unreadable: {TOTAL_WEEK_PATH}")
        print(f"Mounted WebDAV folder missing or unreadable: {TOTAL_COVERAGE_MOUNT_DIR}")
        print("Trying WebDAV API weekly total-coverage folder...")
        total_week = load_weekly_total_coverage_from_webdav(TOTAL_COVERAGE_RD_DIR)
        total_week.to_csv(TOTAL_WEEK_PATH, index=False)
        print(f"Cached WebDAV weekly denominator locally: {TOTAL_WEEK_PATH}")

    # Always derive months from the canonical week bins. The political-corruption
    # numerator is aggregated through the same bins below.
    total_month = total_week.copy()
    total_month["month"] = total_month["week"].dt.to_period("M").dt.to_timestamp()
    total_month = (
        total_month.groupby(["country", "month"], observed=True)["total_news_articles"]
        .sum()
        .reset_index()
    )
    total_month.to_csv(TOTAL_MONTH_PATH, index=False)
    print(f"Derived and cached monthly denominator from weekly bins: {TOTAL_MONTH_PATH}")

    print("Monthly total-news denominator:", total_month.shape)
    display(total_month.head())
    print("Weekly total-news denominator:", total_week.shape)
    display(total_week.head())

    use_columns = {
        "country",
        "date_parsed",
        "year",
        "month",
        "week",
        "pred_political_corruption",
        "source_uri",
        "source.uri",
        "source_title",
        "source.title",
    }
    frames = []

    for country in COUNTRY_ORDER:
        path = CLASSIFIED_DIR / f"{country}_classified.csv.gz"
        if not path.exists():
            print(f"Missing classified file: {path}")
            continue

        data = pd.read_csv(path, usecols=lambda column: column in use_columns)
        data["country"] = country
        data = data[data["pred_political_corruption"].eq(1)].copy()
        frames.append(data)
        print(f"Loaded {country}: {len(data):,} source-eligible political-corruption articles")

    if not frames:
        raise FileNotFoundError(f"No classified files found in {CLASSIFIED_DIR}")

    pc_articles_unfiltered = pd.concat(frames, ignore_index=True)
    print(f"Total source-eligible political-corruption articles: {len(pc_articles_unfiltered):,}")

    if APPLY_SOURCE_INCLUSION_FILTER:
        source_decisions = load_source_decisions(SOURCE_DECISION_FILE)
        pc_articles, source_filter_merged = apply_source_inclusion_filter(
            pc_articles_unfiltered,
            source_decisions,
            country_column="country",
        )
        source_filter_report = source_filter_summary(source_filter_merged, group_columns=["country"])
        source_filter_report.to_csv(TABLE_DIR / "political_corruption_source_filter_summary_by_country.csv", index=False)
        source_filter_merged[[
            "country",
            "source_filter_decision",
            "source_filter_source_column",
        ]].value_counts().reset_index(name="rows").to_csv(
            TABLE_DIR / "political_corruption_source_filter_decision_counts.csv",
            index=False,
        )
        print(f"Total political-corruption articles after source filtering: {len(pc_articles):,}")
        display(source_filter_report)
    else:
        pc_articles = pc_articles_unfiltered.copy()

    pc_articles["date_parsed"] = pd.to_datetime(pc_articles["date_parsed"], errors="coerce", utc=True)
    pc_articles["date_naive"] = pc_articles["date_parsed"].dt.tz_convert(None)

    if "month" in pc_articles.columns:
        pc_articles["month"] = pd.to_datetime(pc_articles["month"], errors="coerce")
    else:
        pc_articles["month"] = pc_articles["date_naive"].dt.to_period("M").dt.to_timestamp()

    if "week" in pc_articles.columns:
        pc_articles["week"] = pd.to_datetime(pc_articles["week"], errors="coerce")
    else:
        pc_articles["week"] = pc_articles["date_naive"].dt.to_period("W").dt.start_time

    pc_articles["country"] = pd.Categorical(pc_articles["country"], COUNTRY_ORDER, ordered=True)

    print(f"Final political-corruption articles loaded: {len(pc_articles):,}")
    display(pc_articles.head())

    pc_week = (
        pc_articles.groupby(["country", "week"], observed=True)
        .size()
        .reset_index(name="political_corruption_articles")
    )

    # Aggregate both numerator and denominator from identical country-week bins.
    # This avoids assigning a boundary week differently on the two sides.
    pc_month = pc_week.copy()
    pc_month["month"] = pd.to_datetime(pc_month["week"]).dt.to_period("M").dt.to_timestamp()
    pc_month = (
        pc_month.groupby(["country", "month"], observed=True)["political_corruption_articles"]
        .sum()
        .reset_index()
    )

    attention_month = total_month.merge(pc_month, on=["country", "month"], how="left")
    attention_month["political_corruption_articles"] = attention_month["political_corruption_articles"].fillna(0).astype(int)
    attention_month["relative_attention"] = attention_month["political_corruption_articles"] / attention_month["total_news_articles"]
    attention_month["relative_attention_pct"] = attention_month["relative_attention"] * 100
    attention_month["country_label"] = attention_month["country"].astype(str).replace(COUNTRY_LABELS)

    if total_week is not None:
        attention_week = total_week.merge(pc_week, on=["country", "week"], how="left")
        attention_week["political_corruption_articles"] = attention_week["political_corruption_articles"].fillna(0).astype(int)
        attention_week["relative_attention"] = attention_week["political_corruption_articles"] / attention_week["total_news_articles"]
        attention_week["relative_attention_pct"] = attention_week["relative_attention"] * 100
        attention_week["country_label"] = attention_week["country"].astype(str).replace(COUNTRY_LABELS)
        attention_week.to_csv(TABLE_DIR / "political_corruption_attention_total_news_week.csv", index=False)
    else:
        attention_week = None

    attention_month.to_csv(TABLE_DIR / "political_corruption_attention_total_news_month.csv", index=False)

    print(f"Saved attention tables to: {TABLE_DIR}")
    display(attention_month.head())

    country_summary = (
        attention_month.groupby(["country", "country_label"], observed=True)
        .agg(
            total_news_articles=("total_news_articles", "sum"),
            political_corruption_articles=("political_corruption_articles", "sum"),
        )
        .reset_index()
    )
    country_summary["relative_attention"] = (
        country_summary["political_corruption_articles"] / country_summary["total_news_articles"]
    )
    country_summary["relative_attention_pct"] = country_summary["relative_attention"] * 100
    country_summary = country_summary.sort_values("relative_attention_pct", ascending=False)

    overall_total = country_summary["total_news_articles"].sum()
    overall_pc = country_summary["political_corruption_articles"].sum()
    overall_rate = overall_pc / overall_total

    print(f"Total news articles:                   {overall_total:,}")
    print(f"Total political-corruption articles:   {overall_pc:,}")
    print(f"Overall relative attention:            {overall_rate:.4%}")

    country_summary.to_csv(TABLE_DIR / "political_corruption_attention_total_news_country_summary.csv", index=False)
    display(country_summary)

    query_month_path = PIPELINE_DIR / "denominator_country_month.csv"
    if query_month_path.exists():
        query_month = pd.read_csv(query_month_path)
        query_month["month"] = pd.to_datetime(query_month["month"])
        query_month = query_month.rename(columns={"total_articles": "corruption_query_articles"})
        query_month["country"] = pd.Categorical(query_month["country"].astype(str), COUNTRY_ORDER, ordered=True)
        query_attention_month = query_month.merge(pc_month, on=["country", "month"], how="left")
        query_attention_month["political_corruption_articles"] = query_attention_month["political_corruption_articles"].fillna(0).astype(int)
        query_attention_month["share_within_corruption_query"] = (
            query_attention_month["political_corruption_articles"] / query_attention_month["corruption_query_articles"]
        )
        query_attention_month.to_csv(TABLE_DIR / "political_corruption_share_within_corruption_query_month.csv", index=False)
        display(query_attention_month.head())
    else:
        print(f"No corruption-query diagnostic denominator found at {query_month_path}")

    COUNTRY_COLORS = {
        "Bulgaria": "#3B6FB6",
        "France": "#E6862E",
        "Hungary": "#C83E4D",
        "Italy": "#2A9D8F",
        "Netherlands": "#5C9E3F",
        "Serbia": "#C9A227",
        "Sweden": "#8E6BBE",
        "Ukraine": "#D95F8D",
        "United_Kingdom": "#8A6A55",
    }


    def period_data(level=PLOT_LEVEL):
        if level == "week":
            if attention_week is None:
                raise ValueError("Weekly plot requested, but no weekly total-news denominator was loaded.")
            data = attention_week.copy()
            period = "week"
            label = "Weekly"
        elif level == "month":
            data = attention_month.copy()
            period = "month"
            label = "Monthly"
        else:
            raise ValueError("level must be 'week' or 'month'")
        return data, period, label


    def add_rolling_average(data, period, window=3):
        data = data.sort_values(["country", period]).copy()
        data[f"relative_attention_pct_roll{window}"] = (
            data.groupby("country", observed=True)["relative_attention_pct"]
            .transform(lambda series: series.rolling(window=window, min_periods=1).mean())
        )
        return data


    def save_figure(fig, name):
        png = FIGURE_DIR / f"{name}.png"
        pdf = FIGURE_DIR / f"{name}.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Saved {png}")
        print(f"Saved {pdf}")


    def format_time_axis(ax):
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", rotation=0)
        return ax


    def percent_axis(ax):
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        return ax


    def add_source_note(fig, text="Relative attention = predicted political-corruption articles / total news articles."):
        fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=8, color="#555555")

    data, period, level_label = period_data(PLOT_LEVEL)

    fig, ax = plt.subplots(figsize=(12, 6))

    for country in COUNTRY_ORDER:
        country_data = data[data["country"].astype(str).eq(country)].sort_values(period)
        if country_data.empty:
            continue
        ax.plot(
            country_data[period],
            country_data["relative_attention_pct"],
            label=COUNTRY_LABELS.get(country, country),
            color=COUNTRY_COLORS[country],
            linewidth=1.8,
            alpha=0.9,
        )

    ax.set_title(f"{level_label} Relative Attention To Political Corruption By Country")
    ax.set_ylabel("Political-corruption articles (% of total news coverage)")
    ax.set_xlabel("")
    format_time_axis(ax)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    save_figure(fig, f"political_corruption_relative_attention_total_news_{period}_country_lines")

    data, period, level_label = period_data(PLOT_LEVEL)

    fig, axes = plt.subplots(3, 3, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, country in zip(axes, COUNTRY_ORDER):
        country_data = data[data["country"].astype(str).eq(country)].sort_values(period)
        ax.plot(
            country_data[period],
            country_data["relative_attention_pct"],
            color=COUNTRY_COLORS[country],
            linewidth=1.8,
        )
        ax.fill_between(
            country_data[period],
            country_data["relative_attention_pct"],
            color=COUNTRY_COLORS[country],
            alpha=0.16,
        )
        ax.set_title(COUNTRY_LABELS.get(country, country), loc="left", fontweight="bold")
        format_time_axis(ax)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"{level_label} Relative Attention To Political Corruption", y=1.02, fontsize=14)
    fig.text(0.5, -0.01, "Year", ha="center")
    fig.text(0.0, 0.5, "% of total news coverage", va="center", rotation="vertical")
    fig.tight_layout()
    save_figure(fig, f"political_corruption_relative_attention_total_news_{period}_small_multiples")

    data, period, level_label = period_data(PLOT_LEVEL)

    wide_counts = (
        data.pivot_table(
            index=period,
            columns="country",
            values="political_corruption_articles",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        .reindex(columns=COUNTRY_ORDER)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        wide_counts.index,
        [wide_counts[country].to_numpy() for country in COUNTRY_ORDER],
        labels=[COUNTRY_LABELS.get(country, country) for country in COUNTRY_ORDER],
        colors=[COUNTRY_COLORS[country] for country in COUNTRY_ORDER],
        alpha=0.88,
    )
    ax.set_title(f"{level_label} Volume Of Political-Corruption Coverage")
    ax.set_ylabel("Predicted political-corruption articles")
    ax.set_xlabel("")
    format_time_axis(ax)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save_figure(fig, f"political_corruption_absolute_volume_{period}_stacked")

    yearly_attention = attention_month.copy()
    yearly_attention["year"] = yearly_attention["month"].dt.year
    yearly_attention = (
        yearly_attention.groupby(["country", "country_label", "year"], observed=True)
        .agg(
            total_news_articles=("total_news_articles", "sum"),
            political_corruption_articles=("political_corruption_articles", "sum"),
        )
        .reset_index()
    )
    yearly_attention["relative_attention_pct"] = (
        yearly_attention["political_corruption_articles"] / yearly_attention["total_news_articles"] * 100
    )

    heatmap_data = yearly_attention.pivot(index="country_label", columns="year", values="relative_attention_pct")
    heatmap_data = heatmap_data.reindex([COUNTRY_LABELS.get(country, country) for country in COUNTRY_ORDER])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns.astype(int))
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title("Relative Attention To Political Corruption By Country-Year")

    for y in range(heatmap_data.shape[0]):
        for x in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[y, x]
            if pd.notna(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="black")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Political-corruption articles (% of total news coverage)")
    fig.tight_layout()
    save_figure(fig, "political_corruption_relative_attention_total_news_country_year_heatmap")

    yearly_attention.to_csv(TABLE_DIR / "political_corruption_attention_total_news_country_year.csv", index=False)
    display(yearly_attention.head())

    ranking = country_summary.sort_values("relative_attention_pct", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9, 5.8))
    colors = [COUNTRY_COLORS.get(country, "#777777") for country in ranking["country"].astype(str)]
    ax.barh(ranking["country_label"], ranking["relative_attention_pct"], color=colors, alpha=0.92)
    ax.set_title("Average Attention To Political Corruption By Country")
    ax.set_xlabel("Political-corruption articles (% of total news coverage)")
    ax.set_ylabel("")
    percent_axis(ax)
    for y, value in enumerate(ranking["relative_attention_pct"]):
        ax.text(value + ranking["relative_attention_pct"].max() * 0.01, y, f"{value:.2f}%", va="center", fontsize=8)
    ax.set_xlim(0, ranking["relative_attention_pct"].max() * 1.18)
    add_source_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "political_corruption_attention_country_ranking_total_news")

    data, period, level_label = period_data("month")
    indexed = add_rolling_average(data, "month", window=3)
    indexed["country_mean_attention"] = indexed.groupby("country", observed=True)["relative_attention_pct"].transform("mean")
    indexed["attention_index"] = indexed["relative_attention_pct_roll3"] / indexed["country_mean_attention"] * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for country in COUNTRY_ORDER:
        country_data = indexed[indexed["country"].astype(str).eq(country)].sort_values("month")
        ax.plot(
            country_data["month"],
            country_data["attention_index"],
            label=COUNTRY_LABELS.get(country, country),
            color=COUNTRY_COLORS[country],
            linewidth=1.7,
            alpha=0.88,
        )
    ax.axhline(100, color="#222222", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_title("Indexed Political-Corruption Attention Over Time")
    ax.set_ylabel("Index, country average = 100")
    ax.set_xlabel("")
    format_time_axis(ax)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    add_source_note(fig, "Three-month rolling average; each country indexed to its own mean attention level.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "political_corruption_attention_monthly_indexed_country_lines")

    # Generate LaTeX attention tables from saved CSV outputs.
    #
    # This section deliberately reads the CSV files written above rather than using
    # in-memory data frames. That makes the reproducible chain explicit:
    # CSV outputs -> LaTeX tables -> upload to WebDAV/Overleaf.

    ATTENTION_LATEX_DIR = TABLE_DIR / "latex"
    ATTENTION_LATEX_DIR.mkdir(parents=True, exist_ok=True)

    COUNTRY_SUMMARY_CSV = TABLE_DIR / "political_corruption_attention_total_news_country_summary.csv"
    COUNTRY_YEAR_CSV = TABLE_DIR / "political_corruption_attention_total_news_country_year.csv"
    MONTH_CSV = TABLE_DIR / "political_corruption_attention_total_news_month.csv"
    country_summary_from_csv = pd.read_csv(COUNTRY_SUMMARY_CSV)
    country_year_from_csv = pd.read_csv(COUNTRY_YEAR_CSV)
    month_from_csv = pd.read_csv(MONTH_CSV, parse_dates=["month"])


    TABLE_FONT_SIZE = "\\scriptsize"
    TABLE_COL_SEP = "3pt"


    def should_resize_latex_table(dataframe, resize="auto"):
        """Resize only broad tables; never scale narrow tables up to text width."""
        if isinstance(resize, bool):
            return resize
        if resize != "auto":
            raise ValueError("resize must be True, False, or 'auto'")
        max_header_len = max((len(str(column)) for column in dataframe.columns), default=0)
        return len(dataframe.columns) >= 7 or max_header_len >= 22


    def add_table_font(latex, font_size=TABLE_FONT_SIZE, tabcolsep=TABLE_COL_SEP):
        begin = "\\begin{tabular}"
        if begin not in latex:
            return latex
        prefix = (
            "\\centering\n"
            f"{font_size}\n"
            f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n"
        )
        return latex.replace(begin, prefix + begin, 1)


    def resize_latex_tabular(latex):
        begin = "\\begin{tabular}"
        end = "\\end{tabular}"
        if begin not in latex or end not in latex:
            return latex
        latex = latex.replace(begin, "\\begin{adjustbox}{max width=\\textwidth}\n" + begin, 1)
        latex = latex.replace(end, end + "\n\\end{adjustbox}", 1)
        return latex


    def add_table_note(latex, note):
        return latex.replace("\\end{table}\n", f"\\par\\smallskip\\footnotesize{{{note}}}\n\\end{{table}}\n")


    def save_attention_latex_table(dataframe, filename, caption, label, note, resize="auto"):
        path = ATTENTION_LATEX_DIR / filename
        latex = dataframe.to_latex(
            index=False,
            escape=True,
            caption=caption,
            label=label,
            float_format="%.3f",
            bold_rows=False,
        )
        latex = add_table_font(latex)
        if should_resize_latex_table(dataframe, resize=resize):
            latex = resize_latex_tabular(latex)
        latex = add_table_note(latex, note)
        path.write_text(latex)
        print(f"Saved {path}")
        return path

    summary_table = country_summary_from_csv.copy()
    summary_table["share_of_pc_articles_pct"] = (
        summary_table["political_corruption_articles"]
        / summary_table["political_corruption_articles"].sum()
        * 100
    )
    summary_table = summary_table[
        [
            "country_label",
            "total_news_articles",
            "political_corruption_articles",
            "relative_attention_pct",
            "share_of_pc_articles_pct",
        ]
    ].rename(
        columns={
            "country_label": "Country",
            "total_news_articles": "Total news",
            "political_corruption_articles": "PC articles",
            "relative_attention_pct": "PC share of total news (%)",
            "share_of_pc_articles_pct": "Share of PC corpus (%)",
        }
    )

    save_attention_latex_table(
        summary_table,
        "table_attention_country_summary.tex",
        "Political-corruption attention by country relative to total news coverage.",
        "tab:pc-attention-country-summary",
        "Note. PC = political corruption. Relative attention is the percentage of total news coverage classified as political corruption.",
    )
    display(summary_table)


    # Source/outlet descriptives for the classified political-corruption corpus.
    SOURCE_COLUMN_CANDIDATES = ["source_uri", "source.uri", "source", "outlet", "source_title", "source.title"]
    source_column = next((column for column in SOURCE_COLUMN_CANDIDATES if column in pc_articles.columns), None)

    if source_column is None:
        print("No source/outlet column found in pc_articles. Available columns:")
        print(list(pc_articles.columns))
    else:
        print(f"Using source/outlet column: {source_column}")
        source_data = pc_articles.copy()
        source_data["source_clean"] = (
            source_data[source_column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"": "unknown", "nan": "unknown", "none": "unknown"})
        )
        source_data["country_label"] = source_data["country"].astype(str).map(COUNTRY_LABELS).fillna(source_data["country"].astype(str))

        source_summary = (
            source_data.groupby("source_clean", observed=True)
            .agg(
                political_corruption_articles=("source_clean", "size"),
                country_count=("country", "nunique"),
                first_year=("year", "min"),
                last_year=("year", "max"),
            )
            .reset_index()
            .sort_values("political_corruption_articles", ascending=False)
        )
        source_summary["share_of_pc_corpus_pct"] = (
            source_summary["political_corruption_articles"] / len(source_data) * 100
        )
        source_summary.to_csv(TABLE_DIR / "political_corruption_source_summary.csv", index=False)

        top_sources_overall = source_summary.head(30).copy()
        top_sources_overall_table = top_sources_overall[
            ["source_clean", "political_corruption_articles", "share_of_pc_corpus_pct", "country_count"]
        ].rename(
            columns={
                "source_clean": "Source",
                "political_corruption_articles": "PC articles",
                "share_of_pc_corpus_pct": "Share of PC corpus (%)",
                "country_count": "Countries",
            }
        )
        save_attention_latex_table(
            top_sources_overall_table.head(20),
            "table_attention_top_sources_overall.tex",
            "Largest sources in the classified political-corruption corpus.",
            "tab:pc-top-sources-overall",
            "Note. PC = political corruption. Sources are standardized from the source URI/title field in the classified corpus. The table reports the 20 largest sources overall.",
        )
        display(top_sources_overall_table)

        source_country_summary = (
            source_data.groupby(["country", "country_label", "source_clean"], observed=True)
            .size()
            .reset_index(name="political_corruption_articles")
            .sort_values(["country", "political_corruption_articles"], ascending=[True, False])
        )
        country_totals_for_sources = source_data.groupby("country", observed=True).size().rename("country_pc_total")
        source_country_summary = source_country_summary.merge(
            country_totals_for_sources,
            on="country",
            how="left",
        )
        source_country_summary["share_of_country_pc_pct"] = (
            source_country_summary["political_corruption_articles"]
            / source_country_summary["country_pc_total"]
            * 100
        )
        source_country_summary["rank_within_country"] = source_country_summary.groupby("country", observed=True)[
            "political_corruption_articles"
        ].rank(method="first", ascending=False).astype(int)
        source_country_summary.to_csv(TABLE_DIR / "political_corruption_source_country_summary.csv", index=False)

        top_sources_by_country = source_country_summary[source_country_summary["rank_within_country"].le(5)].copy()
        top_sources_by_country.to_csv(TABLE_DIR / "political_corruption_top_sources_by_country.csv", index=False)
        top_sources_by_country_table = top_sources_by_country[
            ["country_label", "rank_within_country", "source_clean", "political_corruption_articles", "share_of_country_pc_pct"]
        ].rename(
            columns={
                "country_label": "Country",
                "rank_within_country": "Rank",
                "source_clean": "Source",
                "political_corruption_articles": "PC articles",
                "share_of_country_pc_pct": "Share of country PC corpus (%)",
            }
        )
        save_attention_latex_table(
            top_sources_by_country_table,
            "table_attention_top_sources_by_country.tex",
            "Top sources in the classified political-corruption corpus by country.",
            "tab:pc-top-sources-by-country",
            "Note. PC = political corruption. The table reports the five largest sources within each country-specific political-corruption corpus.",
        )
        display(top_sources_by_country_table)

        source_concentration_rows = []
        for country, country_df in source_country_summary.groupby("country", observed=True):
            ordered = country_df.sort_values("political_corruption_articles", ascending=False)
            total = ordered["political_corruption_articles"].sum()
            source_concentration_rows.append(
                {
                    "country": country,
                    "country_label": COUNTRY_LABELS.get(str(country), str(country)),
                    "pc_articles": int(total),
                    "unique_sources": int(ordered["source_clean"].nunique()),
                    "top_1_source_share_pct": ordered.head(1)["political_corruption_articles"].sum() / total * 100,
                    "top_5_source_share_pct": ordered.head(5)["political_corruption_articles"].sum() / total * 100,
                    "top_10_source_share_pct": ordered.head(10)["political_corruption_articles"].sum() / total * 100,
                }
            )
        source_concentration = pd.DataFrame(source_concentration_rows).sort_values("top_10_source_share_pct", ascending=False)
        source_concentration.to_csv(TABLE_DIR / "political_corruption_source_concentration_by_country.csv", index=False)
        source_concentration_table = source_concentration[
            [
                "country_label",
                "pc_articles",
                "unique_sources",
                "top_1_source_share_pct",
                "top_5_source_share_pct",
                "top_10_source_share_pct",
            ]
        ].rename(
            columns={
                "country_label": "Country",
                "pc_articles": "PC articles",
                "unique_sources": "Unique sources",
                "top_1_source_share_pct": "Top 1 share (%)",
                "top_5_source_share_pct": "Top 5 share (%)",
                "top_10_source_share_pct": "Top 10 share (%)",
            }
        )
        save_attention_latex_table(
            source_concentration_table,
            "table_attention_source_concentration_by_country.tex",
            "Source concentration in the classified political-corruption corpus by country.",
            "tab:pc-source-concentration-country",
            "Note. PC = political corruption. Source shares are calculated within each country-specific political-corruption corpus.",
        )
        display(source_concentration_table)


    overall_year_table = (
        country_year_from_csv.groupby("year", observed=True)
        .agg(
            total_news_articles=("total_news_articles", "sum"),
            political_corruption_articles=("political_corruption_articles", "sum"),
        )
        .reset_index()
    )
    overall_year_table["relative_attention_pct"] = (
        overall_year_table["political_corruption_articles"]
        / overall_year_table["total_news_articles"]
        * 100
    )
    overall_year_latex = overall_year_table.rename(
        columns={
            "year": "Year",
            "total_news_articles": "Total news",
            "political_corruption_articles": "PC articles",
            "relative_attention_pct": "PC share of total news (%)",
        }
    )

    save_attention_latex_table(
        overall_year_latex,
        "table_attention_year_summary.tex",
        "Political-corruption attention by year across all countries.",
        "tab:pc-attention-year-summary",
        "Note. PC = political corruption. Counts are summed across countries.",
    )
    display(overall_year_latex)


    country_year_matrix = country_year_from_csv.pivot(
        index="country_label",
        columns="year",
        values="relative_attention_pct",
    ).reindex([COUNTRY_LABELS.get(country, country) for country in COUNTRY_ORDER])
    country_year_matrix = country_year_matrix.reset_index().rename(columns={"country_label": "Country"})

    save_attention_latex_table(
        country_year_matrix,
        "table_attention_country_year_matrix.tex",
        "Political-corruption attention by country-year as a percentage of total news coverage.",
        "tab:pc-attention-country-year-matrix",
        "Note. Entries are percentages of total news coverage classified as political corruption.",
    )
    display(country_year_matrix)


    peak_months = month_from_csv.copy()
    peak_months = peak_months[peak_months["total_news_articles"].gt(0)].copy()
    peak_months["rank"] = peak_months.groupby("country", observed=True)["relative_attention_pct"].rank(
        method="first",
        ascending=False,
    )
    peak_months = peak_months[peak_months["rank"].le(3)].copy()
    peak_months["month_label"] = peak_months["month"].dt.strftime("%Y-%m")
    peak_months = peak_months.sort_values(["country", "rank"])
    peak_months_table = peak_months[
        [
            "country_label",
            "rank",
            "month_label",
            "total_news_articles",
            "political_corruption_articles",
            "relative_attention_pct",
        ]
    ].rename(
        columns={
            "country_label": "Country",
            "rank": "Rank",
            "month_label": "Month",
            "total_news_articles": "Total news",
            "political_corruption_articles": "PC articles",
            "relative_attention_pct": "PC share of total news (%)",
        }
    )

    save_attention_latex_table(
        peak_months_table,
        "table_attention_peak_months.tex",
        "Peak months of political-corruption attention by country.",
        "tab:pc-attention-peak-months",
        "Note. PC = political corruption. Peak months are ranked within country by relative attention.",
    )
    display(peak_months_table)

    print(f"LaTeX attention tables written to: {ATTENTION_LATEX_DIR}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build political-corruption attention outputs."
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_attention_analysis(args.pipeline_dir)


if __name__ == "__main__":
    main()
