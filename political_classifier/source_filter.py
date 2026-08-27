"""Source-exclusion filtering for the political-corruption sample.

Articles are removed only when their country/source pair has an explicit
``conventional_journalism == "No"`` decision. Yes, unresolved, missing, and
unknown sources remain eligible; missing pairs are retained but audited.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_SOURCE_DECISION_FILE = (
    DEFAULT_PIPELINE_DIR
    / "source_inclusion"
    / "political_corruption_all_sources_classified.xlsx"
)

SOURCE_COLUMN_CANDIDATES = [
    "source_clean",
    "source_uri",
    "source.uri",
    "source",
    "outlet",
    "source_title",
    "source.title",
]

DECISION_COLUMN_CANDIDATES = [
    "conventional_journalism",
    "conventional journalism",
]

COUNTRY_ALIASES = {
    "bulgaria": "Bulgaria",
    "france": "France",
    "hungary": "Hungary",
    "italy": "Italy",
    "netherlands": "Netherlands",
    "the_netherlands": "Netherlands",
    "serbia": "Serbia",
    "sweden": "Sweden",
    "ukraine": "Ukraine",
    "uk": "United_Kingdom",
    "u_k": "United_Kingdom",
    "united_kingdom": "United_Kingdom",
    "great_britain": "United_Kingdom",
}


def normalize_country(value) -> str:
    text = str(value).strip().casefold()
    key = "_".join(text.replace("-", " ").split())
    return COUNTRY_ALIASES.get(key, key)


def normalize_source(value) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().casefold()
    if text in {"", "nan", "none", "<na>"}:
        return "unknown"
    parsed = urlsplit(text if "://" in text else "//" + text)
    host = (parsed.hostname or "").strip(".")
    if host:
        if host.startswith("www."):
            host = host[4:]
        try:
            host = host.encode("ascii").decode("idna")
        except (UnicodeError, UnicodeEncodeError):
            pass
        return host
    return text.rstrip("/")


def choose_source_column(dataframe) -> str:
    for column in SOURCE_COLUMN_CANDIDATES:
        if column in dataframe.columns:
            return column
    raise ValueError(
        "No source/outlet column found. Expected one of: "
        + ", ".join(SOURCE_COLUMN_CANDIDATES)
    )


def choose_decision_column(dataframe) -> str:
    normalized_columns = {
        str(column).strip().lower().replace(" ", "_"): column
        for column in dataframe.columns
    }
    for column in DECISION_COLUMN_CANDIDATES:
        key = column.strip().lower().replace(" ", "_")
        if key in normalized_columns:
            return normalized_columns[key]
    raise ValueError(
        "No conventional-journalism decision column found. Expected one of: "
        + ", ".join(DECISION_COLUMN_CANDIDATES)
    )


def load_source_decisions(path: Path = DEFAULT_SOURCE_DECISION_FILE):
    """Load Yes country/source pairs from the source annotation file."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Source decision file not found: {path}. "
            "Place political_corruption_all_sources_classified.xlsx there, "
            "or pass --source-decision-file."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet_name = (
            "all_sources_classified"
            if "all_sources_classified" in workbook.sheet_names
            else workbook.sheet_names[0]
        )
        decisions = pd.read_excel(path, sheet_name=sheet_name)
    else:
        decisions = pd.read_csv(path)

    decision_column = choose_decision_column(decisions)
    required = {"country", "source_clean", decision_column}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"Source decision file is missing required columns: {sorted(missing)}")

    decisions = decisions.copy()
    decisions["country"] = decisions["country"].map(normalize_country)
    decisions["source_clean"] = decisions["source_clean"].map(normalize_source)
    decisions["source_filter_decision_clean"] = (
        decisions[decision_column].astype(str).str.strip().str.upper()
    )
    decisions["source_include"] = decisions["source_filter_decision_clean"].ne("NO")

    # If duplicate country/source rows conflict, any explicit No excludes the
    # pair. Otherwise Yes, unresolved, and blank decisions remain eligible.
    grouped = (
        decisions.groupby(["country", "source_clean"], as_index=False)
        .agg(
            source_include=("source_include", "all"),
            source_filter_decision_clean=(
                "source_filter_decision_clean",
                lambda values: ";".join(sorted(set(values))),
            ),
        )
    )
    return grouped


def apply_source_inclusion_filter(
    dataframe,
    decisions,
    country_column: str = "country",
    source_column: str | None = None,
):
    """Return rows except country/source pairs explicitly coded No."""
    data = dataframe.copy()
    if source_column is None:
        source_column = choose_source_column(data)

    data["_source_filter_country"] = data[country_column].map(normalize_country)
    data["_source_filter_source"] = data[source_column].map(normalize_source)

    filtered = data.merge(
        decisions,
        left_on=["_source_filter_country", "_source_filter_source"],
        right_on=["country", "source_clean"],
        how="left",
        suffixes=("", "_decision"),
    )
    filtered["source_include"] = (
        filtered["source_include"].astype("boolean").fillna(True).astype(bool)
    )
    filtered["source_filter_decision"] = filtered["source_filter_decision_clean"].fillna("MISSING")
    filtered["source_filter_source_column"] = source_column

    keep = filtered[filtered["source_include"]].copy()
    drop_columns = [
        "_source_filter_country",
        "_source_filter_source",
        "country_decision",
        "source_clean",
        "source_include",
        "source_filter_decision_clean",
    ]
    keep = keep.drop(columns=[column for column in drop_columns if column in keep.columns])
    return keep, filtered


def source_filter_summary(merged, group_columns=None):
    """Summarize retained/excluded rows after applying the source filter."""
    import pandas as pd

    group_columns = group_columns or ["country"]
    summary = (
        merged.groupby(group_columns, dropna=False)
        .agg(
            rows_before=("source_filter_decision", "size"),
            rows_after=("source_include", "sum"),
        )
        .reset_index()
    )
    summary["rows_excluded"] = summary["rows_before"] - summary["rows_after"]
    summary["included_share_pct"] = summary["rows_after"] / summary["rows_before"] * 100

    decision_counts = (
        merged.groupby(group_columns + ["source_filter_decision"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    decision_wide = decision_counts.pivot_table(
        index=group_columns,
        columns="source_filter_decision",
        values="rows",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    decision_wide.columns.name = None

    return pd.merge(summary, decision_wide, on=group_columns, how="left")
