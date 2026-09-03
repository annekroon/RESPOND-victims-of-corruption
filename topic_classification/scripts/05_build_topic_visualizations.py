"""Build interactive topic visualizations from BERTopic document assignments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topic_classification.provenance import validate_topic_label_outputs
from topic_classification.scripts._impl.reproducibility import write_run_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Plotly topic visualizations.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Maximum directly labelled topics to display.",
    )
    parser.add_argument(
        "--analysis-level",
        choices=["fine-grained"],
        default="fine-grained",
        help="Plot the directly labelled BERTopic topics.",
    )
    parser.add_argument(
        "--include-outlier",
        action="store_true",
        help="Include BERTopic outlier topic -1 in visualizations.",
    )
    parser.add_argument(
        "--time-unit",
        choices=["year", "month"],
        default="year",
        help="Time aggregation for trend charts.",
    )
    return parser.parse_args()


def load_analysis_labels(topic_info, labels_path):
    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(labels_path)

    import pandas as pd

    labels = pd.read_csv(labels_path)
    label_col = "llm_topic_short_label" if "llm_topic_short_label" in labels.columns else "llm_topic_label"
    labels["analysis_label"] = labels[label_col].fillna("").astype(str)
    labels = topic_info[["Topic", "Name"]].merge(
        labels[["Topic", "analysis_label"]], on="Topic", how="left"
    )
    labels = labels[labels["Topic"].ne(-1)].copy()
    labels.loc[
        labels["analysis_label"].fillna("").str.strip().eq(""), "analysis_label"
    ] = labels["Name"]

    return labels[["Topic", "analysis_label"]]


def weighted_group_share(data, group_cols, weight_col):
    grouped = data.groupby(group_cols, dropna=False)[weight_col].sum().rename("weighted_articles").reset_index()
    denominator_cols = group_cols[:-1]
    grouped["share"] = grouped["weighted_articles"] / grouped.groupby(denominator_cols)["weighted_articles"].transform("sum")
    return grouped


def main() -> None:
    args = parse_args()

    import pandas as pd
    import plotly.express as px

    document_topics_path = args.bertopic_dir / "document_topics.csv.gz"
    topic_info_path = args.bertopic_dir / "topic_info.csv"
    if not document_topics_path.exists():
        raise FileNotFoundError(document_topics_path)
    if not topic_info_path.exists():
        raise FileNotFoundError(topic_info_path)

    provenance = validate_topic_label_outputs(args.bertopic_dir)

    output_dir = args.output_dir or (args.bertopic_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(document_topics_path)
    topic_info = pd.read_csv(topic_info_path)
    labels_path = args.labels or (args.bertopic_dir / "topic_labels_llm.csv")
    labels = load_analysis_labels(topic_info, labels_path)

    docs = docs.merge(labels, left_on="topic", right_on="Topic", how="left")
    analysis_label = "analysis_label"
    docs[analysis_label] = docs[analysis_label].fillna(docs["topic"].astype(str))
    if not args.include_outlier:
        docs = docs[docs["topic"].ne(-1)].copy()

    if "analysis_weight" not in docs.columns:
        docs["analysis_weight"] = 1.0

    topic_totals = (
        docs.groupby([analysis_label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
        .sort_values("weighted_articles", ascending=False)
    )
    top_labels = topic_totals.head(args.top_n)[analysis_label].tolist()
    docs_top = docs[docs[analysis_label].isin(top_labels)].copy()
    display_n = len(top_labels)
    display_level = "topics"

    topic_totals_path = output_dir / "topic_weighted_totals.csv"
    topic_totals.to_csv(topic_totals_path, index=False)

    country_topic = weighted_group_share(docs_top, ["country", analysis_label], "analysis_weight")
    fig_country = px.imshow(
        country_topic.pivot(index="country", columns=analysis_label, values="share").fillna(0),
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": "Within-country share"},
        title=f"{display_n} political-corruption {display_level} by country",
    )
    fig_country.update_layout(height=650)
    country_heatmap_path = output_dir / "country_topic_heatmap.html"
    fig_country.write_html(country_heatmap_path)

    if args.time_unit == "month":
        if "date_parsed" not in docs_top.columns:
            raise ValueError("Monthly trends require date_parsed in document_topics.csv.gz.")
        docs_top["date_parsed"] = pd.to_datetime(docs_top["date_parsed"], errors="coerce", utc=True)
        docs_top["period"] = docs_top["date_parsed"].dt.tz_convert(None).dt.to_period("M").astype(str)
    else:
        docs_top["period"] = docs_top["year"].astype("Int64").astype(str)

    time_topic = weighted_group_share(docs_top, ["period", analysis_label], "analysis_weight")
    fig_time = px.area(
        time_topic.sort_values("period"),
        x="period",
        y="share",
        color=analysis_label,
        title=f"Political-corruption {display_level} over time",
        labels={"period": args.time_unit.title(), "share": "Topic share", analysis_label: "Topic"},
    )
    fig_time.update_layout(height=650, hovermode="x unified")
    time_path = output_dir / "topic_shares_over_time.html"
    fig_time.write_html(time_path)

    country_time = (
        docs_top.groupby(["country", "period", analysis_label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    country_time["share"] = country_time["weighted_articles"] / country_time.groupby(
        ["country", "period"]
    )["weighted_articles"].transform("sum")
    fig_country_time = px.line(
        country_time.sort_values("period"),
        x="period",
        y="share",
        color=analysis_label,
        facet_row="country",
        title=f"Political-corruption {display_level} by country over time",
        labels={
            "period": args.time_unit.title(),
            "share": "Within-country topic share",
            analysis_label: "Topic",
        },
        height=1400,
    )
    fig_country_time.update_yaxes(matches=None)
    country_time_path = output_dir / "country_topic_trends.html"
    fig_country_time.write_html(country_time_path)

    write_run_manifest(
        output_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs={
            "document_topics": document_topics_path,
            "topic_info": topic_info_path,
            "topic_labels": labels_path,
            "topic_label_manifest": provenance["manifest_path"],
        },
        outputs={
            "topic_weighted_totals": topic_totals_path,
            "country_topic_heatmap": country_heatmap_path,
            "topic_shares_over_time": time_path,
            "country_topic_trends": country_time_path,
        },
        extra={
            "analysis_level": args.analysis_level,
            "upstream_classifier": provenance["upstream_classifier"],
        },
        manifest_name="visualizations_run_manifest.json",
    )

    print(f"Saved visualizations under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
