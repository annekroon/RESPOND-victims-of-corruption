"""Build interactive topic visualizations from BERTopic document assignments."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Plotly topic visualizations.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument(
        "--label-mode",
        choices=["topic", "primary-domain", "generic-domain"],
        default="topic",
        help="Use GPT inductive topic labels by default. Legacy domain-label modes are optional.",
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


def load_topic_labels(topic_info, labels_path):
    if labels_path is None or not labels_path.exists():
        topic_info["topic_label"] = topic_info["Name"]
        return topic_info[["Topic", "topic_label"]]

    import pandas as pd

    labels = pd.read_csv(labels_path)
    label_col = "llm_topic_short_label" if "llm_topic_short_label" in labels.columns else "llm_topic_label"
    labels["topic_label"] = labels[label_col].fillna("").astype(str)
    labels.loc[labels["topic_label"].str.strip().eq(""), "topic_label"] = labels["Name"]
    if "llm_primary_domain" in labels.columns:
        labels["primary_domain"] = labels["llm_primary_domain"].fillna("").astype(str)
    elif "llm_corruption_type" in labels.columns:
        labels["primary_domain"] = labels["llm_corruption_type"].fillna("").astype(str)
    else:
        labels["primary_domain"] = ""
    labels.loc[labels["primary_domain"].str.strip().eq(""), "primary_domain"] = labels["topic_label"]

    if "llm_generic_domain_short_label" in labels.columns:
        labels["generic_domain_label"] = labels["llm_generic_domain_short_label"].fillna("").astype(str)
    elif "llm_corruption_type" in labels.columns:
        labels["generic_domain_label"] = labels["llm_corruption_type"].fillna("").astype(str)
    else:
        labels["generic_domain_label"] = ""
    labels.loc[labels["generic_domain_label"].str.strip().eq(""), "generic_domain_label"] = labels["primary_domain"]
    return labels[["Topic", "topic_label", "primary_domain", "generic_domain_label"]]


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

    output_dir = args.output_dir or (args.bertopic_dir / "visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(document_topics_path)
    topic_info = pd.read_csv(topic_info_path)
    labels = load_topic_labels(topic_info, args.labels or (args.bertopic_dir / "topic_labels_llm.csv"))

    docs = docs.merge(labels, left_on="topic", right_on="Topic", how="left")
    docs["topic_label"] = docs["topic_label"].fillna(docs["topic"].astype(str))
    docs["primary_domain"] = docs.get("primary_domain", docs["topic_label"]).fillna(docs["topic_label"])
    docs["generic_domain_label"] = docs.get("generic_domain_label", docs["topic_label"]).fillna(docs["topic_label"])
    if args.label_mode == "generic-domain":
        analysis_label = "generic_domain_label"
    elif args.label_mode == "primary-domain":
        analysis_label = "primary_domain"
    else:
        analysis_label = "topic_label"
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

    topic_totals.to_csv(output_dir / "topic_weighted_totals.csv", index=False)

    country_topic = weighted_group_share(docs_top, ["country", analysis_label], "analysis_weight")
    fig_country = px.imshow(
        country_topic.pivot(index="country", columns=analysis_label, values="share").fillna(0),
        aspect="auto",
        color_continuous_scale="Viridis",
        labels={"color": "Within-country share"},
        title=f"Top {args.top_n} political-corruption topics by country",
    )
    fig_country.update_layout(height=650)
    fig_country.write_html(output_dir / "country_topic_heatmap.html")

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
        title=f"Top {args.top_n} political-corruption topic shares over time",
        labels={"period": args.time_unit.title(), "share": "Topic share", analysis_label: "Topic"},
    )
    fig_time.update_layout(height=650, hovermode="x unified")
    fig_time.write_html(output_dir / "topic_shares_over_time.html")

    country_time = (
        docs_top.groupby(["country", "period", analysis_label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    fig_country_time = px.line(
        country_time.sort_values("period"),
        x="period",
        y="weighted_articles",
        color=analysis_label,
        facet_row="country",
        title=f"Top {args.top_n} political-corruption topic volume by country over time",
        labels={"period": args.time_unit.title(), "weighted_articles": "Weighted articles", analysis_label: "Topic"},
        height=1400,
    )
    fig_country_time.update_yaxes(matches=None)
    fig_country_time.write_html(output_dir / "country_topic_trends.html")

    print(f"Saved visualizations under: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
