"""Build interactive topic visualizations from BERTopic document assignments."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
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


def wrapped(value: object, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))


def save_static_figure(fig, output_dir: Path, stem: str) -> dict[str, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    return {f"{stem}_png": png, f"{stem}_pdf": pdf}


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
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
    topic_totals["share"] = topic_totals["weighted_articles"] / topic_totals[
        "weighted_articles"
    ].sum()
    top_labels = topic_totals.head(args.top_n)[analysis_label].tolist()
    docs_top = docs[docs[analysis_label].isin(top_labels)].copy()
    display_n = len(top_labels)
    display_level = "topics"

    topic_totals_path = output_dir / "topic_weighted_totals.csv"
    topic_totals.to_csv(topic_totals_path, index=False)

    country_topic = weighted_group_share(
        docs, ["country", analysis_label], "analysis_weight"
    )
    country_topic = country_topic[
        country_topic[analysis_label].isin(top_labels)
    ].copy()
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

    docs["period"] = (
        pd.to_datetime(docs["date_parsed"], errors="coerce", utc=True)
        .dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
        if args.time_unit == "month"
        else docs["year"].astype("Int64").astype(str)
    )
    time_topic = weighted_group_share(
        docs, ["period", analysis_label], "analysis_weight"
    )
    time_topic = time_topic[time_topic[analysis_label].isin(top_labels)].copy()
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
        docs.groupby(["country", "period", analysis_label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    country_time["share"] = country_time["weighted_articles"] / country_time.groupby(
        ["country", "period"]
    )["weighted_articles"].transform("sum")
    country_time = country_time[
        country_time[analysis_label].isin(top_labels)
    ].copy()
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

    static_outputs: dict[str, Path] = {}

    prevalence = topic_totals.head(args.top_n).sort_values("share")
    prevalence_labels = [wrapped(value, 30) for value in prevalence[analysis_label]]
    fig, ax = plt.subplots(
        figsize=(8.2, max(4.2, 0.52 * len(prevalence) + 1.4)),
        constrained_layout=True,
    )
    bars = ax.barh(
        prevalence_labels,
        100 * prevalence["share"],
        color="#4C78A8",
        edgecolor="#1f1f1f",
        linewidth=0.45,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
    ax.set_xlabel("Weighted share of modelled articles (%)")
    ax.set_ylabel("")
    ax.set_title("Recurring themes in political-corruption coverage", loc="left")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    static_outputs.update(
        save_static_figure(fig, output_dir, "figure_topic_prevalence")
    )
    plt.close(fig)

    heatmap = country_topic.pivot(
        index="country", columns=analysis_label, values="share"
    ).fillna(0)
    heatmap = heatmap.reindex(columns=top_labels)
    heatmap.index = heatmap.index.astype(str).str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(
        figsize=(max(8.5, 0.9 * len(top_labels) + 3.0), 5.8),
        constrained_layout=True,
    )
    image = ax.imshow(100 * heatmap.to_numpy(), aspect="auto", cmap="cividis")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(
        [wrapped(value, 18) for value in heatmap.columns],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=9)
    ax.set_title("Topic composition within each country", loc="left")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82, pad=0.02)
    colorbar.set_label("Share of country coverage (%)")
    static_outputs.update(
        save_static_figure(fig, output_dir, "figure_topic_country_heatmap")
    )
    plt.close(fig)

    trend_labels = top_labels[: min(8, len(top_labels))]
    trend_data = time_topic[time_topic[analysis_label].isin(trend_labels)].copy()
    periods = sorted(trend_data["period"].dropna().astype(str).unique())
    ncols = 2
    nrows = int(np.ceil(len(trend_labels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(9.0, max(4.5, 2.25 * nrows)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, label in zip(axes, trend_labels):
        values = (
            trend_data[trend_data[analysis_label].eq(label)]
            .set_index("period")["share"]
            .reindex(periods)
            .fillna(0)
        )
        x_positions = np.arange(len(periods))
        ax.plot(x_positions, 100 * values, color="#4C78A8", linewidth=1.8)
        ax.fill_between(
            x_positions, 0, 100 * values, color="#4C78A8", alpha=0.14
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(periods)
        ax.set_title(wrapped(label, 42), loc="left", fontsize=9)
        ax.set_ylabel("Share (%)", fontsize=8)
        ax.grid(axis="y", color="#e1e1e1", linewidth=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
    for ax in axes[len(trend_labels) :]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle("Topic prevalence over time", x=0.0, ha="left", fontsize=13)
    static_outputs.update(
        save_static_figure(fig, output_dir, "figure_topic_trends")
    )
    plt.close(fig)

    candidate_path = args.bertopic_dir / "hdbscan_stability_candidates.csv"
    selection_path = args.bertopic_dir / "hdbscan_stability_selection.json"
    if candidate_path.exists() and selection_path.exists():
        candidates = pd.read_csv(candidate_path)
        selection = json.loads(selection_path.read_text(encoding="utf-8"))[
            "selected"
        ]
        fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
        adequate = candidates["adequate"].astype(str).str.lower().eq("true")
        scatter = ax.scatter(
            100 * candidates["outlier_share"],
            candidates["mean_resample_ari"],
            c=candidates["topics"],
            cmap="cividis",
            s=24,
            marker="o",
            alpha=0.28,
            edgecolors="none",
        )
        ax.scatter(
            100 * candidates.loc[adequate, "outlier_share"],
            candidates.loc[adequate, "mean_resample_ari"],
            c=candidates.loc[adequate, "topics"],
            cmap="cividis",
            vmin=candidates["topics"].min(),
            vmax=candidates["topics"].max(),
            s=48,
            marker="o",
            alpha=0.85,
            edgecolors="#222222",
            linewidths=0.4,
        )
        ax.scatter(
            100 * float(selection["outlier_share"]),
            float(selection["mean_resample_ari"]),
            marker="*",
            s=210,
            color="#D1495B",
            edgecolor="#111111",
            linewidth=0.8,
            label="Selected specification",
            zorder=4,
        )
        ax.set_xlabel("Outlier share (%)")
        ax.set_ylabel("Mean resample agreement (adjusted Rand index)")
        ax.set_title("Stability-based BERTopic specification selection", loc="left")
        ax.grid(color="#e1e1e1", linewidth=0.55)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="best")
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Number of topics")
        static_outputs.update(
            save_static_figure(fig, output_dir, "figure_topic_model_selection")
        )
        plt.close(fig)

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
            **static_outputs,
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
