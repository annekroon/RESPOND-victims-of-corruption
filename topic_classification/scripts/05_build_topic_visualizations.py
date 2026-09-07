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

from topic_classification.provenance import (
    apply_publication_label_overrides,
    validate_topic_label_outputs,
)
from topic_classification.scripts._impl.reproducibility import write_run_manifest


PUBLICATION_WIDTH_IN = 7.2
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Plotly topic visualizations.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--label-overrides", type=Path, default=None)
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


def load_analysis_labels(topic_info, labels_path, overrides_path=None):
    if labels_path is None or not labels_path.exists():
        raise FileNotFoundError(labels_path)

    import pandas as pd

    labels = pd.read_csv(labels_path)
    labels = apply_publication_label_overrides(labels, overrides_path)
    label_col = "publication_topic_label"
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


def configure_publication_style(plt) -> None:
    """Set a restrained, vector-safe style for manuscript figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def save_static_figure(fig, output_dir: Path, stem: str) -> dict[str, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={"Creator": "RESPOND reproducible topic workflow"},
    )
    return {f"{stem}_png": png, f"{stem}_pdf": pdf}


def main() -> None:
    args = parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from matplotlib.ticker import PercentFormatter

    configure_publication_style(plt)

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
    labels = load_analysis_labels(topic_info, labels_path, args.label_overrides)

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
        color_continuous_scale="Cividis",
        labels={"color": "Within-country share"},
        title=f"{display_n} political-corruption {display_level} by country",
    )
    fig_country.update_layout(
        height=650,
        template="simple_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 13},
    )
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
        color_discrete_sequence=OKABE_ITO,
        title=f"Political-corruption {display_level} over time",
        labels={"period": args.time_unit.title(), "share": "Topic share", analysis_label: "Topic"},
    )
    fig_time.update_layout(
        height=650,
        hovermode="x unified",
        template="simple_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 13},
    )
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
        color_discrete_sequence=OKABE_ITO,
        facet_row="country",
        title=f"Political-corruption {display_level} by country over time",
        labels={
            "period": args.time_unit.title(),
            "share": "Within-country topic share",
            analysis_label: "Topic",
        },
        height=1400,
    )
    fig_country_time.update_layout(
        template="simple_white",
        font={"family": "Arial, Helvetica, sans-serif", "size": 12},
    )
    fig_country_time.update_yaxes(matches=None)
    country_time_path = output_dir / "country_topic_trends.html"
    fig_country_time.write_html(country_time_path)

    static_outputs: dict[str, Path] = {}
    topic_colors = {
        label: OKABE_ITO[index % len(OKABE_ITO)]
        for index, label in enumerate(top_labels)
    }

    prevalence = topic_totals.head(args.top_n).sort_values("share")
    prevalence_labels = [wrapped(value, 34) for value in prevalence[analysis_label]]
    fig, ax = plt.subplots(
        figsize=(PUBLICATION_WIDTH_IN, max(2.7, 0.48 * len(prevalence) + 0.7)),
        constrained_layout=True,
    )
    bars = ax.barh(
        prevalence_labels,
        100 * prevalence["share"],
        color=[topic_colors[label] for label in prevalence[analysis_label]],
        edgecolor="none",
        height=0.68,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8.2)
    upper = max(40.0, 1.18 * float(100 * prevalence["share"].max()))
    ax.set_xlim(0, upper)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Weighted share among assigned abstractions")
    ax.set_ylabel("")
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#D9DDE2", linewidth=0.55)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
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
        figsize=(PUBLICATION_WIDTH_IN, 4.25),
        constrained_layout=True,
    )
    heatmap_values = 100 * heatmap.to_numpy()
    heatmap_max = max(30.0, 5 * np.ceil(float(heatmap_values.max()) / 5))
    image = ax.imshow(
        heatmap_values,
        aspect="auto",
        cmap="cividis",
        vmin=0,
        vmax=heatmap_max,
    )
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels(
        [wrapped(value, 17) for value in heatmap.columns],
        rotation=0,
        ha="center",
        fontsize=7.7,
    )
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index, fontsize=8.3)
    ax.set_ylabel("Publication country")
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(heatmap.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heatmap.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(heatmap_values.shape[0]):
        for column_index in range(heatmap_values.shape[1]):
            value = float(heatmap_values[row_index, column_index])
            color = "white" if value >= 0.53 * heatmap_max else "#1C232B"
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.4,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.018)
    colorbar.set_label("Within-country share (%)")
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=2.5, width=0.5)
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
        figsize=(PUBLICATION_WIDTH_IN, max(3.6, 1.85 * nrows)),
        sharex=True,
        sharey=True,
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
        color = topic_colors[label]
        ax.plot(
            x_positions,
            100 * values,
            color=color,
            linewidth=1.65,
            marker="o",
            markersize=2.8,
            markeredgewidth=0,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(periods)
        ax.set_title(wrapped(label, 38), loc="left", fontsize=8.5, pad=4)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=7.5)
    for ax in axes[len(trend_labels) :]:
        ax.set_visible(False)
    for ax in axes[-ncols:]:
        ax.tick_params(axis="x", rotation=0)
    fig.supylabel("Weighted share among assigned abstractions", fontsize=8.5)
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
        fig, ax = plt.subplots(
            figsize=(PUBLICATION_WIDTH_IN, 4.15), constrained_layout=True
        )
        adequate = candidates["adequate"].astype(str).str.lower().eq("true")
        ax.scatter(
            100 * candidates.loc[~adequate, "outlier_share"],
            candidates.loc[~adequate, "mean_resample_ari"],
            color="#C5CBD1",
            s=20,
            marker="o",
            alpha=0.62,
            edgecolors="none",
            label="Did not meet all criteria",
        )
        ax.scatter(
            100 * candidates.loc[adequate, "outlier_share"],
            candidates.loc[adequate, "mean_resample_ari"],
            color="#0072B2",
            s=38,
            marker="o",
            alpha=0.82,
            edgecolors="white",
            linewidths=0.45,
            label="Met all criteria",
        )
        selected_x = 100 * float(selection["outlier_share"])
        selected_y = float(selection["mean_resample_ari"])
        ax.scatter(
            selected_x,
            selected_y,
            marker="D",
            s=74,
            color="#D55E00",
            edgecolor="#111111",
            linewidth=0.65,
            label="Selected specification",
            zorder=4,
        )
        ax.annotate(
            f"Selected: {int(selection['topics'])} topics\n"
            f"ARI {selected_y:.3f}; {selected_x:.1f}% outliers",
            xy=(selected_x, selected_y),
            xytext=(10, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.8,
            color="#31363B",
            arrowprops={"arrowstyle": "-", "color": "#737A82", "lw": 0.6},
        )
        ax.axvline(45, color="#737A82", linewidth=0.75, linestyle=(0, (3, 2)))
        ax.text(
            45,
            0.015,
            "45% outlier criterion",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7.2,
            color="#626970",
        )
        ax.set_xlabel("Outlier share (%)")
        ax.set_ylabel("Mean resample stability (adjusted Rand index)")
        ax.grid(color="#D9DDE2", linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="lower left", ncols=1, handletextpad=0.5)
        static_outputs.update(
            save_static_figure(fig, output_dir, "figure_topic_model_selection")
        )
        plt.close(fig)

    manifest_inputs = {
        "document_topics": document_topics_path,
        "topic_info": topic_info_path,
        "topic_labels": labels_path,
        "topic_label_manifest": provenance["manifest_path"],
    }
    if args.label_overrides is not None:
        manifest_inputs["label_overrides"] = args.label_overrides

    write_run_manifest(
        output_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs=manifest_inputs,
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
