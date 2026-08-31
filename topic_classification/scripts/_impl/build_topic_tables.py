"""Build reproducible CSV and LaTeX summaries for the final topic solution."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topic_classification.provenance import validate_topic_group_outputs
from topic_classification.scripts._impl.reproducibility import write_run_manifest


DEFAULT_TOPIC_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "topic_classification/bertopic_political_corruption_source_filtered_200_min10"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weighted topic tables from the verified final topic run."
    )
    parser.add_argument("--bertopic-dir", type=Path, default=DEFAULT_TOPIC_DIR)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "--",
        "\u2014": "---",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in clean_text(value))


def compact_summary(value: object, max_words: int = 58) -> str:
    words = clean_text(value).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def weighted_analyses(docs, groups):
    import pandas as pd

    label = "topic_group_short_label"
    inliers = docs[docs["topic"].ne(-1)].copy()
    inliers = inliers.merge(
        groups[
            [
                "Topic",
                "llm_topic_short_label",
                "llm_topic_summary",
                "topic_group_id",
                "topic_group_short_label",
                "topic_group_label",
                "topic_group_summary",
                "topic_group_assignment_rationale",
            ]
        ],
        left_on="topic",
        right_on="Topic",
        how="left",
        validate="many_to_one",
    )
    if inliers[label].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("At least one inlier document lacks a higher-order topic.")
    if "analysis_weight" not in inliers.columns:
        inliers["analysis_weight"] = 1.0
    inliers["analysis_weight"] = pd.to_numeric(
        inliers["analysis_weight"], errors="raise"
    )

    topic_totals = (
        inliers.groupby(label, dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
        .sort_values("weighted_articles", ascending=False)
    )
    topic_totals["weighted_share"] = (
        topic_totals["weighted_articles"] / topic_totals["weighted_articles"].sum()
    )

    country_topic = (
        inliers.groupby(["country", label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    country_topic["share"] = country_topic["weighted_articles"] / country_topic.groupby(
        "country"
    )["weighted_articles"].transform("sum")

    time_topic = (
        inliers.groupby(["year", label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    time_topic["share"] = time_topic["weighted_articles"] / time_topic.groupby(
        "year"
    )["weighted_articles"].transform("sum")

    country_time = (
        inliers.groupby(["country", "year", label], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    country_time["share"] = country_time["weighted_articles"] / country_time.groupby(
        ["country", "year"]
    )["weighted_articles"].transform("sum")

    overall = topic_totals[[label, "weighted_share"]].rename(
        columns={"weighted_share": "overall_share"}
    )
    country_lift = country_topic.merge(overall, on=label, how="left")
    country_lift["lift"] = country_lift["share"] / country_lift["overall_share"]
    time_lift = time_topic.merge(overall, on=label, how="left")
    time_lift["lift"] = time_lift["share"] / time_lift["overall_share"]
    country_time_lift = country_time.merge(overall, on=label, how="left")
    country_time_lift["lift"] = (
        country_time_lift["share"] / country_time_lift["overall_share"]
    )

    country_within_group = (
        inliers.groupby([label, "country"], dropna=False)["analysis_weight"]
        .sum()
        .rename("country_weighted_articles")
        .reset_index()
    )
    country_within_group["group_total"] = country_within_group.groupby(label)[
        "country_weighted_articles"
    ].transform("sum")
    country_within_group["country_share_within_group"] = (
        country_within_group["country_weighted_articles"]
        / country_within_group["group_total"]
    )
    top_countries = (
        country_within_group.sort_values(
            [label, "country_weighted_articles"], ascending=[True, False]
        )
        .groupby(label)
        .head(3)
        .assign(
            country_piece=lambda frame: frame["country"].astype(str).str.replace(
                "_", " ", regex=False
            )
            + " ("
            + (100 * frame["country_share_within_group"]).round(1).astype(str)
            + "%)"
        )
        .groupby(label)["country_piece"]
        .apply(lambda values: "; ".join(values))
        .rename("top_countries")
        .reset_index()
    )

    group_details = (
        groups.groupby(label, dropna=False)
        .agg(
            topic_group_label=("topic_group_label", "first"),
            substantive_description=("topic_group_summary", "first"),
            fine_grained_topics=("Topic", "nunique"),
        )
        .reset_index()
    )
    summary = (
        topic_totals.merge(group_details, on=label, how="left")
        .merge(top_countries, on=label, how="left")
        .sort_values("weighted_articles", ascending=False)
    )

    raw_topic_totals = (
        inliers.groupby("topic", dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    raw_topic_totals["weighted_share"] = (
        raw_topic_totals["weighted_articles"]
        / raw_topic_totals["weighted_articles"].sum()
    )
    raw_country = (
        inliers.groupby(["topic", "country"], dropna=False)["analysis_weight"]
        .sum()
        .rename("country_weighted_articles")
        .reset_index()
    )
    raw_country["topic_total"] = raw_country.groupby("topic")[
        "country_weighted_articles"
    ].transform("sum")
    raw_country["country_share_within_topic"] = (
        raw_country["country_weighted_articles"] / raw_country["topic_total"]
    )
    raw_specificity = (
        raw_country.groupby("topic", dropna=False)
        .agg(
            max_country_share=("country_share_within_topic", "max"),
            countries_ge_5pct=(
                "country_share_within_topic",
                lambda values: int((values >= 0.05).sum()),
            ),
        )
        .reset_index()
    )
    largest_country = (
        raw_country.sort_values(
            ["topic", "country_weighted_articles"], ascending=[True, False]
        )
        .groupby("topic")
        .head(1)[["topic", "country"]]
        .rename(columns={"country": "largest_country"})
    )
    inventory = (
        groups.rename(columns={"Topic": "topic"})
        .merge(raw_topic_totals, on="topic", how="left")
        .merge(raw_specificity, on="topic", how="left")
        .merge(largest_country, on="topic", how="left")
        .sort_values([label, "weighted_share"], ascending=[True, False])
    )

    return {
        "inliers": inliers,
        "summary": summary,
        "topic_totals": topic_totals,
        "country_topic": country_topic,
        "time_topic": time_topic,
        "country_time": country_time,
        "country_lift": country_lift,
        "time_lift": time_lift,
        "country_time_lift": country_time_lift,
        "raw_specificity": raw_specificity,
        "inventory": inventory,
    }


def higher_order_latex(summary) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\caption{\textit{Substantive inventory of higher-order topics in political-corruption news}}",
        r"\label{tab:topic_higher_order_summary}",
        r"\begin{tabularx}{\linewidth}{p{0.18\linewidth} p{0.10\linewidth} p{0.21\linewidth} X}",
        r"\toprule",
        r"\textbf{Higher-order topic} & \textbf{Weighted share} & \textbf{Main countries} & \textbf{Substantive interpretation} \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{latex_escape(row['topic_group_short_label'])} & "
            f"{100 * float(row['weighted_share']):.1f}\\% & "
            f"{latex_escape(row['top_countries'])} & "
            f"{latex_escape(row['substantive_description'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\par\smallskip",
            r"\scriptsize \textit{Note.} Weighted shares use country-year sampling weights. Main countries are the largest contributors within each higher-order topic. The higher-order topics are LLM-assisted interpretive aggregations of the fine-grained BERTopic topics.",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def inventory_latex(inventory, group_order: list[str]) -> str:
    inventory = inventory.copy()
    inventory["group_order"] = inventory["topic_group_short_label"].map(
        {name: index for index, name in enumerate(group_order)}
    )
    inventory = inventory.sort_values(
        ["group_order", "weighted_share"], ascending=[True, False]
    )
    caption = "Fine-grained BERTopic topics and LLM higher-order topic classification"
    lines = [
        r"\begingroup",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\LTleft}{0pt}",
        r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{p{0.15\linewidth} p{0.13\linewidth} p{0.13\linewidth} p{0.09\linewidth} p{0.42\linewidth}}",
        rf"\caption{{\textit{{{latex_escape(caption)}}}}}\label{{tab:all_topics_llm_higher_order_topics}}\\",
        r"\toprule",
        r"\textbf{Original BERTopic topic} & \textbf{Higher-order topic} & \textbf{Weighted articles / share} & \textbf{Largest country} & \textbf{Interpretive summary} \\",
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{\textit{{{latex_escape(caption + ' (continued)')}}}}}\\",
        r"\toprule",
        r"\textbf{Original BERTopic topic} & \textbf{Higher-order topic} & \textbf{Weighted articles / share} & \textbf{Largest country} & \textbf{Interpretive summary} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{\scriptsize Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\multicolumn{5}{p{0.94\linewidth}}{\scriptsize \textit{Note.} This fine-grained inventory is retained for reproducibility and is not inserted into the manuscript by default. Weighted values use country-year sampling weights.}\\",
        r"\endlastfoot",
    ]
    previous_group = None
    for _, row in inventory.iterrows():
        group = clean_text(row["topic_group_short_label"])
        if previous_group is not None and group != previous_group:
            lines.append(r"\addlinespace")
        previous_group = group
        weighted = (
            f"{float(row['weighted_articles']):,.0f} "
            f"({100 * float(row['weighted_share']):.2f}\\%)"
        )
        largest_country = clean_text(row["largest_country"]).replace("_", " ")
        lines.append(
            f"{latex_escape(row['llm_topic_short_label'])} & "
            f"{latex_escape(group)} & {weighted} & "
            f"{latex_escape(largest_country)} & "
            f"{latex_escape(compact_summary(row['llm_topic_summary']))} \\\\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    bertopic_dir = args.bertopic_dir.resolve()
    provenance = validate_topic_group_outputs(bertopic_dir)

    import pandas as pd

    documents_path = bertopic_dir / "document_topics.csv.gz"
    groups_path = bertopic_dir / "topic_groups_llm.csv"
    docs = pd.read_csv(documents_path)
    groups = pd.read_csv(groups_path)
    if groups["Topic"].duplicated().any():
        raise ValueError("Fine-grained topic assignments are not unique.")

    analyses = weighted_analyses(docs, groups)
    output_dir = bertopic_dir / "inspection_tables"
    latex_dir = output_dir / "latex"
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    csv_outputs = {
        "substantive_topic_summary": (
            analyses["summary"],
            output_dir / "substantive_topic_summary.csv",
        ),
        "topic_group_overview": (
            analyses["summary"],
            output_dir / "topic_group_overview.csv",
        ),
        "topic_group_assignment_explanation": (
            analyses["inventory"],
            output_dir / "topic_group_assignment_explanation.csv",
        ),
        "topic_weighted_totals": (
            analyses["topic_totals"],
            output_dir / "topic_weighted_totals.csv",
        ),
        "country_topic_shares": (
            analyses["country_topic"],
            output_dir / "country_topic_shares_top_topics.csv",
        ),
        "topic_shares_over_time": (
            analyses["time_topic"],
            output_dir / "topic_shares_over_time_top_topics.csv",
        ),
        "country_topic_trends": (
            analyses["country_time"],
            output_dir / "country_topic_trends_top_topics.csv",
        ),
        "country_topic_lift": (
            analyses["country_lift"],
            output_dir / "country_topic_lift_top_topics.csv",
        ),
        "topic_lift_over_time": (
            analyses["time_lift"],
            output_dir / "topic_lift_over_time_top_topics.csv",
        ),
        "country_time_topic_lift": (
            analyses["country_time_lift"],
            output_dir / "country_time_topic_lift_top_topics.csv",
        ),
        "raw_topic_country_specificity": (
            analyses["raw_specificity"],
            output_dir / "raw_topic_country_specificity.csv",
        ),
        "raw_topic_to_label_mapping": (
            analyses["inventory"],
            output_dir / "raw_topic_to_label_mapping.csv",
        ),
    }
    for frame, path in csv_outputs.values():
        frame.to_csv(path, index=False)

    summary_table_path = latex_dir / "table_topic_higher_order_summary.tex"
    inventory_table_path = (
        latex_dir / "table_all_topics_llm_higher_order_topics.tex"
    )
    summary_table_path.write_text(
        higher_order_latex(analyses["summary"]), encoding="utf-8"
    )
    inventory_table_path.write_text(
        inventory_latex(
            analyses["inventory"],
            analyses["summary"]["topic_group_short_label"].tolist(),
        ),
        encoding="utf-8",
    )

    manifest_outputs = {name: path for name, (_, path) in csv_outputs.items()}
    manifest_outputs.update(
        {
            "higher_order_summary_table": summary_table_path,
            "fine_grained_inventory_table": inventory_table_path,
        }
    )
    write_run_manifest(
        output_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs={
            "document_topics": documents_path,
            "topic_groups": groups_path,
            "topic_group_manifest": provenance["manifest_path"],
        },
        outputs=manifest_outputs,
        extra={
            "analysis_level": "higher-order",
            "upstream_classifier": provenance["upstream_classifier"],
        },
        manifest_name="topic_tables_run_manifest.json",
    )

    print(f"Saved topic analysis tables: {output_dir}", flush=True)
    print(f"Saved manuscript table: {summary_table_path}", flush=True)
    print(f"Saved archive inventory: {inventory_table_path}", flush=True)


if __name__ == "__main__":
    main()
