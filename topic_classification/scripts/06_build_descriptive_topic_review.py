"""Build direct descriptive outputs for a compact BERTopic solution.

Unlike the archived higher-order workflow, this script reports the small set of
BERTopic topics directly. It also surfaces country concentration and the exact
abstractions used for LLM labelling so the solution can be manually audited.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from topic_classification.provenance import validate_topic_label_outputs
from topic_classification.scripts._impl.reproducibility import (
    command_output,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build auditable direct-topic summaries for descriptive BERTopic."
    )
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--examples-per-topic", type=int, default=5)
    parser.add_argument("--country-dominance-threshold", type=float, default=0.80)
    return parser.parse_args()


def clean(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def latex_escape(value: object) -> str:
    text = clean(value)
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
    return "".join(replacements.get(char, char) for char in text)


def compact(value: object, max_words: int = 42) -> str:
    words = clean(value).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def direct_topic_latex(summary) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Descriptive topics in political-corruption coverage.}",
        r"\label{tab:pc-descriptive-topics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabularx}{\textwidth}{@{}p{0.18\textwidth}r p{0.22\textwidth} X@{}}",
        r"\toprule",
        r"\textbf{Topic} & \textbf{Share} & \textbf{Largest contributors} & \textbf{Interpretation} \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{latex_escape(row['llm_topic_short_label'])} & "
            f"{100 * float(row['weighted_share']):.1f}\\% & "
            f"{latex_escape(row['top_countries'])} & "
            f"{latex_escape(compact(row['llm_topic_summary']))} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\par\smallskip",
            r"\scriptsize \textit{Note.} Shares use inverse country-year sampling weights and exclude BERTopic outliers. Articles were represented by language-neutral English abstractions before clustering. Topics are exploratory descriptions, not validated corruption-type measures.",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_values(path: Path, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"\\newcommand{{\\{name}}}{{{value}}}"
        for name, value in values.items()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    import pandas as pd
    from sklearn.metrics import normalized_mutual_info_score

    root = args.bertopic_dir.resolve()
    provenance = validate_topic_label_outputs(root)
    documents_path = root / "document_topics.csv.gz"
    labels_path = root / "topic_labels_llm.csv"
    audit_path = root / "topic_labels_llm_audit.jsonl"
    model_manifest_path = root / "run_manifest.json"

    documents = pd.read_csv(documents_path)
    labels = pd.read_csv(labels_path)
    labels = labels[labels["Topic"].ne(-1)].copy()
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if "analysis_weight" not in documents.columns:
        documents["analysis_weight"] = 1.0
    documents["analysis_weight"] = pd.to_numeric(
        documents["analysis_weight"], errors="raise"
    )

    inliers = documents[documents["topic"].ne(-1)].copy()
    inliers = inliers.merge(
        labels[
            [
                "Topic",
                "llm_topic_label",
                "llm_topic_short_label",
                "llm_topic_summary",
                "llm_inclusion_rule",
                "llm_exclusion_rule",
                "llm_confidence",
            ]
        ],
        left_on="topic",
        right_on="Topic",
        how="left",
        validate="many_to_one",
    )
    if inliers["llm_topic_short_label"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("At least one inlier document lacks a descriptive topic label.")

    totals = (
        inliers.groupby("topic", dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    totals["weighted_share"] = totals["weighted_articles"] / totals[
        "weighted_articles"
    ].sum()

    country = (
        inliers.groupby(["topic", "country"], dropna=False)
        .agg(
            weighted_articles=("analysis_weight", "sum"),
            sampled_abstractions=("topic", "size"),
        )
        .reset_index()
    )
    country["country_share_within_topic"] = country["weighted_articles"] / country.groupby(
        "topic"
    )["weighted_articles"].transform("sum")
    country["sample_country_share_within_topic"] = country[
        "sampled_abstractions"
    ] / country.groupby("topic")["sampled_abstractions"].transform("sum")
    country["country_share_within_corpus"] = country["weighted_articles"] / country.groupby(
        "country"
    )["weighted_articles"].transform("sum")

    top_country_rows = (
        country.sort_values(["topic", "weighted_articles"], ascending=[True, False])
        .groupby("topic")
        .head(3)
        .copy()
    )
    top_country_rows["piece"] = (
        top_country_rows["country"].astype(str).str.replace("_", " ", regex=False)
        + " ("
        + (100 * top_country_rows["country_share_within_topic"]).round(1).astype(str)
        + "%)"
    )
    top_countries = (
        top_country_rows.groupby("topic")["piece"]
        .apply(lambda values: "; ".join(values))
        .rename("top_countries")
        .reset_index()
    )
    specificity = (
        country.groupby("topic")
        .agg(
            max_weighted_country_share=("country_share_within_topic", "max"),
            max_sample_country_share=("sample_country_share_within_topic", "max"),
            countries_ge_5pct=(
                "sample_country_share_within_topic",
                lambda values: int((values >= 0.05).sum()),
            ),
        )
        .reset_index()
    )

    summary = (
        labels.rename(columns={"Topic": "topic"})
        .merge(totals, on="topic", how="left")
        .merge(top_countries, on="topic", how="left")
        .merge(specificity, on="topic", how="left")
        .sort_values("weighted_articles", ascending=False)
    )
    summary["country_dominated"] = summary["max_sample_country_share"].ge(
        args.country_dominance_threshold
    )

    year = (
        inliers.groupby(["year", "topic"], dropna=False)["analysis_weight"]
        .sum()
        .rename("weighted_articles")
        .reset_index()
    )
    year["share_within_year"] = year["weighted_articles"] / year.groupby("year")[
        "weighted_articles"
    ].transform("sum")
    year = year.merge(
        labels[["Topic", "llm_topic_short_label"]],
        left_on="topic",
        right_on="Topic",
        how="left",
    ).drop(columns="Topic")
    country = country.merge(
        labels[["Topic", "llm_topic_short_label"]],
        left_on="topic",
        right_on="Topic",
        how="left",
    ).drop(columns="Topic")

    review_rows: list[dict] = []
    label_lookup = labels.set_index("Topic")
    with audit_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            topic = int(record["topic_id"])
            if topic == -1 or topic not in label_lookup.index:
                continue
            for number, example in enumerate(
                record.get("examples", [])[: args.examples_per_topic], 1
            ):
                review_rows.append(
                    {
                        "topic": topic,
                        "topic_label": label_lookup.loc[
                            topic, "llm_topic_short_label"
                        ],
                        "example_number": number,
                        "example": clean(example),
                    }
                )
    review = pd.DataFrame(
        review_rows,
        columns=["topic", "topic_label", "example_number", "example"],
    ).sort_values(["topic", "example_number"])

    topic_country_nmi = float(
        normalized_mutual_info_score(
            inliers["country"].fillna("missing").astype(str),
            inliers["topic"].astype(str),
        )
    )
    model_extra = model_manifest.get("extra") or {}
    source_rows = int(model_extra.get("input_rows_before_status_filter", len(documents)))
    usable_rows = int(model_extra.get("documents_for_model", len(documents)))
    outlier_rows = int(documents["topic"].eq(-1).sum())
    diagnostics = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_git_commit": command_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ),
        "source_sample_rows": source_rows,
        "usable_abstraction_rows": usable_rows,
        "boundary_or_unclear_rows": source_rows - usable_rows,
        "inlier_rows": int(len(inliers)),
        "outlier_rows": outlier_rows,
        "outlier_share": outlier_rows / len(documents),
        "observed_topics": int(summary["topic"].nunique()),
        "largest_weighted_topic_share": float(summary["weighted_share"].max()),
        "country_dominated_topics": int(summary["country_dominated"].sum()),
        "country_dominance_threshold": args.country_dominance_threshold,
        "topic_country_normalized_mutual_information_unweighted": topic_country_nmi,
        "interpretation_warning": (
            "Topic-country NMI and country-dominated-topic counts diagnose residual "
            "country structure using the balanced sample rather than population "
            "weights; they are not inferential tests."
        ),
    }

    output_dir = root / "descriptive_outputs"
    latex_dir = output_dir / "latex"
    output_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "descriptive_topic_summary.csv"
    country_path = output_dir / "descriptive_topic_country_shares.csv"
    year_path = output_dir / "descriptive_topic_year_shares.csv"
    review_path = output_dir / "descriptive_topic_manual_review.csv"
    diagnostics_path = output_dir / "descriptive_topic_diagnostics.json"
    table_path = latex_dir / "table_topic_descriptive_summary.tex"
    values_path = latex_dir / "descriptive_topic_manuscript_values.tex"
    latest_path = root / "00_LATEST_DESCRIPTIVE_TOPIC_BUILD.txt"

    summary.to_csv(summary_path, index=False)
    country.to_csv(country_path, index=False)
    year.to_csv(year_path, index=False)
    review.to_csv(review_path, index=False)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    table_path.write_text(direct_topic_latex(summary), encoding="utf-8")
    write_values(
        values_path,
        {
            "DescriptiveTopicSourceSampleN": f"{source_rows:,}",
            "DescriptiveTopicUsableN": f"{usable_rows:,}",
            "DescriptiveTopicBoundaryN": f"{source_rows - usable_rows:,}",
            "DescriptiveTopicInlierN": f"{len(inliers):,}",
            "DescriptiveTopicOutlierN": f"{outlier_rows:,}",
            "DescriptiveTopicOutlierShare": f"{100 * outlier_rows / len(documents):.1f}",
            "DescriptiveTopicN": f"{summary['topic'].nunique():,}",
        },
    )
    latest_path.write_text(
        "\n".join(
            [
                "LATEST DESCRIPTIVE RESPOND TOPIC-MODEL BUILD",
                "============================================",
                "",
                f"Built (UTC): {diagnostics['built_at_utc']}",
                f"Generator Git commit: {diagnostics['generator_git_commit']}",
                f"Source sample: {source_rows:,}",
                f"Usable abstractions: {usable_rows:,}",
                f"Boundary/unclear abstractions: {source_rows - usable_rows:,}",
                f"Observed topics: {diagnostics['observed_topics']}",
                f"Inliers: {len(inliers):,}",
                f"Outliers: {outlier_rows:,} ({100 * outlier_rows / len(documents):.1f}%)",
                f"Largest weighted topic: {100 * diagnostics['largest_weighted_topic_share']:.1f}%",
                f"Country-dominated topics: {diagnostics['country_dominated_topics']}",
                f"Topic-country NMI: {topic_country_nmi:.3f}",
                "",
                f"Review first: {review_path}",
                f"Direct topic table: {table_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = write_run_manifest(
        root,
        script_name=Path(__file__).name,
        args=args,
        inputs={
            "topic_label_manifest": provenance["manifest_path"],
            "document_topics": documents_path,
            "topic_labels": labels_path,
            "topic_label_audit": audit_path,
        },
        outputs={
            "topic_summary": summary_path,
            "country_shares": country_path,
            "year_shares": year_path,
            "manual_review": review_path,
            "diagnostics": diagnostics_path,
            "latex_table": table_path,
            "manuscript_values": values_path,
            "latest_index": latest_path,
        },
        extra={
            **diagnostics,
            "upstream_classifier": provenance["upstream_classifier"],
        },
        manifest_name="descriptive_topic_output_manifest.json",
    )
    print(f"Saved descriptive outputs: {output_dir}", flush=True)
    print(f"Saved direct topic table: {table_path}", flush=True)
    print(f"Saved review packet: {review_path}", flush=True)
    print(f"Saved manifest: {manifest_path}", flush=True)
    print(
        f"Topics={diagnostics['observed_topics']}; "
        f"largest share={100 * diagnostics['largest_weighted_topic_share']:.1f}%; "
        f"country-dominated={diagnostics['country_dominated_topics']}; "
        f"country-topic NMI={topic_country_nmi:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
