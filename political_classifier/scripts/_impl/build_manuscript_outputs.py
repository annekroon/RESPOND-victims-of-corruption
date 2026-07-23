"""Build current political-classifier manuscript tables and the pipeline figure.

All values are read from saved pipeline CSV/text outputs. This keeps the
tracked notebooks optional and prevents hand-edited manuscript numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
RAW_CORRUPTION_QUERY_ARTICLES = 3_092_051
REPORT_LABEL_SOURCES = {"silver_labelled_training_set", "human_5fold_cv"}
TABLE_NOTE = (
    "Note. PC = political corruption. Precision, recall, and PC F1 refer to "
    "the political-corruption class."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build current political-classifier manuscript outputs."
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    return parser.parse_args()


def read_required_csv(path: Path):
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_float_text(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    return float(path.read_text(encoding="utf-8").strip())


def cleaned_country_totals(pipeline_dir: Path, source_filtered):
    """Return cleaned/deduplicated totals without requiring one specific file."""
    import pandas as pd

    if "input_rows" in source_filtered.columns:
        return source_filtered[["country", "input_rows"]].rename(
            columns={"input_rows": "total_articles"}
        )

    total_path = pipeline_dir / "denominator_country_total.csv"
    if total_path.exists():
        return read_required_csv(total_path)[["country", "total_articles"]]

    for filename in [
        "denominator_country_year.csv",
        "denominator_country_month.csv",
        "denominator_country_week.csv",
    ]:
        path = pipeline_dir / filename
        if not path.exists():
            continue
        data = read_required_csv(path)
        count_column = next(
            (
                column
                for column in ["total_articles", "corruption_query_articles", "count", "n"]
                if column in data.columns
            ),
            None,
        )
        if count_column is None or "country" not in data.columns:
            continue
        return (
            data.groupby("country", dropna=False)[count_column]
            .sum()
            .reset_index(name="total_articles")
        )

    raise FileNotFoundError(
        "Could not determine cleaned country totals. Expected input_rows in "
        "source_inclusion/cleaned_source_filter_output_summary.csv or a "
        "denominator_country_{total,year,month,week}.csv file."
    )


def model_label(row) -> str:
    if row["model"] == "tfidf_char_ngrams_logreg":
        return "TF-IDF char. n-grams"
    labels = {
        "intfloat/multilingual-e5-large": "E5-large",
        "intfloat/multilingual-e5-base": "E5-base",
        "BAAI/bge-m3": "BGE-M3",
        "sentence-transformers/LaBSE": "LaBSE",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": "mMPNet",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "mMiniLM",
    }
    embedding = str(row.get("embedding_model", ""))
    return labels.get(embedding, embedding or row["model"])


def training_label(value: object) -> str:
    labels = {
        "silver_labelled_training_set": "Silver-labelled set",
        "human_5fold_cv": "Human 5-fold CV",
    }
    return labels.get(str(value), str(value))


def latex_escape(value: object) -> str:
    text = str(value)
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
    return "".join(replacements.get(character, character) for character in text)


def latex_cell(value: object, integer_column: bool) -> str:
    if value is None:
        return ""
    if isinstance(value, numbers.Real) and math.isnan(float(value)):
        return ""
    if integer_column and isinstance(value, numbers.Real):
        return f"{int(value):,}"
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return latex_escape(value)


def format_latex_table(data, caption: str, label: str, note: str) -> str:
    integer_columns = {
        column: getattr(data[column].dtype, "kind", "") in {"i", "u"}
        for column in data.columns
    }
    alignments = "".join(
        "r" if getattr(data[column].dtype, "kind", "") in {"i", "u", "f"} else "l"
        for column in data.columns
    )
    lines = [
        r"\begin{table}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
    ]
    resize = len(data.columns) >= 7
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignments}}}",
            r"\toprule",
            " & ".join(latex_escape(column) for column in data.columns) + r" \\",
            r"\midrule",
        ]
    )
    for _, row in data.iterrows():
        lines.append(
            " & ".join(
                latex_cell(row[column], integer_columns[column])
                for column in data.columns
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append("}")
    lines.extend(
        [
            rf"\par\smallskip\footnotesize{{{note}}}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_latex_table(
    data,
    path: Path,
    caption: str,
    label: str,
    note: str = TABLE_NOTE,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_latex_table(data, caption, label, note),
        encoding="utf-8",
    )
    print(f"Saved {path}", flush=True)
    return path


def select_final_classifier(best_results):
    selected = best_results[
        best_results["label_source"].eq("silver_labelled_training_set")
    ].copy()
    if selected.empty:
        raise ValueError(
            "No silver_labelled_training_set row found in best_model_results.csv"
        )
    return selected.sort_values(
        ["political_f1", "political_recall", "political_precision"],
        ascending=False,
    ).iloc[0]


def binary_metrics(y_true, y_pred) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    pairs = [
        (int(actual), int(predicted))
        for actual, predicted in zip(y_true, y_pred)
    ]
    if not pairs:
        raise ValueError("Cannot calculate metrics for an empty country group.")
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in pairs)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in pairs)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in pairs)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in pairs)

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    political_precision = ratio(tp, tp + fp)
    political_recall = ratio(tp, tp + fn)
    political_f1 = ratio(
        2 * political_precision * political_recall,
        political_precision + political_recall,
    )
    no_precision = ratio(tn, tn + fn)
    no_recall = ratio(tn, tn + fp)
    no_f1 = ratio(2 * no_precision * no_recall, no_precision + no_recall)
    no_support = tn + fp
    political_support = tp + fn
    total = len(pairs)
    return {
        "accuracy": ratio(tp + tn, total),
        "political_precision": political_precision,
        "political_recall": political_recall,
        "political_f1": political_f1,
        "macro_f1": (no_f1 + political_f1) / 2,
        "weighted_f1": (
            no_f1 * no_support + political_f1 * political_support
        )
        / total,
        "predicted_positive_rate": ratio(tp + fp, total),
        "political_support": political_support,
    }


def classifier_tables(pipeline_dir: Path) -> list[Path]:
    import pandas as pd

    comparison_dir = pipeline_dir / "classifier_comparison"
    table_dir = pipeline_dir / "manuscript_tables"
    best = read_required_csv(comparison_dir / "best_model_results.csv")
    thresholds = read_required_csv(comparison_dir / "all_threshold_results.csv")
    predictions = read_required_csv(
        comparison_dir / "validation_prediction_comparison.csv"
    )
    selected = select_final_classifier(best)
    final_threshold = read_float_text(
        pipeline_dir / "silver_classifier" / "selected_threshold.txt"
    )
    if abs(float(selected["threshold"]) - final_threshold) > 1e-9:
        raise ValueError(
            "Classifier comparison and final scoring use different thresholds: "
            f"{float(selected['threshold']):.2f} versus {final_threshold:.2f}. "
            "Rerun steps 05 and 06 before rebuilding manuscript outputs."
        )

    report_rows = best[best["label_source"].isin(REPORT_LABEL_SOURCES)].copy()
    report_rows["Model"] = report_rows.apply(model_label, axis=1)
    report_rows["Labels"] = report_rows["label_source"].map(training_label)
    order = {
        "E5-large": 0,
        "BGE-M3": 1,
        "E5-base": 2,
        "LaBSE": 3,
        "mMPNet": 4,
        "mMiniLM": 5,
        "TF-IDF char. n-grams": 6,
    }
    report_rows["_order"] = report_rows["Model"].map(order).fillna(99)
    report_rows["_label_order"] = report_rows["label_source"].map(
        {
            "silver_labelled_training_set": 0,
            "human_5fold_cv": 1,
        }
    ).fillna(99)
    report_rows = report_rows.sort_values(
        ["_order", "_label_order", "political_f1"],
        ascending=[True, True, False],
    )

    metric_columns = [
        "Model",
        "Labels",
        "train_rows",
        "threshold",
        "accuracy",
        "political_precision",
        "political_recall",
        "political_f1",
        "macro_f1",
        "weighted_f1",
    ]
    rename_metrics = {
        "train_rows": "N train",
        "threshold": "Thr.",
        "accuracy": "Acc.",
        "political_precision": "PC Prec.",
        "political_recall": "PC Rec.",
        "political_f1": "PC F1",
        "macro_f1": "Macro F1",
        "weighted_f1": "Wtd. F1",
    }
    main_table = report_rows[metric_columns].rename(columns=rename_metrics)

    outputs = [
        write_latex_table(
            main_table,
            table_dir / "table_pc_classifier_comparison_main.tex",
            "Validation performance of political-corruption classification models.",
            "tab:pc-classifier-comparison-main",
        )
    ]

    appendix_table = report_rows[
        metric_columns + ["predicted_positive_rate"]
    ].rename(
        columns={
            **rename_metrics,
            "predicted_positive_rate": "Pred. pos.",
        }
    )
    outputs.append(
        write_latex_table(
            appendix_table,
            table_dir / "table_pc_classifier_comparison_appendix.tex",
            "Full validation comparison of political-corruption classification models.",
            "tab:pc-classifier-comparison-appendix",
        )
    )

    selected_thresholds = thresholds[
        thresholds["embedding_model"].fillna("").eq(
            str(selected["embedding_model"])
        )
        & thresholds["label_source"].eq(selected["label_source"])
    ].copy().sort_values("threshold")
    if selected_thresholds.empty:
        raise ValueError("No threshold sweep found for the selected classifier.")
    threshold_table = selected_thresholds[
        [
            "threshold",
            "accuracy",
            "political_precision",
            "political_recall",
            "political_f1",
            "macro_f1",
            "weighted_f1",
            "predicted_positive_rate",
        ]
    ].rename(
        columns={
            "threshold": "Thr.",
            "accuracy": "Acc.",
            "political_precision": "PC Prec.",
            "political_recall": "PC Rec.",
            "political_f1": "PC F1",
            "macro_f1": "Macro F1",
            "weighted_f1": "Wtd. F1",
            "predicted_positive_rate": "Pred. pos.",
        }
    )
    outputs.append(
        write_latex_table(
            threshold_table,
            table_dir / "table_pc_classifier_threshold_sweep_appendix.tex",
            "Decision-threshold sweep for the selected political-corruption classifier.",
            "tab:pc-classifier-threshold-sweep",
        )
    )

    selected_predictions = predictions[
        predictions["embedding_model"].fillna("").eq(
            str(selected["embedding_model"])
        )
        & predictions["label_source"].eq(selected["label_source"])
    ].copy()
    if selected_predictions.empty:
        raise ValueError(
            "No validation predictions found for the selected classifier."
        )
    country_rows = []
    for country, group in selected_predictions.groupby("country"):
        y_true = group["y"].astype(int).tolist()
        y_pred = group["pred_best_threshold"].astype(int).tolist()
        metrics = binary_metrics(y_true, y_pred)
        country_rows.append(
            {
                "Country": str(country).replace("United_Kingdom", "UK"),
                "N": len(group),
                "PC support": metrics["political_support"],
                "Acc.": metrics["accuracy"],
                "PC Prec.": metrics["political_precision"],
                "PC Rec.": metrics["political_recall"],
                "PC F1": metrics["political_f1"],
                "Macro F1": metrics["macro_f1"],
                "Wtd. F1": metrics["weighted_f1"],
                "Pred. pos.": metrics["predicted_positive_rate"],
            }
        )
    country_table = pd.DataFrame(country_rows).sort_values(
        "PC F1",
        ascending=False,
    )
    outputs.append(
        write_latex_table(
            country_table,
            table_dir / "table_pc_classifier_country_validation_appendix.tex",
            "Country-level validation performance of the selected political-corruption classifier.",
            "tab:pc-classifier-country-validation",
        )
    )
    return outputs


def pipeline_counts(pipeline_dir: Path) -> dict[str, int | float]:
    import pandas as pd

    source_filtered = read_required_csv(
        pipeline_dir
        / "source_inclusion"
        / "cleaned_source_filter_output_summary.csv"
    )
    cleaned = cleaned_country_totals(pipeline_dir, source_filtered)
    classified = read_required_csv(
        pipeline_dir / "silver_classifier" / "classified_country_summary.csv"
    )
    attention = read_required_csv(
        pipeline_dir
        / "attention_tables"
        / "political_corruption_attention_total_news_country_summary.csv"
    )

    counts = {
        "raw_query": RAW_CORRUPTION_QUERY_ARTICLES,
        "cleaned_query": int(
            pd.to_numeric(cleaned["total_articles"], errors="raise").sum()
        ),
        "source_filtered_query": int(
            pd.to_numeric(source_filtered["output_rows"], errors="raise").sum()
        ),
        "classified_input": int(
            pd.to_numeric(classified["total_articles"], errors="raise").sum()
        ),
        "political_corruption": int(
            pd.to_numeric(
                classified["predicted_political_corruption"],
                errors="raise",
            ).sum()
        ),
        "total_news": int(
            pd.to_numeric(attention["total_news_articles"], errors="raise").sum()
        ),
        "threshold": read_float_text(
            pipeline_dir / "silver_classifier" / "selected_threshold.txt"
        ),
    }
    if counts["classified_input"] != counts["source_filtered_query"]:
        raise ValueError(
            "The classifier summary does not match the source-filtered corpus: "
            f"{counts['classified_input']:,} versus "
            f"{counts['source_filtered_query']:,}. Rebuild step 06."
        )
    return counts


def corpus_construction_table(pipeline_dir: Path) -> list[Path]:
    import pandas as pd

    source_filtered = read_required_csv(
        pipeline_dir
        / "source_inclusion"
        / "cleaned_source_filter_output_summary.csv"
    )
    cleaned = cleaned_country_totals(pipeline_dir, source_filtered)
    classified = read_required_csv(
        pipeline_dir / "silver_classifier" / "classified_country_summary.csv"
    )
    attention = read_required_csv(
        pipeline_dir
        / "attention_tables"
        / "political_corruption_attention_total_news_country_summary.csv"
    )

    cleaned = cleaned[["country", "total_articles"]].rename(
        columns={"total_articles": "cleaned_corruption_query_articles"}
    )
    source_filtered = source_filtered[["country", "output_rows"]].rename(
        columns={"output_rows": "source_filtered_query_articles"}
    )
    classified = classified[
        ["country", "total_articles", "predicted_political_corruption"]
    ].rename(
        columns={
            "total_articles": "classified_input_articles",
            "predicted_political_corruption": "political_corruption_articles",
        }
    )
    attention = attention[
        [
            "country",
            "country_label",
            "political_corruption_articles",
            "total_news_articles",
        ]
    ].rename(
        columns={
            "political_corruption_articles": "attention_pc_articles",
        }
    )

    data = (
        cleaned.merge(source_filtered, on="country", validate="one_to_one")
        .merge(classified, on="country", validate="one_to_one")
        .merge(attention, on="country", validate="one_to_one")
    )
    mismatch = data[
        data["source_filtered_query_articles"].ne(
            data["classified_input_articles"]
        )
    ]
    if not mismatch.empty:
        raise ValueError(
            "Source-filtered and classifier-input counts differ for: "
            + ", ".join(mismatch["country"].astype(str))
        )
    stale_attention = data[
        data["political_corruption_articles"].ne(data["attention_pc_articles"])
    ]
    if not stale_attention.empty:
        raise ValueError(
            "Attention summaries do not match the final classified corpus for: "
            + ", ".join(stale_attention["country"].astype(str))
            + ". Rerun step 07 without --skip-notebook."
        )
    data["pc_share_of_source_filtered_query_pct"] = (
        data["political_corruption_articles"]
        / data["source_filtered_query_articles"]
        * 100
    )
    data["pc_share_of_total_news_pct"] = (
        data["political_corruption_articles"]
        / data["total_news_articles"]
        * 100
    )
    data = data.sort_values("country").reset_index(drop=True)

    table_dir = pipeline_dir / "attention_tables"
    csv_path = (
        table_dir
        / "political_corruption_corpus_construction_country_summary.csv"
    )
    data.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}", flush=True)

    display = data[
        [
            "country_label",
            "cleaned_corruption_query_articles",
            "source_filtered_query_articles",
            "political_corruption_articles",
            "pc_share_of_source_filtered_query_pct",
            "total_news_articles",
            "pc_share_of_total_news_pct",
        ]
    ].rename(
        columns={
            "country_label": "Country",
            "cleaned_corruption_query_articles": "Cleaned N",
            "source_filtered_query_articles": "Source-filtered N",
            "political_corruption_articles": "PC N",
            "pc_share_of_source_filtered_query_pct": "PC share filtered (%)",
            "total_news_articles": "Total news N",
            "pc_share_of_total_news_pct": "PC share total (%)",
        }
    )
    tex_path = (
        table_dir
        / "latex"
        / "table_attention_corpus_construction_country_summary.tex"
    )
    note = (
        "Note. PC = political corruption. Cleaned N is the deduplicated "
        "corruption-query corpus. Source-filtered N retains only country-source "
        "pairs coded \\texttt{conventional\\_journalism = Yes}. Total news N "
        "comes from the separate NewsAPI country-period count series and is not "
        "source-specific."
    )
    write_latex_table(
        display,
        tex_path,
        "Country-level construction of the final political-corruption corpus.",
        "tab:pc-corpus-construction-country-summary",
        note,
    )
    return [csv_path, tex_path]


def pipeline_tikz(counts: dict[str, int | float]) -> str:
    threshold = float(counts["threshold"])
    template = r"""% Requires \usepackage{tikz} and
% \usetikzlibrary{arrows.meta,positioning}
\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  node distance=0.42cm and 1.05cm,
  box/.style={
    draw=black,
    line width=0.55pt,
    rounded corners=2pt,
    align=center,
    font=\scriptsize,
    text width=4.45cm,
    minimum height=0.82cm,
    inner sep=4pt
  },
  key/.style={
    box,
    line width=0.9pt,
    fill=black!7
  },
  output/.style={
    box,
    line width=0.9pt,
    fill=black!4,
    minimum height=1.18cm
  },
  arrow/.style={
    -{Latex[length=1.8mm]},
    line width=0.65pt
  }
]

\node[key] (api) at (0,0) {\textbf{NewsAPI}};

\node[box] (query) at (-3.0,-1.35)
  {\textbf{Article retrieval}\\Corruption-keyword queries};
\node[box] (counts) at (3.0,-1.35)
  {\textbf{News-volume retrieval}\\Total-news count request};

\node[box] (raw) at (-3.0,-2.70)
  {Raw corruption-query corpus\\$N = __RAW_QUERY__$};
\node[box] (clean) at (-3.0,-4.05)
  {Cleaned and deduplicated corpus\\$N = __CLEANED_QUERY__$};
\node[box] (screen) at (-3.0,-5.40)
  {Source-inclusion screen\\Eligible conventional news outlets};
\node[key] (filtered) at (-3.0,-6.80)
  {\textbf{Source-filtered query corpus}\\$N = __SOURCE_FILTERED_QUERY__$};
\node[box] (classifier) at (-3.0,-8.20)
  {Political-corruption classifier\\E5-large embeddings\\Logistic regression; $p \geq __THRESHOLD__$};
\node[key] (corpus) at (-3.0,-9.70)
  {\textbf{Final political-corruption corpus}\\$N = __POLITICAL_CORRUPTION__$};

\node[box] (denominator) at (3.0,-2.70)
  {Total-news denominator\\$N = __TOTAL_NEWS__$};
\node[output] (attention) at (3.0,-9.70)
  {\textbf{Relative attention}\\[1mm]
    Political-corruption articles\\
    divided by all news articles\\
    in country $c$ and period $t$};
\node[output] (coding) at (-3.0,-11.35)
  {\textbf{Article-level coding}\\[1mm]
    Victim visibility, frame,\\
    case location, and accused actor};

\draw[arrow] (api.south) -- (0,-0.68) -| (query.north);
\draw[arrow] (api.south) -- (0,-0.68) -| (counts.north);
\draw[arrow] (query) -- (raw);
\draw[arrow] (raw) -- (clean);
\draw[arrow] (clean) -- (screen);
\draw[arrow] (screen) -- (filtered);
\draw[arrow] (filtered) -- (classifier);
\draw[arrow] (classifier) -- (corpus);
\draw[arrow] (corpus) -- (coding);
\draw[arrow] (counts) -- (denominator);
\draw[arrow] (denominator) -- (attention);
\draw[arrow] (corpus) -- (attention);

\end{tikzpicture}
\caption{Construction and analytical uses of the final political-corruption
corpus. Corruption-keyword queries produce an article corpus that is cleaned,
deduplicated, restricted to conventional journalistic outlets, and classified.
A separate NewsAPI count request supplies the total-news denominator used only
for relative-attention estimates.}
\label{fig:pc-data-pipeline}
\end{figure}
"""
    replacements = {
        "__RAW_QUERY__": f"{int(counts['raw_query']):,}",
        "__CLEANED_QUERY__": f"{int(counts['cleaned_query']):,}",
        "__SOURCE_FILTERED_QUERY__": (
            f"{int(counts['source_filtered_query']):,}"
        ),
        "__THRESHOLD__": f"{threshold:.2f}",
        "__POLITICAL_CORRUPTION__": (
            f"{int(counts['political_corruption']):,}"
        ),
        "__TOTAL_NEWS__": f"{int(counts['total_news']):,}",
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def write_pipeline_tikz(pipeline_dir: Path) -> Path:
    counts = pipeline_counts(pipeline_dir)
    path = (
        pipeline_dir
        / "attention_tables"
        / "latex"
        / "figure_political_corruption_data_pipeline_tikz.tex"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipeline_tikz(counts), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    print(
        "Pipeline totals: "
        f"{counts['raw_query']:,} raw -> "
        f"{counts['cleaned_query']:,} cleaned -> "
        f"{counts['source_filtered_query']:,} source-filtered -> "
        f"{counts['political_corruption']:,} political corruption",
        flush=True,
    )
    return path


def write_build_manifest(pipeline_dir: Path) -> Path:
    counts = pipeline_counts(pipeline_dir)
    path = pipeline_dir / "manuscript_tables" / "manuscript_output_manifest.json"
    payload = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "political_classifier/scripts/07_build_attention_outputs.py",
        "raw_corruption_query_articles": counts["raw_query"],
        "cleaned_corruption_query_articles": counts["cleaned_query"],
        "source_filtered_query_articles": counts["source_filtered_query"],
        "final_political_corruption_articles": counts["political_corruption"],
        "total_news_articles": counts["total_news"],
        "selected_threshold": counts["threshold"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)
    return path


def main() -> None:
    args = parse_args()
    manifest = (
        args.pipeline_dir
        / "manuscript_tables"
        / "manuscript_output_manifest.json"
    )
    manifest.unlink(missing_ok=True)
    classifier_tables(args.pipeline_dir)
    corpus_construction_table(args.pipeline_dir)
    write_pipeline_tikz(args.pipeline_dir)
    write_build_manifest(args.pipeline_dir)


if __name__ == "__main__":
    main()
