"""Build current political-classifier manuscript tables and the pipeline figure.

All values are read from saved pipeline CSV/text outputs. This keeps the
tracked notebooks optional and prevents hand-edited manuscript numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
REPORT_LABEL_SOURCES = {"silver_labelled_training_set", "human_5fold_cv"}
TABLE_NOTE = (
    "Note. PC = political corruption. Precision, recall, and PC F1 refer to "
    "the political-corruption class. The silver model threshold was selected "
    "on the calibration partition and its performance is reported on the "
    "held-out test partition; human-trained rows are five-fold CV diagnostics."
)
SELECTED_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"


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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_required_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def require_record_matches(path: Path, record: dict | None, label: str) -> None:
    if not record or not record.get("sha256"):
        raise ValueError(f"Completed manifest lacks a checksum for {label}.")
    if not path.exists():
        raise FileNotFoundError(path)
    current = sha256_file(path)
    if current != record["sha256"]:
        raise ValueError(
            f"{label} differs from its completed run manifest. Rerun the "
            "upstream numbered step before building manuscript outputs."
        )


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
    constrain_width = len(data.columns) >= 7
    if constrain_width:
        lines.append(r"\begin{adjustbox}{max width=\textwidth}")
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
    if constrain_width:
        lines.append(r"\end{adjustbox}")
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
        & best_results["embedding_model"].fillna("").eq(SELECTED_EMBEDDING_MODEL)
    ].copy()
    if selected.empty:
        raise ValueError(
            "No silver_labelled_training_set row found in best_model_results.csv"
        )
    if "evaluation_split" in selected.columns:
        selected = selected[selected["evaluation_split"].eq("held_out_test")]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one held-out silver result for the pre-specified "
            f"model {SELECTED_EMBEDDING_MODEL}; found {len(selected)}."
        )
    return selected.iloc[0]


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
    report_rows["Evaluation"] = report_rows["evaluation_split"].map(
        {
            "held_out_test": "Held-out test",
            "full_human_5fold_cv": "5-fold CV",
        }
    )
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
        "Evaluation",
        "train_rows",
        "validation_rows",
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
        "validation_rows": "N eval",
        "threshold": "Thr.",
        "accuracy": "Acc.",
        "political_precision": "PC Prec.",
        "political_recall": "PC Rec.",
        "political_f1": "PC F1",
        "macro_f1": "Macro F1",
        "weighted_f1": "Wtd. F1",
    }
    main_row = report_rows[
        report_rows["label_source"].eq("silver_labelled_training_set")
        & report_rows["evaluation_split"].eq("held_out_test")
    ].copy()
    if len(main_row) != 1:
        raise ValueError(
            "Expected exactly one held-out silver-model row for the main table; "
            f"found {len(main_row)}."
        )
    main_table = main_row[metric_columns].rename(columns=rename_metrics)

    outputs = [
        write_latex_table(
            main_table,
            table_dir / "table_pc_classifier_comparison_main.tex",
            "Held-out validation performance of the selected political-corruption classifier.",
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
            "Note. PC = political corruption. This sweep uses only the threshold-calibration partition; the selected threshold is evaluated separately on the held-out test partition.",
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

    clean_manifest = read_required_json(
        pipeline_dir / "clean_dedupe_run_manifest.json"
    )
    source_manifest = read_required_json(
        pipeline_dir / "source_inclusion" / "source_filter_run_manifest.json"
    )
    comparison_manifest_path = (
        pipeline_dir
        / "classifier_comparison"
        / "classifier_comparison_run_manifest.json"
    )
    read_required_json(comparison_manifest_path)
    classifier_manifest = read_required_json(
        pipeline_dir / "silver_classifier" / "classifier_run_manifest.json"
    )
    source_summary_path = (
        pipeline_dir
        / "source_inclusion"
        / "cleaned_source_filter_output_summary.csv"
    )
    clean_audit_path = pipeline_dir / "clean_dedupe_audit.csv"
    classified_summary_path = (
        pipeline_dir / "silver_classifier" / "classified_country_summary.csv"
    )
    require_record_matches(clean_audit_path, clean_manifest.get("audit"), "cleaning audit")
    require_record_matches(
        source_summary_path,
        source_manifest.get("output_summary"),
        "source-filter summary",
    )
    require_record_matches(
        classified_summary_path,
        classifier_manifest.get("classified_country_summary"),
        "classified-country summary",
    )
    require_record_matches(
        comparison_manifest_path,
        classifier_manifest.get("comparison_manifest"),
        "classifier-comparison manifest",
    )
    source_decision_record = source_manifest.get("source_decision_file")
    if source_decision_record and source_decision_record.get("path"):
        require_record_matches(
            Path(source_decision_record["path"]),
            source_decision_record,
            "reviewed source-decision workbook",
        )
    if (
        (classifier_manifest.get("source_decision_file") or {}).get("sha256")
        != (source_decision_record or {}).get("sha256")
    ):
        raise ValueError(
            "Source workbook differs between source filtering and final classifier "
            "runs. Rerun steps 02-06."
        )

    source_filtered = read_required_csv(source_summary_path)
    clean_audit = read_required_csv(clean_audit_path)
    cleaned = cleaned_country_totals(pipeline_dir, source_filtered)
    classified = read_required_csv(classified_summary_path)
    attention = read_required_csv(
        pipeline_dir
        / "attention_tables"
        / "political_corruption_attention_total_news_country_summary.csv"
    )

    counts = {
        "raw_query": int(pd.to_numeric(clean_audit["raw_rows"], errors="raise").sum()),
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
    manifest_threshold = float(classifier_manifest["threshold"])
    if abs(counts["threshold"] - manifest_threshold) > 1e-12:
        raise ValueError(
            "Saved threshold differs from the completed classifier manifest. "
            "Rerun step 06 before building manuscript outputs."
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
            + ". Rerun step 07 without --skip-analysis."
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
        "corruption-query corpus. Source-filtered N removes country-source "
        "pairs coded \\texttt{conventional\\_journalism = No}. Total news N "
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
  box/.style={
    draw=black,
    line width=0.55pt,
    rounded corners=2pt,
    align=center,
    font=\scriptsize,
    text width=4.15cm,
    minimum height=0.88cm,
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
    minimum height=1.08cm
  },
  arrow/.style={
    -{Latex[length=1.8mm]},
    line width=0.65pt
  }
]

\node[key, text width=3.1cm] (api) at (0,0) {\textbf{NewsAPI}};

\node[box] (query) at (-2.75,-1.25)
  {\textbf{Corruption-query articles}\\
   Keyword retrieval: $N = __RAW_QUERY__$};
\node[box] (counts) at (2.75,-1.25)
  {\textbf{Total-news counts}\\
   $N = __TOTAL_NEWS__$};

\node[box] (clean) at (-2.75,-2.65)
  {\textbf{Cleaning and exact deduplication}\\
   Within country: $N = __CLEANED_QUERY__$};

\node[box] (sources) at (-2.75,-4.05)
  {\textbf{Source exclusions removed}\\
   Explicit outlet exclusions: $N = __SOURCE_FILTERED_QUERY__$};

\node[key] (classifier) at (-2.75,-5.55)
  {\textbf{Political-corruption classifier}\\
   E5-large + logit model\\
   $p \geq __THRESHOLD__$; final $N = __POLITICAL_CORRUPTION__$};

\node[output] (coding) at (-2.75,-7.25)
  {\textbf{Article-level coding}\\
   Victim visibility, frame, location,\\
   and accused actor};
\node[output] (attention) at (2.75,-7.25)
  {\textbf{Relative attention}\\
   Political-corruption share of all news\\
   by country-period};

\draw[arrow] (api.south) -- (0,-0.68) -| (query.north);
\draw[arrow] (api.south) -- (0,-0.68) -| (counts.north);
\draw[arrow] (query) -- (clean);
\draw[arrow] (clean) -- (sources);
\draw[arrow] (sources) -- (classifier);
\draw[arrow] (classifier) -- (coding);
\draw[arrow] (counts) -- (attention);
\draw[arrow] (classifier.east) -- (0,-5.55) |- (attention.west);

\end{tikzpicture}
\caption{Political-corruption corpus construction and analysis.
Corruption-query articles were cleaned and deduplicated, explicitly excluded
outlets were removed, and the remaining articles were classified. Separate total-news counts provide the
denominator for relative attention.}
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


def write_manuscript_values(pipeline_dir: Path) -> Path:
    """Write LaTeX commands sourced from the current validated pipeline run."""
    import pandas as pd

    counts = pipeline_counts(pipeline_dir)
    comparison_dir = pipeline_dir / "classifier_comparison"
    best = read_required_csv(comparison_dir / "best_model_results.csv")
    selected = select_final_classifier(best)
    split = read_required_csv(comparison_dir / "human_benchmark_split.csv")
    if "evaluation_split" not in split.columns or "y" not in split.columns:
        raise ValueError("human_benchmark_split.csv lacks evaluation_split or y.")

    calibration = split[split["evaluation_split"].eq("threshold_calibration")]
    held_out = split[split["evaluation_split"].eq("held_out_test")]
    final_share = (
        float(counts["political_corruption"])
        / float(counts["source_filtered_query"])
        * 100
    )
    total_news_share = (
        float(counts["political_corruption"])
        / float(counts["total_news"])
        * 100
    )
    values = {
        "PCRawQueryN": f"{int(counts['raw_query']):,}",
        "PCCleanedQueryN": f"{int(counts['cleaned_query']):,}",
        "PCSourceFilteredN": f"{int(counts['source_filtered_query']):,}",
        "PCFinalN": f"{int(counts['political_corruption']):,}",
        "PCTotalNewsN": f"{int(counts['total_news']):,}",
        "PCFinalShare": f"{final_share:.2f}",
        "PCTotalNewsShare": f"{total_news_share:.2f}",
        "PCSilverTrainN": f"{int(selected['train_rows']):,}",
        "PCHumanBenchmarkN": f"{len(split):,}",
        "PCHumanPositiveN": f"{int(pd.to_numeric(split['y']).sum()):,}",
        "PCCalibrationN": f"{len(calibration):,}",
        "PCHeldOutN": f"{len(held_out):,}",
        "PCHeldOutPositiveN": f"{int(pd.to_numeric(held_out['y']).sum()):,}",
        "PCThreshold": f"{float(selected['threshold']):.2f}",
        "PCTestAccuracy": f"{float(selected['accuracy']):.3f}",
        "PCTestPrecision": f"{float(selected['political_precision']):.3f}",
        "PCTestRecall": f"{float(selected['political_recall']):.3f}",
        "PCTestFOne": f"{float(selected['political_f1']):.3f}",
        "PCTestMacroFOne": f"{float(selected['macro_f1']):.3f}",
        "PCTestWeightedFOne": f"{float(selected['weighted_f1']):.3f}",
    }
    lines = [
        "% Generated by political_classifier/scripts/07_build_attention_outputs.py.",
        "% Do not edit these values by hand.",
    ]
    lines.extend(
        rf"\providecommand{{\{name}}}{{{value}}}" for name, value in values.items()
    )
    lines.append("")
    path = (
        pipeline_dir
        / "attention_tables"
        / "latex"
        / "political_corruption_manuscript_values.tex"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}", flush=True)
    return path


def write_build_manifest(pipeline_dir: Path) -> Path:
    counts = pipeline_counts(pipeline_dir)
    path = pipeline_dir / "manuscript_tables" / "manuscript_output_manifest.json"
    artifact_patterns = [
        "manuscript_tables/table_pc_classifier*.tex",
        "attention_tables/*.csv",
        "attention_tables/latex/*.tex",
        "attention_figures/*.png",
        "attention_figures/*.pdf",
        "attention_figures/*.svg",
        "manuscript_tables/00_LATEST_MANUSCRIPT_BUILD.txt",
    ]
    artifacts = []
    for pattern in artifact_patterns:
        for artifact in sorted(pipeline_dir.glob(pattern)):
            if artifact.is_file():
                artifacts.append(
                    {
                        "relative_path": artifact.relative_to(pipeline_dir).as_posix(),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": sha256_file(artifact),
                    }
                )
    payload = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "political_classifier/scripts/07_build_attention_outputs.py",
        "generator_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "raw_corruption_query_articles": counts["raw_query"],
        "cleaned_corruption_query_articles": counts["cleaned_query"],
        "source_filtered_query_articles": counts["source_filtered_query"],
        "final_political_corruption_articles": counts["political_corruption"],
        "total_news_articles": counts["total_news"],
        "selected_threshold": counts["threshold"],
        "artifacts": artifacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {path}", flush=True)
    return path


def write_latest_build_index(pipeline_dir: Path) -> Path:
    counts = pipeline_counts(pipeline_dir)
    classifier_manifest = read_required_json(
        pipeline_dir / "silver_classifier" / "classifier_run_manifest.json"
    )
    generator_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        text=True,
    ).strip()
    path = (
        pipeline_dir
        / "manuscript_tables"
        / "00_LATEST_MANUSCRIPT_BUILD.txt"
    )
    lines = [
        "LATEST RESPOND VICTIMS-OF-CORRUPTION MANUSCRIPT BUILD",
        "====================================================",
        "",
        f"Built (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"Generator Git commit: {generator_commit}",
        f"Classifier Git commit: {classifier_manifest.get('git_commit', 'unknown')}",
        f"Final political-corruption N: {int(counts['political_corruption']):,}",
        f"Source-filtered query N: {int(counts['source_filtered_query']):,}",
        f"Selected threshold: {float(counts['threshold']):.2f}",
        "",
        "CANONICAL LATEST FILES ON RESEARCH DRIVE",
        "----------------------------------------",
        "output/manuscript/method.tex",
        "output/manuscript/results_political_corruption_attention.tex",
        "output/manuscript/appendix_political_corruption.tex",
        "output/tables/table_pc_classifier_comparison_main.tex",
        "output/tables/table_pc_classifier_comparison_appendix.tex",
        "output/tables/table_pc_classifier_threshold_sweep_appendix.tex",
        "output/tables/table_pc_classifier_country_validation_appendix.tex",
        "output/tables/attention/political_corruption_manuscript_values.tex",
        "output/tables/attention/figure_political_corruption_data_pipeline_tikz.tex",
        "output/tables/attention/table_attention_corpus_construction_country_summary.tex",
        "output/tables/attention/table_attention_country_summary.tex",
        "output/tables/attention/table_attention_country_year_matrix.tex",
        "output/tables/attention/table_attention_peak_months.tex",
        "output/figures/attention/political_corruption_relative_attention_total_news_month_small_multiples.png",
        "output/figures/attention/political_corruption_absolute_volume_month_stacked.png",
        "output/manuscript_output_manifest.json",
        "",
        "OVERLEAF INPUTS",
        "---------------",
        r"\input{output/manuscript/method}",
        r"\input{output/manuscript/results_political_corruption_attention}",
        r"\appendix",
        r"\input{output/manuscript/appendix_political_corruption}",
        "",
        "The output/ tree is overwritten with the latest validated build.",
        "Immutable timestamped runs are stored under derived_data/political_classifier/runs/.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
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
    write_manuscript_values(args.pipeline_dir)
    write_latest_build_index(args.pipeline_dir)
    write_build_manifest(args.pipeline_dir)


if __name__ == "__main__":
    main()
