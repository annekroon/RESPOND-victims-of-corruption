"""Compare candidate political-corruption classifiers on human labels.

This script writes clean comparison tables for deciding the final classifier.
It evaluates:

1. TF-IDF trained/evaluated with cross-validation on the human labels.
2. Multilingual embeddings trained/evaluated with cross-validation on human labels.
3. Multilingual embeddings trained on the combined LLM silver-labelled training
   set, validated on human labels.

By default, supplied silver-label files are pooled into one training set. Use
--include-silver-source-diagnostics only when you explicitly want to inspect
individual source-file diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.source_filter import (
    DEFAULT_SOURCE_DECISION_FILE,
    load_source_decisions,
)
from political_classifier.classifier_data import (
    choose_integrity_text_series,
    choose_text_series,
    filter_frame_by_source,
    format_texts_for_embedding,
    load_human_validation,
    normalize_text,
)
from political_classifier.split_integrity import (
    assert_no_validation_overlap,
    calibration_test_split,
)
from political_classifier.reproducibility import (
    embedding_revision,
    file_record,
    frame_fingerprint,
    git_commit,
    package_versions,
)


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_SILVER_LABEL_DIR = DEFAULT_PIPELINE_DIR / "active_learning"
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "classifier_comparison"
DEFAULT_UK_REVIEWED_VALIDATION_PATH = DEFAULT_SILVER_LABEL_DIR / "uk_human_validation_reviewed.csv"
DEFAULT_SILVER_LABEL_PATHS = [
    DEFAULT_SILVER_LABEL_DIR / "silver_training_source_filtered_with_llm_suggestions.csv",
]
DEFAULT_EMBEDDING_MODELS = [
    "intfloat/multilingual-e5-large",
]
RECOMMENDED_EMBEDDING_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "sentence-transformers/LaBSE",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
]

SILVER_LABEL_MAP = {
    "Yes": 1,
    "No": 0,
    "Mentioned but not central": 0,
}
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate classifier models.")
    parser.add_argument(
        "--silver-labels",
        type=Path,
        nargs="+",
        default=DEFAULT_SILVER_LABEL_PATHS,
        help="LLM-labelled CSV files that will be pooled into one silver-labelled training set.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embedding-models",
        nargs="+",
        default=DEFAULT_EMBEDDING_MODELS,
        help=(
            "One or more sentence-transformers embedding models to compare. "
            "For a broader check, use: " + " ".join(RECOMMENDED_EMBEDDING_MODELS)
        ),
    )
    parser.add_argument(
        "--embedding-model",
        dest="embedding_model",
        default=None,
        help="Backward-compatible alias for running one embedding model.",
    )
    parser.add_argument(
        "--extra-human-validation",
        type=Path,
        nargs="+",
        default=[DEFAULT_UK_REVIEWED_VALIDATION_PATH],
        help=(
            "Optional manually reviewed validation CSV files to append to the "
            "original human validation set. Expected label column: "
            "human_final_label, corruption_label_m, or label."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--threshold-calibration-fraction",
        type=float,
        default=0.40,
        help=(
            "Fraction of the human benchmark used only to choose the decision "
            "threshold. Remaining rows form the held-out reporting set."
        ),
    )
    parser.add_argument(
        "--include-silver-source-diagnostics",
        action="store_true",
        help=(
            "Also evaluate each supplied silver-label file separately. This is "
            "for diagnostics only; manuscript tables use the pooled silver-labelled set."
        ),
    )
    parser.add_argument(
        "--source-decision-file",
        type=Path,
        default=DEFAULT_SOURCE_DECISION_FILE,
        help="Source decision workbook/CSV. Only conventional_journalism == No is excluded.",
    )
    parser.add_argument(
        "--no-source-filter",
        action="store_true",
        help="Do not remove explicitly excluded sources from silver-label training data.",
    )
    parser.add_argument(
        "--keep-unfiltered-validation",
        action="store_true",
        help=(
            "Diagnostic override: retain human-validation rows from explicitly "
            "excluded sources. By default the benchmark matches the production "
            "source-eligible population."
        ),
    )
    return parser.parse_args()


def load_one_silver_file(path: Path, source_decisions=None):
    import pandas as pd

    data = pd.read_csv(path)
    dedupe_key = "silver_row_id" if "silver_row_id" in data.columns else "uri"
    if dedupe_key in data.columns:
        data = data.drop_duplicates(subset=[dedupe_key], keep="last").copy()
    if "llm_error" in data.columns:
        data = data[data["llm_error"].fillna("").astype(str).str.strip().eq("")].copy()
    data["silver_source_file"] = str(path)
    data = data[data["llm_label_suggestion"].isin(SILVER_LABEL_MAP)].copy()
    data["y"] = data["llm_label_suggestion"].map(SILVER_LABEL_MAP).astype(int)
    data["model_text"] = choose_text_series(data).fillna("").astype(str).map(normalize_text)
    data["integrity_text"] = (
        choose_integrity_text_series(data).fillna("").astype(str).map(normalize_text)
    )
    data = data[data["model_text"].str.strip().ne("")].copy()
    data["_training_text_hash"] = data["model_text"].map(
        lambda value: hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()
    )
    data = data.drop_duplicates(subset=["_training_text_hash"], keep="last").copy()
    if source_decisions is not None:
        data = filter_frame_by_source(data, source_decisions, path.name)
    return data


def load_silver_variants(paths, include_source_diagnostics=False, source_decisions=None):
    import pandas as pd

    variants = {}
    frames = []
    for idx, path in enumerate(paths, start=1):
        if not path.exists():
            raise FileNotFoundError(f"Required silver-label file not found: {path}")
        complete_path = path.with_name(path.name + ".complete.json")
        if not complete_path.exists():
            raise FileNotFoundError(
                f"Silver-label completion marker not found: {complete_path}. "
                "Resume 04_label_silver_batch.py before model comparison."
            )
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        current_record = file_record(path)
        if (
            completion.get("status") != "complete"
            or (completion.get("output") or {}).get("sha256")
            != current_record["sha256"]
        ):
            raise ValueError(
                f"Silver-label file does not match its completion marker: {path}"
            )
        data = load_one_silver_file(path, source_decisions=source_decisions)
        if include_source_diagnostics:
            variants[f"silver_source_file_{idx}"] = data
        frames.append(data)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["_training_text_hash"], keep="last").copy()
        variants["silver_labelled_training_set"] = combined

    if not variants:
        raise FileNotFoundError("No silver-label files were found.")

    return variants


def resolve_embedding_models(args):
    if args.embedding_model:
        return [args.embedding_model]
    return args.embedding_models


def safe_model_slug(model_name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")


def metrics_from_predictions(
    y_true,
    y_pred,
    model_name,
    threshold=None,
    train_rows=None,
    label_source=None,
    embedding_model=None,
):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "model": model_name,
        "embedding_model": embedding_model or "",
        "label_source": label_source or "",
        "train_rows": train_rows,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "no_precision": precision[0],
        "no_recall": recall[0],
        "no_f1": f1[0],
        "political_precision": precision[1],
        "political_recall": recall[1],
        "political_f1": f1[1],
        "macro_f1": (f1[0] + f1[1]) / 2,
        "weighted_f1": None,
        "predicted_positive_rate": y_pred.mean(),
        "validation_rows": len(y_true),
        "political_support": int((y_true == 1).sum()),
    }


def add_weighted_f1(row, y_true):
    no_support = int((y_true == 0).sum())
    political_support = int((y_true == 1).sum())
    total = no_support + political_support
    row["weighted_f1"] = (
        row["no_f1"] * no_support + row["political_f1"] * political_support
    ) / total
    return row


def threshold_table(y_true, probabilities, model_name, train_rows, label_source, embedding_model=None):
    import pandas as pd

    rows = []
    for threshold in THRESHOLDS:
        pred = (probabilities >= threshold).astype(int)
        row = metrics_from_predictions(
            y_true=y_true,
            y_pred=pred,
            model_name=model_name,
            threshold=threshold,
            train_rows=train_rows,
            label_source=label_source,
            embedding_model=embedding_model,
        )
        rows.append(add_weighted_f1(row, y_true))
    return pd.DataFrame(rows)


def country_table(
    valid_df,
    pred_column,
    model_name,
    threshold,
    train_rows,
    label_source,
    embedding_model=None,
):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    rows = []
    for country, country_df in valid_df.groupby("country"):
        y_country = country_df["y"]
        pred_country = country_df[pred_column]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_country,
            pred_country,
            labels=[1],
            zero_division=0,
        )
        rows.append(
            {
                "model": model_name,
                "embedding_model": embedding_model or "",
                "label_source": label_source,
                "threshold": threshold,
                "train_rows": train_rows,
                "country": country,
                "n": len(country_df),
                "political_support": int((y_country == 1).sum()),
                "accuracy": accuracy_score(y_country, pred_country),
                "political_precision": precision[0],
                "political_recall": recall[0],
                "political_f1": f1[0],
                "predicted_positive_rate": pred_country.mean(),
            }
        )
    return pd.DataFrame(rows)


def evaluate_tfidf_human_cv(valid_df, random_state):
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=200_000,
                    lowercase=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=random_state,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        model,
        valid_df["model_text"],
        valid_df["y"],
        cv=cv,
        method="predict_proba",
    )[:, 1]
    table = threshold_table(
        valid_df["y"].to_numpy(),
        probabilities,
        model_name="tfidf_char_ngrams_logreg",
        train_rows=len(valid_df),
        label_source="human_5fold_cv",
    )
    pred = (probabilities >= 0.5).astype(int)
    predictions = pd.DataFrame(
        {
            "row_index": valid_df.index,
            "model": "tfidf_char_ngrams_logreg",
            "embedding_model": "",
            "prob": probabilities,
            "pred_t05": pred,
            "y": valid_df["y"].to_numpy(),
            "country": valid_df["country"].to_numpy(),
        }
    )
    return table, predictions


def evaluate_embedding_human_cv(valid_df, embeddings, random_state, embedding_model):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probabilities = cross_val_predict(
        model,
        embeddings,
        valid_df["y"],
        cv=cv,
        method="predict_proba",
    )[:, 1]
    table = threshold_table(
        valid_df["y"].to_numpy(),
        probabilities,
        model_name="multilingual_embeddings_logreg",
        train_rows=len(valid_df),
        label_source="human_5fold_cv",
        embedding_model=embedding_model,
    )
    predictions = pd.DataFrame(
        {
            "row_index": valid_df.index,
            "model": "multilingual_embeddings_logreg",
            "embedding_model": embedding_model,
            "prob": probabilities,
            "pred_t05": (probabilities >= 0.5).astype(int),
            "y": valid_df["y"].to_numpy(),
            "country": valid_df["country"].to_numpy(),
        }
    )
    return table, predictions


def evaluate_silver_variant(
    name,
    train_df,
    valid_df,
    valid_embeddings,
    embedder,
    embedding_model,
    batch_size,
    random_state,
):
    from sklearn.linear_model import LogisticRegression

    train_embeddings = embedder.encode(
        format_texts_for_embedding(train_df["model_text"].tolist(), embedding_model),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    model.fit(train_embeddings, train_df["y"])
    probabilities = model.predict_proba(valid_embeddings)[:, 1]
    table = threshold_table(
        valid_df["y"].to_numpy(),
        probabilities,
        model_name="multilingual_embeddings_logreg",
        train_rows=len(train_df),
        label_source=name,
        embedding_model=embedding_model,
    )
    return table, probabilities, model


def select_best_rows(all_threshold_results):
    sort_columns = ["political_f1", "political_recall", "political_precision", "accuracy"]
    best_rows = []
    for _, group in all_threshold_results.groupby(
        ["model", "embedding_model", "label_source"],
        dropna=False,
    ):
        best_rows.append(group.sort_values(sort_columns, ascending=False).iloc[0])
    return (
        all_threshold_results.__class__(best_rows)
        .sort_values(["political_f1", "political_recall", "political_precision"], ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    import pandas as pd
    from sentence_transformers import SentenceTransformer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "classifier_comparison_run_manifest.json").unlink(
        missing_ok=True
    )
    embedding_models = resolve_embedding_models(args)

    source_decisions = None
    if not args.no_source_filter:
        source_decisions = load_source_decisions(args.source_decision_file)
        print(f"Loaded source decisions: {args.source_decision_file}", flush=True)

    validation_source_decisions = (
        None if args.keep_unfiltered_validation else source_decisions
    )
    valid_df = load_human_validation(
        args.extra_human_validation,
        source_decisions=validation_source_decisions,
    )
    valid_df = valid_df.reset_index(drop=True)
    if source_decisions is not None and validation_source_decisions is None:
        print(
            "Source filter is applied to silver training data only; "
            "human validation is left unfiltered.",
            flush=True,
        )
    print(f"Human validation rows: {len(valid_df):,}", flush=True)
    print(valid_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    silver_variants = load_silver_variants(
        args.silver_labels,
        include_source_diagnostics=args.include_silver_source_diagnostics,
        source_decisions=source_decisions,
    )
    for name, data in silver_variants.items():
        overlap_audit = args.output_dir / f"{name}_human_benchmark_overlap.csv"
        assert_no_validation_overlap(
            data,
            valid_df,
            text_column="integrity_text",
            audit_path=overlap_audit,
        )
        print(f"{name}: {len(data):,} rows", flush=True)
        print(data["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    calibration_df, test_df = calibration_test_split(
        valid_df,
        calibration_fraction=args.threshold_calibration_fraction,
        random_state=args.random_state,
    )
    split_assignment = valid_df[[column for column in ["uri", "country", "y"] if column in valid_df.columns]].copy()
    split_assignment["evaluation_split"] = ""
    split_assignment.loc[calibration_df.index, "evaluation_split"] = "threshold_calibration"
    split_assignment.loc[test_df.index, "evaluation_split"] = "held_out_test"
    split_path = args.output_dir / "human_benchmark_split.csv"
    split_assignment.to_csv(split_path, index_label="row_index")
    print(
        f"Threshold calibration rows: {len(calibration_df):,}; "
        f"held-out reporting rows: {len(test_df):,}",
        flush=True,
    )

    threshold_tables = []
    best_result_rows = []
    prediction_tables = []
    model_revisions: dict[str, str | None] = {}

    tfidf_table, tfidf_predictions = evaluate_tfidf_human_cv(valid_df, args.random_state)
    tfidf_table = tfidf_table[tfidf_table["threshold"].eq(0.50)].copy()
    tfidf_table["evaluation_split"] = "full_human_5fold_cv"
    threshold_tables.append(tfidf_table)
    best_result_rows.append(tfidf_table.iloc[0].to_dict())
    prediction_tables.append(tfidf_predictions)

    country_tables = []
    silver_prediction_frames = []

    for embedding_model in embedding_models:
        print(f"\nLoading embedding model: {embedding_model}", flush=True)
        embedder = SentenceTransformer(embedding_model)
        model_revisions[embedding_model] = embedding_revision(embedder)

        print(f"Encoding human validation set with {embedding_model}", flush=True)
        valid_embeddings = embedder.encode(
            format_texts_for_embedding(valid_df["model_text"].tolist(), embedding_model),
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        embedding_cv_table, embedding_cv_predictions = evaluate_embedding_human_cv(
            valid_df=valid_df,
            embeddings=valid_embeddings,
            random_state=args.random_state,
            embedding_model=embedding_model,
        )
        embedding_cv_table = embedding_cv_table[embedding_cv_table["threshold"].eq(0.50)].copy()
        embedding_cv_table["evaluation_split"] = "full_human_5fold_cv"
        threshold_tables.append(embedding_cv_table)
        best_result_rows.append(embedding_cv_table.iloc[0].to_dict())
        prediction_tables.append(embedding_cv_predictions)

        calibration_positions = valid_df.index.get_indexer(calibration_df.index)
        test_positions = valid_df.index.get_indexer(test_df.index)
        calibration_embeddings = valid_embeddings[calibration_positions]
        test_embeddings = valid_embeddings[test_positions]

        for name, train_df in silver_variants.items():
            print(f"\nEvaluating silver model: {name} with {embedding_model}", flush=True)
            table, _, fitted_model = evaluate_silver_variant(
                name=name,
                train_df=train_df,
                valid_df=calibration_df,
                valid_embeddings=calibration_embeddings,
                embedder=embedder,
                embedding_model=embedding_model,
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
            table["evaluation_split"] = "threshold_calibration"
            threshold_tables.append(table)

            best_row = table.sort_values(
                ["political_f1", "political_recall", "political_precision", "accuracy"],
                ascending=False,
            ).iloc[0]
            test_probabilities = fitted_model.predict_proba(test_embeddings)[:, 1]
            test_predictions = (test_probabilities >= best_row["threshold"]).astype(int)
            test_result = metrics_from_predictions(
                y_true=test_df["y"].to_numpy(),
                y_pred=test_predictions,
                model_name="multilingual_embeddings_logreg",
                threshold=float(best_row["threshold"]),
                train_rows=len(train_df),
                label_source=name,
                embedding_model=embedding_model,
            )
            test_result = add_weighted_f1(test_result, test_df["y"].to_numpy())
            test_result["evaluation_split"] = "held_out_test"
            best_result_rows.append(test_result)

            pred_column = f"pred_{safe_model_slug(embedding_model)}_{name}"
            test_with_predictions = test_df.copy()
            test_with_predictions[pred_column] = test_predictions
            country_tables.append(
                country_table(
                    test_with_predictions,
                    pred_column=pred_column,
                    model_name="multilingual_embeddings_logreg",
                    threshold=best_row["threshold"],
                    train_rows=len(train_df),
                    label_source=name,
                    embedding_model=embedding_model,
                )
            )
            silver_prediction_frames.append(
                pd.DataFrame(
                    {
                        "row_index": test_df.index,
                        "model": "multilingual_embeddings_logreg",
                        "embedding_model": embedding_model,
                        "label_source": name,
                        "prob": test_probabilities,
                        "best_threshold": best_row["threshold"],
                        "pred_best_threshold": test_predictions,
                        "y": test_df["y"].to_numpy(),
                        "country": test_df["country"].to_numpy(),
                        "evaluation_split": "held_out_test",
                    }
                )
            )

    all_threshold_results = pd.concat(threshold_tables, ignore_index=True)
    best_results = pd.DataFrame(best_result_rows).sort_values(
        ["political_f1", "political_recall", "political_precision"],
        ascending=False,
    ).reset_index(drop=True)
    all_predictions = pd.concat(prediction_tables + silver_prediction_frames, ignore_index=True)

    all_threshold_results.to_csv(args.output_dir / "all_threshold_results.csv", index=False)
    best_results.to_csv(args.output_dir / "best_model_results.csv", index=False)
    all_predictions.to_csv(args.output_dir / "validation_prediction_comparison.csv", index=False)

    if country_tables:
        pd.concat(country_tables, ignore_index=True).to_csv(
            args.output_dir / "country_results_for_best_silver_thresholds.csv",
            index=False,
        )

    primary = best_results[
        best_results["embedding_model"].fillna("").eq(embedding_models[0])
        & best_results["label_source"].eq("silver_labelled_training_set")
        & best_results["evaluation_split"].eq("held_out_test")
    ]
    if len(primary) != 1:
        raise ValueError(
            "Expected exactly one held-out result for the primary embedding model "
            f"{embedding_models[0]!r}; found {len(primary)}."
        )
    selected_threshold = float(primary.iloc[0]["threshold"])
    (args.output_dir / "selected_threshold.txt").write_text(
        f"{selected_threshold:.10g}\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "embedding_models": embedding_models,
        "embedding_model_revisions": model_revisions,
        "primary_embedding_model": embedding_models[0],
        "selected_threshold": selected_threshold,
        "threshold_selection_split": "threshold_calibration",
        "reported_performance_split": "held_out_test",
        "threshold_calibration_fraction": args.threshold_calibration_fraction,
        "random_state": args.random_state,
        "batch_size": args.batch_size,
        "source_filter_applied_to_silver": not args.no_source_filter,
        "source_filter_applied_to_validation": (
            not args.no_source_filter and not args.keep_unfiltered_validation
        ),
        "human_benchmark_rows": len(valid_df),
        "human_benchmark_sha256": frame_fingerprint(
            valid_df,
            ["country", "uri", "integrity_text", "model_text", "y"],
        ),
        "threshold_calibration_rows": len(calibration_df),
        "held_out_test_rows": len(test_df),
        "benchmark_split_file": file_record(split_path),
        "silver_label_files": [file_record(path) for path in args.silver_labels],
        "silver_label_completion_markers": [
            file_record(path.with_name(path.name + ".complete.json"))
            for path in args.silver_labels
        ],
        "source_decision_file": (
            None if args.no_source_filter else file_record(args.source_decision_file)
        ),
        "package_versions": package_versions(),
    }
    (args.output_dir / "classifier_comparison_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Primary pre-specified model: {embedding_models[0]}; "
        f"calibration-selected threshold: {selected_threshold:.2f}",
        flush=True,
    )

    print("\nBest model results:", flush=True)
    print(best_results, flush=True)
    print(f"\nSaved comparison tables to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
