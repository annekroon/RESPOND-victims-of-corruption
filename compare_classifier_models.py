"""Compare candidate political-corruption classifiers on human labels.

This script writes clean comparison tables for deciding the final classifier.
It evaluates:

1. TF-IDF trained/evaluated with cross-validation on the human labels.
2. Multilingual embeddings trained/evaluated with cross-validation on human labels.
3. Multilingual embeddings trained on LLM silver labels, validated on human labels.

The silver-label variants are evaluated per supplied silver-label file and for
the combined set of all supplied files.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_ACTIVE_LEARNING_DIR = DEFAULT_PIPELINE_DIR / "active_learning"
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "classifier_comparison"
DEFAULT_SILVER_LABEL_PATHS = [
    DEFAULT_ACTIVE_LEARNING_DIR / "active_learning_batch_with_llm_suggestions.csv",
    DEFAULT_ACTIVE_LEARNING_DIR / "active_learning_batch_2_with_llm_suggestions.csv",
]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SILVER_LABEL_MAP = {
    "Yes": 1,
    "No": 0,
    "Mentioned but not central": 0,
}
HUMAN_LABEL_MAP = {
    "political corruption": 1,
    "no political corruption": 0,
}
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]


def fix_mojibake(text):
    if not isinstance(text, str):
        return ""

    candidates = [text]
    try:
        candidates.append(text.encode("latin1").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    try:
        candidates.append(text.encode("cp1252").decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    def badness(s):
        markers = ["Ð", "Ñ", "Ã", "Â", "Ä", "Å", "�"]
        return sum(s.count(m) for m in markers)

    return min(candidates, key=badness)


def normalize_text(text):
    text = fix_mojibake(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare candidate classifier models.")
    parser.add_argument(
        "--silver-labels",
        type=Path,
        nargs="+",
        default=DEFAULT_SILVER_LABEL_PATHS,
        help="LLM-labelled active-learning CSV files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def choose_text_series(data):
    if "translated_text" in data.columns:
        return data["translated_text"]
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]

    title = data["title"].fillna("").astype(str) if "title" in data.columns else ""
    body = data["body"].fillna("").astype(str) if "body" in data.columns else ""
    return title + "\n" + body


def load_human_validation():
    from dataloader import load_human_annotated_for_translation_webdav

    data = load_human_annotated_for_translation_webdav()
    data["label_clean"] = data["corruption_label_m"].astype(str).str.strip().str.lower()
    data["y"] = data["label_clean"].map(HUMAN_LABEL_MAP)
    data = data[data["y"].notna()].copy()
    data["y"] = data["y"].astype(int)
    data["model_text"] = choose_text_series(data).fillna("").astype(str).map(normalize_text)
    data = data[data["model_text"].str.strip().ne("")].copy()
    return data


def load_one_silver_file(path: Path):
    import pandas as pd

    data = pd.read_csv(path)
    data["silver_source_file"] = str(path)
    data = data[data["llm_label_suggestion"].isin(SILVER_LABEL_MAP)].copy()
    data["y"] = data["llm_label_suggestion"].map(SILVER_LABEL_MAP).astype(int)
    data["model_text"] = choose_text_series(data).fillna("").astype(str).map(normalize_text)
    data = data[data["model_text"].str.strip().ne("")].copy()
    return data


def load_silver_variants(paths):
    import pandas as pd

    variants = {}
    frames = []
    for idx, path in enumerate(paths, start=1):
        if not path.exists():
            print(f"Skipping missing silver-label file: {path}", flush=True)
            continue
        name = f"silver_batch_{idx}"
        data = load_one_silver_file(path)
        variants[name] = data
        frames.append(data)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if "uri" in combined.columns:
            combined = combined.drop_duplicates(subset=["uri"], keep="first").copy()
        variants["silver_combined"] = combined

    if not variants:
        raise FileNotFoundError("No silver-label files were found.")

    return variants


def metrics_from_predictions(y_true, y_pred, model_name, threshold=None, train_rows=None, label_source=None):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )
    return {
        "model": model_name,
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


def threshold_table(y_true, probabilities, model_name, train_rows, label_source):
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
        )
        rows.append(add_weighted_f1(row, y_true))
    return pd.DataFrame(rows)


def country_table(valid_df, pred_column, model_name, threshold, train_rows, label_source):
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
            "prob": probabilities,
            "pred_t05": pred,
            "y": valid_df["y"].to_numpy(),
            "country": valid_df["country"].to_numpy(),
        }
    )
    return table, predictions


def evaluate_embedding_human_cv(valid_df, embeddings, random_state):
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
    )
    predictions = pd.DataFrame(
        {
            "row_index": valid_df.index,
            "model": "multilingual_embeddings_logreg",
            "prob": probabilities,
            "pred_t05": (probabilities >= 0.5).astype(int),
            "y": valid_df["y"].to_numpy(),
            "country": valid_df["country"].to_numpy(),
        }
    )
    return table, predictions


def evaluate_silver_variant(name, train_df, valid_df, valid_embeddings, embedder, batch_size, random_state):
    from sklearn.linear_model import LogisticRegression

    train_embeddings = embedder.encode(
        train_df["model_text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)
    model.fit(train_embeddings, train_df["y"])
    probabilities = model.predict_proba(valid_embeddings)[:, 1]
    return threshold_table(
        valid_df["y"].to_numpy(),
        probabilities,
        model_name="multilingual_embeddings_logreg",
        train_rows=len(train_df),
        label_source=name,
    ), probabilities


def select_best_rows(all_threshold_results):
    sort_columns = ["political_f1", "political_recall", "political_precision", "accuracy"]
    best_rows = []
    for _, group in all_threshold_results.groupby(["model", "label_source"], dropna=False):
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

    valid_df = load_human_validation()
    print(f"Human validation rows: {len(valid_df):,}", flush=True)
    print(valid_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    silver_variants = load_silver_variants(args.silver_labels)
    for name, data in silver_variants.items():
        print(f"{name}: {len(data):,} rows", flush=True)
        print(data["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    threshold_tables = []
    prediction_tables = []

    tfidf_table, tfidf_predictions = evaluate_tfidf_human_cv(valid_df, args.random_state)
    threshold_tables.append(tfidf_table)
    prediction_tables.append(tfidf_predictions)

    embedder = SentenceTransformer(args.embedding_model)
    valid_embeddings = embedder.encode(
        valid_df["model_text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embedding_cv_table, embedding_cv_predictions = evaluate_embedding_human_cv(
        valid_df,
        valid_embeddings,
        args.random_state,
    )
    threshold_tables.append(embedding_cv_table)
    prediction_tables.append(embedding_cv_predictions)

    country_tables = []
    silver_prediction_frames = []
    for name, train_df in silver_variants.items():
        print(f"\nEvaluating silver model: {name}", flush=True)
        table, probabilities = evaluate_silver_variant(
            name=name,
            train_df=train_df,
            valid_df=valid_df,
            valid_embeddings=valid_embeddings,
            embedder=embedder,
            batch_size=args.batch_size,
            random_state=args.random_state,
        )
        threshold_tables.append(table)

        best_row = table.sort_values(
            ["political_f1", "political_recall", "political_precision", "accuracy"],
            ascending=False,
        ).iloc[0]
        pred_column = f"pred_{name}"
        valid_df[pred_column] = (probabilities >= best_row["threshold"]).astype(int)
        country_tables.append(
            country_table(
                valid_df,
                pred_column=pred_column,
                model_name="multilingual_embeddings_logreg",
                threshold=best_row["threshold"],
                train_rows=len(train_df),
                label_source=name,
            )
        )
        silver_prediction_frames.append(
            pd.DataFrame(
                {
                    "row_index": valid_df.index,
                    "model": "multilingual_embeddings_logreg",
                    "label_source": name,
                    "prob": probabilities,
                    "best_threshold": best_row["threshold"],
                    "pred_best_threshold": valid_df[pred_column].to_numpy(),
                    "y": valid_df["y"].to_numpy(),
                    "country": valid_df["country"].to_numpy(),
                }
            )
        )

    all_threshold_results = pd.concat(threshold_tables, ignore_index=True)
    best_results = select_best_rows(all_threshold_results)
    all_predictions = pd.concat(prediction_tables + silver_prediction_frames, ignore_index=True)

    all_threshold_results.to_csv(args.output_dir / "all_threshold_results.csv", index=False)
    best_results.to_csv(args.output_dir / "best_model_results.csv", index=False)
    all_predictions.to_csv(args.output_dir / "validation_prediction_comparison.csv", index=False)

    if country_tables:
        pd.concat(country_tables, ignore_index=True).to_csv(
            args.output_dir / "country_results_for_best_silver_thresholds.csv",
            index=False,
        )

    print("\nBest model results:", flush=True)
    print(best_results, flush=True)
    print(f"\nSaved comparison tables to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
