"""Train and apply a political-corruption classifier from LLM silver labels.

This script reads one or more LLM-labelled active-learning CSVs, trains an
embedding classifier on those silver labels, validates against the human-labelled
set, and can optionally classify the cleaned country files.

Examples:
    python3 05_train_final_classifier.py

    nohup python3 -u 05_train_final_classifier.py --score-corpus --threshold 0.30 \
      > silver_classifier_scoring.log 2>&1 &
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
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier"
DEFAULT_SILVER_LABEL_PATHS = [
    DEFAULT_ACTIVE_LEARNING_DIR / "active_learning_batch_with_llm_suggestions.csv",
    DEFAULT_ACTIVE_LEARNING_DIR / "active_learning_batch_2_with_llm_suggestions.csv",
]
DEFAULT_COUNTRIES = [
    "Bulgaria",
    "France",
    "Hungary",
    "Italy",
    "Netherlands",
    "Serbia",
    "Sweden",
    "Ukraine",
    "United_Kingdom",
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

CLASSIFIED_OUTPUT_COLUMNS = [
    "uri",
    "country",
    "date_parsed",
    "year",
    "month",
    "week",
    "source_uri",
    "word_count",
    "prob_political_corruption",
    "pred_political_corruption",
    "article_text",
]


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
    parser = argparse.ArgumentParser(
        description="Train a silver-label classifier and optionally classify the cleaned corpus."
    )
    parser.add_argument(
        "--silver-labels",
        type=Path,
        nargs="+",
        default=DEFAULT_SILVER_LABEL_PATHS,
        help="One or more LLM-labelled active-learning CSV files.",
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument(
        "--select-threshold",
        action="store_true",
        help="Pick the threshold with best political-corruption F1 on the human validation set.",
    )
    parser.add_argument("--score-corpus", action="store_true")
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
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


def load_silver_labels(paths):
    import pandas as pd

    frames = []
    for path in paths:
        if not path.exists():
            print(f"Skipping missing silver-label file: {path}", flush=True)
            continue
        data = pd.read_csv(path)
        data["silver_source_file"] = str(path)
        frames.append(data)

    if not frames:
        raise FileNotFoundError("No silver-label files were found.")

    silver = pd.concat(frames, ignore_index=True)
    if "uri" in silver.columns:
        before = len(silver)
        silver = silver.drop_duplicates(subset=["uri"], keep="first").copy()
        print(f"Dropped duplicate silver-label URIs: {before - len(silver):,}", flush=True)

    silver = silver[silver["llm_label_suggestion"].isin(SILVER_LABEL_MAP)].copy()
    silver["y"] = silver["llm_label_suggestion"].map(SILVER_LABEL_MAP).astype(int)
    silver["model_text"] = choose_text_series(silver).fillna("").astype(str).map(normalize_text)
    silver = silver[silver["model_text"].str.strip().ne("")].copy()
    return silver


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


def evaluate_thresholds(y_true, probabilities):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    rows = []
    for threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        pred = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            labels=[1],
            zero_division=0,
        )
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, pred),
                "political_precision": precision[0],
                "political_recall": recall[0],
                "political_f1": f1[0],
                "predicted_positive_rate": pred.mean(),
            }
        )
    return pd.DataFrame(rows)


def country_metrics(valid_df, pred_column):
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
    return pd.DataFrame(rows).sort_values("political_f1", ascending=False)


def save_validation_outputs(valid_df, threshold_results, country_results, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_df.to_csv(output_dir / "human_validation_predictions.csv", index=False)
    threshold_results.to_csv(output_dir / "threshold_validation_results.csv", index=False)
    country_results.to_csv(output_dir / "country_validation_results.csv", index=False)


def cleaned_country_path(pipeline_dir: Path, country: str) -> Path:
    return pipeline_dir / f"{country}_cleaned_deduped.csv.gz"


def score_texts(texts, embedder, clf, batch_size: int):
    import numpy as np
    from tqdm.auto import tqdm

    all_probs = []
    starts = range(0, len(texts), batch_size * 8)
    for start in tqdm(list(starts), desc="Embedding + scoring"):
        end = min(start + batch_size * 8, len(texts))
        batch_texts = texts[start:end]
        embeddings = embedder.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_probs.append(clf.predict_proba(embeddings)[:, 1])
    return np.concatenate(all_probs) if all_probs else np.array([])


def score_country_file(country: str, path: Path, output_dir: Path, embedder, clf, threshold: float, batch_size: int):
    import pandas as pd

    if not path.exists():
        print(f"Skipping missing cleaned file for {country}: {path}", flush=True)
        return None

    print(f"\nScoring {country}: {path}", flush=True)
    data = pd.read_csv(path)
    data["model_text"] = data["article_text"].fillna("").astype(str).map(normalize_text)
    probs = score_texts(data["model_text"].tolist(), embedder, clf, batch_size)

    data["prob_political_corruption"] = probs
    data["pred_political_corruption"] = (data["prob_political_corruption"] >= threshold).astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{country}_classified.csv.gz"
    output_cols = [column for column in CLASSIFIED_OUTPUT_COLUMNS if column in data.columns]
    data[output_cols].to_csv(output_path, index=False, compression="gzip")

    summary = {
        "country": country,
        "total_articles": len(data),
        "predicted_political_corruption": int(data["pred_political_corruption"].sum()),
        "predicted_rate": float(data["pred_political_corruption"].mean()),
        "output_path": str(output_path),
    }
    print(
        f"{country}: {summary['predicted_political_corruption']:,} / "
        f"{summary['total_articles']:,} predicted positive "
        f"({summary['predicted_rate']:.2%})",
        flush=True,
    )
    return summary


def score_corpus(args, embedder, clf, threshold: float) -> None:
    import pandas as pd

    classified_dir = args.output_dir / "classified_country_files"
    summaries = []
    for country in args.countries:
        summary = score_country_file(
            country=country,
            path=cleaned_country_path(args.pipeline_dir, country),
            output_dir=classified_dir,
            embedder=embedder,
            clf=clf,
            threshold=threshold,
            batch_size=args.batch_size,
        )
        if summary:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No country files were scored.")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_dir / "classified_country_summary.csv", index=False)

    # Build country-year counts from the saved classified files.
    year_counts = []
    for country in args.countries:
        path = classified_dir / f"{country}_classified.csv.gz"
        if not path.exists():
            continue
        data = pd.read_csv(path, usecols=lambda col: col in {"country", "year", "pred_political_corruption"})
        counts = (
            data.groupby(["country", "year"], dropna=False)
            .agg(
                total_articles=("pred_political_corruption", "size"),
                predicted_political_corruption=("pred_political_corruption", "sum"),
            )
            .reset_index()
        )
        counts["predicted_rate"] = counts["predicted_political_corruption"] / counts["total_articles"]
        year_counts.append(counts)

    if year_counts:
        pd.concat(year_counts, ignore_index=True).to_csv(
            args.output_dir / "classified_country_year_counts.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()

    import joblib
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix

    args.output_dir.mkdir(parents=True, exist_ok=True)

    silver_df = load_silver_labels(args.silver_labels)
    print(f"Silver training rows: {len(silver_df):,}", flush=True)
    print(silver_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    valid_df = load_human_validation()
    print(f"\nHuman validation rows: {len(valid_df):,}", flush=True)
    print(valid_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    embedder = SentenceTransformer(args.embedding_model)
    x_train = embedder.encode(
        silver_df["model_text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    x_valid = embedder.encode(
        valid_df["model_text"].tolist(),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)
    clf.fit(x_train, silver_df["y"])

    valid_probs = clf.predict_proba(x_valid)[:, 1]
    threshold_results = evaluate_thresholds(valid_df["y"].to_numpy(), valid_probs)

    threshold = args.threshold
    if args.select_threshold:
        threshold = float(
            threshold_results.sort_values(
                ["political_f1", "political_recall", "political_precision"],
                ascending=False,
            ).iloc[0]["threshold"]
        )

    valid_df["silver_prob_political_corruption"] = valid_probs
    valid_df["silver_pred_political_corruption"] = (
        valid_df["silver_prob_political_corruption"] >= threshold
    ).astype(int)

    print(f"\nUsing threshold: {threshold:.2f}", flush=True)
    print(
        classification_report(
            valid_df["y"],
            valid_df["silver_pred_political_corruption"],
            target_names=["No political corruption", "Political corruption"],
            zero_division=0,
        ),
        flush=True,
    )
    print(
        pd.DataFrame(
            confusion_matrix(valid_df["y"], valid_df["silver_pred_political_corruption"]),
            index=["True No", "True Political"],
            columns=["Pred No", "Pred Political"],
        ),
        flush=True,
    )

    country_results = country_metrics(valid_df, "silver_pred_political_corruption")
    print("\nThreshold results:", flush=True)
    print(threshold_results, flush=True)
    print("\nCountry validation results:", flush=True)
    print(country_results, flush=True)

    save_validation_outputs(valid_df, threshold_results, country_results, args.output_dir)
    joblib.dump(clf, args.output_dir / "silver_logistic_regression.joblib")
    (args.output_dir / "embedding_model_name.txt").write_text(args.embedding_model + "\n")
    (args.output_dir / "selected_threshold.txt").write_text(f"{threshold}\n")

    print(f"\nSaved validation/model outputs to: {args.output_dir}", flush=True)

    if args.score_corpus:
        score_corpus(args, embedder, clf, threshold)
        print(f"\nSaved corpus classifications to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
