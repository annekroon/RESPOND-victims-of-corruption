"""Train and apply a political-corruption classifier from LLM silver labels.

This script reads one or more LLM-labelled silver-label CSVs, trains an
embedding classifier on those silver labels, validates against the human-labelled
set, and can optionally classify the cleaned country files.

Examples:
    python3 political_classifier/scripts/06_train_final_classifier.py

    nohup python3 -u political_classifier/scripts/06_train_final_classifier.py --score-corpus \
      > silver_classifier_final_scoring.log 2>&1 &
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
    apply_source_inclusion_filter,
    load_source_decisions,
    source_filter_summary,
)
from political_classifier.split_integrity import assert_no_validation_overlap
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
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier"
DEFAULT_UK_REVIEWED_VALIDATION_PATH = DEFAULT_SILVER_LABEL_DIR / "uk_human_validation_reviewed.csv"
DEFAULT_SILVER_LABEL_PATHS = [
    DEFAULT_SILVER_LABEL_DIR / "silver_training_source_filtered_with_llm_suggestions.csv",
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
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_THRESHOLD_FILE = DEFAULT_PIPELINE_DIR / "classifier_comparison" / "selected_threshold.txt"
DEFAULT_BENCHMARK_SPLIT_FILE = (
    DEFAULT_PIPELINE_DIR / "classifier_comparison" / "human_benchmark_split.csv"
)
DEFAULT_COMPARISON_MANIFEST = (
    DEFAULT_PIPELINE_DIR
    / "classifier_comparison"
    / "classifier_comparison_run_manifest.json"
)
DEFAULT_CORPUS_DIR = DEFAULT_PIPELINE_DIR / "cleaned_deduped_source_filtered"
DEFAULT_BATCH_SIZE = 32

SILVER_LABEL_MAP = {
    "Yes": 1,
    "No": 0,
    "Mentioned but not central": 0,
}
HUMAN_LABEL_MAP = {
    "political corruption": 1,
    "no political corruption": 0,
    "mentioned but not central": 0,
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


def format_texts_for_embedding(texts, embedding_model):
    """Apply model-family-specific text formatting when needed."""
    if "multilingual-e5" in embedding_model.lower():
        return [f"passage: {text}" for text in texts]
    return texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a silver-label classifier and optionally classify the cleaned corpus."
    )
    parser.add_argument(
        "--silver-labels",
        type=Path,
        nargs="+",
        default=DEFAULT_SILVER_LABEL_PATHS,
        help="One or more LLM-labelled silver-label CSV files.",
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional explicit assertion of the final threshold. It must equal "
            "the calibration-selected value from 05_compare_models.py."
        ),
    )
    parser.add_argument("--threshold-file", type=Path, default=DEFAULT_THRESHOLD_FILE)
    parser.add_argument(
        "--benchmark-split-file",
        type=Path,
        default=DEFAULT_BENCHMARK_SPLIT_FILE,
        help="Calibration/test assignment written by 05_compare_models.py.",
    )
    parser.add_argument(
        "--comparison-manifest",
        type=Path,
        default=DEFAULT_COMPARISON_MANIFEST,
        help="Run manifest written by 05_compare_models.py.",
    )
    parser.add_argument(
        "--select-threshold",
        action="store_true",
        help=argparse.SUPPRESS,
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
    parser.add_argument("--score-corpus", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--score-chunksize", type=int, default=25_000)
    parser.add_argument(
        "--overwrite-classifications",
        action="store_true",
        help="Replace existing country classification files when scoring.",
    )
    parser.add_argument("--countries", nargs="+", default=DEFAULT_COUNTRIES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--source-decision-file",
        type=Path,
        default=DEFAULT_SOURCE_DECISION_FILE,
        help="Source decision workbook/CSV. Only conventional_journalism == No is excluded.",
    )
    parser.add_argument(
        "--no-source-filter",
        action="store_true",
        help=(
            "Diagnostic override: do not remove explicitly excluded sources from "
            "silver training data or validation. Corpus scoring still reads the "
            "explicit --corpus-dir, whose default is the source-filtered corpus."
        ),
    )
    parser.add_argument(
        "--keep-unfiltered-validation",
        action="store_true",
        help=(
            "Diagnostic override: leave the human benchmark unfiltered. By default "
            "validation uses the same source eligibility rule as production data."
        ),
    )
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


def choose_integrity_text_series(data):
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]
    return choose_text_series(data)


def filter_frame_by_source(data, decisions, label):
    filtered, merged = apply_source_inclusion_filter(data, decisions, country_column="country")
    summary = source_filter_summary(merged, group_columns=["country"])
    print(f"\nSource filter for {label}: {len(filtered):,} / {len(data):,} rows retained", flush=True)
    print(summary, flush=True)
    return filtered


def load_silver_labels(paths, source_decisions=None):
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
    dedupe_key = "silver_row_id" if "silver_row_id" in silver.columns else "uri"
    if dedupe_key in silver.columns:
        before = len(silver)
        silver = silver.drop_duplicates(subset=[dedupe_key], keep="last").copy()
        print(f"Dropped duplicate silver-label rows: {before - len(silver):,}", flush=True)

    if "llm_error" in silver.columns:
        silver = silver[silver["llm_error"].fillna("").astype(str).str.strip().eq("")].copy()

    silver = silver[silver["llm_label_suggestion"].isin(SILVER_LABEL_MAP)].copy()
    silver["y"] = silver["llm_label_suggestion"].map(SILVER_LABEL_MAP).astype(int)
    silver["model_text"] = choose_text_series(silver).fillna("").astype(str).map(normalize_text)
    silver["integrity_text"] = (
        choose_integrity_text_series(silver).fillna("").astype(str).map(normalize_text)
    )
    silver = silver[silver["model_text"].str.strip().ne("")].copy()
    silver["_training_text_hash"] = silver["model_text"].map(
        lambda value: hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()
    )
    before = len(silver)
    silver = silver.drop_duplicates(subset=["_training_text_hash"], keep="last").copy()
    print(f"Dropped duplicate silver-label texts: {before - len(silver):,}", flush=True)
    if source_decisions is not None:
        silver = filter_frame_by_source(silver, source_decisions, "silver training")
    return silver


def prepare_human_validation_frame(data, source_name):
    label_column = next(
        (
            column
            for column in ["corruption_label_m", "human_final_label", "label"]
            if column in data.columns
        ),
        None,
    )
    if label_column is None:
        raise ValueError(
            f"No human validation label column found in {source_name}. "
            "Expected one of: corruption_label_m, human_final_label, label."
        )

    data = data.copy()
    data["label_clean"] = data[label_column].astype(str).str.strip().str.lower()
    data["y"] = data["label_clean"].map(HUMAN_LABEL_MAP)
    data = data[data["y"].notna()].copy()
    data["y"] = data["y"].astype(int)
    data["model_text"] = choose_text_series(data).fillna("").astype(str).map(normalize_text)
    data["integrity_text"] = (
        choose_integrity_text_series(data).fillna("").astype(str).map(normalize_text)
    )
    data = data[data["model_text"].str.strip().ne("")].copy()
    data["human_validation_source"] = source_name
    return data


def load_human_validation(extra_paths=None, source_decisions=None):
    import pandas as pd
    from dataloader import load_human_annotated_for_translation_webdav

    data = load_human_annotated_for_translation_webdav()
    frames = [prepare_human_validation_frame(data, "original_human_validation")]

    for path in extra_paths or []:
        if not path.exists():
            raise FileNotFoundError(
                f"Required extra human validation file not found: {path}"
            )
        extra = pd.read_csv(path)
        frames.append(prepare_human_validation_frame(extra, path.name))

    combined = pd.concat(frames, ignore_index=True)
    if "uri" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["uri"], keep="first").copy()
        dropped = before - len(combined)
        if dropped:
            print(f"Dropped duplicate human-validation URIs: {dropped:,}", flush=True)
    if source_decisions is not None:
        combined = filter_frame_by_source(combined, source_decisions, "human validation")
    return combined


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


def cleaned_country_path(corpus_dir: Path, country: str) -> Path:
    return corpus_dir / f"{country}_cleaned_deduped_source_filtered.csv.gz"


def score_texts(texts, embedder, clf, embedding_model: str, batch_size: int):
    import numpy as np
    from tqdm.auto import tqdm

    all_probs = []
    starts = range(0, len(texts), batch_size * 8)
    for start in tqdm(list(starts), desc="Embedding + scoring"):
        end = min(start + batch_size * 8, len(texts))
        batch_texts = format_texts_for_embedding(texts[start:end], embedding_model)
        embeddings = embedder.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_probs.append(clf.predict_proba(embeddings)[:, 1])
    return np.concatenate(all_probs) if all_probs else np.array([])


def score_country_file(
    country: str,
    path: Path,
    output_dir: Path,
    embedder,
    clf,
    embedding_model: str,
    threshold: float,
    batch_size: int,
    chunksize: int,
    overwrite: bool,
):
    import pandas as pd

    if not path.exists():
        print(f"Skipping missing cleaned file for {country}: {path}", flush=True)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{country}_classified.csv.gz"
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Classification output already exists: {output_path}. "
            "Pass --overwrite-classifications to replace it."
        )
    temporary = output_path.with_name(output_path.name + ".part")
    if temporary.exists():
        temporary.unlink()

    print(f"\nScoring {country}: {path}", flush=True)
    total_articles = predicted_positive = 0
    first_chunk = True
    for chunk_number, data in enumerate(
        pd.read_csv(path, chunksize=chunksize, low_memory=False), start=1
    ):
        data["model_text"] = data["article_text"].fillna("").astype(str).map(normalize_text)
        probs = score_texts(
            data["model_text"].tolist(), embedder, clf, embedding_model, batch_size
        )
        data["prob_political_corruption"] = probs
        data["pred_political_corruption"] = (
            data["prob_political_corruption"] >= threshold
        ).astype(int)
        output_cols = [column for column in CLASSIFIED_OUTPUT_COLUMNS if column in data.columns]
        data[output_cols].to_csv(
            temporary,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            compression="gzip",
        )
        first_chunk = False
        total_articles += len(data)
        predicted_positive += int(data["pred_political_corruption"].sum())
        print(
            f"{country} chunk {chunk_number}: {total_articles:,} rows scored",
            flush=True,
        )

    if first_chunk:
        raise RuntimeError(f"No rows found in {path}.")
    temporary.replace(output_path)

    summary = {
        "country": country,
        "total_articles": total_articles,
        "predicted_political_corruption": predicted_positive,
        "predicted_rate": predicted_positive / total_articles,
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
    input_paths = {
        country: cleaned_country_path(args.corpus_dir, country)
        for country in args.countries
    }
    missing_inputs = [
        str(path) for path in input_paths.values() if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            "Cannot start a complete corpus-scoring run; missing country files: "
            + ", ".join(missing_inputs)
        )
    existing_outputs = [
        classified_dir / f"{country}_classified.csv.gz"
        for country in args.countries
        if (classified_dir / f"{country}_classified.csv.gz").exists()
    ]
    if existing_outputs and not args.overwrite_classifications:
        raise FileExistsError(
            f"Found {len(existing_outputs)} existing classification file(s). "
            "Pass --overwrite-classifications for a deliberate complete rebuild."
        )
    (args.output_dir / "classified_country_summary.csv").unlink(missing_ok=True)
    (args.output_dir / "classified_country_year_counts.csv").unlink(missing_ok=True)

    summaries = []
    for country in args.countries:
        summary = score_country_file(
            country=country,
            path=input_paths[country],
            output_dir=classified_dir,
            embedder=embedder,
            clf=clf,
            embedding_model=args.embedding_model,
            threshold=threshold,
            batch_size=args.batch_size,
            chunksize=args.score_chunksize,
            overwrite=args.overwrite_classifications,
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
    (args.output_dir / "classifier_run_manifest.json").unlink(missing_ok=True)
    if args.select_threshold:
        raise ValueError(
            "--select-threshold is no longer allowed here. Run 05_compare_models.py; "
            "this script reads its calibration-selected threshold."
        )
    if args.threshold is not None:
        threshold = float(args.threshold)
    else:
        if not args.threshold_file.exists():
            raise FileNotFoundError(
                f"Selected threshold file not found: {args.threshold_file}. "
                "Run 05_compare_models.py first."
            )
        threshold = float(args.threshold_file.read_text(encoding="utf-8").strip())

    if not args.comparison_manifest.exists():
        raise FileNotFoundError(
            f"Classifier comparison manifest not found: {args.comparison_manifest}. "
            "Run 05_compare_models.py first."
        )
    comparison_manifest = json.loads(
        args.comparison_manifest.read_text(encoding="utf-8")
    )
    manifest_threshold = float(comparison_manifest["selected_threshold"])
    if abs(threshold - manifest_threshold) > 1e-12:
        raise ValueError(
            "Final threshold and comparison manifest disagree: "
            f"{threshold} versus {manifest_threshold}. The final model must use "
            "the calibration-selected threshold from 05_compare_models.py."
        )
    expected_model = comparison_manifest.get("primary_embedding_model")
    if expected_model != args.embedding_model:
        raise ValueError(
            "Final embedding model does not match the pre-specified comparison "
            f"model: {args.embedding_model!r} versus {expected_model!r}. Rerun "
            "05_compare_models.py with the intended model first."
        )
    expected_silver = sorted(
        record["sha256"]
        for record in comparison_manifest.get("silver_label_files", [])
    )
    current_silver = sorted(file_record(path)["sha256"] for path in args.silver_labels)
    if current_silver != expected_silver:
        raise ValueError(
            "Silver-label inputs differ from the comparison run. Rerun "
            "05_compare_models.py before fitting the final classifier."
        )
    expected_completion = sorted(
        record["sha256"]
        for record in comparison_manifest.get(
            "silver_label_completion_markers", []
        )
    )
    current_completion = sorted(
        file_record(path.with_name(path.name + ".complete.json"))["sha256"]
        for path in args.silver_labels
    )
    if current_completion != expected_completion:
        raise ValueError(
            "Silver-label completion markers differ from the comparison run. "
            "Rerun 05_compare_models.py before fitting the final classifier."
        )
    expected_source = comparison_manifest.get("source_decision_file")
    current_source = (
        None if args.no_source_filter else file_record(args.source_decision_file)
    )
    if (expected_source or {}).get("sha256") != (current_source or {}).get("sha256"):
        raise ValueError(
            "The source-decision workbook differs from the comparison run. "
            "Rebuild steps 02-05 before fitting the final classifier."
        )
    current_validation_source_filter = (
        not args.no_source_filter and not args.keep_unfiltered_validation
    )
    if (
        comparison_manifest.get("source_filter_applied_to_validation")
        != current_validation_source_filter
    ):
        raise ValueError(
            "Validation source-filter setting differs from the comparison run. "
            "Use the same --keep-unfiltered-validation setting as 05_compare_models.py."
        )
    if not args.benchmark_split_file.exists():
        raise FileNotFoundError(
            f"Human benchmark split not found: {args.benchmark_split_file}. "
            "Run 05_compare_models.py first."
        )
    expected_split = comparison_manifest.get("benchmark_split_file")
    current_split = file_record(args.benchmark_split_file)
    if (expected_split or {}).get("sha256") != current_split.get("sha256"):
        raise ValueError(
            "The calibration/held-out benchmark split differs from the comparison "
            "run. Rerun 05_compare_models.py before fitting the final classifier."
        )

    source_decisions = None
    if not args.no_source_filter:
        source_decisions = load_source_decisions(args.source_decision_file)
        print(f"Loaded source decisions: {args.source_decision_file}", flush=True)

    silver_df = load_silver_labels(args.silver_labels, source_decisions=source_decisions)
    print(f"Silver training rows: {len(silver_df):,}", flush=True)
    print(silver_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    validation_source_decisions = (
        None if args.keep_unfiltered_validation else source_decisions
    )
    valid_df = load_human_validation(
        args.extra_human_validation,
        source_decisions=validation_source_decisions,
    )
    valid_df = valid_df.reset_index(drop=True)
    current_benchmark_fingerprint = frame_fingerprint(
        valid_df,
        ["country", "uri", "integrity_text", "model_text", "y"],
    )
    if current_benchmark_fingerprint != comparison_manifest.get("human_benchmark_sha256"):
        raise ValueError(
            "The human benchmark contents differ from the comparison run. Rerun "
            "05_compare_models.py before fitting the final classifier."
        )
    split = pd.read_csv(args.benchmark_split_file)
    if len(split) != len(valid_df):
        raise ValueError(
            "Human benchmark split is stale: "
            f"{len(split):,} split rows versus {len(valid_df):,} loaded rows."
        )
    for column in ["country", "y"]:
        if column in split.columns and not split[column].astype(str).reset_index(drop=True).equals(
            valid_df[column].astype(str).reset_index(drop=True)
        ):
            raise ValueError(
                f"Human benchmark split is stale: column {column!r} no longer matches."
            )
    valid_df["evaluation_split"] = split["evaluation_split"].to_numpy()
    if source_decisions is not None and validation_source_decisions is None:
        print(
            "Source filter is applied to silver training/corpus scoring only; "
            "human validation is left unfiltered.",
            flush=True,
        )
    print(f"\nHuman validation rows: {len(valid_df):,}", flush=True)
    print(valid_df["y"].value_counts().rename(index={0: "No", 1: "Political corruption"}), flush=True)

    overlap_audit = args.output_dir / "silver_human_benchmark_overlap.csv"
    assert_no_validation_overlap(
        silver_df,
        valid_df,
        text_column="integrity_text",
        audit_path=overlap_audit,
    )

    embedder = SentenceTransformer(args.embedding_model)
    loaded_embedding_revision = embedding_revision(embedder)
    expected_revision = comparison_manifest.get("embedding_model_revisions", {}).get(
        args.embedding_model
    )
    if expected_revision and loaded_embedding_revision != expected_revision:
        raise ValueError(
            "The loaded embedding-model revision differs from the comparison run: "
            f"{loaded_embedding_revision!r} versus {expected_revision!r}."
        )
    x_train = embedder.encode(
        format_texts_for_embedding(silver_df["model_text"].tolist(), args.embedding_model),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    x_valid = embedder.encode(
        format_texts_for_embedding(valid_df["model_text"].tolist(), args.embedding_model),
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.random_state)
    clf.fit(x_train, silver_df["y"])

    valid_probs = clf.predict_proba(x_valid)[:, 1]
    calibration_mask = valid_df["evaluation_split"].eq("threshold_calibration")
    test_mask = valid_df["evaluation_split"].eq("held_out_test")
    if not calibration_mask.any() or not test_mask.any():
        raise ValueError("Benchmark split must contain threshold_calibration and held_out_test rows.")
    threshold_results = evaluate_thresholds(
        valid_df.loc[calibration_mask, "y"].to_numpy(),
        valid_probs[calibration_mask.to_numpy()],
    )
    threshold_results["evaluation_split"] = "threshold_calibration"

    valid_df["silver_prob_political_corruption"] = valid_probs
    valid_df["silver_pred_political_corruption"] = (
        valid_df["silver_prob_political_corruption"] >= threshold
    ).astype(int)

    print(f"\nUsing threshold: {threshold:.2f}", flush=True)
    report_df = valid_df.loc[test_mask].copy()
    print("Held-out test performance:", flush=True)
    print(
        classification_report(
            report_df["y"],
            report_df["silver_pred_political_corruption"],
            target_names=["No political corruption", "Political corruption"],
            zero_division=0,
        ),
        flush=True,
    )
    print(
        pd.DataFrame(
            confusion_matrix(report_df["y"], report_df["silver_pred_political_corruption"]),
            index=["True No", "True Political"],
            columns=["Pred No", "Pred Political"],
        ),
        flush=True,
    )

    country_results = country_metrics(report_df, "silver_pred_political_corruption")
    print("\nThreshold-calibration results:", flush=True)
    print(threshold_results, flush=True)
    print("\nCountry validation results:", flush=True)
    print(country_results, flush=True)

    save_validation_outputs(valid_df, threshold_results, country_results, args.output_dir)
    joblib.dump(clf, args.output_dir / "silver_logistic_regression.joblib")
    (args.output_dir / "embedding_model_name.txt").write_text(args.embedding_model + "\n")
    (args.output_dir / "selected_threshold.txt").write_text(f"{threshold}\n")
    model_path = args.output_dir / "silver_logistic_regression.joblib"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "embedding_model": args.embedding_model,
        "embedding_model_revision": loaded_embedding_revision,
        "threshold": threshold,
        "threshold_source": str(args.threshold_file) if args.threshold is None else "command_line",
        "random_state": args.random_state,
        "silver_training_rows": len(silver_df),
        "human_validation_rows": len(valid_df),
        "human_benchmark_sha256": frame_fingerprint(
            valid_df,
            [
                "country",
                "uri",
                "integrity_text",
                "model_text",
                "y",
                "evaluation_split",
            ],
        ),
        "silver_label_files": [file_record(path) for path in args.silver_labels],
        "source_decision_file": (
            None if args.no_source_filter else file_record(args.source_decision_file)
        ),
        "comparison_manifest": file_record(args.comparison_manifest),
        "benchmark_split_file": file_record(args.benchmark_split_file),
        "corpus_dir": str(args.corpus_dir),
        "model_artifact": file_record(model_path),
        "package_versions": package_versions(),
    }

    if args.score_corpus:
        score_corpus(args, embedder, clf, threshold)
        manifest["classified_country_summary"] = file_record(
            args.output_dir / "classified_country_summary.csv"
        )
        print(f"\nSaved corpus classifications to: {args.output_dir}", flush=True)

    (args.output_dir / "classifier_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nSaved validation/model outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
