"""Shared text preparation and human-benchmark loading for the classifier."""

from __future__ import annotations

import re

from political_classifier.source_filter import (
    apply_source_inclusion_filter,
    source_filter_summary,
)


HUMAN_LABEL_MAP = {
    "political corruption": 1,
    "no political corruption": 0,
    "mentioned but not central": 0,
}


def fix_mojibake(text):
    if not isinstance(text, str):
        return ""

    candidates = [text]
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

    def badness(candidate):
        markers = ["Ð", "Ñ", "Ã", "Â", "Ä", "Å", "�"]
        return sum(candidate.count(marker) for marker in markers)

    return min(candidates, key=badness)


def normalize_text(text):
    text = fix_mojibake(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def format_texts_for_embedding(texts, embedding_model):
    """Apply the document prefix expected by multilingual E5 models."""
    if "multilingual-e5" in embedding_model.lower():
        return [f"passage: {text}" for text in texts]
    return texts


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


def filter_frame_by_source(data, decisions, label):
    filtered, merged = apply_source_inclusion_filter(
        data,
        decisions,
        country_column="country",
    )
    summary = source_filter_summary(merged, group_columns=["country"])
    print(
        f"\nSource filter for {label}: {len(filtered):,} / {len(data):,} rows retained",
        flush=True,
    )
    print(summary, flush=True)
    return filtered


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
        combined = filter_frame_by_source(
            combined,
            source_decisions,
            "human validation",
        )
    return combined
