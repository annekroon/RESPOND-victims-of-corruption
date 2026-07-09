"""Merge reviewed UK validation rows into a copy of the human annotation file.

The original annotation CSV is kept unchanged. This script writes an augmented
annotation file that contains the original human validation rows plus manually
reviewed UK supplement rows from the Streamlit annotation interface.

Default input:
    annotations/classified_pol_corruption_validation_gabriele.csv on Research Drive

Default reviewed UK input:
    /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/
    political_corruption_pipeline/active_learning/uk_human_validation_reviewed.csv

Default output:
    annotations/classified_pol_corruption_validation_gabriele_plus_uk_review.csv
    on Research Drive
"""

from __future__ import annotations

import argparse
import posixpath
from pathlib import Path

from config import ANNOTATION_ENCODING, ANNOTATION_FILE, RD_BASE_DIR


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_REVIEWED_UK_PATH = (
    DEFAULT_PIPELINE_DIR
    / "active_learning"
    / "uk_human_validation_reviewed.csv"
)
DEFAULT_OUTPUT_FILE = "classified_pol_corruption_validation_gabriele_plus_uk_review.csv"
ANNOTATION_DIR = f"{RD_BASE_DIR}/annotations"

VALID_HUMAN_LABELS = {
    "political corruption": "political corruption",
    "no political corruption": "no political corruption",
    "mentioned but not central": "no political corruption",
}


def rd_join(*parts: str) -> str:
    clean = [p.strip("/ ") for p in parts if p]
    return posixpath.join(*clean)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append reviewed UK validation rows to a copy of the human annotation file."
    )
    parser.add_argument(
        "--reviewed-uk",
        type=Path,
        default=DEFAULT_REVIEWED_UK_PATH,
        help="Reviewed UK CSV from the Streamlit annotation interface.",
    )
    parser.add_argument(
        "--original-rd-path",
        default=rd_join(ANNOTATION_DIR, ANNOTATION_FILE),
        help="Research Drive path to the original annotation CSV.",
    )
    parser.add_argument(
        "--output-rd-path",
        default=rd_join(ANNOTATION_DIR, DEFAULT_OUTPUT_FILE),
        help="Research Drive path for the merged output CSV.",
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        default=None,
        help="Optional local copy of the merged output CSV.",
    )
    parser.add_argument(
        "--keep-unsure",
        action="store_true",
        help="Keep 'unsure / revisit' rows with corruption_label_m='unsure / revisit'.",
    )
    return parser.parse_args()


def normalize_label(value, keep_unsure=False):
    label = str(value).strip().lower()
    if label in VALID_HUMAN_LABELS:
        return VALID_HUMAN_LABELS[label]
    if keep_unsure and label == "unsure / revisit":
        return "unsure / revisit"
    return None


def make_title_and_body(text: str) -> tuple[str, str]:
    text = "" if text is None else str(text).strip()
    if not text:
        return "", ""
    parts = text.splitlines()
    title = parts[0].strip()[:300] if parts else text[:300]
    body = "\n".join(part.strip() for part in parts[1:] if part.strip()).strip()
    if not body:
        body = text
    return title, body


def prepare_uk_review_rows(reviewed, original_columns, reviewed_path, keep_unsure=False):
    import pandas as pd

    reviewed = reviewed.copy()
    if "human_final_label" not in reviewed.columns:
        raise ValueError("Reviewed UK file must contain a human_final_label column.")

    reviewed["corruption_label_m"] = reviewed["human_final_label"].map(
        lambda value: normalize_label(value, keep_unsure=keep_unsure)
    )
    reviewed = reviewed[reviewed["corruption_label_m"].notna()].copy()

    if reviewed.empty:
        raise ValueError("No reviewed UK rows with valid human_final_label values were found.")

    if "country" not in reviewed.columns:
        reviewed["country"] = "United_Kingdom"
    reviewed["country"] = "United_Kingdom"

    text_source = None
    for candidate in ["article_text", "combined_text", "translated_text"]:
        if candidate in reviewed.columns:
            text_source = reviewed[candidate].fillna("").astype(str)
            break
    if text_source is None:
        raise ValueError("Reviewed UK file must contain article_text, combined_text, or translated_text.")

    reviewed["combined_text"] = text_source
    title_body = reviewed["combined_text"].map(make_title_and_body)
    reviewed["title"] = title_body.map(lambda item: item[0])
    reviewed["body"] = title_body.map(lambda item: item[1])

    reviewed["annotation_source"] = "uk_human_validation_review"
    reviewed["review_source_file"] = str(reviewed_path)
    reviewed["original_llm_label_suggestion"] = reviewed.get("llm_label_suggestion", "")
    reviewed["original_llm_confidence"] = reviewed.get("llm_confidence", "")

    for column in original_columns:
        if column not in reviewed.columns:
            reviewed[column] = pd.NA

    extra_columns = [
        "annotation_source",
        "review_source_file",
        "validation_bucket",
        "al_bucket",
        "prob_political_corruption",
        "model_pred",
        "translated_text",
        "human_final_label",
        "human_notes",
        "original_llm_label_suggestion",
        "original_llm_confidence",
        "llm_rationale",
        "llm_evidence",
    ]
    output_columns = list(original_columns)
    for column in extra_columns:
        if column in reviewed.columns and column not in output_columns:
            output_columns.append(column)

    return reviewed[output_columns].copy(), output_columns


def main() -> None:
    args = parse_args()

    import pandas as pd
    from rd_io import rd_read_csv_df, rd_write_csv_df

    original = rd_read_csv_df(
        args.original_rd_path,
        encoding=ANNOTATION_ENCODING,
        encoding_errors="replace",
    )
    original["annotation_source"] = original.get("annotation_source", "original_human_validation")

    reviewed = pd.read_csv(args.reviewed_uk)
    uk_rows, output_columns = prepare_uk_review_rows(
        reviewed,
        original_columns=original.columns.tolist(),
        reviewed_path=args.reviewed_uk,
        keep_unsure=args.keep_unsure,
    )

    for column in output_columns:
        if column not in original.columns:
            original[column] = pd.NA
    original = original[output_columns].copy()

    merged = pd.concat([original, uk_rows], ignore_index=True)
    if "uri" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["uri"], keep="first").copy()
        dropped = before - len(merged)
    else:
        dropped = 0

    rd_write_csv_df(merged, args.output_rd_path, index=False)
    if args.local_output:
        args.local_output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.local_output, index=False)

    print(f"Original rows: {len(original):,}", flush=True)
    print(f"Reviewed UK rows appended: {len(uk_rows):,}", flush=True)
    print(f"Duplicate URIs dropped: {dropped:,}", flush=True)
    print(f"Merged rows: {len(merged):,}", flush=True)
    print(f"Saved merged annotation file to Research Drive: {args.output_rd_path}", flush=True)
    if args.local_output:
        print(f"Saved local copy: {args.local_output}", flush=True)
    print("\nLabel counts:", flush=True)
    print(merged["corruption_label_m"].value_counts(dropna=False), flush=True)
    if "country" in merged.columns:
        print("\nCountry counts:", flush=True)
        print(merged["country"].value_counts(dropna=False), flush=True)


if __name__ == "__main__":
    main()
