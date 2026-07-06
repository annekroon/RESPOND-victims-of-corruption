"""Create a UK human-validation review batch from LLM-labelled UK cases.

This script samples from the UK calibration file that already contains English
translations and LLM label suggestions. The output is meant for manual review
in the Streamlit annotation interface. Reviewed rows can then be appended to
the human validation set with:

    python3 04_compare_classifier_models.py \
      --extra-human-validation /path/to/uk_human_validation_reviewed.csv

Important: do not also use these reviewed rows as silver training labels in the
same evaluation. They are intended as an external UK validation supplement.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_ACTIVE_LEARNING_DIR = DEFAULT_PIPELINE_DIR / "active_learning"
DEFAULT_INPUT_PATH = DEFAULT_ACTIVE_LEARNING_DIR / "uk_calibration_batch_with_llm_suggestions.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_ACTIVE_LEARNING_DIR / "uk_human_validation_review_batch.csv"

VALID_LLM_LABELS = ["Yes", "Mentioned but not central", "No", "Unsure"]

OUTPUT_COLUMNS = [
    "validation_bucket",
    "al_bucket",
    "uri",
    "country",
    "date_parsed",
    "year",
    "source_uri",
    "prob_political_corruption",
    "model_pred",
    "llm_label_suggestion",
    "llm_confidence",
    "llm_rationale",
    "llm_evidence",
    "human_final_label",
    "human_notes",
    "translated_text",
    "article_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manually reviewable UK validation supplement from LLM-labelled UK cases."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-n", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Write all valid UK LLM-labelled rows instead of sampling target-n rows.",
    )
    return parser.parse_args()


def sample_group(data, n, random_state):
    if n <= 0 or data.empty:
        return data.head(0).copy()
    return data.sample(n=min(n, len(data)), random_state=random_state).copy()


def balanced_review_sample(data, target_n, random_state):
    """Oversample likely positives/ambiguous cases for a useful validation check."""
    import pandas as pd

    label_targets = {
        "Yes": round(target_n * 0.40),
        "Mentioned but not central": round(target_n * 0.25),
        "No": round(target_n * 0.30),
    }
    label_targets["Unsure"] = max(0, target_n - sum(label_targets.values()))

    selected = []
    selected_uris: set[str] = set()
    for offset, (label, n) in enumerate(label_targets.items()):
        pool = data[data["llm_label_suggestion"].eq(label)].copy()
        if selected_uris and "uri" in pool.columns:
            pool = pool[~pool["uri"].astype(str).isin(selected_uris)].copy()
        sample = sample_group(pool, n, random_state + offset)
        selected.append(sample)
        if "uri" in sample.columns:
            selected_uris.update(sample["uri"].dropna().astype(str))

    result = pd.concat(selected, ignore_index=True)
    if len(result) < target_n:
        topup = data.copy()
        if selected_uris and "uri" in topup.columns:
            topup = topup[~topup["uri"].astype(str).isin(selected_uris)].copy()
        result = pd.concat(
            [result, sample_group(topup, target_n - len(result), random_state + 99)],
            ignore_index=True,
        )

    return result.sample(frac=1, random_state=random_state + 100).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    import pandas as pd

    data = pd.read_csv(args.input)
    data = data[data["country"].astype(str).eq("United_Kingdom")].copy()
    data = data[data["llm_label_suggestion"].isin(VALID_LLM_LABELS)].copy()
    data = data[data["article_text"].fillna("").astype(str).str.strip().ne("")].copy()

    if data.empty:
        raise ValueError(f"No valid UK LLM-labelled rows found in {args.input}")

    if args.include_all:
        review = data.copy()
    else:
        review = balanced_review_sample(data, args.target_n, args.random_state)

    review["validation_bucket"] = "uk_human_validation_review"
    review["human_final_label"] = ""
    review["human_notes"] = ""

    output_columns = [column for column in OUTPUT_COLUMNS if column in review.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    review[output_columns].to_csv(args.output, index=False)

    print(f"Saved UK human-validation review batch: {args.output}", flush=True)
    print(f"Rows: {len(review):,}", flush=True)
    print("\nLLM suggestions:", flush=True)
    print(review["llm_label_suggestion"].value_counts(), flush=True)
    if "al_bucket" in review.columns:
        print("\nOriginal UK calibration buckets:", flush=True)
        print(review["al_bucket"].value_counts(), flush=True)
    print("\nAfter review, use this file as the annotation interface input.", flush=True)


if __name__ == "__main__":
    main()
