"""Create a UK-focused silver-label calibration batch.

This script samples from the already classified United Kingdom corpus, using
the final classifier probabilities. It is intended to audit and improve the
political-corruption classifier where validation performance was weakest.

Example:
    python3 political_classifier/scripts/create_uk_silver_batch.py

Then label with:
    nohup python3 -u political_classifier/scripts/label_silver_batch.py \
      --input /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_calibration_batch_for_annotation.csv \
      --output /home/akroon/data/1t_storage/RESPOND-victims-of-corruption/political_corruption_pipeline/active_learning/uk_calibration_batch_with_llm_suggestions.csv \
      --max-chars 3000 \
      > llm_uk_calibration_batch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_CLASSIFIED_PATH = (
    DEFAULT_PIPELINE_DIR
    / "silver_classifier"
    / "classified_country_files"
    / "United_Kingdom_classified.csv.gz"
)
DEFAULT_SILVER_LABEL_DIR = DEFAULT_PIPELINE_DIR / "active_learning"
DEFAULT_OUTPUT_PATH = DEFAULT_SILVER_LABEL_DIR / "uk_calibration_batch_for_annotation.csv"
DEFAULT_EXCLUDE_PATHS = [
    DEFAULT_SILVER_LABEL_DIR / "active_learning_batch_with_llm_suggestions.csv",
    DEFAULT_SILVER_LABEL_DIR / "active_learning_batch_2_with_llm_suggestions.csv",
]

OUTPUT_COLUMNS = [
    "al_bucket",
    "uri",
    "country",
    "date_parsed",
    "year",
    "source_uri",
    "prob_political_corruption",
    "model_pred",
    "human_final_label",
    "human_notes",
    "article_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a UK-focused silver-label calibration batch.")
    parser.add_argument("--classified-path", type=Path, default=DEFAULT_CLASSIFIED_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--target-n", type=int, default=1500)
    parser.add_argument("--boundary-share", type=float, default=0.40)
    parser.add_argument("--positive-share", type=float, default=0.30)
    parser.add_argument("--negative-share", type=float, default=0.20)
    parser.add_argument("--random-share", type=float, default=0.10)
    parser.add_argument(
        "--boundary-low",
        type=float,
        default=0.30,
        help="Lower probability bound for the threshold-boundary pool.",
    )
    parser.add_argument(
        "--boundary-high",
        type=float,
        default=0.50,
        help="Upper probability bound for the threshold-boundary pool.",
    )
    parser.add_argument(
        "--high-positive",
        type=float,
        default=0.60,
        help="Probability cutoff for high predicted positives.",
    )
    parser.add_argument(
        "--high-negative",
        type=float,
        default=0.20,
        help="Probability cutoff for high predicted negatives.",
    )
    parser.add_argument(
        "--exclude",
        type=Path,
        action="append",
        default=None,
        help="CSV whose uri values should be excluded. Can be passed multiple times.",
    )
    return parser.parse_args()


def load_seen_uris(paths):
    import pandas as pd

    seen = set()
    for path in paths:
        if not path.exists():
            continue
        data = pd.read_csv(path, usecols=lambda column: column == "uri")
        if "uri" in data.columns:
            seen.update(data["uri"].dropna().astype(str))
    return seen


def take_sample(pool, selected_uris: set[str], n: int, random_state: int):
    if n <= 0 or pool.empty:
        return pool.head(0).copy()

    if "uri" in pool.columns:
        pool = pool[~pool["uri"].astype(str).isin(selected_uris)].copy()

    if pool.empty:
        return pool

    n = min(n, len(pool))
    sample = pool.sample(n=n, random_state=random_state).copy()

    if "uri" in sample.columns:
        selected_uris.update(sample["uri"].dropna().astype(str))

    return sample


def main() -> None:
    args = parse_args()

    import pandas as pd

    shares = [args.boundary_share, args.positive_share, args.negative_share, args.random_share]
    if abs(sum(shares) - 1.0) > 1e-6:
        raise ValueError(f"Sampling shares must sum to 1. Got {sum(shares):.3f}")

    exclude_paths = DEFAULT_EXCLUDE_PATHS.copy()
    if args.exclude:
        exclude_paths.extend(args.exclude)

    seen_uris = load_seen_uris(exclude_paths)
    print(f"Excluding previously labelled URIs: {len(seen_uris):,}", flush=True)

    columns = [
        "uri",
        "country",
        "date_parsed",
        "year",
        "source_uri",
        "prob_political_corruption",
        "pred_political_corruption",
        "article_text",
    ]
    data = pd.read_csv(args.classified_path, usecols=lambda column: column in columns)
    data["country"] = "United_Kingdom"
    data = data[data["article_text"].fillna("").astype(str).str.strip().ne("")].copy()

    if seen_uris and "uri" in data.columns:
        before = len(data)
        data = data[~data["uri"].astype(str).isin(seen_uris)].copy()
        print(f"Dropped previously labelled UK rows: {before - len(data):,}", flush=True)

    data["model_pred"] = (data["prob_political_corruption"] >= args.threshold).astype(int)
    data["distance_to_threshold"] = (data["prob_political_corruption"] - args.threshold).abs()

    n_boundary = int(args.target_n * args.boundary_share)
    n_positive = int(args.target_n * args.positive_share)
    n_negative = int(args.target_n * args.negative_share)
    n_random = args.target_n - n_boundary - n_positive - n_negative

    boundary = data[
        data["prob_political_corruption"].between(args.boundary_low, args.boundary_high)
    ].copy()
    boundary = boundary.sort_values("distance_to_threshold")
    boundary["al_bucket"] = "uk_threshold_boundary"

    high_positive = data[data["prob_political_corruption"] >= args.high_positive].copy()
    high_positive = high_positive.sort_values("prob_political_corruption", ascending=False)
    high_positive["al_bucket"] = "uk_high_predicted_positive"

    high_negative = data[data["prob_political_corruption"] <= args.high_negative].copy()
    high_negative = high_negative.sort_values("prob_political_corruption", ascending=True)
    high_negative["al_bucket"] = "uk_high_predicted_negative"

    random_pool = data.copy()
    random_pool["al_bucket"] = "uk_random_check"

    selected_uris: set[str] = set()
    parts = [
        take_sample(boundary, selected_uris, n_boundary, args.random_state),
        take_sample(high_positive, selected_uris, n_positive, args.random_state + 1),
        take_sample(high_negative, selected_uris, n_negative, args.random_state + 2),
        take_sample(random_pool, selected_uris, n_random, args.random_state + 3),
    ]

    batch = pd.concat(parts, ignore_index=True)
    if len(batch) < args.target_n:
        topup = data.sort_values("distance_to_threshold").copy()
        topup["al_bucket"] = "uk_topup_boundary"
        batch = pd.concat(
            [
                batch,
                take_sample(topup, selected_uris, args.target_n - len(batch), args.random_state + 4),
            ],
            ignore_index=True,
        )

    batch["human_final_label"] = ""
    batch["human_notes"] = ""

    output_cols = [column for column in OUTPUT_COLUMNS if column in batch.columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch[output_cols].to_csv(args.output, index=False)

    print(f"Saved UK calibration batch: {args.output}", flush=True)
    print(f"Rows: {len(batch):,}", flush=True)
    print("\nBy bucket:", flush=True)
    print(batch["al_bucket"].value_counts(), flush=True)
    print("\nProbability summary:", flush=True)
    print(batch["prob_political_corruption"].describe(), flush=True)


if __name__ == "__main__":
    main()
