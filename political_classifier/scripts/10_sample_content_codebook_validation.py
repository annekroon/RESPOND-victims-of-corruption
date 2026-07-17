"""Sample classified political-corruption articles for content-codebook validation.

This is the first bridge from Part 1 (political-corruption corpus construction)
to the later article-level content coding work. It samples from the classified
country files produced by ``scripts/06_train_final_classifier.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_CLASSIFIED_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files"
DEFAULT_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "content_codebook_validation"

ANNOTATION_COLUMNS = [
    "human_victim_visibility",
    "human_corruption_domain",
    "human_case_scope",
    "human_accused_actor_type",
    "human_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible sample of predicted political-corruption "
            "articles for content-codebook validation."
        )
    )
    parser.add_argument("--country", default="France")
    parser.add_argument("--n", type=int, default=108)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-all-predicted",
        action="store_true",
        help=(
            "Sample from all classified rows instead of only rows with "
            "pred_political_corruption == 1. Mostly useful for diagnostics."
        ),
    )
    return parser.parse_args()


def probability_band(probability: float) -> str:
    if probability < 0.60:
        return "threshold_near_0.50_0.60"
    if probability < 0.75:
        return "medium_0.60_0.75"
    return "high_0.75_plus"


def add_strata(data):
    data = data.copy()
    if "year" not in data.columns and "date_parsed" in data.columns:
        data["year"] = data["date_parsed"].astype(str).str[:4]
    data["year"] = data.get("year", "unknown").fillna("unknown").astype(str)
    data["probability_band"] = data["prob_political_corruption"].map(probability_band)
    data["sample_stratum"] = data["year"] + "__" + data["probability_band"]
    return data


def stratified_sample(data, n, seed):
    """Draw a broad sample across years and confidence bands."""
    if len(data) <= n:
        return data.sample(frac=1, random_state=seed).copy()

    grouped = data.groupby("sample_stratum", dropna=False)
    group_sizes = grouped.size()
    allocation = (group_sizes / group_sizes.sum() * n).round().astype(int)
    allocation = allocation.clip(lower=1)

    while allocation.sum() > n:
        idx = allocation[allocation > 1].idxmax()
        allocation.loc[idx] -= 1
    while allocation.sum() < n:
        idx = (group_sizes - allocation).idxmax()
        allocation.loc[idx] += 1

    pieces = []
    for stratum, take_n in allocation.items():
        group = grouped.get_group(stratum)
        take_n = min(int(take_n), len(group))
        pieces.append(group.sample(n=take_n, random_state=seed))

    sample = (
        __import__("pandas")
        .concat(pieces, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(sample) < n:
        remaining = data.loc[~data["uri"].astype(str).isin(sample["uri"].astype(str))]
        extra = remaining.sample(n=n - len(sample), random_state=seed)
        sample = __import__("pandas").concat([sample, extra], ignore_index=True)

    return sample.reset_index(drop=True)


def main() -> None:
    import pandas as pd

    args = parse_args()
    input_path = args.classified_dir / f"{args.country}_classified.csv.gz"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing classified country file: {input_path}\n"
            "Run 06_train_final_classifier.py for this country first."
        )

    data = pd.read_csv(input_path)
    print(f"Loaded {len(data):,} classified rows from {input_path}", flush=True)

    if not args.include_all_predicted:
        if "pred_political_corruption" not in data.columns:
            raise ValueError("Expected column pred_political_corruption in classified file.")
        data = data[data["pred_political_corruption"].astype(int).eq(1)].copy()
        print(f"Retained {len(data):,} predicted political-corruption rows", flush=True)

    if len(data) == 0:
        raise ValueError("No rows available for sampling.")

    data = add_strata(data)
    sample = stratified_sample(data, args.n, args.seed)
    sample.insert(0, "content_sample_id", [f"{args.country}_{i:04d}" for i in range(1, len(sample) + 1)])

    for column in ANNOTATION_COLUMNS:
        if column not in sample.columns:
            sample[column] = ""

    output_columns = [
        "content_sample_id",
        "uri",
        "country",
        "date_parsed",
        "year",
        "month",
        "week",
        "source_uri",
        "prob_political_corruption",
        "pred_political_corruption",
        "probability_band",
        "sample_stratum",
        *ANNOTATION_COLUMNS,
        "article_text",
    ]
    output_columns = [column for column in output_columns if column in sample.columns]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.country}_content_codebook_validation_n{len(sample)}.csv"
    sample[output_columns].to_csv(output_path, index=False)

    summary_path = args.output_dir / f"{args.country}_content_codebook_validation_summary.csv"
    summary = (
        sample.groupby(["country", "year", "probability_band"], dropna=False)
        .size()
        .reset_index(name="sample_n")
    )
    summary.to_csv(summary_path, index=False)

    print(f"Saved sample:  {output_path}", flush=True)
    print(f"Saved summary: {summary_path}", flush=True)
    print("\nSample by probability band:", flush=True)
    print(sample["probability_band"].value_counts().to_string(), flush=True)
    print("\nSample by year:", flush=True)
    print(sample["year"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
