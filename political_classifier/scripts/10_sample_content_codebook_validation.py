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

ANNOTATION_COLUMNS = [
    "human_victim_visibility",
    "human_corruption_frame",
    "human_case_location",
    "human_abroad_case",
    "human_accused_actor_visibility",
    "human_accused_actor_visible",
    "human_notes",
    "human_coder_id",
    "human_coder_first_name",
    "human_code_session_id",
    "human_coded_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible sample of predicted political-corruption "
            "articles for content-codebook validation."
        )
    )
    parser.add_argument("--country", default=None, help="Single country to sample.")
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Multiple countries to sample and combine. Defaults to all countries when --all-countries is used.",
    )
    parser.add_argument(
        "--all-countries",
        action="store_true",
        help="Sample all project countries into one combined file.",
    )
    parser.add_argument("--n", type=int, default=108)
    parser.add_argument(
        "--n-per-country",
        type=int,
        default=None,
        help="Rows per country for combined samples. Overrides --n in multi-country mode.",
    )
    parser.add_argument("--output-name", default=None)
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
    import pandas as pd

    if len(data) <= n:
        return data.sample(frac=1, random_state=seed).copy()

    grouped = data.groupby("sample_stratum", dropna=False)
    group_sizes = grouped.size()

    if len(group_sizes) >= n:
        selected_strata = group_sizes.sample(
            n=n,
            weights=group_sizes,
            random_state=seed,
            replace=False,
        ).index
        allocation = pd.Series(1, index=selected_strata)
    else:
        allocation = pd.Series(1, index=group_sizes.index)
        remaining_n = n - int(allocation.sum())
        fractional = group_sizes / group_sizes.sum() * remaining_n
        extra = fractional.astype(int)
        allocation = allocation.add(extra, fill_value=0).astype(int)

        while allocation.sum() < n:
            remainder = fractional - extra
            idx = remainder.sort_values(ascending=False).index[0]
            allocation.loc[idx] += 1
            fractional.loc[idx] = 0
            extra.loc[idx] = 0

    pieces = []
    for stratum, take_n in allocation.items():
        group = grouped.get_group(stratum)
        take_n = min(int(take_n), len(group))
        pieces.append(group.sample(n=take_n, random_state=seed))

    sample = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed).reset_index(drop=True)
    elif len(sample) < n:
        remaining = data.loc[~data["uri"].astype(str).isin(sample["uri"].astype(str))]
        extra = remaining.sample(n=n - len(sample), random_state=seed)
        sample = pd.concat([sample, extra], ignore_index=True)

    return sample.reset_index(drop=True)


def sample_country(args: argparse.Namespace, country: str, n: int) -> pd.DataFrame:
    import pandas as pd

    input_path = args.classified_dir / f"{country}_classified.csv.gz"
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
    sample = stratified_sample(data, n, args.seed)
    sample.insert(0, "content_sample_id", [f"{country}_{i:04d}" for i in range(1, len(sample) + 1)])
    return sample


def selected_countries(args: argparse.Namespace) -> list[str]:
    if args.all_countries:
        return DEFAULT_COUNTRIES
    if args.countries:
        return args.countries
    if args.country:
        return [args.country]
    return ["France"]


def main() -> None:
    args = parse_args()
    import pandas as pd

    countries = selected_countries(args)
    multi_country = len(countries) > 1
    n_per_country = args.n_per_country if args.n_per_country is not None else args.n

    samples = []
    for country in countries:
        take_n = n_per_country if multi_country else args.n
        samples.append(sample_country(args, country, take_n))

    sample = pd.concat(samples, ignore_index=True)
    sample = sample.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if multi_country:
        sample["content_sample_id"] = [f"CCV_{i:04d}" for i in range(1, len(sample) + 1)]

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
    if args.output_name:
        output_name = args.output_name
    elif multi_country:
        output_name = f"content_codebook_validation_sample_{len(sample)}_total_{n_per_country}_per_country.csv"
    else:
        output_name = f"{countries[0]}_content_codebook_validation_n{len(sample)}.csv"
    output_path = args.output_dir / output_name
    sample[output_columns].to_csv(output_path, index=False)

    if output_path.name.endswith(".csv"):
        summary_name = output_path.name.replace(".csv", "_summary.csv")
    else:
        summary_name = output_path.name + "_summary.csv"
    summary_path = args.output_dir / summary_name
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
    print("\nSample by country:", flush=True)
    print(sample["country"].value_counts().sort_index().to_string(), flush=True)
    print("\nSample by year:", flush=True)
    print(sample["year"].value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    main()
