"""Compare GPT content coding against human codebook-development annotations."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_CODEBOOK_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline/content_codebook_validation"
)

VARIABLES = {
    "victim_visibility": ("human_victim_visibility", "victim_visibility"),
    "corruption_frame": ("human_corruption_frame", "corruption_frame"),
    "case_location": ("human_case_location", "case_location"),
    "accused_actor_visibility": ("human_accused_actor_visibility", "accused_actor_visibility"),
}

GPT_CLASSIFIER_FILES = {
    "victim_visibility": "victim_visibility",
    "corruption_frame": "corruption_frame",
    "case_location": "abroad_case",
    "accused_actor_visibility": "accused_actor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GPT-5.1 content labels against human-coded codebook samples."
    )
    parser.add_argument("--codebook-dir", type=Path, default=DEFAULT_CODEBOOK_DIR)
    parser.add_argument(
        "--gpt-dir",
        type=Path,
        default=None,
        help="Defaults to CODEBOOK_DIR/gpt51_test_labels.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["Bulgaria", "France", "Hungary", "Serbia"],
    )
    parser.add_argument(
        "--coder-id",
        default="anne",
        help="Coder suffix in files such as *_english_anne.csv.",
    )
    parser.add_argument("--output-prefix", default="first4")
    return parser.parse_args()


def read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer")


def key_columns(data) -> list[str]:
    if "article_id" in data.columns:
        return ["article_id"]
    if "uri" in data.columns:
        return ["uri"]
    if "content_sample_id" in data.columns:
        return ["content_sample_id"]
    raise ValueError("Could not find article_id, uri, or content_sample_id.")


def normalize_label(value: object) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none"}:
        return ""
    return value


def human_path(codebook_dir: Path, country: str, coder_id: str) -> Path:
    return codebook_dir / f"{country}_content_codebook_validation_n12_english_{coder_id}.csv"


def gpt_path(gpt_dir: Path, country: str, variable: str) -> Path:
    classifier_name = GPT_CLASSIFIER_FILES[variable]
    return (
        gpt_dir
        / country
        / f"{country}_content_codebook_validation_n12_english_{classifier_name}_gpt_labels.csv.gz"
    )


def load_country_comparison(codebook_dir: Path, gpt_dir: Path, country: str, coder_id: str):
    human_file = human_path(codebook_dir, country, coder_id)
    if not human_file.exists():
        raise FileNotFoundError(human_file)

    human = read_csv(human_file)
    keys = key_columns(human)
    keep = list(dict.fromkeys(keys + ["content_sample_id", "uri", "country", *[v[0] for v in VARIABLES.values()]]))
    keep = [column for column in keep if column in human.columns]
    merged = human[keep].copy()

    for variable, (_, gpt_column) in VARIABLES.items():
        path = gpt_path(gpt_dir, country, variable)
        if not path.exists():
            raise FileNotFoundError(path)
        gpt = read_csv(path)
        gpt_keys = key_columns(gpt)
        if gpt_keys != keys and "uri" in gpt.columns and "uri" in merged.columns:
            gpt_keys = ["uri"]
            merge_keys = ["uri"]
        else:
            merge_keys = keys
        gpt_keep = list(dict.fromkeys(gpt_keys + [gpt_column]))
        gpt_keep = [column for column in gpt_keep if column in gpt.columns]
        gpt = gpt[gpt_keep].rename(columns={gpt_column: f"gpt_{gpt_column}"})
        merged = merged.merge(gpt, on=merge_keys, how="left")

    merged["country"] = country
    return merged


def evaluate(data):
    import pandas as pd

    rows = []
    disagreements = []
    for variable, (human_column, gpt_column) in VARIABLES.items():
        gpt_column = f"gpt_{gpt_column}"
        valid = data[[human_column, gpt_column]].copy()
        valid[human_column] = valid[human_column].map(normalize_label)
        valid[gpt_column] = valid[gpt_column].map(normalize_label)
        valid = valid[valid[human_column].ne("") & valid[gpt_column].ne("")]
        correct = valid[human_column].eq(valid[gpt_column])

        rows.append(
            {
                "variable": variable,
                "n": len(valid),
                "agreement": float(correct.mean()) if len(valid) else None,
                "matches": int(correct.sum()),
                "mismatches": int((~correct).sum()),
            }
        )

        bad = data.loc[valid.index[~correct]].copy()
        if not bad.empty:
            bad["variable"] = variable
            bad["human_label"] = bad[human_column].map(normalize_label)
            bad["gpt_label"] = bad[gpt_column].map(normalize_label)
            disagreements.append(bad)

    summary = pd.DataFrame(rows)
    disagreement_df = pd.concat(disagreements, ignore_index=True) if disagreements else pd.DataFrame()
    return summary, disagreement_df


def main() -> None:
    args = parse_args()
    import pandas as pd

    gpt_dir = args.gpt_dir or (args.codebook_dir / "gpt51_test_labels")

    frames = []
    for country in args.countries:
        print(f"Loading {country}", flush=True)
        frames.append(load_country_comparison(args.codebook_dir, gpt_dir, country, args.coder_id))

    data = pd.concat(frames, ignore_index=True)
    summary, disagreements = evaluate(data)

    country_summaries = []
    for country, country_df in data.groupby("country", dropna=False):
        country_summary, _ = evaluate(country_df)
        country_summary.insert(0, "country", country)
        country_summaries.append(country_summary)
    country_summary = pd.concat(country_summaries, ignore_index=True)

    output_dir = gpt_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.output_prefix}_human_gpt_agreement_summary.csv"
    country_path = output_dir / f"{args.output_prefix}_human_gpt_agreement_by_country.csv"
    disagreement_path = output_dir / f"{args.output_prefix}_human_gpt_disagreements.csv"

    summary.to_csv(summary_path, index=False)
    country_summary.to_csv(country_path, index=False)
    disagreements.to_csv(disagreement_path, index=False)

    print("\nOverall agreement:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nAgreement by country:", flush=True)
    print(country_summary.to_string(index=False), flush=True)
    print(f"\nSaved summary:       {summary_path}", flush=True)
    print(f"Saved country table: {country_path}", flush=True)
    print(f"Saved disagreements: {disagreement_path}", flush=True)


if __name__ == "__main__":
    main()
