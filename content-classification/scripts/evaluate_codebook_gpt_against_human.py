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

HUMAN_REVIEW_COLUMNS = [
    "content_sample_id",
    "uri",
    "country",
    "source_uri",
    "year",
    "translated_text_en",
    "translated_text",
    "article_text",
    "human_notes",
]

GPT_CLASSIFIER_FILES = {
    "victim_visibility": "victim_visibility",
    "corruption_frame": "corruption_frame",
    "case_location": "abroad_case",
    "accused_actor_visibility": "accused_actor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot LLM content labels against human-coded codebook samples."
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
    parser.add_argument(
        "--variables",
        nargs="+",
        choices=list(VARIABLES),
        default=list(VARIABLES),
        help="Evaluate only the selected variables. Defaults to all four.",
    )
    parser.add_argument("--output-prefix", default="first4")
    return parser.parse_args()


def read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer")


def key_columns(data) -> list[str]:
    if "content_sample_id" in data.columns:
        return ["content_sample_id"]
    if "article_id" in data.columns:
        return ["article_id"]
    if "uri" in data.columns:
        return ["uri"]
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


def load_country_comparison(
    codebook_dir: Path,
    gpt_dir: Path,
    country: str,
    coder_id: str,
    variables: dict[str, tuple[str, str]],
):
    human_file = human_path(codebook_dir, country, coder_id)
    if not human_file.exists():
        raise FileNotFoundError(human_file)

    human = read_csv(human_file)
    keys = key_columns(human)
    if human.duplicated(subset=keys).any():
        raise ValueError(f"Human file contains duplicate keys {keys}: {human_file}")
    keep = list(
        dict.fromkeys(
            keys
            + HUMAN_REVIEW_COLUMNS
            + [value[0] for value in variables.values()]
        )
    )
    keep = [column for column in keep if column in human.columns]
    merged = human[keep].copy()

    for variable, (_, gpt_column) in variables.items():
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
        if gpt.duplicated(subset=gpt_keys).any():
            raise ValueError(f"GPT file contains duplicate keys {gpt_keys}: {path}")
        detail_columns = [
            column
            for column in gpt.columns
            if any(
                token in column
                for token in [
                    "evidence",
                    "reasoning",
                    "confidence",
                    "harm_cause",
                    "harm_status",
                    "victim_entity",
                    "deprived_entity",
                    "harmed_entity",
                    "explicit_harm_statement",
                    "coercive_demand",
                    "concrete_victim_visible",
                    "institutional_societal_victim_visible",
                ]
            )
        ]
        gpt_keep = list(dict.fromkeys(gpt_keys + [gpt_column] + detail_columns))
        gpt_keep = [column for column in gpt_keep if column in gpt.columns]
        rename_columns = {gpt_column: f"gpt_{gpt_column}"}
        rename_columns.update(
            {
                column: f"gpt_{variable}_{column}"
                for column in detail_columns
                if column != gpt_column
            }
        )
        gpt = gpt[gpt_keep].rename(columns=rename_columns)
        merged = merged.merge(
            gpt,
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
        missing_labels = merged[f"gpt_{gpt_column}"].isna()
        if missing_labels.any():
            raise ValueError(
                f"GPT file is missing {missing_labels.sum():,} human-coded rows: {path}"
            )

    merged["country"] = country
    return merged


def evaluate(data, variables):
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score, f1_score

    rows = []
    disagreements = []
    for variable, (human_column, gpt_column) in variables.items():
        gpt_column = f"gpt_{gpt_column}"
        valid = data[[human_column, gpt_column]].copy()
        valid[human_column] = valid[human_column].map(normalize_label)
        valid[gpt_column] = valid[gpt_column].map(normalize_label)
        valid = valid[valid[human_column].ne("") & valid[gpt_column].ne("")]
        correct = valid[human_column].eq(valid[gpt_column])
        labels = sorted(set(valid[human_column]) | set(valid[gpt_column]))

        rows.append(
            {
                "variable": variable,
                "n": len(valid),
                "agreement": float(correct.mean()) if len(valid) else None,
                "cohen_kappa": (
                    float(cohen_kappa_score(valid[human_column], valid[gpt_column], labels=labels))
                    if len(valid) and len(labels) > 1
                    else None
                ),
                "macro_f1": (
                    float(f1_score(valid[human_column], valid[gpt_column], labels=labels, average="macro", zero_division=0))
                    if len(valid)
                    else None
                ),
                "weighted_f1": (
                    float(f1_score(valid[human_column], valid[gpt_column], labels=labels, average="weighted", zero_division=0))
                    if len(valid)
                    else None
                ),
                "matches": int(correct.sum()),
                "mismatches": int((~correct).sum()),
            }
        )

        bad = data.loc[valid.index[~correct]].copy()
        if not bad.empty:
            bad["variable"] = variable
            bad["human_label"] = bad[human_column].map(normalize_label)
            bad["gpt_label"] = bad[gpt_column].map(normalize_label)
            if variable == "victim_visibility":
                positive_labels = {
                    "concrete_victim",
                    "institutional_societal_victim",
                    "both_concrete_and_institutional",
                }

                def victim_disagreement_type(row) -> str:
                    human_label = row["human_label"]
                    gpt_label = row["gpt_label"]
                    if "unclear" in {human_label, gpt_label}:
                        return "unclear_boundary"
                    if (human_label == "no_victim") != (gpt_label == "no_victim"):
                        return "victim_visibility_gate"
                    if human_label in positive_labels and gpt_label in positive_labels:
                        return "victim_type_boundary"
                    return "other"

                bad["disagreement_type"] = bad.apply(victim_disagreement_type, axis=1)
            else:
                bad["disagreement_type"] = "label_boundary"
            bad["adjudicated_label"] = ""
            bad["adjudication_notes"] = ""
            disagreements.append(bad)

    summary = pd.DataFrame(rows)
    disagreement_df = pd.concat(disagreements, ignore_index=True) if disagreements else pd.DataFrame()
    return summary, disagreement_df


def confusion_table(data, variables):
    import pandas as pd

    rows = []
    for variable, (human_column, gpt_column) in variables.items():
        gpt_column = f"gpt_{gpt_column}"
        valid = data[[human_column, gpt_column]].copy()
        valid[human_column] = valid[human_column].map(normalize_label)
        valid[gpt_column] = valid[gpt_column].map(normalize_label)
        valid = valid[valid[human_column].ne("") & valid[gpt_column].ne("")]
        counts = (
            valid.groupby([human_column, gpt_column], dropna=False)
            .size()
            .rename("n")
            .reset_index()
            .rename(
                columns={
                    human_column: "human_label",
                    gpt_column: "gpt_label",
                }
            )
        )
        counts.insert(0, "variable", variable)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    import pandas as pd

    gpt_dir = args.gpt_dir or (args.codebook_dir / "gpt51_test_labels")
    variables = {name: VARIABLES[name] for name in args.variables}

    frames = []
    for country in args.countries:
        print(f"Loading {country}", flush=True)
        frames.append(
            load_country_comparison(
                args.codebook_dir,
                gpt_dir,
                country,
                args.coder_id,
                variables,
            )
        )

    data = pd.concat(frames, ignore_index=True)
    summary, disagreements = evaluate(data, variables)
    confusion = confusion_table(data, variables)

    country_summaries = []
    for country, country_df in data.groupby("country", dropna=False):
        country_summary, _ = evaluate(country_df, variables)
        country_summary.insert(0, "country", country)
        country_summaries.append(country_summary)
    country_summary = pd.concat(country_summaries, ignore_index=True)

    output_dir = gpt_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.output_prefix}_human_gpt_agreement_summary.csv"
    country_path = output_dir / f"{args.output_prefix}_human_gpt_agreement_by_country.csv"
    disagreement_path = output_dir / f"{args.output_prefix}_human_gpt_disagreements.csv"
    confusion_path = output_dir / f"{args.output_prefix}_human_gpt_confusion.csv"

    summary.to_csv(summary_path, index=False)
    country_summary.to_csv(country_path, index=False)
    disagreements.to_csv(disagreement_path, index=False)
    confusion.to_csv(confusion_path, index=False)

    print("\nOverall agreement:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nAgreement by country:", flush=True)
    print(country_summary.to_string(index=False), flush=True)
    print(f"\nSaved summary:       {summary_path}", flush=True)
    print(f"Saved country table: {country_path}", flush=True)
    print(f"Saved disagreements: {disagreement_path}", flush=True)
    print(f"Saved confusion table: {confusion_path}", flush=True)


if __name__ == "__main__":
    main()
