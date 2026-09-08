"""Compare LLM content coding against human annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from content_classifier_common import (
    validate_completed_content_output,
    validate_content_sample,
)
from config import ALL_COUNTRIES
from political_classifier.reproducibility import sha256_file


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
    "sample_purpose",
    "validation_weight",
    "classifier_threshold",
    "classifier_manifest_sha256",
    "source_filter_policy",
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
        "--human-file",
        type=Path,
        default=None,
        help=(
            "One combined human-coded sample. When omitted, use the legacy "
            "per-country development-sample convention."
        ),
    )
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
    parser.add_argument(
        "--gpt-input-stem",
        default=None,
        help=(
            "Input stem used by classify_all_content_categories.py. It is normally "
            "inferred from --human-file by removing the coder suffix."
        ),
    )
    parser.add_argument(
        "--require-final-validation",
        action="store_true",
        help="Require final-validation provenance and complete human labels.",
    )
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


def path_stem(path: Path) -> str:
    name = path.name
    for suffix in [".csv.gz", ".csv"]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def gpt_sample_path(gpt_dir: Path, input_stem: str, variable: str) -> Path:
    classifier_name = GPT_CLASSIFIER_FILES[variable]
    return gpt_dir / f"{input_stem}_{classifier_name}_gpt_labels.csv.gz"


def merge_comparison(
    human,
    human_file: Path,
    gpt_files: dict[str, Path],
    variables: dict[str, tuple[str, str]],
):
    keys = key_columns(human)
    if human.duplicated(subset=keys).any():
        raise ValueError(f"Human file contains duplicate keys {keys}: {human_file}")
    keep = list(
        dict.fromkeys(
            keys + HUMAN_REVIEW_COLUMNS + [value[0] for value in variables.values()]
        )
    )
    keep = [column for column in keep if column in human.columns]
    merged = human[keep].copy()

    for variable, (_, gpt_column) in variables.items():
        path = gpt_files[variable]
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
            raise ValueError(f"LLM file contains duplicate keys {gpt_keys}: {path}")
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
                f"LLM file is missing {missing_labels.sum():,} human-coded rows: {path}"
            )
    return merged


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
    merged = merge_comparison(
        human,
        human_file,
        {
            variable: gpt_path(gpt_dir, country, variable)
            for variable in variables
        },
        variables,
    )

    merged["country"] = country
    return merged


def load_sample_comparison(
    human_file: Path,
    gpt_dir: Path,
    input_stem: str,
    variables: dict[str, tuple[str, str]],
):
    if not human_file.exists():
        raise FileNotFoundError(human_file)
    human = read_csv(human_file)
    return merge_comparison(
        human,
        human_file,
        {
            variable: gpt_sample_path(gpt_dir, input_stem, variable)
            for variable in variables
        },
        variables,
    )


def validate_final_human_sample(data, variables) -> None:
    required = {
        "sample_purpose",
        "validation_weight",
        "classifier_manifest_sha256",
        "source_filter_policy",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            "Final human validation file is missing provenance columns: "
            f"{sorted(missing)}"
        )
    purposes = set(data["sample_purpose"].dropna().astype(str))
    if purposes != {"final_validation"}:
        raise ValueError(
            f"Expected sample_purpose=final_validation, found {sorted(purposes)}."
        )
    policies = set(data["source_filter_policy"].dropna().astype(str))
    expected_policy = "exclude_explicit_no_retain_all_other_decisions"
    if policies != {expected_policy}:
        raise ValueError(
            f"Final validation uses unexpected source policies: {sorted(policies)}."
        )
    classifier_hashes = set(
        data["classifier_manifest_sha256"].dropna().astype(str)
    )
    if len(classifier_hashes) != 1:
        raise ValueError("Final validation is not tied to one classifier manifest.")
    countries = set(data["country"].dropna().astype(str))
    if countries != set(ALL_COUNTRIES):
        raise ValueError(
            "Final validation must contain all nine publication countries; "
            f"found {sorted(countries)}."
        )
    for variable, (human_column, _) in variables.items():
        missing_labels = data[human_column].map(normalize_label).eq("")
        if missing_labels.any():
            raise ValueError(
                f"Final validation has {int(missing_labels.sum()):,} uncoded "
                f"{variable} row(s)."
            )


def validate_final_files(
    human_file: Path,
    gpt_files: dict[str, Path],
    data,
) -> None:
    annotation_manifest_path = human_file.with_name(
        human_file.name + ".annotation_manifest.json"
    )
    if not annotation_manifest_path.exists():
        raise FileNotFoundError(
            "Final human file has no annotation manifest: "
            f"{annotation_manifest_path}"
        )
    annotation_manifest = json.loads(
        annotation_manifest_path.read_text(encoding="utf-8")
    )
    output_record = annotation_manifest.get("output") or {}
    if output_record.get("sha256") != sha256_file(human_file):
        raise ValueError("Human annotations differ from their annotation manifest.")
    if int(annotation_manifest.get("rows", -1)) != len(data):
        raise ValueError("Human annotation manifest has an incorrect row count.")
    if int(annotation_manifest.get("reviewed_rows", -1)) != len(data):
        raise ValueError("Final human annotation file is not completely reviewed.")

    input_record = annotation_manifest.get("input") or {}
    if not input_record.get("path") or not input_record.get("sha256"):
        raise ValueError("Annotation manifest has no recorded input sample.")
    input_path = Path(str(input_record.get("path", "")))
    if input_record.get("sha256") != sha256_file(input_path):
        raise ValueError("Annotated input differs from its annotation manifest.")
    sample_provenance = validate_content_sample(input_path)
    if sample_provenance is None:
        raise ValueError("Annotated final sample has no sample manifest.")
    sample_manifest = sample_provenance["manifest"]
    if sample_manifest.get("sample_purpose") != "final_validation":
        raise ValueError("Annotated sample manifest is not final validation.")
    if set(sample_manifest.get("countries") or []) != set(ALL_COUNTRIES):
        raise ValueError("Annotated sample manifest does not cover all countries.")
    if int(sample_manifest.get("sample_rows", -1)) != len(data):
        raise ValueError("Annotated sample manifest has an incorrect row count.")

    completions = {
        variable: validate_completed_content_output(path)
        for variable, path in gpt_files.items()
    }
    if {item.get("production_full_corpus") for item in completions.values()} != {False}:
        raise ValueError("Held-out evaluation expects validation-sample LLM runs.")
    if len({item.get("article_id_sha256") for item in completions.values()}) != 1:
        raise ValueError("Held-out LLM outputs contain different article sets.")
    if len({item.get("model") for item in completions.values()}) != 1:
        raise ValueError("Held-out LLM outputs use different models.")
    if {item.get("source") for item in completions.values()} != {"csv"}:
        raise ValueError("Held-out LLM outputs were not coded from the saved sample.")
    if {int(item.get("rows", -1)) for item in completions.values()} != {len(data)}:
        raise ValueError("Held-out LLM completion markers have incorrect row counts.")

    expected_classifier_hashes = set(
        data["classifier_manifest_sha256"].dropna().astype(str)
    )
    llm_classifier_hashes = {
        (item.get("upstream_classifier") or {}).get("classifier_manifest_sha256")
        for item in completions.values()
    }
    if llm_classifier_hashes != expected_classifier_hashes:
        raise ValueError(
            "Human and LLM held-out samples use different classifier builds."
        )


def evaluate(data, variables):
    import pandas as pd
    from sklearn.metrics import cohen_kappa_score, f1_score

    rows = []
    disagreements = []
    for variable, (human_column, gpt_column) in variables.items():
        gpt_column = f"gpt_{gpt_column}"
        columns = [human_column, gpt_column]
        if "validation_weight" in data.columns:
            columns.append("validation_weight")
        valid = data[columns].copy()
        valid[human_column] = valid[human_column].map(normalize_label)
        valid[gpt_column] = valid[gpt_column].map(normalize_label)
        valid = valid[valid[human_column].ne("") & valid[gpt_column].ne("")]
        correct = valid[human_column].eq(valid[gpt_column])
        labels = sorted(set(valid[human_column]) | set(valid[gpt_column]))
        design_weights = None
        if "validation_weight" in valid.columns:
            design_weights = pd.to_numeric(
                valid["validation_weight"], errors="coerce"
            )
            if design_weights.isna().any() or (design_weights <= 0).any():
                raise ValueError("Validation weights must be positive numeric values.")

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
                "design_weighted_agreement": (
                    float((correct.astype(float) * design_weights).sum() / design_weights.sum())
                    if design_weights is not None and len(valid)
                    else None
                ),
                "design_weighted_macro_f1": (
                    float(
                        f1_score(
                            valid[human_column],
                            valid[gpt_column],
                            labels=labels,
                            average="macro",
                            sample_weight=design_weights,
                            zero_division=0,
                        )
                    )
                    if design_weights is not None and len(valid)
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

    if args.human_file is not None:
        input_stem = args.gpt_input_stem
        if input_stem is None:
            input_stem = path_stem(args.human_file)
            coder_suffix = f"_{args.coder_id}"
            if input_stem.endswith(coder_suffix):
                input_stem = input_stem[: -len(coder_suffix)]
        print(f"Loading {args.human_file}", flush=True)
        gpt_files = {
            variable: gpt_sample_path(gpt_dir, input_stem, variable)
            for variable in variables
        }
        data = load_sample_comparison(
            args.human_file,
            gpt_dir,
            input_stem,
            variables,
        )
    else:
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

    if args.require_final_validation:
        if args.human_file is None:
            raise ValueError(
                "--require-final-validation must be used with --human-file."
            )
        validate_final_human_sample(data, variables)
        validate_final_files(args.human_file, gpt_files, data)
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
