"""Merge complete article-level LLM content labels into one dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from political_classifier.reproducibility import file_record, git_commit
from content_classifier_common import validate_completed_content_output


DEFAULT_OUTPUT_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification/final_gpt51"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge zero-shot content labels and derive analysis variables."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cpi",
        type=Path,
        default=None,
        help="Optional CPI file. Canonical content-label outputs omit covariates.",
    )
    parser.add_argument("--skip-cpi", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Diagnostic mode: retain articles missing one or more classifier outputs.",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Diagnostic mode: merge rows with non-empty LLM errors.",
    )
    return parser.parse_args()


def read_csv(path: Path):
    import pandas as pd

    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer")


def write_csv(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data.to_csv(
        temporary,
        index=False,
        compression="gzip" if path.name.endswith(".gz") else None,
    )
    temporary.replace(path)


def standardize_country(country: object) -> str:
    return str(country).replace("_", " ").strip()


def normalize_yes_no(value: object):
    value = str(value).strip().lower()
    if value in {"yes", "1", "true"}:
        return 1
    if value in {"no", "0", "false"}:
        return 0
    return None


def derive_variables(data):
    data = data.copy()

    data["victim_visible_binary"] = data["victim_visibility"].map(
        {
            "concrete_victim": 1,
            "institutional_societal_victim": 1,
            "no_victim": 0,
        }
    )
    if "concrete_victim_visible" in data.columns:
        data["concrete_victim_visible"] = data["concrete_victim_visible"].map(
            normalize_yes_no
        )
    else:
        data["concrete_victim_visible"] = data["victim_visibility"].map(
            {
                "concrete_victim": 1,
                "institutional_societal_victim": 0,
                "no_victim": 0,
            }
        )
    if "institutional_societal_victim_visible" in data.columns:
        data["institutional_societal_victim_visible"] = data[
            "institutional_societal_victim_visible"
        ].map(normalize_yes_no)
    else:
        data["institutional_societal_victim_visible"] = data[
            "victim_visibility"
        ].map(
            {
                "institutional_societal_victim": 1,
                "concrete_victim": 0,
                "no_victim": 0,
            }
        )
    data["abroad_case_binary"] = data["abroad_case"].map(normalize_yes_no)
    data["accused_actor_visible_binary"] = data["accused_actor_visible"].map(normalize_yes_no)
    data["frame_individualized"] = data["corruption_frame"].map(
        {"individualized": 1, "systemic": 0, "other_or_mixed": 0}
    )
    data["frame_systemic"] = data["corruption_frame"].map(
        {"systemic": 1, "individualized": 0, "other_or_mixed": 0}
    )
    data["frame_other_or_mixed"] = data["corruption_frame"].map(
        {"other_or_mixed": 1, "individualized": 0, "systemic": 0}
    )
    return data


def base_metadata(frames: dict, base_cols: list[str]):
    import pandas as pd

    parts = []
    for frame in frames.values():
        columns = [column for column in base_cols if column in frame.columns]
        if columns:
            parts.append(frame[columns])
    if not parts:
        raise ValueError("No metadata columns found in classifier outputs.")
    return pd.concat(parts, ignore_index=True).drop_duplicates("article_id", keep="first")


def add_cpi_context(data, cpi_path: Path):
    import pandas as pd

    if not cpi_path.exists():
        raise FileNotFoundError(cpi_path)

    cpi = pd.read_csv(cpi_path)
    cpi["country_join"] = cpi["country"].map(standardize_country)
    cpi["year"] = pd.to_numeric(cpi["year"], errors="coerce").astype("Int64")
    cpi["perceived_corruption"] = 100 - pd.to_numeric(cpi["cpi_score"], errors="coerce")
    cpi["year"] = cpi["year"] + 1
    cpi = cpi.rename(
        columns={
            "cpi_score": "cpi_score_lag1",
            "cpi_rank": "cpi_rank_lag1",
            "perceived_corruption": "perceived_corruption_lag1",
        }
    )
    cpi = cpi[
        [
            "country_join",
            "year",
            "cpi_score_lag1",
            "cpi_rank_lag1",
            "perceived_corruption_lag1",
        ]
    ]

    data = data.copy()
    data["country_join"] = data["country"].map(standardize_country)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    data = data.merge(cpi, on=["country_join", "year"], how="left")
    data = data.drop(columns=["country_join"])
    return data


def validate_production_completions(paths: dict[str, Path]) -> dict[str, dict]:
    completions = {
        name: validate_completed_content_output(path)
        for name, path in paths.items()
    }
    article_hashes = {
        completion.get("article_id_sha256")
        for completion in completions.values()
    }
    models = {completion.get("model") for completion in completions.values()}
    production_scopes = {
        completion.get("production_full_corpus")
        for completion in completions.values()
    }
    upstream_hashes = {
        (completion.get("upstream_classifier") or {}).get(
            "classifier_manifest_sha256"
        )
        for completion in completions.values()
    }
    if None in upstream_hashes or len(upstream_hashes) != 1:
        raise ValueError(
            "Content outputs are not tied to one verified final classifier run."
        )
    if None in article_hashes or len(article_hashes) != 1:
        raise ValueError("Content outputs contain different article-ID sets.")
    if None in models or len(models) != 1:
        raise ValueError(
            f"Content outputs use different LLMs: {sorted(str(x) for x in models)}"
        )
    if production_scopes != {True}:
        raise ValueError(
            "Canonical merge requires four complete all-country production runs. "
            "Validation-sample and country-subset outputs must be merged only in "
            "diagnostic mode or kept in their validation directory."
        )
    upstream = next(iter(completions.values()))["upstream_classifier"]
    expected_rows = int(upstream["political_corruption_articles"])
    completed_rows = {int(item.get("rows", -1)) for item in completions.values()}
    if completed_rows != {expected_rows}:
        raise ValueError(
            "Content output row totals do not equal the verified final "
            f"political-corruption N ({expected_rows:,})."
        )
    return completions


def main() -> None:
    args = parse_args()

    victim_path = args.input_dir / "victim_visibility_labels.csv.gz"
    frame_path = args.input_dir / "corruption_frame_labels.csv.gz"
    abroad_path = args.input_dir / "abroad_case_labels.csv.gz"
    accused_path = args.input_dir / "accused_actor_labels.csv.gz"
    output_path = args.output or (
        args.input_dir / "political_corruption_content_categories_final.csv.gz"
    )

    paths = {
        "victim": victim_path,
        "frame": frame_path,
        "abroad": abroad_path,
        "accused": accused_path,
    }
    completions = {}
    if not args.allow_partial and not args.allow_errors:
        completions = validate_production_completions(paths)

    frames = {
        "victim": read_csv(victim_path),
        "frame": read_csv(frame_path),
        "abroad": read_csv(abroad_path),
        "accused": read_csv(accused_path),
    }
    if completions:
        for name, frame in frames.items():
            expected_rows = int(completions[name].get("rows", -1))
            if len(frame) != expected_rows:
                raise ValueError(
                    f"{name} output has {len(frame):,} rows but its completion "
                    f"marker records {expected_rows:,}."
                )

    allowed_labels = {
        "victim": ("victim_visibility", {"no_victim", "concrete_victim", "institutional_societal_victim", "unclear"}),
        "frame": ("corruption_frame", {"individualized", "systemic", "other_or_mixed", "unclear"}),
        "abroad": ("case_location", {"domestic", "abroad", "unclear"}),
        "accused": (
            "accused_actor_visibility",
            {"no_accused_actor", "individual_actor", "organizational_or_institutional_actor", "both_individual_and_organizational", "unclear"},
        ),
    }
    for name, frame in frames.items():
        if "article_id" not in frame.columns:
            raise ValueError(f"{name} output is missing article_id.")
        duplicates = frame["article_id"].duplicated(keep=False)
        if duplicates.any():
            raise ValueError(
                f"{name} output contains {duplicates.sum():,} duplicate article_id rows. "
                "Rerun with the keyed checkpoint implementation."
            )
        if not args.allow_errors and "llm_error" in frame.columns:
            errors = frame["llm_error"].fillna("").astype(str).str.strip().ne("")
            if errors.any():
                raise ValueError(
                    f"{name} output contains {errors.sum():,} failed rows. Retry them before merging."
                )
        label_column, allowed = allowed_labels[name]
        if label_column not in frame.columns:
            raise ValueError(f"{name} output is missing {label_column}.")
        invalid = ~frame[label_column].fillna("").astype(str).isin(allowed)
        if invalid.any():
            values = sorted(frame.loc[invalid, label_column].astype(str).unique())
            raise ValueError(f"{name} output has invalid {label_column} values: {values}")

    if not args.allow_partial:
        reference_name = "victim"
        reference_ids = set(frames[reference_name]["article_id"].astype(str))
        differences = []
        for name, frame in frames.items():
            ids = set(frame["article_id"].astype(str))
            missing = len(reference_ids - ids)
            extra = len(ids - reference_ids)
            if missing or extra:
                differences.append(f"{name}: missing={missing}, extra={extra}")
        if differences:
            raise ValueError(
                "Classifier outputs do not contain the same article IDs: "
                + "; ".join(differences)
                + ". Complete/retry all four outputs before the final merge."
            )

    base_cols = [
        "article_id",
        "uri",
        "country",
        "year",
        "month",
        "week",
        "source_uri",
        "source.uri",
        "word_count",
        "prob_political_corruption",
        "pred_political_corruption",
    ]
    data = base_metadata(frames, base_cols)

    how = "outer" if args.allow_partial else "inner"
    for name, frame in frames.items():
        frame = frame.rename(
            columns={
                "llm_error": f"{name}_llm_error",
                "prompt_version": f"{name}_prompt_version",
            }
        )
        label_cols = [
            column
            for column in frame.columns
            if column not in set(base_cols + ["classifier_name"])
        ]
        keep_cols = ["article_id", *label_cols]
        data = data.merge(frame[keep_cols], on="article_id", how=how, suffixes=("", f"_{name}"))

    data = derive_variables(data)
    if args.cpi is not None and not args.skip_cpi:
        data = add_cpi_context(data, args.cpi)

    write_csv(data, output_path)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "output": file_record(output_path),
        "rows": len(data),
        "allow_partial": args.allow_partial,
        "allow_errors": args.allow_errors,
        "cpi_input": file_record(args.cpi) if args.cpi is not None else None,
        "upstream_classifier": (
            next(iter(completions.values())).get("upstream_classifier")
            if completions
            else None
        ),
        "content_model": (
            next(iter(completions.values())).get("model")
            if completions
            else None
        ),
        "article_id_sha256": (
            next(iter(completions.values())).get("article_id_sha256")
            if completions
            else None
        ),
        "inputs": {
            name: {
                "path": str(path),
                "file": file_record(path),
                "rows": len(frames[name]),
                "prompt_versions": sorted(
                    frames[name].get("prompt_version", []).dropna().astype(str).unique().tolist()
                ) if "prompt_version" in frames[name].columns else [],
                "models": sorted(
                    frames[name].get("llm_model", []).dropna().astype(str).unique().tolist()
                ) if "llm_model" in frames[name].columns else [],
                "completion": (
                    file_record(path.with_name(path.name + ".complete.json"))
                    if name in completions
                    else None
                ),
            }
            for name, path in paths.items()
        },
    }
    output_path.with_name(output_path.name + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved merged content labels: {output_path} ({len(data):,} rows)", flush=True)


if __name__ == "__main__":
    main()
