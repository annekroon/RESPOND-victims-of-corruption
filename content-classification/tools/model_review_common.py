"""Persistence, provenance, and validation for model-assisted content review."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "content-classification" / "scripts"
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd

from content_annotation_common import (
    read_csv,
    safe_coder_id,
    write_csv_atomic,
)
from content_classifier_common import validate_content_sample
from content_codebook import CODEBOOK_PATH, CODEBOOK_SHA256, CODEBOOK_VERSION
from political_classifier.reproducibility import file_record, git_commit


VARIABLES = {
    "victim_visibility": {
        "classifier_name": "victim_visibility",
        "label_column": "victim_visibility",
        "labels": [
            "no_victim",
            "concrete_victim",
            "institutional_societal_victim",
            "unclear",
        ],
        "evidence_columns": [
            "victim_entity",
            "victim_harm_evidence",
            "victim_corruption_harm_link_evidence",
        ],
        "reasoning_column": "victim_reasoning_brief",
        "confidence_column": "victim_confidence",
    },
    "corruption_frame": {
        "classifier_name": "corruption_frame",
        "label_column": "corruption_frame",
        "labels": ["individualized", "systemic", "other_or_mixed", "unclear"],
        "evidence_columns": ["frame_evidence"],
        "reasoning_column": "frame_reasoning_brief",
        "confidence_column": "frame_confidence",
    },
    "case_location": {
        "classifier_name": "abroad_case",
        "label_column": "case_location",
        "labels": ["domestic", "abroad", "unclear"],
        "evidence_columns": ["abroad_evidence"],
        "reasoning_column": "abroad_reasoning_brief",
        "confidence_column": "abroad_confidence",
    },
    "accused_actor_visibility": {
        "classifier_name": "accused_actor",
        "label_column": "accused_actor_visibility",
        "labels": [
            "no_accused_actor",
            "individual_actor",
            "organizational_or_institutional_actor",
            "both_individual_and_organizational",
            "unclear",
        ],
        "evidence_columns": [
            "accused_individual_evidence",
            "accused_organization_evidence",
        ],
        "reasoning_column": "accused_reasoning_brief",
        "confidence_column": "accused_confidence",
    },
}

REVIEW_DECISIONS = ["confirm", "correct", "uncertain"]
REVIEWER_COLUMNS = [
    "model_review_overall_comment",
    "model_review_coder_id",
    "model_review_coder_first_name",
    "model_review_session_id",
    "model_review_codebook_version",
    "model_review_codebook_sha256",
    "model_reviewed_at_utc",
]
for variable in VARIABLES:
    REVIEWER_COLUMNS.extend(
        [
            f"model_review_{variable}_decision",
            f"model_review_{variable}_corrected_label",
            f"model_review_{variable}_final_label",
            f"model_review_{variable}_comment",
        ]
    )


def input_stem(path: Path) -> str:
    for suffix in [".csv.gz", ".csv"]:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def default_review_output_template(input_path: Path) -> str:
    suffix = ".csv.gz" if input_path.name.endswith(".csv.gz") else ".csv"
    return str(
        input_path.with_name(
            f"{input_stem(input_path)}_model_review_{{coder_id}}{suffix}"
        )
    )


def review_output_path(
    *,
    input_path: Path,
    output_path: Path | None,
    output_template: str,
    coder_id: str,
) -> Path:
    if output_template:
        return Path(output_template.format(coder_id=safe_coder_id(coder_id)))
    if output_path is not None:
        return output_path
    return Path(
        default_review_output_template(input_path).format(
            coder_id=safe_coder_id(coder_id)
        )
    )


def model_output_paths(input_path: Path, model_dir: Path) -> dict[str, Path]:
    stem = input_stem(input_path)
    return {
        variable: model_dir
        / f"{stem}_{specification['classifier_name']}_gpt_labels.csv.gz"
        for variable, specification in VARIABLES.items()
    }


def _ensure_article_id(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "article_id" not in data.columns:
        if "uri" in data.columns:
            data["article_id"] = data["uri"].fillna("").astype(str)
        elif "content_sample_id" in data.columns:
            data["article_id"] = data["content_sample_id"].fillna("").astype(str)
        else:
            raise ValueError("Input has no article_id, uri, or content_sample_id.")
    if data["article_id"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Input contains blank article IDs.")
    if data["article_id"].astype(str).duplicated().any():
        raise ValueError("Input contains duplicate article IDs.")
    data["article_id"] = data["article_id"].astype(str)
    return data


def _join_key(source: pd.DataFrame, model: pd.DataFrame, path: Path) -> str:
    for column in ["content_sample_id", "article_id", "uri"]:
        if column not in source.columns or column not in model.columns:
            continue
        source_values = source[column].fillna("").astype(str)
        model_values = model[column].fillna("").astype(str)
        if (
            source_values.str.strip().ne("").all()
            and model_values.str.strip().ne("").all()
            and not source_values.duplicated().any()
            and not model_values.duplicated().any()
        ):
            return column
    raise ValueError(f"No unique shared article key in model output: {path}")


def _single_nonblank(data: pd.DataFrame, column: str, path: Path) -> str:
    if column not in data.columns:
        raise ValueError(f"Model output has no {column}: {path}")
    normalized = data[column].fillna("").astype(str).str.strip()
    if normalized.eq("").any():
        raise ValueError(f"Model output contains blank {column} values: {path}")
    values = sorted(set(normalized))
    if len(values) != 1:
        raise ValueError(
            f"Model output must contain one nonblank {column}; found {values}: {path}"
        )
    return values[0]


def _merge_model_output(
    source: pd.DataFrame,
    *,
    variable: str,
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    model = _ensure_article_id(read_csv(path))
    specification = VARIABLES[variable]
    key = _join_key(source, model, path)

    source_keys = set(source[key].astype(str))
    model_keys = set(model[key].astype(str))
    missing = source_keys - model_keys
    extra = model_keys - source_keys
    if missing or extra:
        raise ValueError(
            f"Model output does not exactly match the review input for {variable}: "
            f"{len(missing)} missing and {len(extra)} extra row(s): {path}"
        )
    if "llm_error" in model.columns:
        errors = model["llm_error"].fillna("").astype(str).str.strip().ne("")
        if errors.any():
            raise ValueError(
                f"Model output contains {int(errors.sum())} error row(s): {path}"
            )

    label_column = str(specification["label_column"])
    if label_column not in model.columns:
        raise ValueError(f"Model output has no {label_column}: {path}")
    labels = model[label_column].fillna("").astype(str).str.strip()
    invalid = ~labels.isin(specification["labels"])
    if invalid.any():
        observed = sorted(set(labels.loc[invalid]))
        raise ValueError(f"Invalid {variable} labels {observed}: {path}")

    _single_nonblank(model, "llm_model", path)
    _single_nonblank(model, "prompt_version", path)
    _single_nonblank(model, "codebook_version", path)
    _single_nonblank(model, "codebook_sha256", path)

    detail_columns = [
        label_column,
        str(specification["reasoning_column"]),
        str(specification["confidence_column"]),
        *[str(column) for column in specification["evidence_columns"]],
        "llm_model",
        "prompt_version",
        "codebook_version",
        "codebook_sha256",
        "llm_coded_at_utc",
        "input_text_sha256",
    ]
    detail_columns = [column for column in dict.fromkeys(detail_columns) if column in model]
    renamed = {
        column: (
            f"model_{variable}_label"
            if column == label_column
            else f"model_{variable}_{column}"
        )
        for column in detail_columns
    }
    model_slice = model[[key, *detail_columns]].rename(columns=renamed)
    return source.merge(model_slice, on=key, how="left", validate="one_to_one")


def ensure_review_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    for column in REVIEWER_COLUMNS:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("")
    return data


def build_review_data(
    input_path: Path,
    model_dir: Path,
) -> tuple[pd.DataFrame, dict | None, dict[str, Path]]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    provenance = validate_content_sample(input_path)
    data = _ensure_article_id(read_csv(input_path))
    if data.empty:
        raise ValueError("Model-review input contains no articles.")
    paths = model_output_paths(input_path, model_dir)
    for variable, path in paths.items():
        data = _merge_model_output(data, variable=variable, path=path)

    models = {
        str(data[f"model_{variable}_llm_model"].iloc[0])
        for variable in VARIABLES
    }
    codebook_versions = {
        str(data[f"model_{variable}_codebook_version"].iloc[0])
        for variable in VARIABLES
    }
    codebook_hashes = {
        str(data[f"model_{variable}_codebook_sha256"].iloc[0])
        for variable in VARIABLES
    }
    if len(models) != 1:
        raise ValueError(f"Model outputs use different LLMs: {sorted(models)}")
    if codebook_versions != {CODEBOOK_VERSION} or codebook_hashes != {
        CODEBOOK_SHA256
    }:
        raise ValueError(
            "Model outputs do not use the current canonical codebook. "
            "Rerun all four model coders before starting review."
        )
    return ensure_review_columns(data), provenance, paths


def review_errors(
    *,
    model_labels: dict[str, str],
    decisions: dict[str, str],
    corrected_labels: dict[str, str],
    comments: dict[str, str],
) -> list[str]:
    errors = []
    for variable, specification in VARIABLES.items():
        decision = decisions.get(variable, "")
        corrected = corrected_labels.get(variable, "")
        comment = comments.get(variable, "").strip()
        if decision not in REVIEW_DECISIONS:
            errors.append(f"Review {variable.replace('_', ' ')}.")
            continue
        if decision == "correct":
            if corrected not in specification["labels"]:
                errors.append(
                    f"Choose a corrected {variable.replace('_', ' ')} label."
                )
            elif corrected == model_labels.get(variable, ""):
                errors.append(
                    f"The corrected {variable.replace('_', ' ')} label must differ "
                    "from the model label."
                )
            if not comment:
                errors.append(
                    f"Explain the {variable.replace('_', ' ')} correction."
                )
        if decision == "uncertain" and not comment:
            errors.append(
                f"Explain why {variable.replace('_', ' ')} cannot be decided."
            )
    return errors


def final_review_label(model_label: str, decision: str, corrected_label: str) -> str:
    if decision == "confirm":
        return model_label
    if decision == "correct":
        return corrected_label
    return "unclear" if decision == "uncertain" else ""


def record_review(
    data: pd.DataFrame,
    row_index: int,
    *,
    decisions: dict[str, str],
    corrected_labels: dict[str, str],
    comments: dict[str, str],
    overall_comment: str,
    coder_id: str,
    coder_first_name: str,
    session_id: str,
) -> None:
    for variable in VARIABLES:
        model_label = str(data.loc[row_index, f"model_{variable}_label"])
        decision = decisions[variable]
        corrected = corrected_labels.get(variable, "")
        data.loc[row_index, f"model_review_{variable}_decision"] = decision
        data.loc[
            row_index, f"model_review_{variable}_corrected_label"
        ] = corrected if decision == "correct" else ""
        data.loc[row_index, f"model_review_{variable}_final_label"] = (
            final_review_label(model_label, decision, corrected)
        )
        data.loc[row_index, f"model_review_{variable}_comment"] = comments.get(
            variable, ""
        ).strip()
    data.loc[row_index, "model_review_overall_comment"] = overall_comment.strip()
    data.loc[row_index, "model_review_coder_id"] = safe_coder_id(coder_id)
    data.loc[row_index, "model_review_coder_first_name"] = coder_first_name.strip()
    data.loc[row_index, "model_review_session_id"] = session_id
    data.loc[row_index, "model_review_codebook_version"] = CODEBOOK_VERSION
    data.loc[row_index, "model_review_codebook_sha256"] = CODEBOOK_SHA256
    data.loc[row_index, "model_reviewed_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()


def reviewed_mask(data: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for variable in VARIABLES:
        decision_column = f"model_review_{variable}_decision"
        correction_column = f"model_review_{variable}_corrected_label"
        final_column = f"model_review_{variable}_final_label"
        model_column = f"model_{variable}_label"
        decision = data[decision_column].fillna("").astype(str)
        correction = data[correction_column].fillna("").astype(str)
        final = data[final_column].fillna("").astype(str)
        model = data[model_column].fillna("").astype(str)
        valid_labels = VARIABLES[variable]["labels"]
        mask &= decision.isin(REVIEW_DECISIONS)
        mask &= final.isin(valid_labels)
        mask &= (
            (decision.eq("confirm") & final.eq(model))
            | (
                decision.eq("correct")
                & correction.isin(valid_labels)
                & correction.ne(model)
                & final.eq(correction)
            )
            | (decision.eq("uncertain") & final.eq("unclear"))
        )
        needs_comment = decision.isin(["correct", "uncertain"])
        has_comment = (
            data[f"model_review_{variable}_comment"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        mask &= ~needs_comment | has_comment
    mask &= data["model_review_codebook_version"].fillna("").astype(str).eq(
        CODEBOOK_VERSION
    )
    mask &= data["model_review_codebook_sha256"].fillna("").astype(str).eq(
        CODEBOOK_SHA256
    )
    return mask


def review_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".model_review_manifest.json")


def _record_matches(observed: dict, path: Path) -> bool:
    current = file_record(path)
    return all(observed.get(key) == current[key] for key in ["size_bytes", "sha256"])


def load_review_data(
    input_path: Path,
    model_dir: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict | None, dict[str, Path], bool]:
    current, provenance, model_paths = build_review_data(input_path, model_dir)
    if not output_path.exists():
        return current, provenance, model_paths, False

    manifest_path = review_manifest_path(output_path)
    if not manifest_path.exists():
        raise ValueError(f"Existing model-review output has no manifest: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _record_matches(manifest.get("input", {}), input_path):
        raise ValueError("The review input changed after this review file was created.")
    for variable, path in model_paths.items():
        if not _record_matches(manifest.get("model_outputs", {}).get(variable, {}), path):
            raise ValueError(
                f"The {variable} model output changed after review began. "
                "Use a new model-review output file."
            )

    saved = ensure_review_columns(read_csv(output_path))
    if len(saved) != len(current) or not saved["article_id"].astype(str).equals(
        current["article_id"].astype(str)
    ):
        raise ValueError("Existing model-review output does not match the input rows.")
    for variable in VARIABLES:
        column = f"model_{variable}_label"
        if not saved[column].astype(str).equals(current[column].astype(str)):
            raise ValueError(f"Saved {variable} model labels no longer match.")
    return saved, provenance, model_paths, True


def save_review_data(
    data: pd.DataFrame,
    *,
    input_path: Path,
    model_paths: dict[str, Path],
    output_path: Path,
    coder_id: str,
    coder_first_name: str,
    session_id: str,
    provenance: dict | None,
) -> Path:
    write_csv_atomic(data, output_path)
    reviewed = reviewed_mask(data)
    model_names = sorted(
        {
            str(data[f"model_{variable}_llm_model"].iloc[0])
            for variable in VARIABLES
        }
    )
    decisions = {
        variable: (
            data.loc[reviewed, f"model_review_{variable}_decision"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        )
        for variable in VARIABLES
    }
    manifest = {
        "schema_version": 1,
        "review_mode": "model_assisted_adjudication",
        "independent_human_validation": False,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "coder_id": safe_coder_id(coder_id),
        "coder_first_name": coder_first_name.strip(),
        "session_id": session_id,
        "models": model_names,
        "codebook": {
            "version": CODEBOOK_VERSION,
            "sha256": CODEBOOK_SHA256,
            "file": file_record(CODEBOOK_PATH),
        },
        "input": file_record(input_path),
        "input_sample_manifest": (
            file_record(provenance["manifest_path"]) if provenance else None
        ),
        "model_outputs": {
            variable: file_record(path) for variable, path in model_paths.items()
        },
        "output": file_record(output_path),
        "rows": len(data),
        "reviewed_rows": int(reviewed.sum()),
        "decisions_on_reviewed_rows": decisions,
    }
    manifest_path = review_manifest_path(output_path)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path
