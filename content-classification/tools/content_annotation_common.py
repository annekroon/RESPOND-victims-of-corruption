"""Shared persistence and provenance helpers for content annotation apps."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "content-classification" / "scripts"
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd

from content_codebook import CODEBOOK_PATH, CODEBOOK_SHA256, CODEBOOK_VERSION
from content_classifier_common import validate_content_sample
from political_classifier.reproducibility import file_record, git_commit


HUMAN_COLUMNS = [
    "human_victim_visibility",
    "human_victim_evidence",
    "human_corruption_frame",
    "human_corruption_frame_evidence",
    "human_case_location",
    "human_abroad_case",
    "human_accused_actor_visibility",
    "human_accused_actor_visible",
    "human_accused_individual_evidence",
    "human_accused_organization_evidence",
    "human_notes",
    "human_coder_id",
    "human_coder_first_name",
    "human_code_session_id",
    "human_codebook_version",
    "human_codebook_sha256",
    "human_coded_at",
]

REQUIRED_HUMAN_COLUMNS = [
    "human_victim_visibility",
    "human_corruption_frame",
    "human_case_location",
    "human_accused_actor_visibility",
]

EVIDENCE_COLUMNS = [
    "human_victim_evidence",
    "human_corruption_frame_evidence",
    "human_accused_individual_evidence",
    "human_accused_organization_evidence",
]


def scalar_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        compression="gzip" if path.name.endswith(".gz") else "infer",
    )


def write_csv_atomic(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data.to_csv(
        temporary,
        index=False,
        compression="gzip" if path.name.endswith(".gz") else None,
    )
    temporary.replace(path)


def annotation_manifest_path(path: Path) -> Path:
    return path.with_name(path.name + ".annotation_manifest.json")


def safe_coder_id(coder_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", coder_id.strip())
    return normalized.strip("_") or "coder"


def default_output_template(input_path: Path) -> str:
    name = input_path.name
    if name.endswith(".csv.gz"):
        name = name[: -len(".csv.gz")] + "_{coder_id}.csv.gz"
    elif name.endswith(".csv"):
        name = name[: -len(".csv")] + "_{coder_id}.csv"
    else:
        name = name + "_{coder_id}.csv.gz"
    return str(input_path.with_name(name))


def output_path_for_coder(
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
        default_output_template(input_path).format(coder_id=safe_coder_id(coder_id))
    )


def ensure_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "article_id" not in data.columns:
        data["article_id"] = (
            data["uri"].fillna("").astype(str)
            if "uri" in data.columns
            else ""
        )
    missing_id = data["article_id"].fillna("").astype(str).str.strip().eq("")
    if missing_id.any():
        fallback = pd.Series(data.index.astype(str), index=data.index)
        if "country" in data.columns:
            fallback = data["country"].fillna("").astype(str) + "::" + fallback
        data.loc[missing_id, "article_id"] = fallback.loc[missing_id]

    if data["article_id"].astype(str).duplicated().any():
        raise ValueError("Annotation input contains duplicate article IDs.")

    for column in HUMAN_COLUMNS:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("")
    return data


def evidence_passages(value: object) -> list[str]:
    """Return one normalized evidence passage per nonblank line."""
    passages = []
    for line in scalar_text(value).splitlines():
        passage = re.sub(r"^\s*[-*]\s+", "", line).strip().strip("\"'")
        if passage:
            passages.append(" ".join(passage.split()))
    return passages


def evidence_is_verbatim(value: object, row: pd.Series) -> bool:
    passages = evidence_passages(value)
    if not passages:
        return False
    article_versions = [
        " ".join(scalar_text(row.get(column, "")).split()).casefold()
        for column in ["translated_text_en", "translated_text", "article_text"]
        if scalar_text(row.get(column, "")).strip()
    ]
    return bool(article_versions) and all(
        any(passage.casefold() in article for article in article_versions)
        for passage in passages
    )


def annotation_evidence_errors(row: pd.Series) -> list[str]:
    """Validate evidence required by the selected substantive labels."""
    required = []
    if str(row.get("human_victim_visibility", "")) in {
        "concrete_victim",
        "institutional_societal_victim",
    }:
        required.append(("human_victim_evidence", "victim evidence"))
    if str(row.get("human_corruption_frame", "")) not in {"", "unclear"}:
        required.append(
            ("human_corruption_frame_evidence", "corruption-frame evidence")
        )
    actor = str(row.get("human_accused_actor_visibility", ""))
    if actor in {"individual_actor", "both_individual_and_organizational"}:
        required.append(
            ("human_accused_individual_evidence", "individual-actor evidence")
        )
    if actor in {
        "organizational_or_institutional_actor",
        "both_individual_and_organizational",
    }:
        required.append(
            ("human_accused_organization_evidence", "organizational-actor evidence")
        )

    errors = []
    for column, label in required:
        if not evidence_passages(row.get(column, "")):
            errors.append(f"Add {label}.")
    for column in EVIDENCE_COLUMNS:
        value = row.get(column, "")
        if evidence_passages(value) and not evidence_is_verbatim(value, row):
            label = column.removeprefix("human_").replace("_", " ")
            errors.append(
                f"The {label} must be copied exactly from the English or original article."
            )
    return errors


def reviewed_mask(data: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=data.index)
    for column in REQUIRED_HUMAN_COLUMNS:
        mask &= data[column].fillna("").astype(str).str.strip().ne("")
    mask &= data["human_codebook_version"].fillna("").astype(str).eq(
        CODEBOOK_VERSION
    )
    mask &= data["human_codebook_sha256"].fillna("").astype(str).eq(
        CODEBOOK_SHA256
    )
    evidence_valid = data.apply(
        lambda row: not annotation_evidence_errors(row),
        axis=1,
    )
    mask &= evidence_valid
    return mask


def derive_abroad_case(case_location: str) -> str:
    return {"domestic": "no", "abroad": "yes", "unclear": "unclear"}.get(
        case_location,
        "",
    )


def derive_accused_actor_visible(accused_actor_visibility: str) -> str:
    if accused_actor_visibility == "no_accused_actor":
        return "no"
    if accused_actor_visibility in {
        "individual_actor",
        "organizational_or_institutional_actor",
        "both_individual_and_organizational",
    }:
        return "yes"
    if accused_actor_visibility == "unclear":
        return "unclear"
    return ""


def load_annotation_data(
    input_path: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict | None, bool]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    provenance = validate_content_sample(input_path)
    source = ensure_columns(read_csv(input_path))
    purposes = set(
        source.get("sample_purpose", pd.Series(dtype=str)).dropna().astype(str)
    )
    if purposes == {"final_validation"} and provenance is None:
        raise ValueError(
            "Final-validation annotation input has no verified sample manifest."
        )
    if not output_path.exists():
        return source, provenance, False

    saved = ensure_columns(read_csv(output_path))
    if len(saved) != len(source) or not saved["article_id"].astype(str).equals(
        source["article_id"].astype(str)
    ):
        raise ValueError(
            "Existing coder output does not match the current annotation input: "
            f"{output_path}"
        )
    return saved, provenance, True


def save_annotation_data(
    data: pd.DataFrame,
    *,
    input_path: Path,
    output_path: Path,
    coder_id: str,
    coder_first_name: str,
    code_session_id: str,
    provenance: dict | None,
) -> Path:
    write_csv_atomic(data, output_path)
    reviewed = reviewed_mask(data)
    row_versions = sorted(
        set(
            data.loc[reviewed, "human_codebook_version"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
        - {""}
    )
    unversioned_reviewed_rows = int(
        data.loc[reviewed, "human_codebook_version"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    manifest = {
        "schema_version": 2,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(PROJECT_ROOT),
        "coder_id": safe_coder_id(coder_id),
        "coder_first_name": coder_first_name.strip(),
        "code_session_id": code_session_id,
        "codebook": {
            "version": CODEBOOK_VERSION,
            "sha256": CODEBOOK_SHA256,
            "file": file_record(CODEBOOK_PATH),
            "row_versions": row_versions,
            "unversioned_reviewed_rows": unversioned_reviewed_rows,
        },
        "input": file_record(input_path),
        "input_sample_manifest": (
            file_record(provenance["manifest_path"]) if provenance else None
        ),
        "output": file_record(output_path),
        "rows": len(data),
        "reviewed_rows": int(reviewed.sum()),
    }
    manifest_path = annotation_manifest_path(output_path)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


def record_annotation(
    data: pd.DataFrame,
    row_index: int,
    *,
    victim_visibility: str,
    victim_evidence: str,
    corruption_frame: str,
    corruption_frame_evidence: str,
    case_location: str,
    accused_actor_visibility: str,
    accused_individual_evidence: str,
    accused_organization_evidence: str,
    notes: str,
    coder_id: str,
    coder_first_name: str,
    code_session_id: str,
) -> None:
    data.loc[row_index, "human_victim_visibility"] = victim_visibility
    data.loc[row_index, "human_victim_evidence"] = victim_evidence.strip()
    data.loc[row_index, "human_corruption_frame"] = corruption_frame
    data.loc[row_index, "human_corruption_frame_evidence"] = (
        corruption_frame_evidence.strip()
    )
    data.loc[row_index, "human_case_location"] = case_location
    data.loc[row_index, "human_abroad_case"] = derive_abroad_case(case_location)
    data.loc[row_index, "human_accused_actor_visibility"] = (
        accused_actor_visibility
    )
    data.loc[row_index, "human_accused_actor_visible"] = (
        derive_accused_actor_visible(accused_actor_visibility)
    )
    data.loc[row_index, "human_accused_individual_evidence"] = (
        accused_individual_evidence.strip()
    )
    data.loc[row_index, "human_accused_organization_evidence"] = (
        accused_organization_evidence.strip()
    )
    data.loc[row_index, "human_notes"] = notes
    data.loc[row_index, "human_coder_id"] = safe_coder_id(coder_id)
    data.loc[row_index, "human_coder_first_name"] = coder_first_name.strip()
    data.loc[row_index, "human_code_session_id"] = code_session_id
    data.loc[row_index, "human_codebook_version"] = CODEBOOK_VERSION
    data.loc[row_index, "human_codebook_sha256"] = CODEBOOK_SHA256
    data.loc[row_index, "human_coded_at"] = datetime.now(timezone.utc).isoformat()
