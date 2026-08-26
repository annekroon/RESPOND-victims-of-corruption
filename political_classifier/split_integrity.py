"""Utilities that keep classifier development and human evaluation disjoint."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


def normalize_identifier(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "<na>"} else text


def normalize_text_for_overlap(value: object) -> str:
    text = normalize_identifier(value).casefold().replace("\u00a0", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def text_overlap_hash(value: object) -> str:
    normalized = normalize_text_for_overlap(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""


def add_integrity_keys(data, text_column: str = "model_text"):
    keyed = data.copy()
    if "uri" in keyed.columns:
        keyed["_integrity_uri"] = keyed["uri"].map(normalize_identifier)
    else:
        keyed["_integrity_uri"] = ""
    keyed["_integrity_text_hash"] = keyed[text_column].map(text_overlap_hash)
    return keyed


def overlap_rows(training, validation, text_column: str = "model_text"):
    import pandas as pd

    train = add_integrity_keys(training, text_column=text_column)
    valid = add_integrity_keys(validation, text_column=text_column)
    validation_uris = set(valid.loc[valid["_integrity_uri"].ne(""), "_integrity_uri"])
    validation_hashes = set(
        valid.loc[valid["_integrity_text_hash"].ne(""), "_integrity_text_hash"]
    )
    uri_overlap = train["_integrity_uri"].ne("") & train["_integrity_uri"].isin(validation_uris)
    text_overlap = train["_integrity_text_hash"].ne("") & train["_integrity_text_hash"].isin(
        validation_hashes
    )
    overlap = train[uri_overlap | text_overlap].copy()
    overlap["overlap_by_uri"] = uri_overlap[uri_overlap | text_overlap].to_numpy()
    overlap["overlap_by_normalized_text"] = text_overlap[uri_overlap | text_overlap].to_numpy()
    return overlap


def remove_validation_overlap(
    training,
    validation,
    *,
    text_column: str = "model_text",
    audit_path: Path | None = None,
):
    overlap = overlap_rows(training, validation, text_column=text_column)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        overlap.to_csv(audit_path, index=False)
    if overlap.empty:
        return training.copy(), overlap

    overlap_indices = set(overlap.index)
    filtered = training.loc[~training.index.isin(overlap_indices)].copy()
    return filtered, overlap


def assert_no_validation_overlap(
    training,
    validation,
    *,
    text_column: str = "model_text",
    audit_path: Path | None = None,
) -> None:
    overlap = overlap_rows(training, validation, text_column=text_column)
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        overlap.to_csv(audit_path, index=False)
    if not overlap.empty:
        uri_n = int(overlap["overlap_by_uri"].sum())
        text_n = int(overlap["overlap_by_normalized_text"].sum())
        raise ValueError(
            "Silver training data overlap the human benchmark: "
            f"{len(overlap):,} training row(s), including {uri_n:,} URI and "
            f"{text_n:,} normalized-text match(es). Recreate the training sample "
            "with benchmark exclusions before evaluating."
        )


def calibration_test_split(validation, calibration_fraction: float, random_state: int):
    """Return disjoint country/label-stratified threshold and test partitions."""
    if not 0.1 <= calibration_fraction <= 0.8:
        raise ValueError("calibration_fraction must be between 0.1 and 0.8.")
    strata = validation["country"].astype(str) + "__" + validation["y"].astype(str)
    counts = strata.value_counts()
    rare = set(counts[counts < 2].index)
    if rare:
        raise ValueError(
            "Cannot create a country/label-stratified threshold split because "
            f"these strata have fewer than two rows: {sorted(rare)}"
        )
    calibration_indices = []
    for stratum_number, (_, group) in enumerate(validation.groupby(strata, sort=True)):
        calibration_n = round(len(group) * calibration_fraction)
        calibration_n = min(len(group) - 1, max(1, calibration_n))
        sampled = group.sample(
            n=calibration_n,
            random_state=random_state + stratum_number,
        )
        calibration_indices.extend(sampled.index.tolist())
    calibration_index = validation.index[validation.index.isin(calibration_indices)]
    test_index = validation.index[~validation.index.isin(calibration_indices)]
    calibration = validation.loc[calibration_index].copy()
    test = validation.loc[test_index].copy()
    return calibration, test
