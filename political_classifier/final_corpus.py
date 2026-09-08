"""Verify downstream inputs against the completed political-corruption corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from political_classifier.reproducibility import sha256_file


FINAL_SOURCE_FILTER_POLICY = "exclude_explicit_no_retain_all_other_decisions"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def assert_file_hash(path: Path, expected: str | None, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(
            f"{label} differs from its recorded manifest hash: {path}. "
            "Rebuild the upstream step before continuing."
        )


@dataclass(frozen=True)
class ClassifierRun:
    output_dir: Path
    classified_dir: Path
    manifest_path: Path
    summary_path: Path
    threshold_path: Path
    source_filter_manifest_path: Path
    manifest: dict[str, Any]
    source_filter_manifest: dict[str, Any]
    summary: Any
    threshold: float
    manifest_sha256: str
    summary_sha256: str
    source_filter_manifest_sha256: str
    total_articles: int
    political_articles: int

    def provenance_record(self) -> dict[str, Any]:
        source_record = self.manifest.get("source_decision_file") or {}
        return {
            "verified_final_classifier": True,
            "classifier_git_commit": self.manifest.get("git_commit"),
            "classifier_manifest_path": str(self.manifest_path),
            "classifier_manifest_sha256": self.manifest_sha256,
            "classified_summary_path": str(self.summary_path),
            "classified_summary_sha256": self.summary_sha256,
            "source_decision_sha256": source_record.get("sha256"),
            "source_filter_manifest_path": str(self.source_filter_manifest_path),
            "source_filter_manifest_sha256": self.source_filter_manifest_sha256,
            "source_filter_policy": self.source_filter_manifest.get(
                "source_filter_policy"
            ),
            "embedding_model": self.manifest.get("embedding_model"),
            "embedding_model_revision": self.manifest.get(
                "embedding_model_revision"
            ),
            "threshold": self.threshold,
            "source_filtered_articles": self.total_articles,
            "political_corruption_articles": self.political_articles,
        }

    def expected_political_articles(self, countries: list[str]) -> int:
        selected = self.summary[
            self.summary["country"].astype(str).isin(countries)
        ]
        found = set(selected["country"].astype(str))
        missing = set(countries) - found
        if missing:
            raise ValueError(
                "Countries are absent from the final classifier summary: "
                + ", ".join(sorted(missing))
            )
        return int(selected["predicted_political_corruption"].sum())


def load_verified_classifier_run(
    output_dir: Path,
    countries: list[str],
    *,
    classified_dir: Path | None = None,
    require_local_country_files: bool = True,
    require_exact_countries: bool = True,
) -> ClassifierRun:
    """Load and validate the production classifier's completion artifacts."""
    import pandas as pd

    output_dir = output_dir.resolve()
    classified_dir = (
        classified_dir or output_dir / "classified_country_files"
    ).resolve()
    manifest_path = output_dir / "classifier_run_manifest.json"
    summary_path = output_dir / "classified_country_summary.csv"
    threshold_path = output_dir / "selected_threshold.txt"
    source_filter_manifest_path = (
        output_dir.parent / "source_inclusion" / "source_filter_run_manifest.json"
    )

    manifest = read_json(manifest_path)
    if "classified_country_summary" not in manifest:
        raise ValueError(
            "The classifier manifest has no classified-country summary. "
            "Step 06 did not finish a complete corpus-scoring run."
        )
    assert_file_hash(
        summary_path,
        manifest["classified_country_summary"].get("sha256"),
        "Classified-country summary",
    )
    if not threshold_path.exists():
        raise FileNotFoundError(threshold_path)
    threshold = float(threshold_path.read_text(encoding="utf-8").strip())
    manifest_threshold = float(manifest.get("threshold"))
    if abs(threshold - manifest_threshold) > 1e-12:
        raise ValueError(
            "Classifier threshold file and classifier manifest disagree: "
            f"{threshold} versus {manifest_threshold}."
        )
    if not manifest.get("source_decision_file"):
        raise ValueError(
            "The classifier manifest does not record a source-decision workbook; "
            "downstream analysis must use the source-screened production run."
        )

    source_filter_manifest = read_json(source_filter_manifest_path)
    if source_filter_manifest.get("source_filter_policy") != FINAL_SOURCE_FILTER_POLICY:
        raise ValueError(
            "The source-filter manifest does not use the final explicit-exclusion "
            f"policy: expected {FINAL_SOURCE_FILTER_POLICY!r}."
        )
    classifier_source_hash = (manifest.get("source_decision_file") or {}).get(
        "sha256"
    )
    filter_source_hash = (
        source_filter_manifest.get("source_decision_file") or {}
    ).get("sha256")
    if not classifier_source_hash or classifier_source_hash != filter_source_hash:
        raise ValueError(
            "The classifier and source-filter manifests use different "
            "source-decision workbooks."
        )
    corpus_dir = Path(str(manifest.get("corpus_dir", "")))
    if corpus_dir.name != "cleaned_deduped_source_filtered":
        raise ValueError(
            "The classifier manifest does not point to the canonical source-filtered "
            f"corpus directory: {corpus_dir}."
        )

    summary = pd.read_csv(summary_path)
    required = {
        "country",
        "total_articles",
        "predicted_political_corruption",
        "predicted_rate",
    }
    missing_columns = required - set(summary.columns)
    if missing_columns:
        raise ValueError(
            "Classified-country summary is missing columns: "
            f"{sorted(missing_columns)}"
        )
    if summary["country"].duplicated().any():
        raise ValueError("Classified-country summary contains duplicate countries.")
    expected_countries = set(countries)
    actual_countries = set(summary["country"].astype(str))
    countries_match = (
        actual_countries == expected_countries
        if require_exact_countries
        else expected_countries.issubset(actual_countries)
    )
    if not countries_match:
        raise ValueError(
            "Classifier summary does not contain the required countries. "
            f"Required {sorted(expected_countries)}, found {sorted(actual_countries)}."
        )
    summary["total_articles"] = pd.to_numeric(
        summary["total_articles"], errors="raise"
    ).astype(int)
    summary["predicted_political_corruption"] = pd.to_numeric(
        summary["predicted_political_corruption"], errors="raise"
    ).astype(int)
    if (summary["total_articles"] <= 0).any():
        raise ValueError("Every classifier country must contain at least one article.")
    if (summary["predicted_political_corruption"] < 0).any() or (
        summary["predicted_political_corruption"] > summary["total_articles"]
    ).any():
        raise ValueError("Classifier country totals contain impossible positive counts.")

    source_summary_record = source_filter_manifest.get("output_summary") or {}
    source_summary_path = Path(str(source_summary_record.get("path", "")))
    if not source_summary_path.is_absolute():
        source_summary_path = (
            source_filter_manifest_path.parent / source_summary_path.name
        )
    assert_file_hash(
        source_summary_path,
        source_summary_record.get("sha256"),
        "Source-filter output summary",
    )
    source_summary = pd.read_csv(source_summary_path)
    if not {"country", "output_rows"}.issubset(source_summary.columns):
        raise ValueError(
            "Source-filter output summary is missing country or output_rows."
        )
    classifier_totals = summary.set_index("country")["total_articles"].astype(int)
    source_totals = source_summary.set_index("country")["output_rows"].astype(int)
    if not classifier_totals.sort_index().equals(source_totals.sort_index()):
        raise ValueError(
            "Classifier country totals differ from the verified source-filter corpus."
        )

    if require_local_country_files:
        missing_files = [
            classified_dir / f"{country}_classified.csv.gz"
            for country in countries
            if not (classified_dir / f"{country}_classified.csv.gz").exists()
        ]
        if missing_files:
            raise FileNotFoundError(
                "Missing classified country files: "
                + ", ".join(str(path) for path in missing_files)
            )

    return ClassifierRun(
        output_dir=output_dir,
        classified_dir=classified_dir,
        manifest_path=manifest_path,
        summary_path=summary_path,
        threshold_path=threshold_path,
        source_filter_manifest_path=source_filter_manifest_path,
        manifest=manifest,
        source_filter_manifest=source_filter_manifest,
        summary=summary,
        threshold=threshold,
        manifest_sha256=sha256_file(manifest_path),
        summary_sha256=sha256_file(summary_path),
        source_filter_manifest_sha256=sha256_file(source_filter_manifest_path),
        total_articles=int(summary["total_articles"].sum()),
        political_articles=int(summary["predicted_political_corruption"].sum()),
    )


def verify_classified_country_frame(
    frame,
    *,
    country: str,
    classifier_run: ClassifierRun,
) -> None:
    """Verify one loaded classifier output against the completion summary."""
    import pandas as pd

    expected_rows = classifier_run.summary.loc[
        classifier_run.summary["country"].astype(str).eq(country)
    ]
    if expected_rows.empty:
        raise ValueError(f"{country} is absent from the classifier summary.")
    expected = expected_rows.iloc[0]
    required = {"pred_political_corruption", "prob_political_corruption"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{country} classified file is missing: {sorted(missing)}")

    predictions = pd.to_numeric(
        frame["pred_political_corruption"], errors="raise"
    ).astype(int)
    probabilities = pd.to_numeric(
        frame["prob_political_corruption"], errors="raise"
    )
    if probabilities.isna().any() or not probabilities.between(0, 1).all():
        raise ValueError(f"{country} contains invalid classifier probabilities.")
    if not predictions.isin([0, 1]).all():
        raise ValueError(f"{country} contains non-binary classifier predictions.")
    if len(frame) != int(expected["total_articles"]):
        raise ValueError(
            f"{country} row count differs from classifier summary: "
            f"{len(frame):,} versus {int(expected['total_articles']):,}."
        )
    if int(predictions.sum()) != int(expected["predicted_political_corruption"]):
        raise ValueError(
            f"{country} positive count differs from classifier summary: "
            f"{int(predictions.sum()):,} versus "
            f"{int(expected['predicted_political_corruption']):,}."
        )
    threshold_predictions = probabilities.ge(classifier_run.threshold).astype(int)
    threshold_predictions = threshold_predictions.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    if not threshold_predictions.equals(predictions):
        mismatches = int(threshold_predictions.ne(predictions).sum())
        raise ValueError(
            f"{country} contains {mismatches:,} predictions inconsistent with "
            f"threshold {classifier_run.threshold:.2f}."
        )


def verify_classified_country_file(
    path: Path,
    *,
    country: str,
    classifier_run: ClassifierRun,
    chunksize: int = 100_000,
) -> None:
    """Stream-verify a country file before an expensive downstream run."""
    import pandas as pd

    expected_rows = classifier_run.summary.loc[
        classifier_run.summary["country"].astype(str).eq(country)
    ]
    if expected_rows.empty:
        raise ValueError(f"{country} is absent from the classifier summary.")
    expected = expected_rows.iloc[0]
    row_count = 0
    positive_count = 0
    threshold_mismatches = 0
    for chunk in pd.read_csv(
        path,
        usecols=["prob_political_corruption", "pred_political_corruption"],
        chunksize=chunksize,
    ):
        predictions = pd.to_numeric(
            chunk["pred_political_corruption"], errors="raise"
        ).astype(int)
        probabilities = pd.to_numeric(
            chunk["prob_political_corruption"], errors="raise"
        )
        if probabilities.isna().any() or not probabilities.between(0, 1).all():
            raise ValueError(f"{country} contains invalid classifier probabilities.")
        if not predictions.isin([0, 1]).all():
            raise ValueError(f"{country} contains non-binary classifier predictions.")
        row_count += len(chunk)
        positive_count += int(predictions.sum())
        threshold_mismatches += int(
            predictions.ne(probabilities.ge(classifier_run.threshold).astype(int)).sum()
        )
    if row_count == 0:
        raise ValueError(f"{country} classified file is empty: {path}")
    if row_count != int(expected["total_articles"]):
        raise ValueError(
            f"{country} row count differs from classifier summary: "
            f"{row_count:,} versus {int(expected['total_articles']):,}."
        )
    if positive_count != int(expected["predicted_political_corruption"]):
        raise ValueError(
            f"{country} positive count differs from classifier summary: "
            f"{positive_count:,} versus "
            f"{int(expected['predicted_political_corruption']):,}."
        )
    if threshold_mismatches:
        raise ValueError(
            f"{country} contains {threshold_mismatches:,} predictions inconsistent "
            f"with threshold {classifier_run.threshold:.2f}."
        )
