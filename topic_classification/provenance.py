"""Integrity checks linking topic outputs to the final classifier run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def load_verified_classifier_run(
    output_dir: Path,
    countries: list[str],
    *,
    classified_dir: Path | None = None,
    require_local_country_files: bool = True,
) -> ClassifierRun:
    """Load and validate the production classifier's completion artifacts."""
    import pandas as pd

    output_dir = output_dir.resolve()
    classified_dir = (classified_dir or output_dir / "classified_country_files").resolve()
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
            "topic modelling must use the source-screened production run."
        )
    source_filter_manifest = read_json(source_filter_manifest_path)
    expected_policy = "exclude_explicit_no_retain_all_other_decisions"
    if source_filter_manifest.get("source_filter_policy") != expected_policy:
        raise ValueError(
            "The source-filter manifest does not use the final explicit-exclusion "
            f"policy: expected {expected_policy!r}."
        )
    classifier_source_hash = (manifest.get("source_decision_file") or {}).get(
        "sha256"
    )
    filter_source_hash = (
        source_filter_manifest.get("source_decision_file") or {}
    ).get("sha256")
    if not classifier_source_hash or classifier_source_hash != filter_source_hash:
        raise ValueError(
            "The classifier and source-filter manifests use different source-decision "
            "workbooks."
        )
    corpus_dir = Path(str(manifest.get("corpus_dir", "")))
    if corpus_dir.name != "cleaned_deduped_source_filtered":
        raise ValueError(
            "The classifier manifest does not point to the canonical source-filtered "
            f"corpus directory: {corpus_dir}."
        )

    summary = pd.read_csv(summary_path)
    source_summary_record = source_filter_manifest.get("output_summary") or {}
    source_summary_path = Path(str(source_summary_record.get("path", "")))
    if not source_summary_path.is_absolute():
        source_summary_path = source_filter_manifest_path.parent / source_summary_path.name
    assert_file_hash(
        source_summary_path,
        source_summary_record.get("sha256"),
        "Source-filter output summary",
    )
    source_summary = pd.read_csv(source_summary_path)
    required = {
        "country",
        "total_articles",
        "predicted_political_corruption",
        "predicted_rate",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            f"Classified-country summary is missing columns: {sorted(missing)}"
        )
    if summary["country"].duplicated().any():
        raise ValueError("Classified-country summary contains duplicate countries.")
    expected_countries = set(countries)
    actual_countries = set(summary["country"].astype(str))
    if actual_countries != expected_countries:
        raise ValueError(
            "Classifier summary does not contain exactly the requested countries. "
            f"Expected {sorted(expected_countries)}, found {sorted(actual_countries)}."
        )
    summary["total_articles"] = pd.to_numeric(
        summary["total_articles"], errors="raise"
    ).astype(int)
    summary["predicted_political_corruption"] = pd.to_numeric(
        summary["predicted_political_corruption"], errors="raise"
    ).astype(int)
    if (summary["total_articles"] <= 0).any():
        raise ValueError("Every classifier country must contain at least one article.")
    if (
        summary["predicted_political_corruption"] < 0
    ).any() or (
        summary["predicted_political_corruption"] > summary["total_articles"]
    ).any():
        raise ValueError("Classifier country totals contain impossible positive counts.")
    required_source_columns = {"country", "output_rows"}
    if not required_source_columns.issubset(source_summary.columns):
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

    expected = classifier_run.summary.loc[
        classifier_run.summary["country"].astype(str).eq(country)
    ].iloc[0]
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
    threshold_predictions = (
        probabilities.ge(classifier_run.threshold).astype(int).reset_index(drop=True)
    )
    predictions = predictions.reset_index(drop=True)
    if not threshold_predictions.equals(predictions):
        mismatches = int(
            threshold_predictions.ne(predictions).sum()
        )
        raise ValueError(
            f"{country} contains {mismatches:,} predictions inconsistent with "
            f"threshold {classifier_run.threshold:.2f}."
        )


def sample_manifest_path(sample_path: Path) -> Path:
    suffix = "_run_manifest.json"
    if sample_path.name.endswith(".csv.gz"):
        return sample_path.with_name(sample_path.name[: -len(".csv.gz")] + suffix)
    return sample_path.with_name(sample_path.name + suffix)


def validate_verified_topic_sample(sample_path: Path) -> dict[str, Any]:
    """Require a sample manifest tied to a verified final classifier run."""
    manifest_path = sample_manifest_path(sample_path)
    manifest = read_json(manifest_path)
    sample_record = (manifest.get("outputs") or {}).get("sample") or {}
    assert_file_hash(sample_path, sample_record.get("sha256"), "Topic sample")
    extra = manifest.get("extra") or {}
    upstream = extra.get("upstream_classifier") or {}
    if not upstream.get("verified_final_classifier"):
        raise ValueError(
            "Topic sample is not tied to a verified final classifier run. "
            "Recreate it with 01_create_stratified_topic_sample.py."
        )
    if not extra.get("political_only"):
        raise ValueError("Final topic modelling requires a political-only sample.")
    if int(extra.get("saved_sample_rows", -1)) <= 0:
        raise ValueError("Topic sample manifest records no sampled rows.")

    abstraction = extra.get("topic_abstraction") or {}
    if abstraction:
        inputs = manifest.get("inputs") or {}
        source_record = inputs.get("source_sample") or {}
        source_path = Path(str(source_record.get("path", "")))
        if not source_path.is_absolute():
            source_path = sample_path.parent / source_path.name
        assert_file_hash(
            source_path,
            source_record.get("sha256"),
            "Topic abstraction source sample",
        )
        source_provenance = validate_verified_topic_sample(source_path)
        source_manifest_record = inputs.get("source_sample_manifest") or {}
        recorded_source_manifest_hash = source_manifest_record.get("sha256")
        expected_source_manifest_hash = source_provenance["manifest_sha256"]
        if recorded_source_manifest_hash != expected_source_manifest_hash:
            raise ValueError(
                "Topic abstractions belong to a different source-sample manifest. "
                "Rerun 02_create_descriptive_abstractions.py."
            )
        if (
            abstraction.get("source_sample_manifest_sha256")
            != expected_source_manifest_hash
        ):
            raise ValueError(
                "Topic abstraction metadata does not match its source sample. "
                "Rerun 02_create_descriptive_abstractions.py."
            )
        source_classifier_hash = source_provenance["upstream_classifier"].get(
            "classifier_manifest_sha256"
        )
        abstraction_classifier_hash = upstream.get("classifier_manifest_sha256")
        if abstraction_classifier_hash != source_classifier_hash:
            raise ValueError(
                "Topic abstractions and their source sample point to different "
                "classifier runs."
            )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "upstream_classifier": upstream,
    }


def validate_topic_model_outputs(bertopic_dir: Path) -> dict[str, Any]:
    """Validate the BERTopic model outputs against their recorded input sample."""
    manifest_path = bertopic_dir / "run_manifest.json"
    manifest = read_json(manifest_path)
    inputs = manifest.get("inputs") or {}
    sample_record = inputs.get("sample") or {}
    sample_path = Path(str(sample_record.get("path", "")))
    if not sample_path.is_absolute():
        sample_path = bertopic_dir.parent / sample_path.name
    assert_file_hash(sample_path, sample_record.get("sha256"), "BERTopic sample")
    sample_provenance = validate_verified_topic_sample(sample_path)
    recorded_sample_manifest = inputs.get("sample_manifest") or {}
    if recorded_sample_manifest.get("sha256") != sample_provenance["manifest_sha256"]:
        raise ValueError(
            "BERTopic model belongs to a different topic sample manifest. "
            "Rerun 03_fit_descriptive_bertopic.py with --overwrite."
        )
    for name in ["topic_info", "document_topics"]:
        record = (manifest.get("outputs") or {}).get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = bertopic_dir / path.name
        assert_file_hash(path, record.get("sha256"), f"BERTopic {name}")
    extra = manifest.get("extra") or {}
    if not (extra.get("upstream_classifier") or {}).get(
        "verified_final_classifier"
    ):
        raise ValueError("BERTopic run is not linked to the final classifier run.")
    if (
        (extra.get("upstream_classifier") or {}).get("classifier_manifest_sha256")
        != sample_provenance["upstream_classifier"].get(
            "classifier_manifest_sha256"
        )
    ):
        raise ValueError(
            "BERTopic and its saved sample point to different classifier runs."
        )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "upstream_classifier": extra["upstream_classifier"],
    }


def validate_topic_label_outputs(bertopic_dir: Path) -> dict[str, Any]:
    """Validate completed GPT labels against the current BERTopic run."""
    model = validate_topic_model_outputs(bertopic_dir)
    manifest_path = bertopic_dir / "topic_labels_run_manifest.json"
    manifest = read_json(manifest_path)
    inputs = manifest.get("inputs") or {}
    recorded_model = inputs.get("topic_model_manifest") or {}
    if recorded_model.get("sha256") != model["manifest_sha256"]:
        raise ValueError(
            "Topic-label manifest belongs to a different BERTopic run. "
            "Rerun 04_label_descriptive_topics_with_llm.py with --overwrite."
        )
    outputs = manifest.get("outputs") or {}
    for name in ["topic_labels", "topic_label_audit"]:
        record = outputs.get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = bertopic_dir / path.name
        assert_file_hash(path, record.get("sha256"), f"Topic-label {name}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "model": model,
        "upstream_classifier": model["upstream_classifier"],
    }


def validate_topic_group_outputs(bertopic_dir: Path) -> dict[str, Any]:
    """Validate higher-order assignments against the current GPT labels."""
    labels = validate_topic_label_outputs(bertopic_dir)
    manifest_path = bertopic_dir / "topic_groups_run_manifest.json"
    manifest = read_json(manifest_path)
    inputs = manifest.get("inputs") or {}
    recorded_labels = inputs.get("topic_labels_manifest") or {}
    if recorded_labels.get("sha256") != labels["manifest_sha256"]:
        raise ValueError(
            "Topic-group manifest belongs to different topic labels. "
            "Rerun 04_group_topics_with_llm.py with --overwrite."
        )
    outputs = manifest.get("outputs") or {}
    for name in ["topic_groups", "topic_group_summaries", "topic_group_audit"]:
        record = outputs.get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = bertopic_dir / path.name
        assert_file_hash(path, record.get("sha256"), f"Topic-group {name}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "labels": labels,
        "upstream_classifier": labels["upstream_classifier"],
    }


def validate_topic_visualization_outputs(bertopic_dir: Path) -> dict[str, Any]:
    """Validate higher-order visualizations against the current group run."""
    groups = validate_topic_group_outputs(bertopic_dir)
    output_dir = bertopic_dir / "visualizations"
    manifest_path = output_dir / "visualizations_run_manifest.json"
    manifest = read_json(manifest_path)
    inputs = manifest.get("inputs") or {}
    recorded_groups = inputs.get("topic_group_manifest") or {}
    if recorded_groups.get("sha256") != groups["manifest_sha256"]:
        raise ValueError(
            "Topic visualizations belong to different higher-order assignments. "
            "Rerun 05_build_topic_visualizations.py."
        )
    if (manifest.get("extra") or {}).get("analysis_level") != "higher-order":
        raise ValueError(
            "Final manuscript visualizations must use the higher-order analysis level."
        )
    outputs = manifest.get("outputs") or {}
    for name in [
        "topic_weighted_totals",
        "country_topic_heatmap",
        "topic_shares_over_time",
        "country_topic_trends",
    ]:
        record = outputs.get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = output_dir / path.name
        assert_file_hash(path, record.get("sha256"), f"Topic visualization {name}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "groups": groups,
        "upstream_classifier": groups["upstream_classifier"],
    }


def validate_topic_table_outputs(bertopic_dir: Path) -> dict[str, Any]:
    """Validate generated CSV/LaTeX analyses against the current group run."""
    groups = validate_topic_group_outputs(bertopic_dir)
    output_dir = bertopic_dir / "inspection_tables"
    manifest_path = output_dir / "topic_tables_run_manifest.json"
    manifest = read_json(manifest_path)
    recorded_groups = ((manifest.get("inputs") or {}).get("topic_group_manifest") or {})
    if recorded_groups.get("sha256") != groups["manifest_sha256"]:
        raise ValueError(
            "Topic tables belong to different higher-order assignments. "
            "Rerun build_topic_tables.py."
        )
    outputs = manifest.get("outputs") or {}
    for name in ["higher_order_summary_table", "fine_grained_inventory_table"]:
        record = outputs.get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = output_dir / "latex" / path.name
        assert_file_hash(path, record.get("sha256"), f"Topic table {name}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "groups": groups,
        "upstream_classifier": groups["upstream_classifier"],
    }


def validate_final_topic_outputs(bertopic_dir: Path) -> dict[str, Any]:
    """Validate manuscript metadata and tables against the final topic run."""
    visualizations = validate_topic_visualization_outputs(bertopic_dir)
    groups = visualizations["groups"]
    tables = validate_topic_table_outputs(bertopic_dir)
    manifest_path = bertopic_dir / "topic_model_output_manifest.json"
    manifest = read_json(manifest_path)
    inputs = manifest.get("inputs") or {}
    recorded_groups = inputs.get("topic_group_manifest") or {}
    if recorded_groups.get("sha256") != groups["manifest_sha256"]:
        raise ValueError(
            "Final topic outputs belong to different higher-order assignments. "
            "Rerun the finalization step."
        )
    recorded_visualizations = inputs.get("visualizations_manifest") or {}
    if recorded_visualizations.get("sha256") != visualizations["manifest_sha256"]:
        raise ValueError(
            "Final topic outputs belong to different visualizations. "
            "Rerun the finalization step."
        )
    recorded_tables = inputs.get("topic_tables_manifest") or {}
    if recorded_tables.get("sha256") != tables["manifest_sha256"]:
        raise ValueError(
            "Final topic outputs belong to different generated tables. "
            "Rerun the finalization step."
        )
    for name in [
        "higher_order_summary_table",
        "fine_grained_inventory_table",
    ]:
        record = inputs.get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = bertopic_dir / "inspection_tables" / "latex" / path.name
        assert_file_hash(path, record.get("sha256"), f"Final topic input {name}")
    for name in ["manuscript_values", "build_summary", "latest_index"]:
        record = (manifest.get("outputs") or {}).get(name) or {}
        path = Path(str(record.get("path", "")))
        if not path.is_absolute():
            path = bertopic_dir / path.name
        assert_file_hash(path, record.get("sha256"), f"Final topic output {name}")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "visualizations": visualizations,
        "tables": tables,
        "groups": groups,
        "upstream_classifier": groups["upstream_classifier"],
    }
