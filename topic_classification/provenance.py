"""Integrity checks linking topic outputs to the final classifier run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from political_classifier.final_corpus import (
    ClassifierRun,
    assert_file_hash,
    load_verified_classifier_run,
    read_json,
    verify_classified_country_frame,
)
from political_classifier.reproducibility import sha256_file


def apply_publication_label_overrides(labels, overrides_path: Path | None):
    """Apply a complete, audited publication-label map without changing topics."""
    import pandas as pd

    result = labels.copy()
    source_column = (
        "llm_topic_short_label"
        if "llm_topic_short_label" in result.columns
        else "llm_topic_label"
    )
    result["publication_topic_label"] = result[source_column]
    result["figure_topic_label"] = result[source_column]
    if overrides_path is None:
        return result
    if not overrides_path.exists():
        raise FileNotFoundError(overrides_path)

    overrides = pd.read_csv(overrides_path)
    required = {
        "Topic",
        "llm_topic_short_label_expected",
        "publication_topic_label",
        "figure_topic_label",
        "publication_label_rationale",
    }
    missing = required - set(overrides.columns)
    if missing:
        raise ValueError(
            f"Publication-label file is missing columns: {sorted(missing)}"
        )
    if overrides["Topic"].duplicated().any():
        raise ValueError("Publication-label file contains duplicate topic IDs.")
    if (
        overrides["publication_topic_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError("Publication-label file contains a blank label.")
    if (
        overrides["figure_topic_label"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError("Publication-label file contains a blank figure label.")

    result_topics = set(result.loc[result["Topic"].ne(-1), "Topic"].astype(int))
    override_topics = set(overrides["Topic"].astype(int))
    if result_topics != override_topics:
        raise ValueError(
            "Publication-label topics do not exactly match the fitted topics: "
            f"model={sorted(result_topics)}, overrides={sorted(override_topics)}."
        )

    overrides = overrides.set_index("Topic")
    for topic in sorted(result_topics):
        mask = result["Topic"].astype(int).eq(topic)
        actual = str(result.loc[mask, source_column].iloc[0]).strip()
        expected = str(
            overrides.loc[topic, "llm_topic_short_label_expected"]
        ).strip()
        if actual != expected:
            raise ValueError(
                f"Topic {topic} has LLM label {actual!r}, but the reviewed map "
                f"expects {expected!r}. Review the new model before publishing."
            )
        result.loc[mask, "publication_topic_label"] = str(
            overrides.loc[topic, "publication_topic_label"]
        ).strip()
        result.loc[mask, "figure_topic_label"] = str(
            overrides.loc[topic, "figure_topic_label"]
        ).strip()
        result.loc[mask, "publication_label_rationale"] = str(
            overrides.loc[topic, "publication_label_rationale"]
        ).strip()
    return result


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
