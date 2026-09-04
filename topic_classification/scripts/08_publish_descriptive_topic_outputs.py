"""Publish an approved descriptive topic table and figures to Research Drive."""

from __future__ import annotations

import argparse
import mimetypes
import posixpath
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RD_BASE_DIR
from rd_utils import webdav_mkdirs, webdav_upload_bytes
from topic_classification.provenance import (
    assert_file_hash,
    read_json,
    validate_topic_label_outputs,
)


DEFAULT_OUTPUT_ROOT = posixpath.join(
    RD_BASE_DIR, "victims-of-corruption-paper", "output"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and publish the approved descriptive topic-model table, "
            "figures, and build diagnostics to Research Drive."
        )
    )
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Perform the upload. Without this flag, print the validated plan.",
    )
    parser.add_argument("--rd-output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def resolve_record_path(record: dict, fallback_dir: Path) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = fallback_dir / path.name
    return path


def validate_manifest_outputs(
    manifest_path: Path,
    *,
    expected_input_name: str,
    expected_input_hash: str,
) -> dict[str, Path]:
    manifest = read_json(manifest_path)
    input_record = (manifest.get("inputs") or {}).get(expected_input_name) or {}
    if input_record.get("sha256") != expected_input_hash:
        raise ValueError(
            f"{manifest_path.name} does not belong to the current topic labels."
        )
    outputs = {}
    for name, record in (manifest.get("outputs") or {}).items():
        path = resolve_record_path(record, manifest_path.parent)
        assert_file_hash(path, record.get("sha256"), f"Published output {name}")
        outputs[name] = path
    return outputs


def upload_file(local_path: Path, remote_path: str) -> None:
    webdav_mkdirs(posixpath.dirname(remote_path))
    content_type = mimetypes.guess_type(local_path.name)[0]
    if content_type is None:
        content_type = (
            "text/plain; charset=utf-8"
            if local_path.suffix in {".tex", ".txt", ".csv"}
            else "application/octet-stream"
        )
    webdav_upload_bytes(remote_path, local_path.read_bytes(), content_type)


def main() -> None:
    args = parse_args()
    root = args.bertopic_dir.resolve()
    labels = validate_topic_label_outputs(root)
    label_manifest_hash = labels["manifest_sha256"]

    descriptive = validate_manifest_outputs(
        root / "descriptive_topic_output_manifest.json",
        expected_input_name="topic_label_manifest",
        expected_input_hash=label_manifest_hash,
    )
    visualizations = validate_manifest_outputs(
        root / "visualizations" / "visualizations_run_manifest.json",
        expected_input_name="topic_label_manifest",
        expected_input_hash=label_manifest_hash,
    )

    required_descriptive = {
        "latex_table",
        "manuscript_values",
        "diagnostics",
        "manual_review",
        "latest_index",
    }
    required_figures = {
        "figure_topic_prevalence_pdf",
        "figure_topic_country_heatmap_pdf",
        "figure_topic_trends_pdf",
        "figure_topic_model_selection_pdf",
        "figure_topic_prevalence_png",
        "figure_topic_country_heatmap_png",
        "figure_topic_trends_png",
        "figure_topic_model_selection_png",
    }
    missing = (required_descriptive - descriptive.keys()) | (
        required_figures - visualizations.keys()
    )
    if missing:
        raise FileNotFoundError(
            "Final descriptive topic outputs are incomplete: "
            + ", ".join(sorted(missing))
        )

    table_dir = posixpath.join(args.rd_output_root, "tables", "topic_models")
    figure_dir = posixpath.join(args.rd_output_root, "figures", "topic_models")
    metadata_dir = posixpath.join(args.rd_output_root, "topic_models")

    publication_plan = []
    for key in ["latex_table", "manuscript_values"]:
        local = descriptive[key]
        publication_plan.append((local, posixpath.join(table_dir, local.name)))
    for key in sorted(required_figures):
        local = visualizations[key]
        publication_plan.append((local, posixpath.join(figure_dir, local.name)))
    for key in ["diagnostics", "manual_review", "latest_index"]:
        local = descriptive[key]
        publication_plan.append((local, posixpath.join(metadata_dir, local.name)))
    for local in [
        root / "hdbscan_stability_candidates.csv",
        root / "hdbscan_stability_selection.json",
        root / "run_manifest.json",
        root / "topic_labels_run_manifest.json",
        root / "descriptive_topic_output_manifest.json",
        root / "visualizations" / "visualizations_run_manifest.json",
    ]:
        if not local.exists():
            raise FileNotFoundError(local)
        publication_plan.append((local, posixpath.join(metadata_dir, local.name)))

    print("Validated final descriptive topic publication plan:")
    for local, remote in publication_plan:
        print(f"  {local} -> {remote}")

    if not args.upload:
        print("Dry run only. Add --upload after approving the final topic review.")
        return

    for local, remote in publication_plan:
        upload_file(local, remote)
        print(f"Uploaded {local.name} -> {remote}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
