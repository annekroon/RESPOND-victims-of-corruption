"""Delete old generated outputs before a clean political-classifier rebuild.

This intentionally preserves the expensive cleaned/deduplicated country files,
total-news denominators, source-inclusion workbook, and validation review files.
It removes generated silver-label batches, classifier comparisons, classifier
outputs, attention tables/figures, and manuscript classifier tables so the next
run starts from a clean derived-output state.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
CONFIRM_TEXT = "DELETE_OLD_POLITICAL_CLASSIFIER_OUTPUTS"


DELETE_PATHS = [
    "classifier_comparison",
    "classifier_comparison_heavy",
    "classifier_comparison_uk_calibration",
    "silver_classifier",
    "attention_tables",
    "attention_figures",
    "cleaned_deduped_source_filtered",
]

DELETE_GLOBS = [
    "active_learning/active_learning_batch*",
    "active_learning/silver_training_source_filtered*",
    "active_learning/uk_calibration_batch*",
    "source_inclusion/cleaned_source_filter_output_summary.csv",
    "source_inclusion/cleaned_source_filter_decision_summary_by_country.csv",
    "source_inclusion/source_filter_run_manifest.json",
    "manuscript_tables/table_pc_classifier*.tex",
    "manuscript_tables/table_classifier*.tex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove old generated political-classifier outputs before a clean rebuild."
    )
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument(
        "--confirm",
        default="",
        help=f"Required exact text to execute deletion: {CONFIRM_TEXT}",
    )
    return parser.parse_args()


def remove_path(path: Path) -> None:
    if not path.exists():
        print(f"Missing, skip: {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    else:
        path.unlink()
        print(f"Deleted file: {path}")


def main() -> None:
    args = parse_args()
    if args.confirm != CONFIRM_TEXT:
        print("Dry run only. The following paths would be removed:")
        for relative in DELETE_PATHS:
            print(f"- {args.pipeline_dir / relative}")
        for pattern in DELETE_GLOBS:
            matches = sorted(args.pipeline_dir.glob(pattern))
            if matches:
                for path in matches:
                    print(f"- {path}")
            else:
                print(f"- {args.pipeline_dir / pattern} (no matches)")
        print(f"\nTo delete, rerun with: --confirm {CONFIRM_TEXT}")
        return

    for relative in DELETE_PATHS:
        remove_path(args.pipeline_dir / relative)

    for pattern in DELETE_GLOBS:
        for path in sorted(args.pipeline_dir.glob(pattern)):
            remove_path(path)

    print("\nClean rebuild outputs removed.")
    print("Preserved cleaned/deduped corpus files, source_inclusion/, and validation files.")


if __name__ == "__main__":
    main()
