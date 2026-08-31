from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from topic_classification.provenance import (
    load_verified_classifier_run,
    sha256_file,
    validate_verified_topic_sample,
    verify_classified_country_frame,
)
from topic_classification.scripts._impl.build_topic_tables import (
    higher_order_latex,
    weighted_analyses,
)


ROOT = Path(__file__).resolve().parents[1]


def record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256_file(path)}


class TopicClassifierProvenanceTests(unittest.TestCase):
    def make_classifier_run(self, root: Path, *, threshold: float = 0.6) -> Path:
        output_dir = root / "silver_classifier"
        output_dir.mkdir(parents=True)
        source_dir = root / "source_inclusion"
        source_dir.mkdir(parents=True)
        source_summary_path = source_dir / "cleaned_source_filter_output_summary.csv"
        pd.DataFrame([{"country": "France", "output_rows": 2}]).to_csv(
            source_summary_path, index=False
        )
        source_record = {"path": "sources.xlsx", "sha256": "source"}
        (source_dir / "source_filter_run_manifest.json").write_text(
            json.dumps(
                {
                    "source_filter_policy": (
                        "exclude_explicit_no_retain_all_other_decisions"
                    ),
                    "source_decision_file": source_record,
                    "output_summary": record(source_summary_path),
                }
            ),
            encoding="utf-8",
        )
        summary_path = output_dir / "classified_country_summary.csv"
        pd.DataFrame(
            [
                {
                    "country": "France",
                    "total_articles": 2,
                    "predicted_political_corruption": 1,
                    "predicted_rate": 0.5,
                }
            ]
        ).to_csv(summary_path, index=False)
        (output_dir / "selected_threshold.txt").write_text(
            f"{threshold}\n", encoding="utf-8"
        )
        manifest = {
            "git_commit": "abc123",
            "threshold": threshold,
            "corpus_dir": str(root / "cleaned_deduped_source_filtered"),
            "source_decision_file": source_record,
            "classified_country_summary": record(summary_path),
            "embedding_model": "intfloat/multilingual-e5-large",
            "embedding_model_revision": "revision",
        }
        (output_dir / "classifier_run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return output_dir

    def test_final_classifier_and_country_predictions_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = self.make_classifier_run(Path(directory))
            run = load_verified_classifier_run(
                output_dir,
                ["France"],
                require_local_country_files=False,
            )
            frame = pd.DataFrame(
                {
                    "prob_political_corruption": [0.7, 0.4],
                    "pred_political_corruption": [1, 0],
                }
            )
            verify_classified_country_frame(
                frame,
                country="France",
                classifier_run=run,
            )
            self.assertEqual(run.total_articles, 2)
            self.assertEqual(run.political_articles, 1)
            self.assertEqual(run.threshold, 0.6)

    def test_threshold_mismatch_stops_topic_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = self.make_classifier_run(Path(directory))
            (output_dir / "selected_threshold.txt").write_text(
                "0.5\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "threshold file"):
                load_verified_classifier_run(
                    output_dir,
                    ["France"],
                    require_local_country_files=False,
                )

    def test_topic_sample_requires_verified_political_only_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "sample.csv.gz"
            pd.DataFrame([{"country": "France", "year": 2024}]).to_csv(
                sample_path,
                index=False,
                compression="gzip",
            )
            manifest_path = root / "sample_run_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "outputs": {"sample": record(sample_path)},
                        "extra": {
                            "saved_sample_rows": 1,
                            "political_only": True,
                            "upstream_classifier": {
                                "verified_final_classifier": True,
                                "threshold": 0.6,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            provenance = validate_verified_topic_sample(sample_path)
            self.assertTrue(
                provenance["upstream_classifier"]["verified_final_classifier"]
            )


class TopicWorkflowStructureTests(unittest.TestCase):
    def test_final_runner_uses_six_groups_and_finalizes_outputs(self):
        runner = (
            ROOT
            / "topic_classification/scripts/07_rerun_final_source_filtered_topic_solution.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--target-groups 6", runner)
        self.assertNotIn("--target-groups 12", runner)
        self.assertIn("finalize_topic_outputs.py", runner)
        self.assertIn("build_topic_tables.py", runner)
        self.assertIn("--analysis-level higher-order", runner)
        self.assertNotIn("nbconvert", runner)
        self.assertNotIn(".ipynb", runner)
        self.assertGreaterEqual(runner.count("--overwrite"), 4)

    def test_appendix_uses_only_compact_canonical_topic_table(self):
        appendix = (ROOT / "docs/appendix_political_corruption.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "output/tables/topic_models/table_topic_higher_order_summary",
            appendix,
        )
        self.assertIn(
            "output/tables/topic_models/topic_model_manuscript_values",
            appendix,
        )
        self.assertNotIn("table_all_topics_llm_higher_order_topics", appendix)

    def test_publisher_uses_canonical_topic_models_folder(self):
        publisher = (
            ROOT
            / "topic_classification/scripts/06_publish_topic_archive_to_webdav.py"
        ).read_text(encoding="utf-8")
        self.assertIn('default="topic_models"', publisher)
        self.assertNotIn('default="topic models"', publisher)
        self.assertIn("validate_final_topic_outputs", publisher)

    def test_readme_has_no_stale_fixed_topic_solution(self):
        readme = (ROOT / "topic_classification/README.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("67 non-outlier", readme)
        self.assertNotIn("target-groups 12", readme)
        self.assertIn("00_LATEST_TOPIC_MODEL_BUILD.txt", readme)


class TopicTableBuilderTests(unittest.TestCase):
    def test_weighted_summaries_exclude_outliers_and_use_compact_latex(self):
        documents = pd.DataFrame(
            {
                "topic": [0, 0, 1, -1],
                "country": ["France", "Italy", "France", "France"],
                "year": [2023, 2023, 2024, 2024],
                "analysis_weight": [2.0, 1.0, 1.0, 50.0],
            }
        )
        groups = pd.DataFrame(
            {
                "Topic": [0, 1],
                "llm_topic_short_label": ["Tender cases", "Court cases"],
                "llm_topic_summary": ["Tender summary", "Court summary"],
                "topic_group_id": ["local", "elite"],
                "topic_group_short_label": ["Local cases", "Elite scandals"],
                "topic_group_label": ["Local cases", "Elite scandals"],
                "topic_group_summary": ["Local summary", "Elite summary"],
                "topic_group_assignment_rationale": ["local", "elite"],
            }
        )
        analyses = weighted_analyses(documents, groups)
        self.assertAlmostEqual(analyses["summary"]["weighted_share"].sum(), 1.0)
        self.assertEqual(analyses["inliers"]["analysis_weight"].sum(), 4.0)
        latex = higher_order_latex(analyses["summary"])
        self.assertIn(r"\scriptsize", latex)
        self.assertIn(r"\begin{tabularx}", latex)
        self.assertNotIn(r"\resizebox", latex)


if __name__ == "__main__":
    unittest.main()
