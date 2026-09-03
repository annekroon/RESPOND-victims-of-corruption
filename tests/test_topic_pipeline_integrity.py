from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

from topic_classification.provenance import (
    load_verified_classifier_run,
    sha256_file,
    validate_verified_topic_sample,
    verify_classified_country_frame,
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
    def test_compact_descriptive_runner_has_no_forced_grouping(self):
        runner = (
            ROOT
            / "topic_classification/scripts/07_run_compact_descriptive_topic_model.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('PER_COUNTRY_YEAR="${PER_COUNTRY_YEAR:-50}"', runner)
        self.assertIn('TARGET_TOPICS="${TARGET_TOPICS:-8}"', runner)
        self.assertIn('--per-country-year "$PER_COUNTRY_YEAR"', runner)
        self.assertIn("02_create_descriptive_abstractions.py", runner)
        self.assertIn('--nr-topics "$TARGET_TOPICS"', runner)
        self.assertIn("04_label_descriptive_topics_with_llm.py", runner)
        self.assertNotIn("--input-kind", runner)
        self.assertIn("--analysis-level fine-grained", runner)
        self.assertIn("06_build_descriptive_topic_review.py", runner)
        self.assertNotIn("04_group_topics_with_llm.py", runner)
        self.assertNotIn("--upload", runner)

    def test_appendix_uses_direct_descriptive_topic_table(self):
        appendix = (ROOT / "docs/appendix_political_corruption.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "output/tables/topic_models/table_topic_descriptive_summary",
            appendix,
        )
        self.assertIn(
            "output/tables/topic_models/descriptive_topic_manuscript_values",
            appendix,
        )
        self.assertNotIn("table_topic_higher_order_summary", appendix)
        self.assertNotIn("table_all_topics_llm_higher_order_topics", appendix)

    def test_readme_has_no_stale_fixed_topic_solution(self):
        readme = (ROOT / "topic_classification/README.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("67 non-outlier", readme)
        self.assertNotIn("target-groups 12", readme)
        self.assertIn("00_LATEST_DESCRIPTIVE_TOPIC_BUILD.txt", readme)
        self.assertIn("language-neutral english", readme.lower())


class TopicTableBuilderTests(unittest.TestCase):
    def test_direct_descriptive_table_is_compact(self):
        script_path = (
            ROOT
            / "topic_classification/scripts/06_build_descriptive_topic_review.py"
        )
        spec = spec_from_file_location("descriptive_topic_review", script_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        summary = pd.DataFrame(
            [
                {
                    "llm_topic_short_label": "Public contracting",
                    "weighted_share": 0.6,
                    "top_countries": "France (40.0%); Italy (30.0%)",
                    "llm_topic_summary": "Cases concerning manipulation of public contracts.",
                }
            ]
        )
        latex = module.direct_topic_latex(summary)
        self.assertIn(r"\scriptsize", latex)
        self.assertIn(r"\begin{tabularx}", latex)
        self.assertNotIn(r"\resizebox", latex)

    def test_abstraction_prompt_removes_case_identifiers(self):
        script_path = (
            ROOT
            / "topic_classification/scripts/02_create_descriptive_abstractions.py"
        )
        spec = spec_from_file_location("descriptive_abstractions", script_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        prompt = module.build_prompt("Article text")
        self.assertIn("Do not include names", prompt)
        self.assertIn("institutional or sectoral setting", prompt)
        self.assertIn("boundary_or_unclear", prompt)

    def test_topic_label_prompt_cannot_reconstruct_case_identifiers(self):
        script_path = (
            ROOT
            / "topic_classification/scripts/04_label_descriptive_topics_with_llm.py"
        )
        spec = spec_from_file_location("descriptive_topic_labels", script_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        prompt = module.build_prompt(0, "keywords", 10, ["Neutral abstraction"])
        self.assertIn("Do not reconstruct removed countries", prompt)
        self.assertIn("Labels must remain country-neutral", prompt)
        self.assertNotIn("acceptable for a label to mention a country", prompt.lower())


if __name__ == "__main__":
    unittest.main()
