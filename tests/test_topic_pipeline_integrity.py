from __future__ import annotations

import json
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd

from topic_classification.provenance import (
    apply_publication_label_overrides,
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
    def test_final_reviewed_topic_map_is_complete(self):
        path = (
            ROOT
            / "topic_classification/manual_labels/final_stability_v1_topic_labels.csv"
        )
        overrides = pd.read_csv(path)
        self.assertEqual(overrides["Topic"].tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(
            overrides["publication_topic_label"].tolist(),
            [
                "Procurement and public-revenue corruption",
                "Prosecution of senior public officials",
                "Electoral corruption and incumbent abuse",
                "Anti-corruption reform and rule-of-law oversight",
                "Campaign and party finance",
            ],
        )

    def test_reviewed_publication_labels_require_exact_topic_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            pd.DataFrame(
                [
                    {
                        "Topic": 0,
                        "llm_topic_short_label_expected": "Raw label",
                        "publication_topic_label": "Reviewed label",
                        "publication_label_rationale": "Neutral wording.",
                    }
                ]
            ).to_csv(path, index=False)
            labels = pd.DataFrame(
                [{"Topic": 0, "llm_topic_short_label": "Raw label"}]
            )
            reviewed = apply_publication_label_overrides(labels, path)
            self.assertEqual(
                reviewed.loc[0, "publication_topic_label"], "Reviewed label"
            )
            labels.loc[0, "llm_topic_short_label"] = "Different topic"
            with self.assertRaisesRegex(ValueError, "expects"):
                apply_publication_label_overrides(labels, path)

    def test_compact_descriptive_runner_has_no_forced_grouping(self):
        runner = (
            ROOT
            / "topic_classification/scripts/07_run_compact_descriptive_topic_model.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('PER_COUNTRY_YEAR="${PER_COUNTRY_YEAR:-50}"', runner)
        self.assertIn('TARGET_TOPICS="${TARGET_TOPICS:-none}"', runner)
        self.assertIn('CLUSTERER="${CLUSTERER:-hdbscan-stability}"', runner)
        self.assertIn(
            'CLUSTER_SELECTION_METHOD="${CLUSTER_SELECTION_METHOD:-leaf}"', runner
        )
        self.assertIn('HDBSCAN_MIN_SAMPLES="${HDBSCAN_MIN_SAMPLES:-10}"', runner)
        self.assertIn('UMAP_NEIGHBORS="${UMAP_NEIGHBORS:-15}"', runner)
        self.assertIn('--per-country-year "$PER_COUNTRY_YEAR"', runner)
        self.assertIn("02_create_descriptive_abstractions.py", runner)
        self.assertIn('--nr-topics "$TARGET_TOPICS"', runner)
        self.assertIn(
            '--cluster-selection-method "$CLUSTER_SELECTION_METHOD"', runner
        )
        self.assertIn('--umap-neighbors "$UMAP_NEIGHBORS"', runner)
        self.assertIn('_u${UMAP_NEIGHBORS}_', runner)
        self.assertIn("CLUSTER_SPEC", runner)
        self.assertIn("04_label_descriptive_topics_with_llm.py", runner)
        self.assertNotIn("--input-kind", runner)
        self.assertIn("--analysis-level fine-grained", runner)
        self.assertIn("06_build_descriptive_topic_review.py", runner)
        self.assertNotIn("04_group_topics_with_llm.py", runner)
        self.assertNotIn("--upload", runner)

    def test_stability_selector_uses_adequacy_guardrails(self):
        script_path = (
            ROOT / "topic_classification/scripts/03_fit_descriptive_bertopic.py"
        )
        spec = spec_from_file_location("descriptive_topic_fit_guardrails", script_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        kwargs = {
            "minimum_topics": 4,
            "maximum_topics": 12,
            "maximum_outlier_share": 0.45,
            "maximum_largest_topic_share": 0.40,
            "maximum_country_nmi": 0.25,
            "minimum_resample_common_share": 0.35,
        }
        self.assertTrue(
            module.candidate_meets_guardrails(
                topic_count=7,
                outlier_share=0.30,
                largest_topic_share=0.25,
                country_nmi=0.10,
                mean_resample_common_share=0.70,
                **kwargs,
            )
        )
        self.assertFalse(
            module.candidate_meets_guardrails(
                topic_count=2,
                outlier_share=0.05,
                largest_topic_share=0.98,
                country_nmi=0.01,
                mean_resample_common_share=0.70,
                **kwargs,
            )
        )
        self.assertFalse(
            module.candidate_meets_guardrails(
                topic_count=10,
                outlier_share=0.60,
                largest_topic_share=0.20,
                country_nmi=0.10,
                mean_resample_common_share=0.70,
                **kwargs,
            )
        )
        self.assertFalse(
            module.candidate_meets_guardrails(
                topic_count=7,
                outlier_share=0.30,
                largest_topic_share=0.25,
                country_nmi=0.10,
                mean_resample_common_share=0.20,
                **kwargs,
            )
        )

    def test_appendix_uses_direct_descriptive_topic_table(self):
        appendix = (ROOT / "docs/appendix_political_corruption.tex").read_text(
            encoding="utf-8"
        )
        method = (ROOT / "docs/method.tex").read_text(encoding="utf-8")
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
        self.assertIn("Purpose and Procedure", appendix)
        self.assertNotIn("Exploratory Topic Modeling", method)
        self.assertNotIn("BERTopic", method)

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
                    "publication_topic_label": "Public contracting",
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

    def test_compact_vectorizer_supports_few_topic_documents(self):
        script = (
            ROOT / "topic_classification/scripts/03_fit_descriptive_bertopic.py"
        ).read_text(encoding="utf-8")
        self.assertIn("min_df=1", script)
        self.assertIn("max_df=1.0", script)
        self.assertIn('default="hdbscan"', script)
        self.assertIn("n_init=20", script)
        self.assertIn("min_samples=args.hdbscan_min_samples", script)
        self.assertIn("selected_umap_neighbors = args.umap_neighbors", script)
        self.assertIn("save_embedding_model=args.embedding_model", script)
        self.assertIn('choices=["kmeans", "hdbscan", "hdbscan-stability"]', script)
        self.assertIn("mean_resample_ari", script)

    def test_appendix_visualizations_have_static_outputs(self):
        script = (
            ROOT / "topic_classification/scripts/05_build_topic_visualizations.py"
        ).read_text(encoding="utf-8")
        for stem in [
            "figure_topic_prevalence",
            "figure_topic_country_heatmap",
            "figure_topic_trends",
            "figure_topic_model_selection",
        ]:
            self.assertIn(stem, script)
        self.assertIn('"pdf.fonttype": 42', script)
        self.assertIn("dpi=600", script)
        self.assertIn("Met all criteria", script)
        self.assertIn("Within-country share (%)", script)
        appendix = (ROOT / "docs/appendix_political_corruption.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn("figure_topic_trends.pdf", appendix)
        self.assertIn("figure_topic_country_heatmap.pdf", appendix)
        self.assertNotIn("figure_topic_model_selection.pdf", appendix)

    def test_descriptive_publisher_targets_canonical_output_tree(self):
        script = (
            ROOT
            / "topic_classification/scripts/08_publish_descriptive_topic_outputs.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"tables", "topic_models"', script)
        self.assertIn('"figures", "topic_models"', script)
        self.assertIn("validate_topic_label_outputs", script)

    def test_topic_reduction_can_be_disabled(self):
        script_path = (
            ROOT / "topic_classification/scripts/03_fit_descriptive_bertopic.py"
        )
        spec = spec_from_file_location("descriptive_topic_fit", script_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertIsNone(module.parse_nr_topics("none"))
        self.assertEqual(module.parse_nr_topics("auto"), "auto")
        self.assertEqual(module.parse_nr_topics("7"), 7)

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

    def test_topic_label_examples_round_robin_across_countries(self):
        script_path = (
            ROOT
            / "topic_classification/scripts/04_label_descriptive_topics_with_llm.py"
        )
        spec = spec_from_file_location("descriptive_topic_labels_diversity", script_path)
        module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        documents = pd.DataFrame(
            [
                {"country": country, "article_text": f"{country}-{row}"}
                for country in ["Bulgaria", "France", "Hungary", "Italy", "Serbia"]
                for row in range(3)
            ]
        )
        examples = module.select_diverse_examples(
            documents,
            text_column="article_text",
            n=5,
            max_chars=100,
            random_state=42,
        )
        self.assertEqual(len(examples), 5)
        self.assertEqual(
            {example.split("-", maxsplit=1)[0] for example in examples},
            {"Bulgaria", "France", "Hungary", "Italy", "Serbia"},
        )


if __name__ == "__main__":
    unittest.main()
