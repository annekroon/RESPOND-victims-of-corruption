from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from political_classifier.source_filter import (
    apply_source_inclusion_filter,
    normalize_country,
    normalize_source,
)
from political_classifier.split_integrity import (
    assert_no_validation_overlap,
    calibration_test_split,
    overlap_rows,
)
from political_classifier.scripts._impl.build_manuscript_outputs import (
    format_latex_table,
    pipeline_counts,
)
from political_classifier.reproducibility import file_record
from political_classifier.classifier_data import (
    format_texts_for_embedding,
    normalize_text,
)


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CLEAN = load_script(
    "clean_dedupe_script",
    "political_classifier/scripts/01_clean_dedupe_data.py",
)
CONTENT_COMMON = load_script(
    "content_classifier_common_test",
    "content-classification/scripts/content_classifier_common.py",
)
CONTENT_PROMPTS = load_script(
    "content_prompts_test",
    "content-classification/scripts/content_prompts.py",
)
SILVER_LABELER = load_script(
    "silver_labeler_test",
    "political_classifier/scripts/_impl/label_silver_batch.py",
)
ATTENTION_UPLOADER = load_script(
    "attention_uploader_test",
    "political_classifier/scripts/_impl/upload_attention_outputs.py",
)
CONTENT_SAMPLER = load_script(
    "content_validation_sampler_test",
    "content-classification/scripts/create_validation_sample.py",
)


class CleaningTests(unittest.TestCase):
    def test_deduplication_state_is_country_scoped(self):
        article = "This is a sufficiently long political corruption article. " * 20
        chunk = pd.DataFrame(
            {
                "uri": ["same-uri"],
                "body": [article],
                "dateTime": ["2024-01-01"],
                "source.uri": ["https://www.example.com/story"],
            }
        )
        bulgaria = CLEAN.prepare_chunk(chunk, "Bulgaria", 1, 0)
        france = CLEAN.prepare_chunk(chunk, "France", 1, 0)
        self.assertEqual(len(CLEAN.remove_seen(bulgaria, set(), set(), set())), 1)
        self.assertEqual(len(CLEAN.remove_seen(france, set(), set(), set())), 1)

    def test_blank_uris_do_not_collapse_distinct_articles(self):
        data = pd.DataFrame(
            {
                "uri": ["", ""],
                "text_hash": ["one", "two"],
                "near_dup_hash": ["", ""],
            }
        )
        self.assertEqual(len(CLEAN.remove_seen(data, set(), set(), set())), 2)


class SourceFilterTests(unittest.TestCase):
    def test_domain_canonicalization(self):
        self.assertEqual(normalize_source("https://WWW.Example.COM/news/1"), "example.com")
        self.assertEqual(normalize_source("example.com/"), "example.com")

    def test_country_aliases_match(self):
        self.assertEqual(normalize_country("United Kingdom"), "United_Kingdom")
        self.assertEqual(normalize_country("UK"), "United_Kingdom")

    def test_only_explicit_no_sources_are_excluded(self):
        data = pd.DataFrame(
            {
                "country": ["France"] * 4,
                "source_uri": ["yes.fr", "no.fr", "review.fr", "missing.fr"],
            }
        )
        decisions = pd.DataFrame(
            {
                "country": ["France"] * 3,
                "source_clean": ["yes.fr", "no.fr", "review.fr"],
                "source_include": [True, False, True],
                "source_filter_decision_clean": ["YES", "NO", "REVIEW"],
            }
        )
        filtered, merged = apply_source_inclusion_filter(data, decisions)
        self.assertEqual(
            set(filtered["source_uri"]),
            {"yes.fr", "review.fr", "missing.fr"},
        )
        missing = merged.loc[merged["source_uri"].eq("missing.fr")].iloc[0]
        self.assertTrue(missing["source_include"])
        self.assertEqual(missing["source_filter_decision"], "MISSING")


class SplitIntegrityTests(unittest.TestCase):
    def test_overlap_detects_uri_and_text(self):
        training = pd.DataFrame(
            {"uri": ["u1", "u2"], "model_text": ["First text", "Shared text"]}
        )
        validation = pd.DataFrame(
            {"uri": ["u1", "u3"], "model_text": ["Different", "shared text!"]}
        )
        overlap = overlap_rows(training, validation)
        self.assertEqual(len(overlap), 2)
        with self.assertRaises(ValueError):
            assert_no_validation_overlap(training, validation)

    def test_calibration_and_test_are_disjoint_and_stratified(self):
        rows = []
        for country in ["A", "B"]:
            for label in [0, 1]:
                rows.extend({"country": country, "y": label} for _ in range(10))
        data = pd.DataFrame(rows)
        calibration, test = calibration_test_split(data, 0.4, 42)
        self.assertFalse(set(calibration.index) & set(test.index))
        self.assertEqual(len(calibration) + len(test), len(data))
        self.assertEqual(set(calibration.groupby(["country", "y"]).size()), {4})


class CheckpointTests(unittest.TestCase):
    def test_successful_retry_replaces_failed_row(self):
        existing = pd.DataFrame(
            [{"article_id": "A::1", "llm_error": "timeout", "label": ""}]
        )
        replacement = [{"article_id": "A::1", "llm_error": "", "label": "yes"}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "labels.csv"
            CONTENT_COMMON.write_checkpoint(existing, replacement, output)
            result = pd.read_csv(output)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "label"], "yes")
        self.assertTrue(pd.isna(result.loc[0, "llm_error"]) or result.loc[0, "llm_error"] == "")

    def test_model_access_errors_are_non_retryable(self):
        error = RuntimeError(
            "key not allowed to access model; code=key_model_access_denied"
        )
        self.assertTrue(CONTENT_COMMON.is_non_retryable_model_error(error))
        self.assertFalse(
            CONTENT_COMMON.is_non_retryable_model_error(RuntimeError("temporary 503"))
        )

    def test_silver_parser_accepts_unescaped_control_characters(self):
        parsed = SILVER_LABELER.extract_json(
            '{"translated_text":"line\nbreak","llm_label_suggestion":"No"}'
        )
        self.assertEqual(parsed["translated_text"], "line\nbreak")

    def test_silver_parse_error_preserves_raw_response(self):
        raw = '{"translated_text":"truncated"'

        class Message:
            content = raw

        class Choice:
            message = Message()

        class Completions:
            @staticmethod
            def create(**_kwargs):
                return type("Response", (), {"choices": [Choice()]})()

        client = type(
            "Client",
            (),
            {"chat": type("Chat", (), {"completions": Completions()})()},
        )()

        with self.assertRaises(SILVER_LABELER.LLMResponseParseError) as context:
            SILVER_LABELER.llm_translate_and_suggest(client, "Article", "model", 100)

        self.assertEqual(context.exception.raw_response, raw)
        self.assertIn("Article", context.exception.prompt)


class ContentSchemaTests(unittest.TestCase):
    def test_blank_uri_article_ids_use_text_hash(self):
        args = type(
            "Args",
            (),
            {"min_words": 1, "keep_non_political": True},
        )()
        frame = pd.DataFrame(
            {
                "country": ["A", "A"],
                "uri": ["", ""],
                "article_text": ["first article", "second article"],
            }
        )
        prepared = CONTENT_COMMON.prepare_frame(frame, None, args)
        self.assertEqual(prepared["article_id"].nunique(), 2)

    def test_invalid_labels_become_unclear(self):
        self.assertEqual(
            CONTENT_PROMPTS.normalize_corruption_frame({"corruption_frame": "typo"})[
                "corruption_frame"
            ],
            "unclear",
        )
        location = CONTENT_PROMPTS.normalize_abroad_case(
            {"case_location": "domestic", "abroad_case": "yes"}
        )
        self.assertEqual(location["abroad_case"], "no")
        actor = CONTENT_PROMPTS.normalize_accused_actor(
            {"accused_actor_visibility": "unexpected"}
        )
        self.assertEqual(actor["accused_actor_visibility"], "unclear")
        self.assertEqual(actor["accused_actor_visible"], "unclear")


class ContentSamplingTests(unittest.TestCase):
    def setUp(self):
        rows = []
        for country in ["A", "B"]:
            for year in [2020, 2021]:
                for index in range(10):
                    rows.append(
                        {
                            "article_id": f"{country}-{year}-{index}",
                            "country": country,
                            "year": year,
                            "prob_political_corruption": 0.51 + index / 20,
                        }
                    )
        self.data = pd.DataFrame(rows)
        self.data["probability_band"] = self.data[
            "prob_political_corruption"
        ].map(CONTENT_SAMPLER.probability_band)

    def test_small_confidence_stratified_sample_has_exact_size(self):
        columns = CONTENT_SAMPLER.sampling_columns(True)
        sample = CONTENT_SAMPLER.sample_total(self.data, 12, 42, columns)
        self.assertEqual(len(sample), 12)

    def test_per_country_sample_has_requested_size(self):
        columns = CONTENT_SAMPLER.sampling_columns(True)
        sample = CONTENT_SAMPLER.sample_per_country(self.data, 3, 42, columns)
        self.assertEqual(sample.groupby("country").size().to_dict(), {"A": 3, "B": 3})


class ClassifierDataTests(unittest.TestCase):
    def test_shared_text_preparation_preserves_model_contract(self):
        self.assertEqual(normalize_text("  Example\u00a0 text  "), "Example text")
        self.assertEqual(
            format_texts_for_embedding(["article"], "intfloat/multilingual-e5-large"),
            ["passage: article"],
        )


class WorkflowStructureTests(unittest.TestCase):
    def test_attention_production_does_not_execute_a_notebook(self):
        entrypoint = (
            ROOT / "political_classifier/scripts/07_build_attention_outputs.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("nbconvert", entrypoint)
        self.assertNotIn(".ipynb", entrypoint)

    def test_attention_country_year_csv_is_written_before_it_is_read(self):
        implementation = (
            ROOT
            / "political_classifier/scripts/_impl/build_attention_analysis.py"
        ).read_text(encoding="utf-8")
        write_position = implementation.index("yearly_attention.to_csv")
        read_position = implementation.index("country_year_from_csv = pd.read_csv")
        self.assertLess(write_position, read_position)


class LatexTests(unittest.TestCase):
    def test_wide_tables_are_never_upscaled(self):
        table = pd.DataFrame([{f"c{i}": i for i in range(8)}])
        latex = format_latex_table(table, "Caption", "tab:test", "Note.")
        self.assertIn(r"\begin{adjustbox}{max width=\textwidth}", latex)
        self.assertNotIn(r"\resizebox", latex)

    def test_published_attention_tex_paths_are_flat(self):
        base = Path("/pipeline/attention_tables")
        latex_path = base / "latex" / "table_attention_country_summary.tex"
        csv_path = base / "political_corruption_attention_total_news_month.csv"
        self.assertEqual(
            ATTENTION_UPLOADER.remote_relative_path(latex_path, base).as_posix(),
            "table_attention_country_summary.tex",
        )
        self.assertEqual(
            ATTENTION_UPLOADER.remote_relative_path(csv_path, base).as_posix(),
            "political_corruption_attention_total_news_month.csv",
        )

    def test_pipeline_counts_use_audited_raw_total_and_completed_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Path(directory)
            source_dir = pipeline / "source_inclusion"
            comparison_dir = pipeline / "classifier_comparison"
            classifier_dir = pipeline / "silver_classifier"
            attention_dir = pipeline / "attention_tables"
            for path in [source_dir, comparison_dir, classifier_dir, attention_dir]:
                path.mkdir(parents=True)

            source_workbook = source_dir / "sources.xlsx"
            source_workbook.write_bytes(b"reviewed source decisions")
            clean_audit = pipeline / "clean_dedupe_audit.csv"
            pd.DataFrame(
                [{"country": "A", "raw_rows": 10, "cleaned_deduplicated_rows": 8}]
            ).to_csv(clean_audit, index=False)
            (pipeline / "clean_dedupe_run_manifest.json").write_text(
                json.dumps({"audit": file_record(clean_audit)})
            )

            source_summary = source_dir / "cleaned_source_filter_output_summary.csv"
            pd.DataFrame(
                [{"country": "A", "input_rows": 8, "output_rows": 7}]
            ).to_csv(source_summary, index=False)
            source_record = file_record(source_workbook)
            (source_dir / "source_filter_run_manifest.json").write_text(
                json.dumps(
                    {
                        "output_summary": file_record(source_summary),
                        "source_decision_file": source_record,
                    }
                )
            )

            comparison_manifest = comparison_dir / "classifier_comparison_run_manifest.json"
            comparison_manifest.write_text("{}")
            classified_summary = classifier_dir / "classified_country_summary.csv"
            pd.DataFrame(
                [
                    {
                        "country": "A",
                        "total_articles": 7,
                        "predicted_political_corruption": 3,
                    }
                ]
            ).to_csv(classified_summary, index=False)
            (classifier_dir / "selected_threshold.txt").write_text("0.5\n")
            (classifier_dir / "classifier_run_manifest.json").write_text(
                json.dumps(
                    {
                        "threshold": 0.5,
                        "classified_country_summary": file_record(classified_summary),
                        "comparison_manifest": file_record(comparison_manifest),
                        "source_decision_file": source_record,
                    }
                )
            )
            pd.DataFrame(
                [
                    {
                        "country": "A",
                        "total_news_articles": 100,
                        "political_corruption_articles": 3,
                    }
                ]
            ).to_csv(
                attention_dir
                / "political_corruption_attention_total_news_country_summary.csv",
                index=False,
            )

            counts = pipeline_counts(pipeline)
            self.assertEqual(counts["raw_query"], 10)
            self.assertEqual(counts["source_filtered_query"], 7)
            self.assertEqual(counts["political_corruption"], 3)


if __name__ == "__main__":
    unittest.main()
