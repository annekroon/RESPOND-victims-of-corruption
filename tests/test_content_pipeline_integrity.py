from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from political_classifier.reproducibility import file_record


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "content-classification" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COMMON = load_script(
    "content_common_integrity_test",
    "content-classification/scripts/content_classifier_common.py",
)
MERGE = load_script(
    "content_merge_integrity_test",
    "content-classification/scripts/merge_content_labels.py",
)
EVALUATE = load_script(
    "content_evaluate_integrity_test",
    "content-classification/scripts/evaluate_codebook_gpt_against_human.py",
)
SAMPLER = load_script(
    "content_sampler_integration_test",
    "content-classification/scripts/create_validation_sample.py",
)
ANNOTATION_COMMON = load_script(
    "content_annotation_common_test",
    "content-classification/tools/content_annotation_common.py",
)


class ContentProvenanceTests(unittest.TestCase):
    def test_canonical_sample_manifest_detects_changed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.csv"
            pd.DataFrame([{"article_id": "A", "country": "France"}]).to_csv(
                sample, index=False
            )
            manifest = {
                "political_only": True,
                "upstream_classifier": {"verified_final_classifier": True},
                "outputs": {"sample": file_record(sample)},
            }
            COMMON.content_sample_manifest_path(sample).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            observed = COMMON.validate_content_sample(sample)
            self.assertTrue(observed["upstream_classifier"]["verified_final_classifier"])

            sample.write_text("article_id,country\nB,France\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                COMMON.validate_content_sample(sample)

    def make_completed_output(self, root: Path) -> Path:
        output = root / "victim_visibility_labels.csv"
        pd.DataFrame(
            [{"article_id": "A", "victim_visibility": "no_victim"}]
        ).to_csv(output, index=False)
        audit = output.with_name(output.stem + "_audit.jsonl")
        audit.write_text('{"article_id":"A"}\n', encoding="utf-8")
        run_manifest = output.with_name(output.name + ".run.json")
        upstream = {
            "verified_final_classifier": True,
            "classifier_manifest_sha256": "classifier-hash",
            "political_corruption_articles": 1,
        }
        run_manifest.write_text(
            json.dumps(
                {
                    "classifier_name": "victim_visibility",
                    "prompt_version": "test-v1",
                    "model": "gpt-5.1",
                    "source": "csv",
                    "countries": ["France"],
                    "keep_non_political": False,
                    "upstream_classifier": upstream,
                }
            ),
            encoding="utf-8",
        )
        completion = {
            "classifier_name": "victim_visibility",
            "prompt_version": "test-v1",
            "model": "gpt-5.1",
            "source": "csv",
            "countries": ["France"],
            "production_full_corpus": False,
            "rows": 1,
            "article_id_sha256": COMMON.hashlib.sha256(b"A").hexdigest(),
            "output": file_record(output),
            "run_manifest": file_record(run_manifest),
            "audit": file_record(audit),
            "upstream_classifier": upstream,
        }
        COMMON.content_completion_path(output).write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return output

    def test_completed_output_checks_output_manifest_and_audit_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.make_completed_output(Path(directory))
            completed = COMMON.validate_completed_content_output(output)
            self.assertFalse(completed["production_full_corpus"])

            completed["audit_path"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit log differs"):
                COMMON.validate_completed_content_output(output)

    def test_content_parser_accepts_control_characters_and_preserves_bad_raw(self):
        parsed = COMMON.extract_json('{"label":"line\nbreak"}')
        self.assertEqual(parsed["label"], "line\nbreak")

        class Message:
            content = '{"broken":'

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
        spec = type(
            "Spec",
            (),
            {
                "name": "test",
                "build_prompt": staticmethod(lambda _text, _row: "test prompt"),
                "normalize_result": staticmethod(lambda result: result),
            },
        )()
        with self.assertRaises(COMMON.LLMResponseParseError) as context:
            COMMON.classify_article(client, spec, {"article_text": "text"}, "model", 100)
        self.assertEqual(context.exception.raw_response, '{"broken":')
        self.assertEqual(context.exception.prompt, "test prompt")

    def test_streamlit_annotation_output_is_resumable_and_manifested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "pilot.csv"
            output_path = root / "pilot_anne.csv"
            pd.DataFrame(
                [
                    {
                        "article_id": "A",
                        "country": "France",
                        "sample_purpose": "codebook_development",
                        "article_text": "Article text",
                    }
                ]
            ).to_csv(input_path, index=False)
            data, provenance, resumed = ANNOTATION_COMMON.load_annotation_data(
                input_path,
                output_path,
            )
            self.assertFalse(resumed)
            ANNOTATION_COMMON.record_annotation(
                data,
                0,
                victim_visibility="no_victim",
                corruption_frame="individualized",
                case_location="domestic",
                accused_actor_visibility="individual_actor",
                notes="checked",
                coder_id="anne",
                coder_first_name="Anne",
                code_session_id="session",
            )
            manifest_path = ANNOTATION_COMMON.save_annotation_data(
                data,
                input_path=input_path,
                output_path=output_path,
                coder_id="anne",
                coder_first_name="Anne",
                code_session_id="session",
                provenance=provenance,
            )
            resumed_data, _, resumed = ANNOTATION_COMMON.load_annotation_data(
                input_path,
                output_path,
            )
            self.assertTrue(resumed)
            self.assertEqual(
                resumed_data.loc[0, "human_victim_visibility"],
                "no_victim",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["reviewed_rows"], 1)
            self.assertEqual(manifest["coder_id"], "anne")


class ContentMergeGuardTests(unittest.TestCase):
    def completion(self, *, production: bool = True, model: str = "gpt-5.1"):
        return {
            "production_full_corpus": production,
            "article_id_sha256": "article-hash",
            "model": model,
            "rows": 326093,
            "upstream_classifier": {
                "verified_final_classifier": True,
                "classifier_manifest_sha256": "classifier-hash",
                "political_corruption_articles": 326093,
            },
        }

    def test_production_merge_accepts_one_complete_shared_corpus(self):
        paths = {name: Path(f"{name}.csv") for name in ["a", "b", "c", "d"]}
        with patch.object(
            MERGE,
            "validate_completed_content_output",
            side_effect=[self.completion() for _ in paths],
        ):
            completions = MERGE.validate_production_completions(paths)
        self.assertEqual(len(completions), 4)

    def test_production_merge_rejects_validation_sample_scope(self):
        paths = {name: Path(f"{name}.csv") for name in ["a", "b", "c", "d"]}
        with patch.object(
            MERGE,
            "validate_completed_content_output",
            side_effect=[self.completion(production=False) for _ in paths],
        ):
            with self.assertRaisesRegex(ValueError, "all-country production"):
                MERGE.validate_production_completions(paths)

    def test_production_merge_rejects_mixed_models(self):
        paths = {name: Path(f"{name}.csv") for name in ["a", "b", "c", "d"]}
        values = [self.completion() for _ in range(3)] + [
            self.completion(model="other-model")
        ]
        with patch.object(
            MERGE,
            "validate_completed_content_output",
            side_effect=values,
        ):
            with self.assertRaisesRegex(ValueError, "different LLMs"):
                MERGE.validate_production_completions(paths)


class ContentEvaluationTests(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("sklearn"),
        "scikit-learn is not installed in the local artifact runtime",
    )
    def test_design_weighted_agreement_uses_saved_sampling_weights(self):
        data = pd.DataFrame(
            {
                "human_victim_visibility": ["no_victim", "no_victim"],
                "gpt_victim_visibility": ["no_victim", "concrete_victim"],
                "validation_weight": [1.0, 9.0],
            }
        )
        summary, _ = EVALUATE.evaluate(
            data,
            {"victim_visibility": ("human_victim_visibility", "victim_visibility")},
        )
        self.assertAlmostEqual(summary.loc[0, "agreement"], 0.5)
        self.assertAlmostEqual(summary.loc[0, "design_weighted_agreement"], 0.1)


class ContentWorkflowStructureTests(unittest.TestCase):
    def test_streamlit_content_app_has_core_workflow_controls(self):
        app = (
            ROOT / "content-classification/tools/annotation_streamlit_app.py"
        ).read_text(encoding="utf-8")
        for text in [
            "Publication country",
            "To code",
            "English translation",
            "Save and continue",
            "Coding complete",
            "Download backup",
        ]:
            self.assertIn(text, app)

    def test_readme_prefers_content_streamlit_app(self):
        readme = (ROOT / "content-classification/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "streamlit run content-classification/tools/annotation_streamlit_app.py",
            readme,
        )

    def test_validation_sampler_defines_streaming_arguments(self):
        with patch.object(sys, "argv", ["create_validation_sample.py"]):
            args = SAMPLER.parse_args()
        self.assertEqual(args.input_chunksize, 25_000)
        self.assertIsNone(args.random_sample)

    def test_final_runner_uses_accessible_model_and_strict_merge(self):
        runner = (
            ROOT
            / "content-classification/scripts/05_run_final_content_classification.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CONTENT_MODEL=${CONTENT_MODEL:-gpt-5.1}", runner)
        self.assertIn("00_verify_final_corpus.py", runner)
        self.assertIn("merge_content_labels.py", runner)
        self.assertNotIn("--allow-partial", runner)
        self.assertNotIn("--allow-errors", runner)

    def test_readme_names_canonical_merged_output(self):
        readme = (ROOT / "content-classification/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("political_corruption_content_categories_final.csv.gz", readme)
        self.assertNotIn("content_silver_labels_merged.csv.gz", readme)


if __name__ == "__main__":
    unittest.main()
