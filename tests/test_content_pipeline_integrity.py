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

import content_codebook as CODEBOOK
import content_prompts as PROMPTS


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
MODEL_REVIEW_COMMON = load_script(
    "model_review_common_test",
    "content-classification/tools/model_review_common.py",
)
CLASSIFY_CONTENT = load_script(
    "content_classify_entry_point_test",
    "content-classification/scripts/classify_content.py",
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
                victim_evidence="",
                corruption_frame="individualized",
                corruption_frame_evidence="Article text",
                case_location="domestic",
                accused_actor_visibility="individual_actor",
                accused_individual_evidence="Article text",
                accused_organization_evidence="",
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
            self.assertEqual(
                resumed_data.loc[0, "human_codebook_version"],
                CODEBOOK.CODEBOOK_VERSION,
            )
            self.assertEqual(
                resumed_data.loc[0, "human_codebook_sha256"],
                CODEBOOK.CODEBOOK_SHA256,
            )
            self.assertEqual(
                resumed_data.loc[0, "human_corruption_frame_evidence"],
                "Article text",
            )
            self.assertTrue(ANNOTATION_COMMON.reviewed_mask(resumed_data).all())
            old_version = resumed_data.copy()
            old_version.loc[0, "human_codebook_version"] = "older-codebook"
            self.assertFalse(ANNOTATION_COMMON.reviewed_mask(old_version).any())
            self.assertEqual(
                manifest["codebook"]["sha256"],
                CODEBOOK.CODEBOOK_SHA256,
            )

    def test_model_review_is_resumable_and_marked_as_model_assisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "review_sample_english.csv.gz"
            model_dir = root / "gpt51_labels"
            output_path = root / "review_sample_english_model_review_anne.csv.gz"
            model_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "article_id": "A",
                        "country": "France",
                        "sample_purpose": "codebook_development",
                        "translated_text_en": "The minister accepted a bribe.",
                        "article_text": "Le ministre a accepte un pot-de-vin.",
                    }
                ]
            ).to_csv(input_path, index=False, compression="gzip")

            model_rows = {
                "victim_visibility": {
                    "victim_visibility": "no_victim",
                    "victim_reasoning_brief": "No explicit harm is stated.",
                    "victim_confidence": 0.9,
                    "victim_entity": "none",
                    "victim_harm_evidence": "none",
                    "victim_corruption_harm_link_evidence": "none",
                },
                "corruption_frame": {
                    "corruption_frame": "individualized",
                    "frame_reasoning_brief": "The report concerns one minister.",
                    "frame_confidence": 0.8,
                    "frame_evidence": "The minister accepted a bribe.",
                },
                "case_location": {
                    "case_location": "domestic",
                    "abroad_reasoning_brief": "The case is in France.",
                    "abroad_confidence": 0.8,
                    "abroad_evidence": "The minister accepted a bribe.",
                },
                "accused_actor_visibility": {
                    "accused_actor_visibility": "individual_actor",
                    "accused_reasoning_brief": "A minister is accused.",
                    "accused_confidence": 0.95,
                    "accused_individual_evidence": "The minister accepted a bribe.",
                    "accused_organization_evidence": "none",
                },
            }
            for variable, row in model_rows.items():
                specification = MODEL_REVIEW_COMMON.VARIABLES[variable]
                record = {
                    "article_id": "A",
                    "llm_model": "gpt-5.1",
                    "prompt_version": f"{variable}-test",
                    "codebook_version": CODEBOOK.CODEBOOK_VERSION,
                    "codebook_sha256": CODEBOOK.CODEBOOK_SHA256,
                    **row,
                }
                path = MODEL_REVIEW_COMMON.model_output_paths(
                    input_path, model_dir
                )[variable]
                pd.DataFrame([record]).to_csv(
                    path,
                    index=False,
                    compression="gzip",
                )

            data, provenance, model_paths, resumed = (
                MODEL_REVIEW_COMMON.load_review_data(
                    input_path,
                    model_dir,
                    output_path,
                )
            )
            self.assertFalse(resumed)
            decisions = {variable: "confirm" for variable in model_rows}
            corrected = {variable: "" for variable in model_rows}
            comments = {variable: "" for variable in model_rows}
            MODEL_REVIEW_COMMON.record_review(
                data,
                0,
                decisions=decisions,
                corrected_labels=corrected,
                comments=comments,
                overall_comment="Reviewed against the article.",
                coder_id="anne",
                coder_first_name="Anne",
                session_id="session",
            )
            manifest_path = MODEL_REVIEW_COMMON.save_review_data(
                data,
                input_path=input_path,
                model_paths=model_paths,
                output_path=output_path,
                coder_id="anne",
                coder_first_name="Anne",
                session_id="session",
                provenance=provenance,
            )
            resumed_data, _, _, resumed = MODEL_REVIEW_COMMON.load_review_data(
                input_path,
                model_dir,
                output_path,
            )
            self.assertTrue(resumed)
            self.assertTrue(MODEL_REVIEW_COMMON.reviewed_mask(resumed_data).all())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["review_mode"], "model_assisted_adjudication")
            self.assertFalse(manifest["independent_human_validation"])
            self.assertEqual(manifest["reviewed_rows"], 1)

    def test_model_review_corrections_require_a_different_label_and_comment(self):
        errors = MODEL_REVIEW_COMMON.review_errors(
            model_labels={
                variable: specification["labels"][0]
                for variable, specification in MODEL_REVIEW_COMMON.VARIABLES.items()
            },
            decisions={
                variable: ("correct" if variable == "victim_visibility" else "confirm")
                for variable in MODEL_REVIEW_COMMON.VARIABLES
            },
            corrected_labels={
                variable: (
                    specification["labels"][0]
                    if variable == "victim_visibility"
                    else ""
                )
                for variable, specification in MODEL_REVIEW_COMMON.VARIABLES.items()
            },
            comments={variable: "" for variable in MODEL_REVIEW_COMMON.VARIABLES},
        )
        self.assertTrue(any("must differ" in error for error in errors))
        self.assertTrue(any("Explain" in error for error in errors))

    def test_machine_output_rows_record_the_canonical_codebook(self):
        spec = type("Spec", (), {"name": "test", "prompt_version": "test-v1"})()
        row = COMMON.base_output_row({}, spec, "gpt-5.1", 100)
        self.assertEqual(row["codebook_version"], CODEBOOK.CODEBOOK_VERSION)
        self.assertEqual(row["codebook_sha256"], CODEBOOK.CODEBOOK_SHA256)


class ContentMergeGuardTests(unittest.TestCase):
    def completion(self, *, production: bool = True, model: str = "gpt-5.1"):
        return {
            "production_full_corpus": production,
            "article_id_sha256": "article-hash",
            "model": model,
            "codebook_version": CODEBOOK.CODEBOOK_VERSION,
            "codebook_sha256": CODEBOOK.CODEBOOK_SHA256,
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
    def test_one_canonical_codebook_drives_all_prompts(self):
        self.assertEqual(
            PROMPTS.CONTENT_CODING_INSTRUCTIONS,
            CODEBOOK.CODEBOOK_MARKDOWN,
        )
        for variable, required_label in {
            "victim_visibility": "concrete_victim",
            "corruption_frame": "other_or_mixed",
            "case_location": "domestic",
            "accused_actor_visibility": "both_individual_and_organizational",
        }.items():
            instructions = CODEBOOK.codebook_for(variable)
            self.assertIn(required_label, instructions)
            self.assertIn(CODEBOOK.CODEBOOK_VERSION, instructions)

        actor_prompt = PROMPTS.build_accused_actor_prompt(
            "A company allegedly paid a bribe.",
            {"country": "France", "year": 2020},
        )
        self.assertIn("Mandatory Two-Test Method", actor_prompt)
        self.assertIn('"individual_actor_evidence"', actor_prompt)
        self.assertIn('"organizational_actor_evidence"', actor_prompt)
        self.assertNotIn("## 1. Victim Visibility", actor_prompt)

    def test_actor_label_is_derived_from_separate_evidence_tests(self):
        normalized = PROMPTS.normalize_accused_actor(
            {
                "individual_actor_visible": "yes",
                "individual_actor_evidence": "The minister accepted a bribe",
                "organizational_actor_visible": "yes",
                "organizational_actor_evidence": "The company paid the bribe",
                "accused_actor_visibility": "individual_actor",
            }
        )
        self.assertEqual(
            normalized["accused_actor_visibility"],
            "both_individual_and_organizational",
        )
        checked = COMMON.enforce_actor_evidence(
            normalized,
            "The minister accepted a bribe. The company paid the bribe.",
        )
        self.assertEqual(
            checked["accused_actor_visibility"],
            "both_individual_and_organizational",
        )
        self.assertEqual(checked["accused_individual_evidence_verbatim"], "yes")
        self.assertEqual(checked["accused_organization_evidence_verbatim"], "yes")

    def test_human_positive_labels_require_verbatim_evidence(self):
        data = ANNOTATION_COMMON.ensure_columns(
            pd.DataFrame(
                [{"article_id": "A", "article_text": "The company paid a bribe."}]
            )
        )
        row = data.loc[0].copy()
        row["human_victim_visibility"] = "no_victim"
        row["human_corruption_frame"] = "individualized"
        row["human_corruption_frame_evidence"] = "Invented frame evidence"
        row["human_case_location"] = "domestic"
        row["human_accused_actor_visibility"] = (
            "organizational_or_institutional_actor"
        )
        row["human_accused_organization_evidence"] = "The company paid a bribe."
        errors = ANNOTATION_COMMON.annotation_evidence_errors(row)
        self.assertTrue(any("frame evidence" in error for error in errors))
        self.assertFalse(any("organization evidence" in error for error in errors))

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
            "codebook_section",
            "Victim evidence",
            "Frame evidence",
            "Individual accused-actor evidence",
            "Organizational accused-actor evidence",
        ]:
            self.assertIn(text, app)

    def test_model_review_app_is_separate_and_exposes_review_controls(self):
        app = (
            ROOT / "content-classification/tools/model_review_streamlit_app.py"
        ).read_text(encoding="utf-8")
        for text in [
            "Model-assisted review",
            "Confirm the model label",
            "Correct the model label",
            "Cannot decide from this article",
            "Reviewer comment or rationale",
            "Download review backup",
            "adjudication data",
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

    def test_unified_classifier_uses_the_four_substantive_variable_names(self):
        self.assertEqual(
            set(PROMPTS.CONTENT_CLASSIFIERS),
            {
                "victim_visibility",
                "corruption_frame",
                "case_location",
                "accused_actor_visibility",
            },
        )
        variable, remaining = CLASSIFY_CONTENT.parse_variable(
            ["--variable", "case_location", "--limit", "2"]
        )
        self.assertEqual(variable, "case_location")
        self.assertEqual(remaining, ["--limit", "2"])

    def test_final_runner_uses_accessible_model_and_strict_merge(self):
        runner = (
            ROOT
            / "content-classification/scripts/05_run_final_content_classification.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CONTENT_MODEL=${CONTENT_MODEL:-gpt-5.1}", runner)
        self.assertIn("00_verify_final_corpus.py", runner)
        self.assertIn("classify_content.py", runner)
        self.assertIn("merge_content_labels.py", runner)
        self.assertNotIn("--allow-partial", runner)
        self.assertNotIn("--allow-errors", runner)

    def test_readme_names_canonical_merged_output(self):
        readme = (ROOT / "content-classification/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("political_corruption_content_categories_final.csv.gz", readme)
        self.assertNotIn("content_silver_labels_merged.csv.gz", readme)

    def test_obsolete_content_entry_points_are_removed(self):
        obsolete = [
            "tools/annotation_flask_app.py",
            "notebooks/01_inspect_content_classification.ipynb",
            "scripts/classify_abroad_case.py",
            "scripts/classify_accused_actor.py",
            "scripts/classify_corruption_frame.py",
            "scripts/classify_victim_visibility.py",
        ]
        root = ROOT / "content-classification"
        self.assertFalse([path for path in obsolete if (root / path).exists()])


if __name__ == "__main__":
    unittest.main()
