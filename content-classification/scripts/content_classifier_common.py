"""Shared runner for article-level GPT content classifiers."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import ALL_COUNTRIES, LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL, RD_BASE_DIR
from content_prompts import ClassifierSpec
from political_classifier.final_corpus import (
    ClassifierRun,
    load_verified_classifier_run,
    verify_classified_country_file,
)
from political_classifier.reproducibility import file_record, git_commit, sha256_file


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_CLASSIFIED_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files"
DEFAULT_CLASSIFIER_OUTPUT_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier"
DEFAULT_CONTENT_ROOT = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CONTENT_ROOT / "final_gpt51"
DEFAULT_RD_CLASSIFIED_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "political_classifier",
    "classifier_outputs",
    "classified_country_files",
)

KEEP_COLUMNS = {
    "uri",
    "country",
    "date_parsed",
    "dateTime",
    "date",
    "published_at",
    "year",
    "month",
    "week",
    "source_uri",
    "source.uri",
    "word_count",
    "prob_political_corruption",
    "pred_political_corruption",
    "content_sample_id",
    "translated_text_en",
    "translated_text",
    "translation_notes",
    "translation_confidence",
    "translation_model",
    "translation_error",
    "article_text",
    "combined_text",
    "title",
    "body",
    "human_victim_visibility",
    "human_corruption_frame",
    "human_case_location",
    "human_abroad_case",
    "human_accused_actor_visibility",
    "human_accused_actor_visible",
    "human_notes",
    "human_coder_id",
    "human_coder_first_name",
    "human_code_session_id",
    "human_coded_at",
}

METADATA_COLUMNS = [
    "article_id",
    "uri",
    "country",
    "year",
    "month",
    "week",
    "source_uri",
    "source.uri",
    "word_count",
    "prob_political_corruption",
    "pred_political_corruption",
    "content_sample_id",
    "translation_confidence",
    "translation_model",
    "translation_error",
]

PRIMARY_LABEL_COLUMNS = {
    "victim_visibility": "victim_visibility",
    "corruption_frame": "corruption_frame",
    "abroad_case": "case_location",
    "accused_actor": "accused_actor_visibility",
}


class LLMResponseParseError(ValueError):
    """Preserve the prompt and raw model response when JSON parsing fails."""

    def __init__(self, message: str, *, prompt: str, raw_response: str):
        super().__init__(message)
        self.prompt = prompt
        self.raw_response = raw_response


def parse_common_args(description: str, default_output_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source",
        choices=["classified", "classified-webdav", "csv"],
        default="classified",
        help="Read local classified country files, archived WebDAV files, or one CSV file.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input CSV/CSV.GZ when --source csv.")
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument(
        "--classifier-output-dir",
        type=Path,
        default=DEFAULT_CLASSIFIER_OUTPUT_DIR,
        help=(
            "Directory containing classifier_run_manifest.json, "
            "classified_country_summary.csv, and selected_threshold.txt."
        ),
    )
    parser.add_argument("--classified-rd-dir", default=DEFAULT_RD_CLASSIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--input-chunksize", type=int, default=25_000)
    parser.add_argument(
        "--min-words",
        type=int,
        default=1,
        help="Minimum non-empty article length. Final production runs use 1.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N unfinished rows.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess rows with a non-empty llm_error in the existing output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start a fresh output and replace any existing labels/audit/manifest.",
    )
    parser.add_argument(
        "--keep-non-political",
        action="store_true",
        help="Do not filter to pred_political_corruption == 1.",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Optional random sample of unfinished rows, useful for pilot checks.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.set_defaults(default_output_name=default_output_name)
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def sha256_text(text: object) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def choose_text(data):
    if "translated_text_en" in data.columns:
        return data["translated_text_en"]
    if "translated_text" in data.columns:
        return data["translated_text"]
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]
    if {"title", "body"}.issubset(data.columns):
        return data["title"].fillna("").astype(str) + "\n" + data["body"].fillna("").astype(str)
    raise ValueError("Input data must contain article_text, combined_text, or title/body columns.")


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0), strict=False)
        raise


def compression_for_path(path: Path):
    return "gzip" if path.name.endswith(".gz") else None


def read_csv(path: Path, **kwargs):
    import pandas as pd

    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer", **kwargs)


def write_csv_atomic(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    data.to_csv(tmp_path, index=False, compression=compression_for_path(output_path))
    tmp_path.replace(output_path)


def append_audit_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def is_non_retryable_model_error(exc: Exception) -> bool:
    """Identify access/configuration failures that retries cannot repair."""
    message = f"{type(exc).__name__}: {exc}".casefold()
    markers = [
        "authenticationerror",
        "permissiondeniederror",
        "invalid_api_key",
        "key_model_access_denied",
        "model_not_found",
        "does not exist or you do not have access",
        "key not allowed to access model",
    ]
    return any(marker in message for marker in markers)


def country_path(args: argparse.Namespace, country: str) -> Path:
    return args.classified_dir / f"{country}_classified.csv.gz"


def country_rd_path(args: argparse.Namespace, country: str) -> str:
    return posixpath.join(args.classified_rd_dir, f"{country}_classified.csv.gz")


def read_country_file(args: argparse.Namespace, country: str):
    import pandas as pd

    if args.source == "classified-webdav":
        from rd_utils import webdav_download_bytes

        rd_path = country_rd_path(args, country)
        data = webdav_download_bytes(rd_path)
        return pd.read_csv(
            io.BytesIO(data),
            compression="gzip" if rd_path.endswith(".gz") else "infer",
            usecols=lambda column: column in KEEP_COLUMNS,
        )

    path = country_path(args, country)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=lambda column: column in KEEP_COLUMNS)


def prepare_frame(data, country: str | None, args: argparse.Namespace):
    import pandas as pd

    data = data.copy()
    if country is not None:
        data["country"] = country
    elif "country" not in data.columns:
        data["country"] = ""

    data["article_text"] = choose_text(data).map(normalize_text)
    data = data[data["article_text"].str.split().str.len().fillna(0).ge(args.min_words)].copy()

    if not args.keep_non_political:
        if "pred_political_corruption" not in data.columns:
            raise ValueError(
                "Input is missing pred_political_corruption. Refusing to classify an "
                "unfiltered corpus; pass --keep-non-political only when intentional."
            )
        data = data[pd.to_numeric(data["pred_political_corruption"], errors="coerce").eq(1)].copy()

    if "year" not in data.columns:
        date_column = next(
            (column for column in ["date_parsed", "dateTime", "date", "published_at"] if column in data.columns),
            None,
        )
        if date_column:
            data["date_parsed"] = pd.to_datetime(data[date_column], errors="coerce", utc=True)
            data["year"] = data["date_parsed"].dt.year
    data["year"] = pd.to_numeric(data.get("year", ""), errors="coerce")

    if "month" not in data.columns:
        date_column = next(
            (column for column in ["date_parsed", "dateTime", "date", "published_at"] if column in data.columns),
            None,
        )
        if date_column:
            date_values = pd.to_datetime(data[date_column], errors="coerce", utc=True)
            data["month"] = date_values.dt.month

    if "source_uri" not in data.columns and "source.uri" in data.columns:
        data["source_uri"] = data["source.uri"]

    if "uri" not in data.columns:
        data["uri"] = ""
    sample_id = (
        data["content_sample_id"].fillna("").astype(str)
        if "content_sample_id" in data.columns
        else pd.Series("", index=data.index)
    )
    uri_id = data["uri"].fillna("").astype(str)
    country_id = data["country"].fillna("").astype(str)
    uri_article_id = country_id + "::" + uri_id
    text_article_id = country_id + "::text::" + data["article_text"].map(sha256_text)
    data["article_id"] = sample_id.where(sample_id.str.strip().ne(""), uri_article_id)
    missing_uri = sample_id.str.strip().eq("") & uri_id.str.strip().eq("")
    data.loc[missing_uri, "article_id"] = text_article_id.loc[missing_uri]
    missing_id = data["article_id"].str.strip().eq("")
    if missing_id.any():
        fallback = (
            data["country"].astype(str)
            + "::"
            + data["year"].fillna("").astype(str)
            + "::"
            + data.index.astype(str)
        )
        data.loc[missing_id, "article_id"] = fallback[missing_id]

    data["input_text_sha256"] = data["article_text"].map(sha256_text)

    return data


def iter_input_frames(args: argparse.Namespace):
    import pandas as pd

    if args.source == "csv":
        if args.input is None:
            raise ValueError("--input is required when --source csv.")
        if args.random_sample is not None:
            data = read_csv(args.input, usecols=lambda column: column in KEEP_COLUMNS)
            yield "csv random-sample pool", prepare_frame(data, None, args)
            return
        for chunk_number, chunk in enumerate(
            read_csv(
                args.input,
                usecols=lambda column: column in KEEP_COLUMNS,
                chunksize=args.input_chunksize,
            ),
            start=1,
        ):
            yield f"csv chunk {chunk_number}", prepare_frame(chunk, None, args)
        return

    for country in args.countries:
        print(f"Loading {country}", flush=True)
        if args.source == "classified-webdav":
            from rd_utils import webdav_download_to_path

            cache_dir = args.output_dir / "_webdav_input_cache"
            local_path = cache_dir / f"{country}_classified.csv.gz"
            if not local_path.exists():
                webdav_download_to_path(country_rd_path(args, country), local_path)
        else:
            local_path = country_path(args, country)
            if not local_path.exists():
                raise FileNotFoundError(local_path)

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                local_path,
                usecols=lambda column: column in KEEP_COLUMNS,
                chunksize=args.input_chunksize,
            ),
            start=1,
        ):
            frame = prepare_frame(chunk, country, args)
            yield f"{country} chunk {chunk_number}", frame


def load_input(args: argparse.Namespace):
    import pandas as pd

    if args.source == "csv":
        if args.input is None:
            raise ValueError("--input is required when --source csv.")
        data = read_csv(args.input, usecols=lambda column: column in KEEP_COLUMNS)
        return prepare_frame(data, None, args)

    frames = []
    for country in args.countries:
        print(f"Loading {country}", flush=True)
        frame = prepare_frame(read_country_file(args, country), country, args)
        frames.append(frame)
        print(f"{country}: {len(frame):,} political-corruption rows after filters", flush=True)

    if not frames:
        raise FileNotFoundError("No classified country files loaded.")
    return pd.concat(frames, ignore_index=True)


def existing_done_ids(existing, retry_errors: bool) -> set[str]:
    if existing.empty or "article_id" not in existing.columns:
        return set()
    if retry_errors and "llm_error" in existing.columns:
        ok = existing[existing["llm_error"].fillna("").astype(str).str.strip().eq("")]
        return set(ok["article_id"].dropna().astype(str))
    return set(existing["article_id"].dropna().astype(str))


def done_ids_for_frame(existing, frame, retry_errors: bool) -> set[str]:
    if existing.empty:
        return set()
    candidates = existing
    if retry_errors and "llm_error" in candidates.columns:
        candidates = candidates[
            candidates["llm_error"].fillna("").astype(str).str.strip().eq("")
        ]
    if "input_text_sha256" not in candidates.columns:
        return set()
    current_hash = frame.set_index("article_id")["input_text_sha256"].astype(str)
    candidates = candidates[candidates["article_id"].astype(str).isin(current_hash.index)].copy()
    matches = candidates["input_text_sha256"].astype(str).eq(
        candidates["article_id"].astype(str).map(current_hash)
    )
    return set(candidates.loc[matches, "article_id"].astype(str))


def deduplicate_output(data):
    if data.empty or "article_id" not in data.columns:
        return data
    return data.drop_duplicates(subset=["article_id"], keep="last").reset_index(drop=True)


def content_sample_manifest_path(sample_path: Path) -> Path:
    return sample_path.with_name(sample_path.name + ".sample_manifest.json")


def content_completion_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".complete.json")


def validate_content_sample(sample_path: Path) -> dict | None:
    """Validate a canonical content sample when its sidecar is present."""
    manifest_path = content_sample_manifest_path(sample_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_record = (manifest.get("outputs") or {}).get("sample") or {}
    if sample_record.get("sha256") != sha256_file(sample_path):
        raise ValueError(
            f"Content sample differs from its manifest: {sample_path}. "
            "Recreate the sample before continuing."
        )
    upstream = manifest.get("upstream_classifier") or {}
    if not upstream.get("verified_final_classifier"):
        raise ValueError(
            "Content sample is not linked to the verified final classifier run."
        )
    if not manifest.get("political_only"):
        raise ValueError("Content samples must contain political-corruption rows only.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "upstream_classifier": upstream,
    }


def validate_completed_content_output(output_path: Path) -> dict:
    """Verify a content label file, its immutable run settings, and completion."""
    completion_path = content_completion_path(output_path)
    if not completion_path.exists():
        raise FileNotFoundError(
            f"Missing completion marker for final content output: {completion_path}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    output_record = completion.get("output") or {}
    if output_record.get("sha256") != sha256_file(output_path):
        raise ValueError(f"Content output differs from its completion marker: {output_path}")
    run_manifest_path = output_path.with_name(output_path.name + ".run.json")
    run_record = completion.get("run_manifest") or {}
    if run_record.get("sha256") != sha256_file(run_manifest_path):
        raise ValueError(
            f"Content run manifest differs from its completion marker: {run_manifest_path}"
        )
    audit_record = completion.get("audit") or {}
    if not audit_record.get("path") or not audit_record.get("sha256"):
        raise ValueError(f"Content completion marker has no audit record: {output_path}")
    audit_path = Path(str(audit_record.get("path", "")))
    if not audit_path.is_absolute():
        audit_path = output_path.parent / audit_path.name
    if audit_record.get("sha256") != sha256_file(audit_path):
        raise ValueError(
            f"Content audit log differs from its completion marker: {audit_path}"
        )
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    for key in ["classifier_name", "prompt_version", "model", "source"]:
        if completion.get(key) != run_manifest.get(key):
            raise ValueError(
                f"Content completion and run manifest disagree on {key}: {output_path}"
            )
    expected_production_scope = bool(
        run_manifest.get("source") in {"classified", "classified-webdav"}
        and not run_manifest.get("keep_non_political")
        and set(run_manifest.get("countries") or []) == set(ALL_COUNTRIES)
    )
    if completion.get("production_full_corpus") != expected_production_scope:
        raise ValueError(
            f"Content completion marker has an inconsistent run scope: {output_path}"
        )
    upstream = completion.get("upstream_classifier") or {}
    if not upstream.get("verified_final_classifier"):
        raise ValueError(
            f"Final content output is not linked to the verified classifier: {output_path}"
        )
    if upstream != (run_manifest.get("upstream_classifier") or {}):
        raise ValueError(
            f"Content completion and run manifest use different upstream runs: {output_path}"
        )
    import pandas as pd

    article_ids = pd.read_csv(
        output_path,
        compression="gzip" if output_path.name.endswith(".gz") else "infer",
        usecols=["article_id"],
    )["article_id"].fillna("").astype(str)
    if article_ids.str.strip().eq("").any():
        raise ValueError(f"Content output contains blank article IDs: {output_path}")
    if article_ids.duplicated().any():
        raise ValueError(f"Content output contains duplicate article IDs: {output_path}")
    if len(article_ids) != int(completion.get("rows", -1)):
        raise ValueError(
            f"Content output row count differs from its completion marker: {output_path}"
        )
    article_id_hash = hashlib.sha256(
        "\n".join(sorted(article_ids)).encode("utf-8")
    ).hexdigest()
    if article_id_hash != completion.get("article_id_sha256"):
        raise ValueError(
            f"Content output article IDs differ from its completion marker: {output_path}"
        )
    return {
        **completion,
        "completion": completion,
        "completion_path": completion_path,
        "completion_sha256": sha256_file(completion_path),
        "run_manifest": run_manifest,
        "run_manifest_path": run_manifest_path,
        "audit_path": audit_path,
        "upstream_classifier": upstream,
    }


def verify_production_input(args: argparse.Namespace) -> ClassifierRun | None:
    """Fail before LLM calls when classified inputs are not the final corpus."""
    if args.source == "csv":
        return None

    require_local = args.source == "classified"
    classifier_run = load_verified_classifier_run(
        args.classifier_output_dir,
        list(ALL_COUNTRIES),
        classified_dir=args.classified_dir,
        require_local_country_files=require_local,
    )
    for country in args.countries:
        if args.source == "classified-webdav":
            from rd_utils import webdav_download_to_path

            cache_dir = args.output_dir / "_webdav_input_cache"
            local_path = cache_dir / f"{country}_classified.csv.gz"
            if not local_path.exists():
                webdav_download_to_path(country_rd_path(args, country), local_path)
        else:
            local_path = country_path(args, country)
        verify_classified_country_file(
            local_path,
            country=country,
            classifier_run=classifier_run,
            chunksize=args.input_chunksize,
        )
        print(
            f"Verified {country} against final classifier threshold "
            f"{classifier_run.threshold:.2f}",
            flush=True,
        )
    print(
        "Verified final source-filtered political-corruption corpus: "
        f"{classifier_run.political_articles:,} articles across "
        f"{len(classifier_run.summary):,} countries",
        flush=True,
    )
    return classifier_run


def output_manifest(
    args: argparse.Namespace,
    spec: ClassifierSpec,
    classifier_run: ClassifierRun | None,
) -> dict:
    input_record = None
    if args.source == "csv" and args.input is not None and args.input.exists():
        input_record = file_record(args.input)
    sample_provenance = (
        validate_content_sample(args.input)
        if args.source == "csv" and args.input is not None
        else None
    )
    classifier_manifest = args.classifier_output_dir / "classifier_run_manifest.json"
    classifier_record = (
        file_record(classifier_manifest)
        if args.source == "classified" and classifier_manifest.exists()
        else None
    )
    return {
        "schema_version": 1,
        "git_commit": git_commit(PROJECT_ROOT),
        "classifier_name": spec.name,
        "prompt_version": spec.prompt_version,
        "model": args.model,
        "temperature": 0,
        "max_chars": args.max_chars,
        "min_words": args.min_words,
        "source": args.source,
        "input": str(args.input) if args.input else None,
        "input_file": input_record,
        "input_sample_manifest": (
            file_record(sample_provenance["manifest_path"])
            if sample_provenance
            else None
        ),
        "classified_dir": str(args.classified_dir),
        "classifier_output_dir": str(args.classifier_output_dir),
        "classifier_run_manifest": classifier_record,
        "upstream_classifier": (
            classifier_run.provenance_record()
            if classifier_run
            else (
                sample_provenance["upstream_classifier"]
                if sample_provenance
                else None
            )
        ),
        "classified_rd_dir": args.classified_rd_dir,
        "countries": list(args.countries),
        "keep_non_political": bool(args.keep_non_political),
    }


def validate_or_write_manifest(path: Path, expected: dict, overwrite: bool) -> None:
    if overwrite or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise ValueError(
            f"Existing output manifest does not match this run: {path}. "
            "Use a new output path or pass --overwrite."
        )


def classify_article(client, spec: ClassifierSpec, row: dict, model: str, max_chars: int) -> tuple[dict, str, str]:
    article_text = normalize_text(row.get("article_text", ""))[:max_chars]
    prompt = spec.build_prompt(article_text, row)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        parsed = extract_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise LLMResponseParseError(
            f"Could not parse {spec.name} response as JSON: {exc}",
            prompt=prompt,
            raw_response=raw,
        ) from exc
    result = spec.normalize_result(parsed)
    if spec.name == "victim_visibility":
        result = enforce_victim_evidence(result, article_text)
    return result, prompt, raw


def evidence_is_verbatim(evidence: object, article_text: str) -> bool:
    evidence = normalize_text(evidence)
    article_text = normalize_text(article_text)
    if evidence.lower() in {
        "",
        "none",
        "no evidence",
        "not stated",
        "not explicit",
        "n/a",
        "nan",
    }:
        return False
    evidence = evidence.strip("\"'“”‘’")
    return bool(evidence) and evidence.casefold() in article_text.casefold()


def enforce_victim_evidence(result: dict, article_text: str) -> dict:
    result = result.copy()
    entity_verbatim = evidence_is_verbatim(
        result.get("victim_entity", ""), article_text
    )
    harm_verbatim = evidence_is_verbatim(
        result.get("victim_harm_evidence", ""), article_text
    )
    link_verbatim = evidence_is_verbatim(
        result.get("victim_corruption_harm_link_evidence", ""), article_text
    )
    result["victim_entity_verbatim"] = "yes" if entity_verbatim else "no"
    result["victim_harm_evidence_verbatim"] = "yes" if harm_verbatim else "no"
    result["victim_corruption_harm_link_evidence_verbatim"] = (
        "yes" if link_verbatim else "no"
    )

    if result.get("victim_visibility") in {
        "concrete_victim",
        "institutional_societal_victim",
    } and not (entity_verbatim and harm_verbatim and link_verbatim):
        result["victim_visibility"] = "no_victim"
        result["victim_visible"] = "no"
        result["concrete_victim_visible"] = "no"
        result["institutional_societal_victim_visible"] = "no"
        reason = str(result.get("victim_reasoning_brief", "")).strip()
        suffix = (
            "Positive label removed because the victim entity and both evidence "
            "fields were not verbatim article quotations."
        )
        result["victim_reasoning_brief"] = f"{reason} {suffix}".strip()
    return result


def base_output_row(row: dict, spec: ClassifierSpec, model: str, max_chars: int) -> dict:
    out = {column: row.get(column, "") for column in METADATA_COLUMNS if column in row}
    out["classifier_name"] = spec.name
    out["prompt_version"] = spec.prompt_version
    out["llm_model"] = model
    out["max_chars"] = max_chars
    out["input_text_sha256"] = row.get("input_text_sha256", "")
    out["llm_coded_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    checkpoint = deduplicate_output(
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    )
    write_csv_atomic(checkpoint, output_path)
    print(f"Saved checkpoint: {output_path} ({len(checkpoint):,} rows)", flush=True)


def select_unfinished(data, done_ids: set[str], args: argparse.Namespace):
    unfinished = data[~data["article_id"].astype(str).isin(done_ids)].copy()
    if args.random_sample is not None:
        unfinished = unfinished.sample(n=min(args.random_sample, len(unfinished)), random_state=args.random_state)
    if args.limit is not None:
        unfinished = unfinished.head(args.limit).copy()
    return unfinished


def run_classifier(spec: ClassifierSpec) -> None:
    args = parse_common_args(
        description=f"Classify political-corruption articles for {spec.name}.",
        default_output_name=spec.default_output_name,
    )

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")
    if args.random_sample is not None and args.source != "csv":
        raise ValueError(
            "--random-sample is supported only for a prepared CSV sample. Use "
            "create_validation_sample.py for a reproducible corpus-level sample."
        )
    if args.retries < 1 or args.save_every < 1 or args.input_chunksize < 1:
        raise ValueError("--retries, --save-every, and --input-chunksize must be positive.")
    if args.max_chars < 1 or args.min_words < 0:
        raise ValueError("--max-chars must be positive and --min-words non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when supplied.")
    if args.random_sample is not None and args.random_sample < 1:
        raise ValueError("--random-sample must be positive when supplied.")

    output_path = args.output or (args.output_dir / spec.default_output_name)
    audit_path = output_path.with_name(output_path.stem + "_audit.jsonl")
    manifest_path = output_path.with_name(output_path.name + ".run.json")
    completion_path = content_completion_path(output_path)
    classifier_run = verify_production_input(args)

    if args.overwrite:
        for path in [output_path, audit_path, manifest_path, completion_path]:
            if path.exists():
                path.unlink()
    if output_path.exists() and not manifest_path.exists():
        raise ValueError(
            f"Existing output has no run manifest: {output_path}. "
            "Use a new output path or pass --overwrite."
        )
    validate_or_write_manifest(
        manifest_path,
        output_manifest(args, spec, classifier_run),
        overwrite=args.overwrite,
    )

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)

    if output_path.exists():
        existing = deduplicate_output(read_csv(output_path))
        print(f"Resuming from {output_path}; existing rows: {len(existing):,}", flush=True)
    else:
        existing = pd.DataFrame()
    print(f"Output: {output_path}", flush=True)
    print(f"Model: {args.model}", flush=True)

    new_rows: list[dict] = []
    processed_this_run = 0
    eligible_article_ids: set[str] = set()
    for input_label, data in iter_input_frames(args):
        eligible_article_ids.update(data["article_id"].dropna().astype(str))
        done_ids = done_ids_for_frame(existing, data, args.retry_errors)
        unfinished = data[~data["article_id"].astype(str).isin(done_ids)].copy()
        if args.random_sample is not None:
            unfinished = unfinished.sample(
                n=min(args.random_sample, len(unfinished)),
                random_state=args.random_state,
            )
        if args.limit is not None:
            remaining = max(0, args.limit - processed_this_run)
            unfinished = unfinished.head(remaining)
        print(
            f"{input_label}: {len(data):,} eligible; {len(unfinished):,} to process",
            flush=True,
        )

        for _, row in tqdm(
            unfinished.iterrows(),
            total=len(unfinished),
            desc=f"LLM {spec.name} ({input_label})",
        ):
            row_dict = row.to_dict()
            out = base_output_row(row_dict, spec, args.model, args.max_chars)
            prompt = spec.build_prompt(
                normalize_text(row_dict.get("article_text", ""))[: args.max_chars],
                row_dict,
            )
            raw_response = ""
            parsed = {}
            fatal_error: Exception | None = None

            for attempt in range(1, args.retries + 1):
                try:
                    parsed, prompt, raw_response = classify_article(
                        client=client,
                        spec=spec,
                        row=row_dict,
                        model=args.model,
                        max_chars=args.max_chars,
                    )
                    out.update(parsed)
                    out["llm_error"] = ""
                    break
                except Exception as exc:
                    if isinstance(exc, LLMResponseParseError):
                        prompt = exc.prompt
                        raw_response = exc.raw_response
                    out["llm_error"] = repr(exc)
                    print(
                        f"Error on article_id={out.get('article_id', '')} "
                        f"attempt {attempt}/{args.retries}: {exc!r}",
                        flush=True,
                    )
                    if is_non_retryable_model_error(exc):
                        fatal_error = exc
                        break
                    if attempt < args.retries:
                        time.sleep(args.retry_sleep * attempt)

            append_audit_record(
                audit_path,
                {
                    "article_id": out.get("article_id", ""),
                    "classifier_name": spec.name,
                    "prompt_version": spec.prompt_version,
                    "model": args.model,
                    "max_chars": args.max_chars,
                    "input_text_sha256": out.get("input_text_sha256", ""),
                    "coded_at_utc": out.get("llm_coded_at_utc", ""),
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "parsed_response": parsed,
                    "error": out.get("llm_error", ""),
                },
            )
            new_rows.append(out)
            processed_this_run += 1

            if fatal_error is not None:
                write_checkpoint(existing, new_rows, output_path)
                raise RuntimeError(
                    f"Non-retryable LLM model/access error for {args.model!r}. "
                    f"The failed request was saved to {audit_path}."
                ) from fatal_error

            if len(new_rows) % args.save_every == 0:
                write_checkpoint(existing, new_rows, output_path)
                time.sleep(args.sleep)

        if new_rows:
            write_checkpoint(existing, new_rows, output_path)
            existing = deduplicate_output(read_csv(output_path))
            new_rows = []
        if args.limit is not None and processed_this_run >= args.limit:
            break

    write_checkpoint(existing, new_rows, output_path)
    final = deduplicate_output(read_csv(output_path))
    errors = (
        final["llm_error"].fillna("").astype(str).str.strip().ne("")
        if "llm_error" in final.columns
        else pd.Series(False, index=final.index)
    )
    label_column = PRIMARY_LABEL_COLUMNS[spec.name]
    invalid_labels = (
        final[label_column].fillna("").astype(str).str.strip().eq("")
        if label_column in final.columns
        else pd.Series(True, index=final.index)
    )
    if errors.any() or invalid_labels.any():
        completion_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.name} run is incomplete: {int(errors.sum()):,} error row(s) "
            f"and {int(invalid_labels.sum()):,} blank-label row(s). Resume with "
            "--retry-errors; final merging will reject this output."
        )

    if args.limit is not None or args.random_sample is not None:
        completion_path.unlink(missing_ok=True)
        print(
            f"Partial diagnostic run finished: {len(final):,} saved row(s); "
            "no completion marker written.",
            flush=True,
        )
        return

    expected_rows = len(eligible_article_ids)
    if len(final) != expected_rows:
        completion_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"{spec.name} output has {len(final):,} rows but the eligible input "
            f"contains {expected_rows:,} unique article IDs."
        )
    if classifier_run is not None:
        selected = classifier_run.summary[
            classifier_run.summary["country"].astype(str).isin(args.countries)
        ]
        expected_classifier_rows = int(
            selected[
                "total_articles"
                if args.keep_non_political
                else "predicted_political_corruption"
            ].sum()
        )
        if expected_rows != expected_classifier_rows:
            completion_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Eligible content rows ({expected_rows:,}) do not equal the "
                f"verified classifier total ({expected_classifier_rows:,}). "
                "Use --min-words 1 and inspect blank article text before continuing."
            )

    article_id_hash = hashlib.sha256(
        "\n".join(sorted(eligible_article_ids)).encode("utf-8")
    ).hexdigest()
    saved_run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "classifier_name": spec.name,
        "prompt_version": spec.prompt_version,
        "model": args.model,
        "source": args.source,
        "countries": list(args.countries),
        "production_full_corpus": bool(
            classifier_run is not None
            and not args.keep_non_political
            and set(args.countries) == set(ALL_COUNTRIES)
        ),
        "rows": len(final),
        "article_id_sha256": article_id_hash,
        "output": file_record(output_path),
        "run_manifest": file_record(manifest_path),
        "audit": file_record(audit_path),
        "upstream_classifier": (
            saved_run_manifest.get("upstream_classifier")
        ),
    }
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved completion marker: {completion_path}", flush=True)
    print(f"Done. Verified {len(final):,} complete rows.", flush=True)
