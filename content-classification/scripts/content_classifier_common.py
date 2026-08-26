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
from political_classifier.reproducibility import file_record, git_commit


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_CLASSIFIED_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files"
DEFAULT_OUTPUT_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification"
)
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
    parser.add_argument("--classified-rd-dir", default=DEFAULT_RD_CLASSIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--input-chunksize", type=int, default=25_000)
    parser.add_argument("--min-words", type=int, default=30)
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
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
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


def output_manifest(args: argparse.Namespace, spec: ClassifierSpec) -> dict:
    input_record = None
    if args.source == "csv" and args.input is not None and args.input.exists():
        input_record = file_record(args.input)
    classifier_manifest = args.classified_dir.parent / "classifier_run_manifest.json"
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
        "classified_dir": str(args.classified_dir),
        "classifier_run_manifest": classifier_record,
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
    result = spec.normalize_result(extract_json(raw))
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

    if args.overwrite:
        for path in [output_path, audit_path, manifest_path]:
            if path.exists():
                path.unlink()
    if output_path.exists() and not manifest_path.exists():
        raise ValueError(
            f"Existing output has no run manifest: {output_path}. "
            "Use a new output path or pass --overwrite."
        )
    validate_or_write_manifest(
        manifest_path,
        output_manifest(args, spec),
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
    for input_label, data in iter_input_frames(args):
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
    print("Done.", flush=True)
