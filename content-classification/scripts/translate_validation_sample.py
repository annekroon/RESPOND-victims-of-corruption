"""Translate sampled political-corruption articles to English with GPT.

The script is resumable, validates canonical sample provenance, and writes
checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL
from content_classifier_common import (
    append_audit_record,
    content_sample_manifest_path,
    deduplicate_output,
    is_non_retryable_model_error,
    normalize_text,
    read_csv,
    sha256_text,
    validate_content_sample,
    validate_or_write_manifest,
    write_csv_atomic,
)
from political_classifier.reproducibility import file_record, git_commit


DEFAULT_INPUT = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification/validation_final/content_validation_final_n500.csv.gz"
)
TRANSLATION_PROMPT_VERSION = "faithful_translation_v1"


class TranslationResponseParseError(ValueError):
    """Preserve a malformed translation response for the audit trail."""

    def __init__(self, message: str, *, prompt: str, raw_response: str):
        super().__init__(message)
        self.prompt = prompt
        self.raw_response = raw_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate a content-validation sample to English with GPT."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--text-column", default="article_text")
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess rows with a non-empty translation_error.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def output_path_from_input(input_path: Path) -> Path:
    if input_path.name.endswith(".csv.gz"):
        return input_path.with_name(input_path.name.replace(".csv.gz", "_english.csv.gz"))
    if input_path.suffix == ".csv":
        return input_path.with_name(input_path.stem + "_english.csv")
    return input_path.with_name(input_path.name + "_english.csv.gz")


def ensure_article_id(data):
    import pandas as pd

    data = data.copy()
    sample_id = (
        data["content_sample_id"].fillna("").astype(str)
        if "content_sample_id" in data.columns
        else pd.Series("", index=data.index)
    )
    uri = (
        data["uri"].fillna("").astype(str)
        if "uri" in data.columns
        else pd.Series("", index=data.index)
    )
    country = (
        data["country"].fillna("").astype(str)
        if "country" in data.columns
        else pd.Series("", index=data.index)
    )
    uri_id = country + "::" + uri
    data["article_id"] = sample_id.where(sample_id.str.strip().ne(""), uri_id)
    missing_uri = sample_id.str.strip().eq("") & uri.str.strip().eq("")
    text_column = next(
        (
            column
            for column in ["article_text", "combined_text", "body", "translated_text"]
            if column in data.columns
        ),
        None,
    )
    if text_column is not None:
        text_id = country + "::text::" + data[text_column].map(sha256_text)
        data.loc[missing_uri, "article_id"] = text_id.loc[missing_uri]
    missing_id = data["article_id"].fillna("").astype(str).str.strip().eq("")
    if missing_id.any():
        fallback = data.index.astype(str)
        if "country" in data.columns:
            fallback = data["country"].fillna("").astype(str) + "::" + fallback
        data.loc[missing_id, "article_id"] = fallback[missing_id]
    return data


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


def confidence_value(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value if value is not None else ""
    if numeric > 1:
        numeric = numeric / 100
    return max(0.0, min(1.0, numeric))


def build_translation_prompt(article_text: str, metadata: dict) -> str:
    country = metadata.get("country", "")
    year = metadata.get("year", "")
    return f"""
Translate the following news article into clear, faithful English.

Instructions:
- Preserve names of people, organizations, parties, institutions, places, and legal cases.
- Preserve uncertainty, allegations, denials, and legal status exactly.
- Do not summarize, interpret, classify, or add background knowledge.
- If a passage is already in English, keep it in fluent English.
- If the article is truncated or contains boilerplate, translate only the substantive article text.

Return valid JSON only:
{{
  "translated_text": "full English translation",
  "translation_notes": "brief note on any truncation, unreadable text, or mixed-language issues",
  "confidence": 0.0
}}

Publication country: {country}
Publication year: {year}

Article text:
{article_text}
""".strip()


def translate_article(client, row: dict, model: str, text_column: str, max_chars: int) -> tuple[dict, str, str]:
    article_text = normalize_text(row.get(text_column, ""))[:max_chars]
    prompt = build_translation_prompt(article_text, row)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    try:
        parsed = extract_json(raw)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise TranslationResponseParseError(
            f"Could not parse translation response as JSON: {exc}",
            prompt=prompt,
            raw_response=raw,
        ) from exc
    return (
        {
            "translated_text_en": parsed.get("translated_text", ""),
            "translation_notes": parsed.get("translation_notes", ""),
            "translation_confidence": confidence_value(parsed.get("confidence", "")),
        },
        prompt,
        raw,
    )


def done_article_ids(existing, data, retry_errors: bool) -> set[str]:
    if existing.empty or "article_id" not in existing.columns:
        return set()
    candidates = existing
    if retry_errors and "translation_error" in candidates.columns:
        candidates = candidates[
            candidates["translation_error"].fillna("").astype(str).str.strip().eq("")
        ]
    if "input_text_sha256" not in candidates.columns:
        return set()
    current = data.set_index("article_id")["input_text_sha256"].astype(str)
    candidates = candidates[candidates["article_id"].astype(str).isin(current.index)].copy()
    matches = candidates["input_text_sha256"].astype(str).eq(
        candidates["article_id"].astype(str).map(current)
    )
    return set(candidates.loc[matches, "article_id"].astype(str))


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    checkpoint = deduplicate_output(
        pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    )
    write_csv_atomic(checkpoint, output_path)
    print(f"Saved checkpoint: {output_path} ({len(checkpoint):,} rows)", flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.retries < 1 or args.save_every < 1 or args.max_chars < 1:
        raise ValueError("--retries, --save-every, and --max-chars must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when supplied.")

    output_path = args.output or output_path_from_input(args.input)
    audit_path = output_path.with_name(output_path.stem + "_translation_audit.jsonl")
    manifest_path = output_path.with_name(output_path.name + ".run.json")
    translated_sample_manifest_path = content_sample_manifest_path(output_path)
    source_sample = validate_content_sample(args.input)

    if args.overwrite:
        for path in [
            output_path,
            audit_path,
            manifest_path,
            translated_sample_manifest_path,
        ]:
            if path.exists():
                path.unlink()
    if output_path.exists() and not manifest_path.exists():
        raise ValueError(
            f"Existing output has no run manifest: {output_path}. "
            "Use a new output path or pass --overwrite."
        )
    validate_or_write_manifest(
        manifest_path,
        {
            "schema_version": 1,
            "git_commit": git_commit(PROJECT_ROOT),
            "task": "translation",
            "prompt_version": TRANSLATION_PROMPT_VERSION,
            "model": args.model,
            "input": str(args.input),
            "input_file": file_record(args.input),
            "input_sample_manifest": (
                file_record(source_sample["manifest_path"])
                if source_sample
                else None
            ),
            "upstream_classifier": (
                source_sample["upstream_classifier"] if source_sample else None
            ),
            "text_column": args.text_column,
            "max_chars": args.max_chars,
            "temperature": 0,
        },
        overwrite=args.overwrite,
    )

    data = ensure_article_id(read_csv(args.input))
    if args.text_column not in data.columns:
        raise ValueError(f"Text column not found: {args.text_column}")
    data["input_text_sha256"] = data[args.text_column].map(sha256_text)

    if output_path.exists():
        existing = deduplicate_output(read_csv(output_path))
        done_ids = done_article_ids(existing, data, args.retry_errors)
        print(f"Resuming from {output_path}; already done: {len(done_ids):,}", flush=True)
    else:
        existing = pd.DataFrame()
        done_ids = set()

    unfinished = data[~data["article_id"].astype(str).isin(done_ids)].copy()
    if args.limit is not None:
        unfinished = unfinished.head(args.limit).copy()

    print(f"Input: {args.input}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Rows to translate now: {len(unfinished):,}", flush=True)

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    new_rows: list[dict] = []

    for _, row in tqdm(unfinished.iterrows(), total=len(unfinished), desc="Translate articles"):
        out = row.to_dict()
        out["translation_prompt_version"] = TRANSLATION_PROMPT_VERSION
        out["translation_max_chars"] = args.max_chars
        out["translation_coded_at_utc"] = datetime.now(timezone.utc).isoformat()
        prompt = build_translation_prompt(
            normalize_text(out.get(args.text_column, ""))[: args.max_chars], out
        )
        raw = ""
        parsed_translation = {}
        fatal_error: Exception | None = None
        for attempt in range(1, args.retries + 1):
            try:
                translated, prompt, raw = translate_article(
                    client=client,
                    row=out,
                    model=args.model,
                    text_column=args.text_column,
                    max_chars=args.max_chars,
                )
                out.update(translated)
                parsed_translation = translated
                out["translation_model"] = args.model
                out["translation_error"] = ""
                break
            except Exception as exc:
                if isinstance(exc, TranslationResponseParseError):
                    prompt = exc.prompt
                    raw = exc.raw_response
                out["translated_text_en"] = ""
                out["translation_notes"] = ""
                out["translation_confidence"] = ""
                out["translation_model"] = args.model
                out["translation_error"] = repr(exc)
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
                "task": "translation",
                "prompt_version": TRANSLATION_PROMPT_VERSION,
                "model": args.model,
                "max_chars": args.max_chars,
                "input_text_sha256": out.get("input_text_sha256", ""),
                "coded_at_utc": out.get("translation_coded_at_utc", ""),
                "prompt": prompt,
                "raw_response": raw,
                "parsed_response": parsed_translation,
                "error": out.get("translation_error", ""),
            },
        )

        new_rows.append(out)
        if fatal_error is not None:
            write_checkpoint(existing, new_rows, output_path)
            raise RuntimeError(
                f"Non-retryable LLM model/access error for {args.model!r}. "
                f"The failed request was saved to {audit_path}."
            ) from fatal_error
        if len(new_rows) % args.save_every == 0:
            write_checkpoint(existing, new_rows, output_path)
            time.sleep(args.sleep)

    write_checkpoint(existing, new_rows, output_path)
    final = deduplicate_output(read_csv(output_path))
    errors = final["translation_error"].fillna("").astype(str).str.strip().ne("")
    missing_translation = (
        final["translated_text_en"].fillna("").astype(str).str.strip().eq("")
    )
    if errors.any() or missing_translation.any():
        translated_sample_manifest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Translation is incomplete: {int(errors.sum()):,} error row(s) and "
            f"{int(missing_translation.sum()):,} blank translation(s). Resume with "
            "--retry-errors."
        )
    if args.limit is not None:
        translated_sample_manifest_path.unlink(missing_ok=True)
        print("Partial translation finished; no sample manifest written.", flush=True)
        return
    if len(final) != len(data):
        translated_sample_manifest_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Translated output has {len(final):,} rows but input has {len(data):,}."
        )
    if source_sample:
        translated_manifest = {
            **source_sample["manifest"],
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(PROJECT_ROOT),
            "translation": {
                "model": args.model,
                "prompt_version": TRANSLATION_PROMPT_VERSION,
                "max_chars": args.max_chars,
                "source_sample_manifest_sha256": source_sample[
                    "manifest_sha256"
                ],
            },
            "inputs": {
                "source_sample": file_record(args.input),
                "source_sample_manifest": file_record(
                    source_sample["manifest_path"]
                ),
            },
            "outputs": {"sample": file_record(output_path)},
        }
        translated_sample_manifest_path.write_text(
            json.dumps(translated_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"Saved translated sample manifest: {translated_sample_manifest_path}",
            flush=True,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
