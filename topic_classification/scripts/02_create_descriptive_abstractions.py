"""Create language-neutral English abstractions for descriptive topic modelling.

The abstraction removes names, countries, outlets, dates, and other case-specific
signals while preserving the political-corruption practice, institutional setting,
and response described by the article. BERTopic can then cluster substantive case
descriptions instead of publication languages or recurring personalities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL
from topic_classification.provenance import (
    sample_manifest_path,
    validate_verified_topic_sample,
)
from topic_classification.scripts._impl.reproducibility import (
    sha256_file,
    write_run_manifest,
)


PROMPT_VERSION = "descriptive_corruption_abstraction_v1"
ALLOWED_STATUSES = {"usable", "boundary_or_unclear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create short English, case-neutral descriptions for descriptive "
            "BERTopic modelling."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-column", default="article_text")
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry rows whose previous abstraction ended with an error.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard abstractions from an earlier run.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    try:
        from ftfy import fix_text

        value = fix_text(value)
    except ImportError:
        pass
    value = unicodedata.normalize("NFKC", value)
    value = "".join(
        char
        for char in value
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )
    return re.sub(r"\s+", " ", value).strip()


def text_hash(value: object) -> str:
    return hashlib.sha256(clean_text(value).encode("utf-8")).hexdigest()


def stable_row_ids(data, text_column: str):
    import pandas as pd

    country = data.get("country", pd.Series("", index=data.index)).fillna("").astype(str)
    uri = data.get("uri", pd.Series("", index=data.index)).fillna("").astype(str)
    identifiers = country + "::" + uri
    missing = uri.str.strip().eq("")
    identifiers.loc[missing] = (
        country.loc[missing]
        + "::text::"
        + data.loc[missing, text_column].map(text_hash)
    )
    return identifiers


def build_prompt(article_text: str) -> str:
    return f"""
You are preparing text for an exploratory BERTopic description of political-corruption news.

Read the article and write one short English abstraction of what the article is primarily about.
The abstraction will be clustered, so retain substantive meaning but remove signals that merely
identify a language, country, outlet, personality, or one-off event.

Include, when stated:
- the alleged corrupt practice or abuse;
- the institutional or sectoral setting;
- the investigation, trial, sanction, reform, or political response when it is central.

Strict rules:
- Use only the article. Do not add facts or infer an unstated corruption mechanism.
- Do not include names of people, parties, companies, countries, cities, outlets, or agencies.
- Do not include dates, exact monetary amounts, quotations, or nationality labels.
- Do not assign a predefined corruption category.
- Do not describe ordinary politics, crime, or misconduct as corruption merely because it appears
  in an article selected by a classifier.
- Write 18-45 words in plain English.
- Use status "usable" when a substantive political-corruption case, allegation, institutional
  pattern, or anti-corruption response is clear.
- Use status "boundary_or_unclear" when political corruption is incidental, absent, or too unclear.
  In that case, still summarize what the article is actually about without inventing corruption.

Return valid JSON only:
{{
  "status": "usable" | "boundary_or_unclear",
  "descriptive_abstract_english": "18-45 word neutral abstraction",
  "reasoning_brief": "brief explanation of what was retained and removed",
  "confidence": 0-100
}}

Article:
{article_text}
""".strip()


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()

    def parse(candidate: str) -> dict:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as strict_error:
            try:
                return json.loads(candidate, strict=False)
            except json.JSONDecodeError:
                raise strict_error

    try:
        return parse(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return parse(match.group(0))


def normalize_result(parsed: dict) -> dict:
    status = str(parsed.get("status", "")).strip()
    abstract = clean_text(parsed.get("descriptive_abstract_english", ""))
    reasoning = clean_text(parsed.get("reasoning_brief", ""))
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"Invalid abstraction status: {status!r}")
    if not abstract:
        raise ValueError("The descriptive abstraction is blank.")
    word_n = len(abstract.split())
    if not 18 <= word_n <= 45:
        raise ValueError(f"Unexpected abstraction length: {word_n} words.")
    try:
        confidence = float(parsed.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric") from exc
    if not 0 <= confidence <= 100:
        raise ValueError("confidence must be between 0 and 100")
    return {
        "topic_abstraction_status": status,
        "topic_description_english": abstract,
        "topic_abstraction_reasoning": reasoning,
        "topic_abstraction_confidence": confidence,
    }


def non_retryable_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in message
        for marker in [
            "authenticationerror",
            "permissiondeniederror",
            "invalid_api_key",
            "key_model_access_denied",
            "model_not_found",
            "key not allowed to access model",
        ]
    )


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    frame = frame.drop_duplicates(subset=["topic_abstraction_row_id"], keep="last")
    temporary = output_path.with_name(output_path.name + ".tmp")
    compression = "gzip" if output_path.name.endswith(".gz") else None
    frame.to_csv(temporary, index=False, compression=compression)
    temporary.replace(output_path)
    print(f"Saved checkpoint: {output_path} ({len(frame):,} rows)", flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in the environment or config_local.py.")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.max_chars < 1 or args.save_every < 1 or args.retries < 1:
        raise ValueError("--max-chars, --save-every, and --retries must be positive.")

    source_provenance = validate_verified_topic_sample(args.input)
    data = pd.read_csv(args.input)
    if args.text_column not in data.columns:
        raise ValueError(f"Text column not found: {args.text_column}")
    data[args.text_column] = data[args.text_column].map(clean_text)
    data["topic_abstraction_row_id"] = stable_row_ids(data, args.text_column)
    data["topic_abstraction_input_sha256"] = data[args.text_column].map(text_hash)
    if data["topic_abstraction_row_id"].duplicated().any():
        raise ValueError("The abstraction input contains duplicate stable row identifiers.")

    audit_path = args.output.with_name(
        args.output.name.removesuffix(".csv.gz") + "_audit.jsonl"
    )
    completion_path = args.output.with_name(args.output.name + ".complete.json")
    output_manifest_path = sample_manifest_path(args.output)
    if args.overwrite:
        for path in [args.output, audit_path, completion_path, output_manifest_path]:
            if path.exists():
                path.unlink()

    existing = pd.DataFrame()
    done_ids: set[str] = set()
    if args.output.exists():
        existing = pd.read_csv(args.output)
        if "topic_abstraction_row_id" not in existing.columns:
            raise ValueError("Existing abstraction output lacks stable row identifiers.")
        expected_hashes = data.set_index("topic_abstraction_row_id")[
            "topic_abstraction_input_sha256"
        ].astype(str)
        valid = existing[
            existing["topic_abstraction_status"].isin(ALLOWED_STATUSES)
            & existing["topic_description_english"].fillna("").astype(str).str.strip().ne("")
        ].copy()
        if "topic_abstraction_error" in valid.columns:
            valid = valid[
                valid["topic_abstraction_error"].fillna("").astype(str).str.strip().eq("")
            ]
        matches = valid["topic_abstraction_input_sha256"].astype(str).eq(
            valid["topic_abstraction_row_id"].astype(str).map(expected_hashes)
        )
        done_ids = set(valid.loc[matches, "topic_abstraction_row_id"].astype(str))
        print(f"Resuming; already complete: {len(done_ids):,}", flush=True)

    unfinished = data[
        ~data["topic_abstraction_row_id"].astype(str).isin(done_ids)
    ].copy()
    if args.limit is not None:
        unfinished = unfinished.head(args.limit)

    print(f"Input: {args.input}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Rows to process now: {len(unfinished):,}", flush=True)

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    new_rows: list[dict] = []
    fatal_error: Exception | None = None
    for _, row in tqdm(
        unfinished.iterrows(), total=len(unfinished), desc="Topic abstractions"
    ):
        out = row.to_dict()
        out.update(
            {
                "topic_abstraction_prompt_version": PROMPT_VERSION,
                "topic_abstraction_model": args.model,
                "topic_abstraction_max_chars": args.max_chars,
                "topic_abstraction_coded_at_utc": datetime.now(timezone.utc).isoformat(),
                "topic_abstraction_error": "",
            }
        )
        text = str(row.get(args.text_column, ""))[: args.max_chars]
        prompt = build_prompt(text)
        raw_response = ""
        for attempt in range(1, args.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw_response = response.choices[0].message.content
                out.update(normalize_result(extract_json(raw_response)))
                out["topic_abstraction_error"] = ""
                break
            except Exception as exc:
                out["topic_abstraction_error"] = repr(exc)
                print(
                    f"Error on row={row['topic_abstraction_row_id']} "
                    f"attempt {attempt}/{args.retries}: {exc!r}",
                    flush=True,
                )
                if non_retryable_error(exc):
                    fatal_error = exc
                    break
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)

        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "row_id": row["topic_abstraction_row_id"],
                        "prompt_version": PROMPT_VERSION,
                        "model": args.model,
                        "temperature": 0,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parsed": {
                            key: out.get(key)
                            for key in [
                                "topic_abstraction_status",
                                "topic_description_english",
                                "topic_abstraction_reasoning",
                                "topic_abstraction_confidence",
                                "topic_abstraction_error",
                            ]
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        new_rows.append(out)
        if len(new_rows) % args.save_every == 0 or fatal_error is not None:
            write_checkpoint(existing, new_rows, args.output)
        if fatal_error is not None:
            raise fatal_error
        time.sleep(args.sleep)

    write_checkpoint(existing, new_rows, args.output)
    completed = pd.read_csv(args.output)
    completed = completed.drop_duplicates(
        subset=["topic_abstraction_row_id"], keep="last"
    )
    expected_ids = set(data["topic_abstraction_row_id"].astype(str))
    observed_ids = set(completed["topic_abstraction_row_id"].astype(str))
    valid_status = completed["topic_abstraction_status"].isin(ALLOWED_STATUSES)
    valid_text = completed["topic_description_english"].fillna("").astype(str).str.strip().ne("")
    no_error = completed["topic_abstraction_error"].fillna("").astype(str).str.strip().eq("")
    if observed_ids != expected_ids or not (valid_status & valid_text & no_error).all():
        raise RuntimeError(
            "Topic abstraction is incomplete: "
            f"missing rows={len(expected_ids - observed_ids)}, "
            f"invalid rows={int((~(valid_status & valid_text & no_error)).sum())}. "
            "Resume with --retry-errors."
        )

    extra = {
        "saved_sample_rows": int(len(completed)),
        "political_only": True,
        "upstream_classifier": source_provenance["upstream_classifier"],
        "topic_abstraction": {
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "temperature": 0,
            "source_sample_manifest_sha256": source_provenance["manifest_sha256"],
            "usable_rows": int(completed["topic_abstraction_status"].eq("usable").sum()),
            "boundary_or_unclear_rows": int(
                completed["topic_abstraction_status"].eq("boundary_or_unclear").sum()
            ),
        },
    }
    write_run_manifest(
        args.output.parent,
        script_name=Path(__file__).name,
        args=args,
        inputs={
            "source_sample": args.input,
            "source_sample_manifest": source_provenance["manifest_path"],
        },
        outputs={"sample": args.output, "abstraction_audit": audit_path},
        extra=extra,
        manifest_name=output_manifest_path.name,
    )
    completion_path.write_text(
        json.dumps(
            {
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "rows": len(completed),
                "output_sha256": sha256_file(args.output),
                "manifest_sha256": sha256_file(output_manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
