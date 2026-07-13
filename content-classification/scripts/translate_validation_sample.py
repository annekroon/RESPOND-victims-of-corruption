"""Translate sampled political-corruption articles to English with GPT.

This is intended for human validation samples, e.g. 900 articles with
100 articles per country. The script is resumable and writes checkpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL
from content_classifier_common import append_audit_record, normalize_text, read_csv, write_csv_atomic


DEFAULT_INPUT = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification/validation/content_validation_sample_100_per_country.csv.gz"
)


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
    return parser.parse_args()


def output_path_from_input(input_path: Path) -> Path:
    if input_path.name.endswith(".csv.gz"):
        return input_path.with_name(input_path.name.replace(".csv.gz", "_english.csv.gz"))
    if input_path.suffix == ".csv":
        return input_path.with_name(input_path.stem + "_english.csv")
    return input_path.with_name(input_path.name + "_english.csv.gz")


def ensure_article_id(data):
    data = data.copy()
    if "article_id" not in data.columns:
        if "uri" in data.columns:
            data["article_id"] = data["uri"].fillna("").astype(str)
        else:
            data["article_id"] = ""
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
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
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
    parsed = extract_json(raw)
    return (
        {
            "translated_text_en": parsed.get("translated_text", ""),
            "translation_notes": parsed.get("translation_notes", ""),
            "translation_confidence": confidence_value(parsed.get("confidence", "")),
        },
        prompt,
        raw,
    )


def done_article_ids(existing, retry_errors: bool) -> set[str]:
    if existing.empty or "article_id" not in existing.columns:
        return set()
    if retry_errors and "translation_error" in existing.columns:
        ok = existing[existing["translation_error"].fillna("").astype(str).str.strip().eq("")]
        return set(ok["article_id"].dropna().astype(str))
    return set(existing["article_id"].dropna().astype(str))


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    checkpoint = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
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

    output_path = args.output or output_path_from_input(args.input)
    audit_path = output_path.with_name(output_path.stem + "_translation_audit.jsonl")

    data = ensure_article_id(read_csv(args.input))
    if args.text_column not in data.columns:
        raise ValueError(f"Text column not found: {args.text_column}")

    if output_path.exists():
        existing = read_csv(output_path)
        done_ids = done_article_ids(existing, args.retry_errors)
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
                out["translation_model"] = args.model
                out["translation_error"] = ""
                append_audit_record(
                    audit_path,
                    {
                        "article_id": out.get("article_id", ""),
                        "model": args.model,
                        "prompt": prompt,
                        "raw_response": raw,
                    },
                )
                break
            except Exception as exc:
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
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)

        new_rows.append(out)
        if len(new_rows) % args.save_every == 0:
            write_checkpoint(existing, new_rows, output_path)
            time.sleep(args.sleep)

    write_checkpoint(existing, new_rows, output_path)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
