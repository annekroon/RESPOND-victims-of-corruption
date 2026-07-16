"""Verify source/outlet inclusion decisions with GPT-5.1.

This script reads ``political_corruption_all_sources_classified.xlsx`` and asks
the configured OpenAI-compatible LLM proxy to assess each country/source row
against the paper's three source-inclusion criteria:

1. The outlet is genuinely based in the assigned country.
2. The outlet produces original reporting or identifiable editorial curation.
3. The outlet is not primarily a press-release distributor, official
   repository, or automated aggregator.

The script does not overwrite the original workbook. It writes a new workbook
and a checkpoint CSV with GPT-5.1 criterion-level decisions and rationale. This
is an audit/review aid based on workbook metadata and the model's general
knowledge; it is not a live web-browsing verification unless the configured API
itself provides browsing.

Example:
    nohup python3 -u political_classifier/scripts/verify_source_workbook_with_llm.py \
      --model gpt-5.1 \
      --save-every 25 \
      > source_workbook_gpt51_verification.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL
from political_classifier.source_filter import DEFAULT_SOURCE_DECISION_FILE


DEFAULT_OUTPUT_PATH = (
    DEFAULT_SOURCE_DECISION_FILE.parent
    / "political_corruption_all_sources_gpt51_verified.xlsx"
)
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_SOURCE_DECISION_FILE.parent
    / "political_corruption_all_sources_gpt51_verified_checkpoint.csv"
)

CONTEXT_COLUMNS = [
    "country",
    "country_label",
    "source_clean",
    "political_corruption_articles",
    "country_pc_total",
    "share_of_country_pc_pct",
    "rank_within_country",
    "publication_country_match",
    "inferred_publication_base",
    "original_or_editorially_curated",
    "official_repository_or_automated_aggregator",
    "source_type",
    "main_sample_decision",
    "decision_reason",
    "classification_method",
    "classification_confidence",
    "verification_note",
    "conventional_journalism",
]

GPT_COLUMNS = [
    "gpt51_country_base",
    "gpt51_original_or_editorial",
    "gpt51_not_repository_or_aggregator",
    "gpt51_final_classification",
    "gpt51_conventional_journalism",
    "gpt51_outlet_scope",
    "gpt51_media_format",
    "gpt51_source_type",
    "gpt51_confidence",
    "gpt51_rationale",
    "gpt51_evidence",
    "gpt51_model",
    "gpt51_checked_at",
    "gpt51_error",
    "INCLUDE",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify source/outlet inclusion decisions in the source workbook with GPT-5.1."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE_DECISION_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--sheet", default=None, help="Workbook sheet to read. Defaults to all_sources_classified if present, otherwise the first sheet.")
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N unfinished rows.")
    parser.add_argument("--start-row", type=int, default=0, help="Skip input rows before this zero-based position.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess rows with non-empty gpt51_error in the checkpoint.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing checkpoint/output and start from scratch.",
    )
    return parser.parse_args()


def read_source_workbook(path: Path, sheet: str | None):
    import pandas as pd

    if path.suffix.lower() in {".xlsx", ".xls"}:
        workbook = pd.ExcelFile(path)
        sheet_name = sheet
        if sheet_name is None:
            sheet_name = (
                "all_sources_classified"
                if "all_sources_classified" in workbook.sheet_names
                else workbook.sheet_names[0]
            )
        data = pd.read_excel(path, sheet_name=sheet_name)
        return data, sheet_name

    return pd.read_csv(path), None


def make_row_id(row) -> str:
    country = str(row.get("country", "")).strip()
    source = str(row.get("source_clean", "")).strip().lower()
    return f"{country}||{source}"


def row_context(row) -> str:
    lines = []
    for column in CONTEXT_COLUMNS:
        if column not in row.index:
            continue
        value = row.get(column, "")
        text = "" if value is None else str(value).strip()
        if text and text.lower() not in {"nan", "none", "<na>"}:
            lines.append(f"- {column}: {text}")
    return "\n".join(lines)


def build_prompt(row) -> str:
    return f"""
You are verifying source/outlet inclusion decisions for a comparative news
corpus about political corruption.

Assess the source in the assigned country using ONLY the information available
in the row below and your general knowledge. If the row does not provide enough
evidence to verify a criterion confidently, use "review" for that criterion.
Be conservative: do not include a source unless all three criteria are met.

Criteria:
1. country_base: The outlet is genuinely based in the assigned country. Exclude
   foreign, international, or diaspora outlets retrieved under a country's
   source list if they do not constitute a domestic journalistic outlet for that
   country.
2. original_or_editorial: The outlet produces original reporting or exercises
   identifiable editorial curation.
3. not_repository_or_aggregator: The outlet is not primarily a press-release
   distributor, official repository, or automated aggregator.

Final classification:
- "include" only when all three criteria are "yes".
- "exclude" when any criterion is clearly "no".
- "review" when available information is insufficient for one or more criteria
  and no criterion is clearly "no".

Source type is descriptive only. It must not determine inclusion.

Return valid JSON only with this schema:
{{
  "country_base": "yes" | "no" | "review",
  "original_or_editorial": "yes" | "no" | "review",
  "not_repository_or_aggregator": "yes" | "no" | "review",
  "final_classification": "include" | "exclude" | "review",
  "conventional_journalism": "Yes" | "No" | "Review",
  "outlet_scope": "national" | "regional/local" | "unclear",
  "media_format": "broadcaster" | "print/digital" | "other/unclear",
  "source_type": "general news outlet" | "regional or local outlet" | "public broadcaster" | "commercial broadcaster" | "news agency" | "specialist or trade publication" | "blog or personal site" | "aggregator or portal" | "official institution" | "press-release distributor" | "other/unclear",
  "confidence": 0-100,
  "rationale": "brief explanation of the decision",
  "evidence": "short statement of the evidence used"
}}

Workbook row:
{row_context(row)}
""".strip()


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


def normalize_choice(value, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    lookup = {choice.lower(): choice for choice in allowed}
    return lookup.get(text.lower(), default)


def normalize_result(parsed: dict, model: str) -> dict:
    final_classification = normalize_choice(
        parsed.get("final_classification"),
        {"include", "exclude", "review"},
        "review",
    )
    conventional = normalize_choice(
        parsed.get("conventional_journalism"),
        {"Yes", "No", "Review"},
        "Review",
    )
    if final_classification == "include":
        conventional = "Yes"
    elif final_classification == "exclude":
        conventional = "No"
    elif conventional not in {"Yes", "No", "Review"}:
        conventional = "Review"

    return {
        "gpt51_country_base": normalize_choice(parsed.get("country_base"), {"yes", "no", "review"}, "review"),
        "gpt51_original_or_editorial": normalize_choice(parsed.get("original_or_editorial"), {"yes", "no", "review"}, "review"),
        "gpt51_not_repository_or_aggregator": normalize_choice(parsed.get("not_repository_or_aggregator"), {"yes", "no", "review"}, "review"),
        "gpt51_final_classification": final_classification,
        "gpt51_conventional_journalism": conventional,
        "gpt51_outlet_scope": normalize_choice(
            parsed.get("outlet_scope"),
            {"national", "regional/local", "unclear"},
            "unclear",
        ),
        "gpt51_media_format": normalize_choice(
            parsed.get("media_format"),
            {"broadcaster", "print/digital", "other/unclear"},
            "other/unclear",
        ),
        "gpt51_source_type": str(parsed.get("source_type", "other/unclear")).strip() or "other/unclear",
        "gpt51_confidence": parsed.get("confidence", ""),
        "gpt51_rationale": str(parsed.get("rationale", "")).strip(),
        "gpt51_evidence": str(parsed.get("evidence", "")).strip(),
        "gpt51_model": model,
        "gpt51_checked_at": datetime.now(timezone.utc).isoformat(),
        "gpt51_error": "",
        "INCLUDE": "Yes" if final_classification == "include" else "No",
    }


def verify_source(client, row, model: str) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(row)}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return normalize_result(extract_json(raw), model=model)


def write_outputs(source_data, checkpoint_rows: list[dict], checkpoint_path: Path, output_path: Path) -> None:
    import pandas as pd

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = pd.DataFrame(checkpoint_rows)
    tmp_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    checkpoint.to_csv(tmp_checkpoint, index=False)
    tmp_checkpoint.replace(checkpoint_path)

    merged = source_data.merge(checkpoint, on="_source_verification_row_id", how="left")
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with pd.ExcelWriter(tmp_output) as writer:
        merged.drop(columns=["_source_verification_row_id"], errors="ignore").to_excel(
            writer,
            sheet_name="gpt51_verified_sources",
            index=False,
        )
        summary = (
            merged.groupby(["country", "gpt51_final_classification"], dropna=False)
            .size()
            .reset_index(name="sources")
        )
        summary.to_excel(writer, sheet_name="summary", index=False)
    tmp_output.replace(output_path)
    print(f"Saved checkpoint: {checkpoint_path} ({len(checkpoint):,} rows)", flush=True)
    print(f"Saved workbook:   {output_path}", flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")

    source_data, sheet_name = read_source_workbook(args.input, args.sheet)
    if "country" not in source_data.columns or "source_clean" not in source_data.columns:
        raise ValueError("Input workbook must contain 'country' and 'source_clean' columns.")

    source_data = source_data.copy()
    source_data["_source_verification_row_id"] = source_data.apply(make_row_id, axis=1)
    source_data = source_data.iloc[args.start_row :].copy()

    if args.overwrite or not args.checkpoint.exists():
        checkpoint_rows: list[dict] = []
        done_ids = set()
    else:
        checkpoint = pd.read_csv(args.checkpoint)
        if args.retry_errors and "gpt51_error" in checkpoint.columns:
            ok_checkpoint = checkpoint[checkpoint["gpt51_error"].fillna("").astype(str).str.strip().eq("")]
            done_ids = set(ok_checkpoint["_source_verification_row_id"].astype(str))
            checkpoint_rows = ok_checkpoint.to_dict("records")
        else:
            done_ids = set(checkpoint["_source_verification_row_id"].astype(str))
            checkpoint_rows = checkpoint.to_dict("records")

    unfinished = source_data[
        ~source_data["_source_verification_row_id"].astype(str).isin(done_ids)
    ].copy()
    if args.limit is not None:
        unfinished = unfinished.head(args.limit).copy()

    print(f"Input:      {args.input}", flush=True)
    print(f"Sheet:      {sheet_name or '(csv)'}", flush=True)
    print(f"Output:     {args.output}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Model:      {args.model}", flush=True)
    print(f"Rows total after start-row: {len(source_data):,}", flush=True)
    print(f"Already done:               {len(done_ids):,}", flush=True)
    print(f"Rows to process now:        {len(unfinished):,}", flush=True)

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)

    for _, row in tqdm(unfinished.iterrows(), total=len(unfinished), desc="Verifying sources"):
        row_id = make_row_id(row)
        result = {"_source_verification_row_id": row_id}

        for attempt in range(1, args.retries + 1):
            try:
                result.update(verify_source(client, row, args.model))
                break
            except Exception as exc:
                result.update({column: "" for column in GPT_COLUMNS})
                result.update(
                    {
                        "gpt51_model": args.model,
                        "gpt51_checked_at": datetime.now(timezone.utc).isoformat(),
                        "gpt51_error": repr(exc),
                        "INCLUDE": "",
                    }
                )
                print(f"Error on {row_id} attempt {attempt}/{args.retries}: {exc!r}", flush=True)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)

        checkpoint_rows.append(result)

        if len(checkpoint_rows) % args.save_every == 0:
            write_outputs(source_data, checkpoint_rows, args.checkpoint, args.output)
            time.sleep(args.sleep)

    write_outputs(source_data, checkpoint_rows, args.checkpoint, args.output)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
