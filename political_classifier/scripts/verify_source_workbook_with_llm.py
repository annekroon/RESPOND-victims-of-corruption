"""LLM-assisted source assessment based on model knowledge.

This script audits source domains in ``political_corruption_all_sources_classified.xlsx``
against the paper's three source-inclusion criteria. It is deliberately an
LLM-assisted assessment tool, not a live web-verification system: the model is
not browsing the web unless the configured endpoint itself provides that
capability. Results are provisional and should be used for triage.

Important methodological guardrails:
- Only the source domain and assigned country are sent to the model.
- Prior workbook classifications are preserved in the output but are not shown
  to the model, avoiding circular validation and anchoring.
- ``review`` means insufficient reliable model knowledge. It is not equivalent
  to exclusion.
- Model confidence is not a calibrated probability.
- Evidence strings are model-generated explanations, not citations.
- The model may know prominent outlets better than obscure local sources.
- High-impact disagreements should be checked through live web research before
  manual source decisions are changed.
- Political orientation, tabloid style, or perceived outlet quality are not
  source-inclusion criteria.

Example test run:
    python3 political_classifier/scripts/verify_source_workbook_with_llm.py \
      --limit 20 \
      --overwrite

Example full run:
    nohup python3 -u political_classifier/scripts/verify_source_workbook_with_llm.py \
      --model gpt-5.1 \
      --save-every 25 \
      > source_workbook_llm_assessment.log 2>&1 &

Diagnostic cases worth inspecting manually after a test run include:
actualno.com — Bulgaria; in.reuters.com — United Kingdom;
af.reuters.com — United Kingdom; courrierinternational.com — France;
it.marketscreener.com — Italy; krstarica.com — Serbia; mignews.info — Bulgaria;
information.tv5monde.com — France.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except Exception:  # pragma: no cover - config import errors are handled below.
    config = None

from political_classifier.source_filter import DEFAULT_SOURCE_DECISION_FILE


DEFAULT_OUTPUT_PATH = (
    DEFAULT_SOURCE_DECISION_FILE.parent
    / "political_corruption_all_sources_llm_assessed.xlsx"
)
DEFAULT_CHECKPOINT_PATH = (
    DEFAULT_SOURCE_DECISION_FILE.parent
    / "political_corruption_all_sources_llm_assessed_checkpoint.csv"
)

VALID_CRITERIA = {"yes", "no", "unclear"}
VALID_DECISIONS = {"include", "exclude", "review"}
VALID_KNOWLEDGE = {"recognized", "inferred", "unknown"}
VALID_OUTLET_SCOPE = {"national", "regional/local", "international", "unclear"}
VALID_MEDIA_FORMAT = {"broadcaster", "print/digital", "news agency", "other/unclear"}
VALID_SOURCE_TYPES = {
    "general news outlet",
    "regional or local outlet",
    "public broadcaster",
    "commercial broadcaster",
    "news agency",
    "specialist or trade publication",
    "partisan or advocacy outlet",
    "aggregator or portal",
    "official institution",
    "press-release distributor",
    "data platform",
    "blog or personal site",
    "other/unclear",
}

LLM_COLUMNS = [
    "llm_c1",
    "llm_c2",
    "llm_c3",
    "llm_decision",
    "llm_include",
    "llm_knowledge_status",
    "llm_outlet_scope",
    "llm_media_format",
    "llm_source_type",
    "llm_reason",
    "llm_evidence",
    "llm_confidence",
    "llm_review_priority",
    "llm_response_format_supported",
    "llm_model",
    "llm_checked_at",
    "llm_error",
]

DISAGREEMENT_COLUMNS = [
    "country_label",
    "country",
    "source_clean",
    "political_corruption_articles",
    "main_sample_decision",
    "conventional_journalism",
    "llm_c1",
    "llm_c2",
    "llm_c3",
    "llm_decision",
    "llm_include",
    "llm_knowledge_status",
    "llm_confidence",
    "llm_review_priority",
    "llm_reason",
    "llm_evidence",
]

SYSTEM_PROMPT = """
You are independently assessing whether a named source domain qualifies as a journalistic outlet for a comparative news corpus.

You are not browsing the web. Use only your own reliable knowledge of the specific outlet. Do not rely on prior classifications or infer facts from the dataset.

Do not infer country base solely from a top-level domain:
- A .com, .net, or .info domain is not evidence that an outlet is foreign.
- A country-code domain is not sufficient evidence that an outlet is domestically based.

Do not exclude an outlet merely because it is:
- partisan or ideologically oriented;
- tabloid or sensationalist;
- digital-native;
- regional or local;
- specialist, financial, trade, or sectoral;
- publicly funded;
- critical of mainstream institutions.

Political orientation and perceived quality are descriptive only and are not inclusion criteria.

Evaluate the specific domain or edition, not merely the parent company.
For example, a regional or country-specific subdomain of an international organization may differ from the parent organization.

Do not invent:
- ownership;
- headquarters;
- editorial offices;
- publication country;
- newsroom practices;
- original-reporting claims.

If you do not recognize the outlet, or cannot establish a criterion from reliable model knowledge, answer "unclear".

Criteria:

C1 — DOMESTIC EDITORIAL OR ORGANIZATIONAL BASE
Does the specific outlet or edition have its principal editorial or organizational publication base in the assigned country, or operate as a distinct domestic newsroom in that country?

Answer:
- yes: clearly domestic to the assigned country;
- no: clearly foreign, international without a domestic newsroom, diaspora-based, or misassigned;
- unclear: insufficient reliable knowledge.

C2 — IDENTIFIABLE JOURNALISTIC EDITORIAL ACTIVITY
Does the outlet publish original reporting or exercise identifiable human editorial selection, editing, translation, commissioning, or curation?

Answer:
- yes: identifiable newsroom or editorial responsibility;
- no: no meaningful journalism or editorial oversight;
- unclear: insufficient reliable knowledge.

C3 — JOURNALISM IS THE PRIMARY FUNCTION
Is the source primarily a journalistic outlet rather than:
- an automated news aggregator;
- an official government or institutional repository;
- a press-release distributor;
- a document archive;
- a data platform;
- a portal that mainly republishes third-party material without substantial editorial processing?

Answer:
- yes: primarily journalistic;
- no: primarily one of the non-journalistic functions above;
- unclear: mixed or insufficiently known.

Decision rule:
- include: C1, C2, and C3 are all yes;
- exclude: at least one criterion is no;
- review: no criterion is no, but at least one is unclear.

Return only valid JSON.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assess source domains with an LLM using only model knowledge of "
            "source_domain and assigned_country. This is not live web verification."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE_DECISION_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--sheet",
        default=None,
        help="Workbook sheet to read. Defaults to all_sources_classified if present, otherwise the first sheet.",
    )
    parser.add_argument("--model", default=os.environ.get("UVA_LLM_MODEL") or getattr(config, "LLMPROXY_MODEL", "gpt-5.1"))
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["Include", "Exclude", "Review"],
        help=(
            "Manual source statuses to audit. Matched against main_sample_decision "
            "when present, otherwise conventional_journalism mapped to Include/Exclude/Review."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N unfinished rows.")
    parser.add_argument("--start-row", type=int, default=0, help="Skip input rows before this zero-based position.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--retry-errors", action="store_true", help="Reprocess rows with non-empty llm_error in the checkpoint.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing checkpoint/output and start from scratch.")
    return parser.parse_args()


def get_llm_credentials() -> tuple[str, str]:
    api_key = os.environ.get("UVA_LLM_API_KEY") or os.environ.get("LLMPROXY_API_KEY")
    base_url = os.environ.get("UVA_LLM_BASE_URL") or os.environ.get("LLMPROXY_BASE_URL")

    if config is not None:
        api_key = api_key or getattr(config, "LLMPROXY_API_KEY", None)
        base_url = base_url or getattr(config, "LLMPROXY_BASE_URL", None)

    if not api_key:
        raise RuntimeError(
            "Missing LLM API key. Set UVA_LLM_API_KEY or LLMPROXY_API_KEY "
            "in the environment, or LLMPROXY_API_KEY in config_local.py."
        )
    if not base_url:
        raise RuntimeError(
            "Missing LLM base URL. Set UVA_LLM_BASE_URL or LLMPROXY_BASE_URL "
            "in the environment, or LLMPROXY_BASE_URL in config_local.py."
        )
    return api_key, base_url


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


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def normalize_source(value: Any) -> str:
    return normalize_text(value).lower()


def make_checkpoint_key(row: Any) -> str:
    country = normalize_text(row.get("country", ""))
    source = normalize_source(row.get("source_clean", ""))
    return f"{country}||{source}"


def manual_status(row: Any) -> str:
    if "main_sample_decision" in row.index:
        value = normalize_text(row.get("main_sample_decision", ""))
        value_lower = value.lower()
        if value_lower in {"include", "included", "yes"}:
            return "Include"
        if value_lower in {"exclude", "excluded", "no"}:
            return "Exclude"
        if value_lower == "review":
            return "Review"

    value = normalize_text(row.get("conventional_journalism", ""))
    value_lower = value.lower()
    if value_lower == "yes":
        return "Include"
    if value_lower == "no":
        return "Exclude"
    return "Review"


def build_user_prompt(source_domain: str, assigned_country: str) -> str:
    payload = {
        "source_domain": source_domain,
        "assigned_country": assigned_country,
    }
    schema = {
        "c1": "yes | no | unclear",
        "c2": "yes | no | unclear",
        "c3": "yes | no | unclear",
        "decision": "include | exclude | review",
        "knowledge_status": "recognized | inferred | unknown",
        "outlet_scope": "national | regional/local | international | unclear",
        "media_format": "broadcaster | print/digital | news agency | other/unclear",
        "source_type": (
            "general news outlet | regional or local outlet | public broadcaster | "
            "commercial broadcaster | news agency | specialist or trade publication | "
            "partisan or advocacy outlet | aggregator or portal | official institution | "
            "press-release distributor | data platform | blog or personal site | other/unclear"
        ),
        "reason": "one concise sentence explaining the decision",
        "evidence": "specific facts known about the outlet, or 'insufficient knowledge'",
        "confidence": 0.0,
    }
    return (
        "Assess this source-domain/country pair independently. Only use your own reliable "
        "knowledge of the specific outlet. If you do not recognize it, say unclear rather "
        "than infer from the top-level domain.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"Return exactly this JSON structure:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


def extract_json(text: str) -> dict[str, Any]:
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


def derive_decision(c1: str, c2: str, c3: str) -> str:
    values = {c1, c2, c3}
    if "no" in values:
        return "exclude"
    if values == {"yes"}:
        return "include"
    return "review"


def normalize_choice(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return min(1.0, max(0.0, confidence))


def derive_review_priority(result: dict[str, Any], manual: str) -> str:
    llm_include = result.get("llm_include", "Review")
    manual_include = {"Include": "Yes", "Exclude": "No", "Review": "Review"}.get(manual, "Review")
    confidence = float(result.get("llm_confidence", 0.0) or 0.0)
    knowledge_status = result.get("llm_knowledge_status", "unknown")
    decision = result.get("llm_decision", "review")
    criteria = {result.get("llm_c1"), result.get("llm_c2"), result.get("llm_c3")}

    if (
        decision == "review"
        or knowledge_status in {"inferred", "unknown"}
        or confidence < 0.70
        or llm_include != manual_include
    ):
        return "high"
    if 0.70 <= confidence <= 0.84:
        return "medium"
    if knowledge_status == "recognized" and confidence >= 0.85 and "unclear" not in criteria:
        return "low"
    return "medium"


def normalize_result(parsed: dict[str, Any], model: str, response_format_supported: bool | None, manual: str) -> dict[str, Any]:
    c1 = normalize_choice(parsed.get("c1"), VALID_CRITERIA, "unclear")
    c2 = normalize_choice(parsed.get("c2"), VALID_CRITERIA, "unclear")
    c3 = normalize_choice(parsed.get("c3"), VALID_CRITERIA, "unclear")
    decision = derive_decision(c1, c2, c3)
    include = {"include": "Yes", "exclude": "No", "review": "Review"}[decision]

    result = {
        "llm_c1": c1,
        "llm_c2": c2,
        "llm_c3": c3,
        "llm_decision": decision,
        "llm_include": include,
        "llm_knowledge_status": normalize_choice(parsed.get("knowledge_status"), VALID_KNOWLEDGE, "unknown"),
        "llm_outlet_scope": normalize_choice(parsed.get("outlet_scope"), VALID_OUTLET_SCOPE, "unclear"),
        "llm_media_format": normalize_choice(parsed.get("media_format"), VALID_MEDIA_FORMAT, "other/unclear"),
        "llm_source_type": normalize_choice(parsed.get("source_type"), VALID_SOURCE_TYPES, "other/unclear"),
        "llm_reason": normalize_text(parsed.get("reason", "")),
        "llm_evidence": normalize_text(parsed.get("evidence", "")) or "insufficient knowledge",
        "llm_confidence": normalize_confidence(parsed.get("confidence", 0.0)),
        "llm_response_format_supported": "unknown" if response_format_supported is None else str(response_format_supported),
        "llm_model": model,
        "llm_checked_at": datetime.now(timezone.utc).isoformat(),
        "llm_error": "",
    }
    result["llm_review_priority"] = derive_review_priority(result, manual)
    return result


def failure_result(model: str, error: Exception | str, manual: str) -> dict[str, Any]:
    result = {
        "llm_c1": "unclear",
        "llm_c2": "unclear",
        "llm_c3": "unclear",
        "llm_decision": "review",
        "llm_include": "Review",
        "llm_knowledge_status": "unknown",
        "llm_outlet_scope": "unclear",
        "llm_media_format": "other/unclear",
        "llm_source_type": "other/unclear",
        "llm_reason": "LLM call failed",
        "llm_evidence": "none",
        "llm_confidence": 0.0,
        "llm_response_format_supported": "unknown",
        "llm_model": model,
        "llm_checked_at": datetime.now(timezone.utc).isoformat(),
        "llm_error": repr(error),
    }
    result["llm_review_priority"] = derive_review_priority(result, manual)
    return result


def response_format_error(exc: Exception) -> bool:
    text = repr(exc).lower()
    return "response_format" in text or "json_object" in text or "extra_forbidden" in text


def call_llm(client: Any, source_domain: str, assigned_country: str, model: str) -> tuple[dict[str, Any], bool | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(source_domain, assigned_country)},
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    try:
        response = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return extract_json(raw), True
    except TypeError:
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        return extract_json(raw), False
    except Exception as exc:
        if not response_format_error(exc):
            raise
        response = client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content
        return extract_json(raw), False


def load_checkpoint(path: Path, retry_errors: bool) -> tuple[list[dict[str, Any]], set[str]]:
    import pandas as pd

    if not path.exists():
        return [], set()
    checkpoint = pd.read_csv(path)
    if "_source_assessment_key" not in checkpoint.columns:
        raise ValueError(
            f"Existing checkpoint has an old/incompatible schema: {path}. "
            "Use --overwrite or provide a new --checkpoint path."
        )
    if retry_errors and "llm_error" in checkpoint.columns:
        checkpoint = checkpoint[checkpoint["llm_error"].fillna("").astype(str).str.strip().eq("")]
    rows = checkpoint.to_dict("records")
    done = set(checkpoint["_source_assessment_key"].astype(str))
    return rows, done


def save_checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = pd.DataFrame(rows)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    checkpoint.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def disagreement_frame(merged: Any):
    import pandas as pd

    data = merged.copy()
    data["_manual_status"] = data.apply(manual_status, axis=1)
    data["_manual_include"] = data["_manual_status"].map({"Include": "Yes", "Exclude": "No", "Review": "Review"})
    disagreements = data[
        data["llm_include"].isin(["Yes", "No", "Review"])
        & data["_manual_include"].ne(data["llm_include"])
    ].copy()
    if "political_corruption_articles" in disagreements.columns:
        disagreements["_sort_articles"] = pd.to_numeric(disagreements["political_corruption_articles"], errors="coerce").fillna(0)
    else:
        disagreements["_sort_articles"] = 0
    disagreements["_sort_confidence"] = pd.to_numeric(disagreements["llm_confidence"], errors="coerce").fillna(0)
    disagreements = disagreements.sort_values(["_sort_articles", "_sort_confidence"], ascending=[False, False])
    return disagreements.drop(columns=["_sort_articles", "_sort_confidence"], errors="ignore")


def summary_frames(merged: Any) -> dict[str, Any]:
    import pandas as pd

    frames: dict[str, Any] = {}
    assessed = merged[merged["llm_decision"].notna()].copy() if "llm_decision" in merged.columns else merged.iloc[0:0].copy()
    summaries = []
    for column in ["llm_decision", "llm_knowledge_status", "llm_c1", "llm_c2", "llm_c3", "llm_review_priority", "llm_error"]:
        if column in assessed.columns:
            table = assessed[column].fillna("").astype(str).value_counts(dropna=False).rename_axis("value").reset_index(name="n")
            table.insert(0, "field", column)
            summaries.append(table)
    frames["summary"] = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    frames["disagreements"] = disagreement_frame(assessed)
    frames["high_priority_review"] = assessed[assessed.get("llm_review_priority", "").eq("high")].copy()
    return frames


def write_outputs(source_data: Any, checkpoint_rows: list[dict[str, Any]], checkpoint_path: Path, output_path: Path) -> None:
    import pandas as pd

    save_checkpoint(checkpoint_rows, checkpoint_path)

    checkpoint = pd.DataFrame(checkpoint_rows)
    merged = source_data.merge(checkpoint, on="_source_assessment_key", how="left")
    frames = summary_frames(merged)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(output_path.stem + ".tmp" + output_path.suffix)
    with pd.ExcelWriter(tmp_output, engine="openpyxl") as writer:
        merged.drop(columns=["_source_assessment_key"], errors="ignore").to_excel(
            writer,
            sheet_name="llm_assessed_sources",
            index=False,
        )
        frames["summary"].to_excel(writer, sheet_name="summary", index=False)

        disagreements = frames["disagreements"]
        disagreement_cols = [column for column in DISAGREEMENT_COLUMNS if column in disagreements.columns]
        disagreements[disagreement_cols].to_excel(writer, sheet_name="disagreements", index=False)

        high_priority = frames["high_priority_review"]
        high_priority_cols = [column for column in DISAGREEMENT_COLUMNS if column in high_priority.columns]
        high_priority[high_priority_cols].to_excel(writer, sheet_name="high_priority_review", index=False)
    tmp_output.replace(output_path)

    print(f"Saved checkpoint: {checkpoint_path} ({len(checkpoint_rows):,} rows)", flush=True)
    print(f"Saved workbook:   {output_path}", flush=True)


def print_run_summary(source_data: Any, checkpoint_rows: list[dict[str, Any]], done_before: int) -> None:
    import pandas as pd

    checkpoint = pd.DataFrame(checkpoint_rows)
    if checkpoint.empty:
        print("No LLM rows in checkpoint.", flush=True)
        return

    failures = checkpoint["llm_error"].fillna("").astype(str).str.strip().ne("").sum()
    print("\nLLM source-assessment summary", flush=True)
    print(f"Processed/checkpoint rows: {len(checkpoint):,}", flush=True)
    print(f"Resumed from checkpoint:   {done_before:,}", flush=True)
    print(f"Failures:                  {failures:,}", flush=True)
    for column in ["llm_decision", "llm_knowledge_status", "llm_c1", "llm_c2", "llm_c3", "llm_review_priority"]:
        if column in checkpoint.columns:
            print(f"\n{column}:", flush=True)
            print(checkpoint[column].fillna("").astype(str).value_counts(dropna=False), flush=True)

    merged = source_data.merge(checkpoint, on="_source_assessment_key", how="inner")
    if not merged.empty:
        merged["manual_status"] = merged.apply(manual_status, axis=1)
        print("\nDisagreement table against manual status:", flush=True)
        print(pd.crosstab(merged["manual_status"], merged["llm_include"], dropna=False), flush=True)
        high_priority = merged[merged["llm_review_priority"].eq("high")]
        print(f"\nHigh-priority review cases: {len(high_priority):,}", flush=True)

        preview_cols = [
            "country",
            "source_clean",
            "llm_decision",
            "llm_include",
            "llm_knowledge_status",
            "llm_confidence",
            "llm_reason",
        ]
        preview_cols = [column for column in preview_cols if column in merged.columns]
        print("\nLast assessed rows:", flush=True)
        print(merged[preview_cols].tail(20).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    api_key, base_url = get_llm_credentials()

    source_data, sheet_name = read_source_workbook(args.input, args.sheet)
    if "country" not in source_data.columns or "source_clean" not in source_data.columns:
        raise ValueError("Input workbook must contain 'country' and 'source_clean' columns.")

    source_data = source_data.copy()
    source_data["_source_assessment_key"] = source_data.apply(make_checkpoint_key, axis=1)
    source_data["_manual_status"] = source_data.apply(manual_status, axis=1)
    statuses = {status.strip().lower() for status in args.statuses}
    source_data = source_data[source_data["_manual_status"].str.lower().isin(statuses)].copy()
    source_data = source_data.iloc[args.start_row :].copy()

    if args.overwrite or not args.checkpoint.exists():
        checkpoint_rows: list[dict[str, Any]] = []
        done_ids: set[str] = set()
    else:
        checkpoint_rows, done_ids = load_checkpoint(args.checkpoint, args.retry_errors)

    unfinished = source_data[~source_data["_source_assessment_key"].astype(str).isin(done_ids)].copy()
    if args.limit is not None:
        unfinished = unfinished.head(args.limit).copy()

    print(f"Input:      {args.input}", flush=True)
    print(f"Sheet:      {sheet_name or '(csv)'}", flush=True)
    print(f"Output:     {args.output}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Model:      {args.model}", flush=True)
    print(f"Statuses:   {', '.join(args.statuses)}", flush=True)
    print("Model is not browsing the web; results are provisional audit/triage labels.", flush=True)
    print(f"Rows total after filters:   {len(source_data):,}", flush=True)
    print(f"Already done:               {len(done_ids):,}", flush=True)
    print(f"Rows to process now:        {len(unfinished):,}", flush=True)

    client = OpenAI(api_key=api_key, base_url=base_url)
    response_format_support: bool | None = None
    done_before = len(done_ids)

    try:
        for _, row in tqdm(unfinished.iterrows(), total=len(unfinished), desc="Assessing sources"):
            source_domain = normalize_source(row.get("source_clean", ""))
            assigned_country = normalize_text(row.get("country", ""))
            manual = manual_status(row)
            result = {"_source_assessment_key": make_checkpoint_key(row)}

            for attempt in range(1, args.retries + 1):
                try:
                    parsed, supported = call_llm(client, source_domain, assigned_country, args.model)
                    if response_format_support is None:
                        response_format_support = supported
                        print(f"response_format supported: {supported}", flush=True)
                    result.update(normalize_result(parsed, args.model, supported, manual))
                    break
                except Exception as exc:
                    result.update(failure_result(args.model, exc, manual))
                    print(
                        f"Error on {assigned_country}||{source_domain} "
                        f"attempt {attempt}/{args.retries}: {exc!r}",
                        flush=True,
                    )
                    if attempt < args.retries:
                        time.sleep(args.retry_sleep * attempt)

            checkpoint_rows.append(result)

            if len(checkpoint_rows) % args.save_every == 0:
                write_outputs(source_data, checkpoint_rows, args.checkpoint, args.output)
                time.sleep(args.sleep)
    finally:
        write_outputs(source_data, checkpoint_rows, args.checkpoint, args.output)

    print_run_summary(source_data, checkpoint_rows, done_before)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
