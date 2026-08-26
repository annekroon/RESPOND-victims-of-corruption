"""Run LLM proxy translations and suggestions for silver-training-set rows.

Example:
    nohup python3 -u political_classifier/scripts/04_label_silver_batch.py \
      > llm_silver_label.log 2>&1 &
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = next(
    path for path in Path(__file__).resolve().parents if (path / "config.py").exists()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL
from political_classifier.reproducibility import file_record, git_commit


DEFAULT_SILVER_LABEL_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline/active_learning"
)
DEFAULT_INPUT_PATH = DEFAULT_SILVER_LABEL_DIR / "silver_training_source_filtered_for_annotation.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_SILVER_LABEL_DIR / "silver_training_source_filtered_with_llm_suggestions.csv"
PROMPT_VERSION = "political_corruption_silver_label_v1"
ALLOWED_LABELS = {"Yes", "Mentioned but not central", "No", "Unsure"}


def build_annotation_prompt(article_text: str) -> str:
    return f"""
You are helping annotate multilingual news articles for a research project.

Tasks:
1. Translate the article into clear, high-quality English.
2. Suggest whether the article is primarily about political corruption.

Strict definition:
Political corruption involves public officials misusing political power for personal or political gain.

Key criteria:
- It must involve public officials in political decision-making roles, such as ministers, members of parliament, presidents, judges, or local council members.
- Exclude police chiefs, military officials, and CEOs of state companies unless they are also acting in a political decision-making role.

Common forms of political corruption:
- Bribery or kickbacks for political influence.
- Embezzlement or theft of public funds by officials.
- Nepotism and cronyism in public appointments.
- Misuse of authority, such as election fraud or shielding political allies.

Important:
- Articles should be labeled as political corruption if they focus on accusations, charges, or suspicions of corruption by political officials, even if not yet proven.
- Do not label articles that focus solely on general crime, private-sector fraud, or misconduct by non-political actors.

Label rules:
- Use "Yes" when political corruption is the article's central focus.
- Use "Mentioned but not central" when political corruption appears but is not the main focus.
- Use "No" for private fraud, ordinary crime, business misconduct, general scandals, policing/military misconduct, or corruption mentioned only generically without political officials being central.
- Use "Unsure" when the article is too ambiguous or lacks enough information.

Return valid JSON only, with these keys:
{{
  "translated_text": "...",
  "llm_label_suggestion": "Yes" | "Mentioned but not central" | "No" | "Unsure",
  "llm_confidence": 0-100,
  "llm_rationale": "short explanation",
  "llm_evidence": "short quote or paraphrase of key evidence"
}}

Article:
{article_text}
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


def normalize_result(parsed: dict) -> dict:
    label = str(parsed.get("llm_label_suggestion", "")).strip()
    if label not in ALLOWED_LABELS:
        raise ValueError(
            f"Invalid llm_label_suggestion {label!r}; expected one of "
            f"{sorted(ALLOWED_LABELS)}."
        )
    try:
        confidence = float(parsed.get("llm_confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("llm_confidence must be numeric.") from exc
    if not 0 <= confidence <= 100:
        raise ValueError("llm_confidence must be between 0 and 100.")
    return {
        "translated_text": parsed.get("translated_text", ""),
        "llm_label_suggestion": label,
        "llm_confidence": confidence,
        "llm_rationale": parsed.get("llm_rationale", ""),
        "llm_evidence": parsed.get("llm_evidence", ""),
    }


def llm_translate_and_suggest(
    client,
    article_text: str,
    model: str,
    max_chars: int,
) -> tuple[dict, str, str]:
    article_text = "" if not isinstance(article_text, str) else article_text[:max_chars]
    if not article_text.strip():
        result = {
            "translated_text": "",
            "llm_label_suggestion": "No",
            "llm_confidence": 0,
            "llm_rationale": "No content.",
            "llm_evidence": "",
        }
        return result, build_annotation_prompt(""), ""

    prompt = build_annotation_prompt(article_text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return normalize_result(extract_json(raw)), prompt, raw


def row_id(data):
    import pandas as pd

    country = (
        data["country"].fillna("").astype(str)
        if "country" in data.columns
        else pd.Series("", index=data.index)
    )
    uri = (
        data["uri"].fillna("").astype(str)
        if "uri" in data.columns
        else pd.Series("", index=data.index)
    )
    base = country + "::" + uri
    missing = uri.str.strip().eq("")
    fallback = data.get(
        "article_text",
        pd.Series("", index=data.index),
    ).map(input_hash)
    base.loc[missing] = country.loc[missing] + "::text::" + fallback.loc[missing]
    return base


def input_hash(value: object) -> str:
    text = value if isinstance(value, str) else ""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def is_non_retryable_model_error(exc: Exception) -> bool:
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


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    checkpoint = checkpoint.drop_duplicates(subset=["silver_row_id"], keep="last")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    checkpoint.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"Saved checkpoint: {output_path} ({len(checkpoint):,} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate and add LLM label suggestions to silver-label rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N unfinished rows.")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess existing output rows with a non-empty llm_error.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Start from scratch and overwrite any existing output file.",
    )
    return parser.parse_args()


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

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)

    al_df = pd.read_csv(args.input)
    al_df["silver_row_id"] = row_id(al_df)
    al_df["input_text_sha256"] = al_df["article_text"].map(input_hash)
    if al_df["silver_row_id"].duplicated().any():
        duplicates = int(al_df["silver_row_id"].duplicated(keep=False).sum())
        raise ValueError(
            f"Input contains {duplicates:,} rows with duplicate silver_row_id values. "
            "Fix the candidate sample before requesting labels."
        )
    print(f"Input:  {args.input}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Model:  {args.model}", flush=True)
    print(f"Loaded {len(al_df):,} silver-label rows.", flush=True)

    audit_path = args.output.with_name(args.output.stem + "_audit.jsonl")
    manifest_path = args.output.with_name(args.output.name + ".run.json")
    complete_path = args.output.with_name(args.output.name + ".complete.json")
    expected_manifest = {
        "schema_version": 1,
        "task": "political_corruption_silver_label",
        "prompt_version": PROMPT_VERSION,
        "model": args.model,
        "max_chars": args.max_chars,
        "temperature": 0,
        "git_commit": git_commit(PROJECT_ROOT),
        "input": file_record(args.input),
        "prompt_sha256": hashlib.sha256(
            build_annotation_prompt("__ARTICLE_TEXT__").encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in [args.output, audit_path, manifest_path, complete_path]:
            if path.exists():
                path.unlink()
    if manifest_path.exists():
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if observed != expected_manifest:
            raise ValueError(
                f"Existing run manifest does not match: {manifest_path}. "
                "Use a new output or pass --overwrite."
            )
    elif args.output.exists():
        raise ValueError(
            f"Existing output has no run manifest: {args.output}. "
            "Use a new output or pass --overwrite."
        )
    else:
        manifest_path.write_text(
            json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if args.overwrite:
        print(f"Overwriting existing output: {args.output}", flush=True)
        existing = pd.DataFrame()
        done_ids = set()
    elif args.output.exists():
        existing = pd.read_csv(args.output)
        existing = existing.drop_duplicates(subset=["silver_row_id"], keep="last")
        if args.retry_errors and "llm_error" in existing.columns:
            ok_existing = existing[existing["llm_error"].fillna("").astype(str).str.strip().eq("")]
        else:
            ok_existing = existing
        current_hash = al_df.set_index("silver_row_id")["input_text_sha256"].astype(str)
        if "input_text_sha256" in ok_existing.columns:
            matches = ok_existing["input_text_sha256"].astype(str).eq(
                ok_existing["silver_row_id"].astype(str).map(current_hash)
            )
            done_ids = set(ok_existing.loc[matches, "silver_row_id"].astype(str))
        else:
            done_ids = set()
        print(f"Resuming from {args.output}; already done: {len(done_ids):,}", flush=True)
    else:
        existing = pd.DataFrame()
        done_ids = set()

    unfinished = al_df[~al_df["silver_row_id"].astype(str).isin(done_ids)].copy()
    if args.limit is not None:
        unfinished = unfinished.head(args.limit).copy()

    print(f"Rows to process now: {len(unfinished):,}", flush=True)

    new_rows: list[dict] = []

    for _, row in tqdm(
        unfinished.iterrows(),
        total=len(unfinished),
        desc="LLM translation + suggestions",
    ):
        out = row.to_dict()
        out["prompt_version"] = PROMPT_VERSION
        out["llm_model"] = args.model
        out["llm_max_chars"] = args.max_chars
        out["llm_coded_at_utc"] = datetime.now(timezone.utc).isoformat()
        prompt = build_annotation_prompt(str(row.get("article_text", ""))[: args.max_chars])
        raw = ""
        parsed_suggestion = {}
        fatal_error: Exception | None = None

        for attempt in range(1, args.retries + 1):
            try:
                suggestion, prompt, raw = llm_translate_and_suggest(
                    client=client,
                    article_text=row.get("article_text", ""),
                    model=args.model,
                    max_chars=args.max_chars,
                )
                parsed_suggestion = suggestion
                out.update(suggestion)
                out["llm_error"] = ""
                break
            except Exception as exc:
                out.update(
                    {
                        "translated_text": "",
                        "llm_label_suggestion": "",
                        "llm_confidence": "",
                        "llm_rationale": "",
                        "llm_evidence": "",
                        "llm_error": repr(exc),
                    }
                )
                print(
                    f"Error on uri={row.get('uri', '')} attempt {attempt}/{args.retries}: {exc!r}",
                    flush=True,
                )
                if is_non_retryable_model_error(exc):
                    fatal_error = exc
                    break
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)

        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "silver_row_id": out.get("silver_row_id", ""),
                        "uri": out.get("uri", ""),
                        "country": out.get("country", ""),
                        "prompt_version": PROMPT_VERSION,
                        "model": args.model,
                        "max_chars": args.max_chars,
                        "input_text_sha256": out.get("input_text_sha256", ""),
                        "coded_at_utc": out.get("llm_coded_at_utc", ""),
                        "prompt": prompt,
                        "raw_response": raw,
                        "parsed_response": parsed_suggestion,
                        "error": out.get("llm_error", ""),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

        out.setdefault("human_final_label", "")
        out.setdefault("human_notes", "")
        new_rows.append(out)

        if fatal_error is not None:
            write_checkpoint(existing, new_rows, args.output)
            raise RuntimeError(
                f"Non-retryable LLM model/access error for {args.model!r}. "
                f"The failed request was saved to {audit_path}."
            ) from fatal_error

        if len(new_rows) % args.save_every == 0:
            write_checkpoint(existing, new_rows, args.output)
            time.sleep(args.sleep)

    write_checkpoint(existing, new_rows, args.output)
    final = pd.read_csv(args.output)
    final = final.drop_duplicates(subset=["silver_row_id"], keep="last")
    errors = (
        final["llm_error"].fillna("").astype(str).str.strip().ne("")
        if "llm_error" in final.columns
        else pd.Series(False, index=final.index)
    )
    valid_labels = final["llm_label_suggestion"].isin(ALLOWED_LABELS)
    missing_ids = set(al_df["silver_row_id"].astype(str)) - set(
        final["silver_row_id"].astype(str)
    )
    if missing_ids or errors.any() or not valid_labels.all():
        complete_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Silver-label run is incomplete: "
            f"{len(missing_ids):,} missing row(s), {int(errors.sum()):,} error "
            f"row(s), and {int((~valid_labels).sum()):,} invalid-label row(s). "
            "Resume with --retry-errors; model comparison will not accept this file yet."
        )
    completion = {
        "schema_version": 1,
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(final),
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "input": file_record(args.input),
        "output": file_record(args.output),
        "run_manifest": file_record(manifest_path),
        "audit": file_record(audit_path),
    }
    complete_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved completion marker: {complete_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
