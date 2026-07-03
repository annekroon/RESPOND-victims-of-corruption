"""Run LLM proxy translations and annotation suggestions for active-learning rows.

Example:
    nohup python3 -u 04_run_llmproxy_active_learning_suggestions.py \
      > llm_active_learning.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL


DEFAULT_AL_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline/active_learning"
)
DEFAULT_INPUT_PATH = DEFAULT_AL_DIR / "active_learning_batch_for_annotation.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_AL_DIR / "active_learning_batch_with_llm_suggestions.csv"


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
    return {
        "translated_text": parsed.get("translated_text", ""),
        "llm_label_suggestion": parsed.get("llm_label_suggestion", ""),
        "llm_confidence": parsed.get("llm_confidence", ""),
        "llm_rationale": parsed.get("llm_rationale", ""),
        "llm_evidence": parsed.get("llm_evidence", ""),
    }


def llm_translate_and_suggest(
    client,
    article_text: str,
    model: str,
    max_chars: int,
) -> dict:
    article_text = "" if not isinstance(article_text, str) else article_text[:max_chars]
    if not article_text.strip():
        return {
            "translated_text": "",
            "llm_label_suggestion": "No",
            "llm_confidence": 0,
            "llm_rationale": "No content.",
            "llm_evidence": "",
        }

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_annotation_prompt(article_text)}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return normalize_result(extract_json(raw))


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    checkpoint.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"Saved checkpoint: {output_path} ({len(checkpoint):,} rows)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate and add LLM label suggestions to active-learning rows."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)

    al_df = pd.read_csv(args.input)
    print(f"Input:  {args.input}", flush=True)
    print(f"Output: {args.output}", flush=True)
    print(f"Model:  {args.model}", flush=True)
    print(f"Loaded {len(al_df):,} active-learning rows.", flush=True)

    if args.output.exists():
        existing = pd.read_csv(args.output)
        if args.retry_errors and "llm_error" in existing.columns and "uri" in existing.columns:
            ok_existing = existing[existing["llm_error"].fillna("").astype(str).str.strip().eq("")]
            done_uris = set(ok_existing["uri"].dropna().astype(str))
        else:
            done_uris = set(existing["uri"].dropna().astype(str)) if "uri" in existing.columns else set()
        print(f"Resuming from {args.output}; already done: {len(done_uris):,}", flush=True)
    else:
        existing = pd.DataFrame()
        done_uris = set()

    unfinished = al_df[~al_df["uri"].astype(str).isin(done_uris)].copy()
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

        for attempt in range(1, args.retries + 1):
            try:
                suggestion = llm_translate_and_suggest(
                    client=client,
                    article_text=row.get("article_text", ""),
                    model=args.model,
                    max_chars=args.max_chars,
                )
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
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)

        out.setdefault("human_final_label", "")
        out.setdefault("human_notes", "")
        new_rows.append(out)

        if len(new_rows) % args.save_every == 0:
            write_checkpoint(existing, new_rows, args.output)
            time.sleep(args.sleep)

    write_checkpoint(existing, new_rows, args.output)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
