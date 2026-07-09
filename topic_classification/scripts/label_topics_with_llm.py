"""Use the UvA LLM proxy to create human-readable topic labels.

The input is a BERTopic output directory containing topic_info.csv and
document_topics.csv.gz. The script asks GPT 5.1 by default to summarize each
non-outlier topic using topic keywords plus representative documents.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL


PROMPT_VERSION = "mechanism_taxonomy_v2"

DOMAIN_TAXONOMY = [
    "public procurement and contracting",
    "public funds, embezzlement, and budget misuse",
    "party finance and campaign money",
    "election manipulation and vote buying",
    "appointments, patronage, nepotism, and cronyism",
    "judicial corruption and prosecution interference",
    "police, security, and coercive-state corruption",
    "local government and municipal corruption",
    "executive abuse of office and impeachment",
    "foreign influence, sanctions, and transnational corruption",
    "state-owned enterprises and privatization",
    "licensing, permits, land, and construction",
    "lobbying, access, and conflict of interest",
    "anti-corruption institutions and rule-of-law enforcement",
    "asset declarations, unexplained wealth, and illicit enrichment",
    "whistleblowing, leaks, and investigative journalism",
    "mixed or unclear",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label BERTopic topics with GPT via the UvA LLM proxy.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--text-column", default="article_text")
    parser.add_argument("--examples-per-topic", type=int, default=8)
    parser.add_argument("--max-example-chars", type=int, default=700)
    parser.add_argument("--min-topic-count", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


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
        "llm_label_prompt_version": PROMPT_VERSION,
        "llm_topic_label": parsed.get("topic_label", ""),
        "llm_topic_short_label": parsed.get("short_label", ""),
        "llm_primary_domain": parsed.get("primary_domain", ""),
        "llm_secondary_domain": parsed.get("secondary_domain", ""),
        "llm_generic_domain_label": parsed.get("generic_domain_label", ""),
        "llm_generic_domain_short_label": parsed.get("generic_domain_short_label", ""),
        "llm_corruption_type": parsed.get("corruption_type", ""),
        "llm_country_event_specific": parsed.get("country_event_specific", ""),
        "llm_topic_summary": parsed.get("summary", ""),
        "llm_inclusion_rule": parsed.get("inclusion_rule", ""),
        "llm_exclusion_rule": parsed.get("exclusion_rule", ""),
        "llm_confidence": parsed.get("confidence", ""),
    }


def compact_text(text: object, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def build_prompt(topic_id: int, topic_name: str, count: int, examples: list[str]) -> str:
    example_block = "\n\n".join(f"Example {i + 1}: {example}" for i, example in enumerate(examples))
    taxonomy_block = "\n".join(f"- {domain}" for domain in DOMAIN_TAXONOMY)
    return f"""
You are helping interpret multilingual news topics for a research project on political corruption.

The documents were already classified as primarily discussing political corruption. Your job is not
to decide whether they are political corruption; your job is to name the topic in a way a human coder
can use.

Important research goal:
- The final labels must support comparison across countries and over time.
- Do not use country names, nationalities, politician names, party names, or one-off event names in
  the generic domain labels unless the topic truly has no cross-country corruption mechanism.
- If the topic looks country-specific or event-specific, abstract upward to the corruption mechanism,
  institutional arena, or scandal type that could appear in other countries.
- Avoid umbrella labels such as "corruption investigations", "elite corruption", "political scandal",
  "corruption probes", "legal proceedings", "accountability", or "rule of law" when a more specific
  mechanism, institution, or resource is visible.
- If the examples mostly discuss investigations/trials, label the underlying alleged conduct if it is
  visible. Use "judicial corruption and prosecution interference" only when courts/prosecutors are the
  alleged corrupt arena, not merely because a case is in court.

Use this domain taxonomy for primary_domain and secondary_domain:
{taxonomy_block}

Topic metadata:
- BERTopic topic id: {topic_id}
- BERTopic keyword/name string: {topic_name}
- Documents assigned to topic in the modeled sample: {count}

Representative documents:
{example_block}

Return valid JSON only with these keys:
{{
  "topic_label": "clear 5-10 word descriptive label; may mention event/country if unavoidable",
  "short_label": "2-4 word chart label for the descriptive topic",
  "primary_domain": "one exact category from the taxonomy",
  "secondary_domain": "one exact category from the taxonomy, or 'none'",
  "generic_domain_label": "country-neutral 4-8 word corruption mechanism/domain label; not a vague investigation label",
  "generic_domain_short_label": "2-4 word country-neutral chart label",
  "corruption_type": "same as primary_domain unless a clearer short category is needed",
  "country_event_specific": true | false,
  "summary": "2-3 sentence interpretation of what binds these articles together",
  "inclusion_rule": "what belongs in this topic",
  "exclusion_rule": "what should not be coded as this topic",
  "confidence": 0-100
}}

Prefer substantive labels such as "Public procurement and contracting scandals" or
"Election fraud and campaign finance allegations". Avoid vague labels such as "corruption news",
"politics", or "legal issues". Avoid country labels such as "Italian scandals" or person labels such
as "Trump/Russia" in generic_domain_label and generic_domain_short_label. Bad labels include
"elite investigations", "corruption probes", "elite prosecutions", and "corruption scandals".
Better labels include "campaign finance violations", "public contracting kickbacks",
"executive abuse of office", "foreign influence allegations", "municipal procurement",
"appointments and patronage", or "asset declarations and wealth".
""".strip()


def llm_label_topic(client, model: str, topic_id: int, topic_name: str, count: int, examples: list[str]) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(topic_id, topic_name, count, examples)}],
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


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")

    topic_info_path = args.bertopic_dir / "topic_info.csv"
    document_topics_path = args.bertopic_dir / "document_topics.csv.gz"
    if not topic_info_path.exists():
        raise FileNotFoundError(topic_info_path)
    if not document_topics_path.exists():
        raise FileNotFoundError(document_topics_path)

    output_path = args.output or (args.bertopic_dir / "topic_labels_llm.csv")
    topic_info = pd.read_csv(topic_info_path)
    docs = pd.read_csv(document_topics_path)
    if args.text_column not in docs.columns:
        raise ValueError(f"Text column not found in document topics: {args.text_column}")

    topic_info = topic_info[topic_info["Topic"].ne(-1)].copy()
    topic_info = topic_info[topic_info["Count"].ge(args.min_topic_count)].copy()
    if args.limit is not None:
        topic_info = topic_info.head(args.limit).copy()

    if output_path.exists():
        existing = pd.read_csv(output_path)
        required_generic_columns = {
            "llm_label_prompt_version",
            "llm_primary_domain",
            "llm_secondary_domain",
            "llm_generic_domain_label",
            "llm_generic_domain_short_label",
            "llm_country_event_specific",
        }
        has_current_prompt = (
            "llm_label_prompt_version" in existing.columns
            and existing["llm_label_prompt_version"].fillna("").astype(str).eq(PROMPT_VERSION).all()
        )
        if not required_generic_columns.issubset(existing.columns) or not has_current_prompt:
            print(
                f"Existing label file is not prompt version {PROMPT_VERSION}; relabelling topics: {output_path}",
                flush=True,
            )
            existing = pd.DataFrame()
            done_topics = set()
        else:
            done_topics = set(existing["Topic"].dropna().astype(int)) if "Topic" in existing.columns else set()
            print(f"Resuming from {output_path}; already done: {len(done_topics):,}", flush=True)
    else:
        existing = pd.DataFrame()
        done_topics = set()

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    new_rows: list[dict] = []

    for _, topic_row in tqdm(topic_info.iterrows(), total=len(topic_info), desc="LLM topic labels"):
        topic_id = int(topic_row["Topic"])
        if topic_id in done_topics:
            continue

        topic_docs = docs[docs["topic"].eq(topic_id)].copy()
        if "analysis_weight" in topic_docs.columns:
            topic_docs = topic_docs.sort_values("analysis_weight", ascending=False)
        examples = [
            compact_text(text, args.max_example_chars)
            for text in topic_docs[args.text_column].dropna().head(args.examples_per_topic)
        ]
        out = topic_row.to_dict()

        for attempt in range(1, args.retries + 1):
            try:
                out.update(
                    llm_label_topic(
                        client=client,
                        model=args.model,
                        topic_id=topic_id,
                        topic_name=str(topic_row.get("Name", "")),
                        count=int(topic_row.get("Count", len(topic_docs))),
                        examples=examples,
                    )
                )
                out["llm_error"] = ""
                break
            except Exception as exc:
                out["llm_error"] = repr(exc)
                print(f"Error on topic={topic_id} attempt {attempt}/{args.retries}: {exc!r}", flush=True)
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
