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
from topic_classification.provenance import validate_topic_model_outputs
from topic_classification.scripts._impl.reproducibility import write_run_manifest


PROMPT_VERSION = "inductive_topic_labels_v1"

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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard labels and audit records from an earlier topic-model run.",
    )
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
        "llm_topic_summary": parsed.get("summary", ""),
        "llm_inclusion_rule": parsed.get("inclusion_rule", ""),
        "llm_exclusion_rule": parsed.get("exclusion_rule", ""),
        "llm_country_event_specific": parsed.get("country_event_specific", ""),
        "llm_cross_country_comparability": parsed.get("cross_country_comparability", ""),
        "llm_label_rationale": parsed.get("label_rationale", ""),
        "llm_confidence": parsed.get("confidence", ""),
    }


def compact_text(text: object, max_chars: int) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def format_example(row, text_column: str, max_chars: int) -> str:
    country = row.get("country", "")
    year = row.get("year", "")
    text = compact_text(row.get(text_column, ""), max_chars)
    return f"Country: {country}; year: {year}; text: {text}"


def select_diverse_examples(topic_docs, text_column: str, n: int, max_chars: int, random_state: int) -> list[str]:
    if topic_docs.empty:
        return []

    sort_cols = [col for col in ["country", "year"] if col in topic_docs.columns]
    if sort_cols:
        topic_docs = topic_docs.sort_values(sort_cols).copy()

    examples = []
    if "country" in topic_docs.columns:
        per_country = topic_docs.groupby("country", group_keys=False).head(2)
        examples.append(per_country)

    remaining = topic_docs.drop(index=examples[0].index, errors="ignore") if examples else topic_docs
    remaining_n = max(0, n - sum(len(frame) for frame in examples))
    if remaining_n:
        examples.append(remaining.sample(n=min(remaining_n, len(remaining)), random_state=random_state))

    selected = topic_docs.head(0) if not examples else __import__("pandas").concat(examples).head(n)
    return [format_example(row, text_column, max_chars) for _, row in selected.iterrows()]


def build_prompt(topic_id: int, topic_name: str, count: int, examples: list[str]) -> str:
    example_block = "\n\n".join(f"Example {i + 1}: {example}" for i, example in enumerate(examples))
    return f"""
You are helping interpret multilingual BERTopic clusters for a research project on political corruption.

The documents were already classified as primarily discussing political corruption. Your job is to
label the topic inductively from the examples and BERTopic keywords. Do not apply a predefined
corruption-type taxonomy. Do not force the topic into categories such as procurement, patronage, or
campaign finance unless that is clearly what the examples themselves show.

Research goal:
- We want topics that can reveal variation across countries and over time.
- Use specific, substantive labels that describe what actually binds the examples together.
- It is acceptable for a label to mention a country, person, institution, or event when the cluster is
  genuinely country/event-specific. In that case, set country_event_specific to true.
- If a cross-country theme is visible, prefer a country-neutral label. If not, do not pretend it is
  cross-country.
- Avoid generic labels such as "corruption investigations", "political corruption", "scandals",
  "legal proceedings", "elite corruption", or "accountability" unless the examples truly contain no
  more specific common thread.

Topic metadata:
- BERTopic topic id: {topic_id}
- BERTopic keyword/name string: {topic_name}
- Documents assigned to topic in the modeled sample: {count}

Representative documents:
{example_block}

Return valid JSON only with these keys:
{{
  "topic_label": "clear 5-12 word inductive label grounded in the examples",
  "short_label": "2-5 word chart label",
  "summary": "2-3 sentence interpretation of what binds these articles together",
  "inclusion_rule": "what belongs in this topic, based only on this cluster",
  "exclusion_rule": "what should not be coded as this topic",
  "country_event_specific": true | false,
  "cross_country_comparability": "high" | "medium" | "low",
  "label_rationale": "brief explanation of the evidence for the label and whether it captures cross-country variation",
  "confidence": 0-100
}}
""".strip()

def llm_label_topic(client, model: str, topic_id: int, topic_name: str, count: int, examples: list[str]) -> tuple[dict, str, str]:
    prompt = build_prompt(topic_id, topic_name, count, examples)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return normalize_result(extract_json(raw)), prompt, raw


def append_audit_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    if "Topic" in checkpoint.columns:
        checkpoint = checkpoint.drop_duplicates(subset=["Topic"], keep="last")
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    checkpoint.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    print(f"Saved checkpoint: {output_path} ({len(checkpoint):,} rows)", flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    model_provenance = validate_topic_model_outputs(args.bertopic_dir)
    model_manifest_sha256 = model_provenance["manifest_sha256"]

    topic_info_path = args.bertopic_dir / "topic_info.csv"
    document_topics_path = args.bertopic_dir / "document_topics.csv.gz"
    if not topic_info_path.exists():
        raise FileNotFoundError(topic_info_path)
    if not document_topics_path.exists():
        raise FileNotFoundError(document_topics_path)

    output_path = args.output or (args.bertopic_dir / "topic_labels_llm.csv")
    audit_path = output_path.with_name(output_path.stem + "_audit.jsonl")
    label_manifest_path = args.bertopic_dir / "topic_labels_run_manifest.json"
    if args.overwrite:
        for path in [output_path, audit_path, label_manifest_path]:
            path.unlink(missing_ok=True)
    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")
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
            "llm_topic_label",
            "llm_topic_short_label",
            "llm_cross_country_comparability",
            "llm_label_rationale",
            "llm_country_event_specific",
        }
        has_current_prompt = (
            "llm_label_prompt_version" in existing.columns
            and existing["llm_label_prompt_version"].fillna("").astype(str).eq(PROMPT_VERSION).all()
        )
        has_current_model_run = (
            "topic_model_manifest_sha256" in existing.columns
            and existing["topic_model_manifest_sha256"]
            .fillna("")
            .astype(str)
            .eq(model_manifest_sha256)
            .all()
        )
        if not required_generic_columns.issubset(existing.columns) or not has_current_prompt or not has_current_model_run:
            raise ValueError(
                "Existing topic labels belong to an older prompt or BERTopic run. "
                f"Rerun with --overwrite: {output_path}"
            )
        else:
            valid_existing = existing[
                existing.get("llm_error", pd.Series("", index=existing.index))
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                & existing["llm_topic_label"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            ]
            done_topics = set(valid_existing["Topic"].dropna().astype(int))
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
        examples = select_diverse_examples(
            topic_docs=topic_docs,
            text_column=args.text_column,
            n=args.examples_per_topic,
            max_chars=args.max_example_chars,
            random_state=args.random_state + topic_id,
        )
        out = topic_row.to_dict()
        out["topic_model_manifest_sha256"] = model_manifest_sha256

        for attempt in range(1, args.retries + 1):
            try:
                parsed_label, prompt, raw_response = llm_label_topic(
                    client=client,
                    model=args.model,
                    topic_id=topic_id,
                    topic_name=str(topic_row.get("Name", "")),
                    count=int(topic_row.get("Count", len(topic_docs))),
                    examples=examples,
                )
                out.update(parsed_label)
                out["llm_model"] = args.model
                out["llm_error"] = ""
                append_audit_record(
                    audit_path,
                    {
                        "prompt_version": PROMPT_VERSION,
                        "topic_id": topic_id,
                        "model": args.model,
                        "temperature": 0,
                        "examples": examples,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parsed": parsed_label,
                    },
                )
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
    completed = pd.read_csv(output_path)
    expected_topics = set(topic_info["Topic"].astype(int))
    completed_topics = set(completed["Topic"].dropna().astype(int))
    error_mask = completed.get(
        "llm_error", pd.Series("", index=completed.index)
    ).fillna("").astype(str).str.strip().ne("")
    label_mask = completed.get(
        "llm_topic_label", pd.Series("", index=completed.index)
    ).fillna("").astype(str).str.strip().ne("")
    if completed_topics != expected_topics or error_mask.any() or not label_mask.all():
        missing_topics = sorted(expected_topics - completed_topics)
        raise RuntimeError(
            "Topic labelling is incomplete: "
            f"missing topics={missing_topics}, errors={int(error_mask.sum())}, "
            f"blank labels={int((~label_mask).sum())}. Rerun without --overwrite "
            "to resume valid checkpoints."
        )
    write_run_manifest(
        args.bertopic_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs={
            "topic_info": topic_info_path,
            "document_topics": document_topics_path,
            "topic_model_manifest": model_provenance["manifest_path"],
        },
        outputs={
            "topic_labels": output_path,
            "topic_label_audit": audit_path,
        },
        extra={
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "temperature": 0,
            "examples_per_topic": args.examples_per_topic,
            "max_example_chars": args.max_example_chars,
            "min_topic_count": args.min_topic_count,
            "topic_model_manifest_sha256": model_manifest_sha256,
            "upstream_classifier": model_provenance["upstream_classifier"],
        },
        manifest_name="topic_labels_run_manifest.json",
    )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
