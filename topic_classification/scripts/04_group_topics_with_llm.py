"""Assign inductive BERTopic topics to higher-order topics with GPT.

This script reads a BERTopic output directory after `03_label_topics_with_llm.py`
has created `topic_labels_llm.csv`. It asks GPT to assign each fine-grained
topic to exactly one higher-order topic, then writes an auditable
topic-to-higher-order-topic mapping with assignment rationales.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL
from topic_classification.scripts._impl.reproducibility import write_run_manifest


PROMPT_VERSION = "higher_order_topic_groups_v1"

HIGHER_ORDER_TOPICS = [
    {
        "higher_order_topic_id": "individualized_elite_scandal",
        "label": "Individualized elite scandal",
        "short_label": "Elite scandals",
        "meaning": (
            "Corruption coverage centered on named politicians, leaders, trials, "
            "accusations, scandals, or personal misconduct."
        ),
    },
    {
        "higher_order_topic_id": "systemic_institutional_corruption",
        "label": "Systemic institutional corruption",
        "short_label": "Systemic corruption",
        "meaning": (
            "Corruption represented as broader institutional dysfunction, state capture, "
            "governance crisis, anti-corruption politics, or abuse of power."
        ),
    },
    {
        "higher_order_topic_id": "transnational_investigative_corruption",
        "label": "Transnational investigative corruption",
        "short_label": "Transnational probes",
        "meaning": (
            "Cross-border probes, international investigations, foreign-linked cases, "
            "offshore money, sanctions, or investigative journalism across borders."
        ),
    },
    {
        "higher_order_topic_id": "boundary_or_nonpolitical_cases",
        "label": "Boundary or less clearly political cases",
        "short_label": "Boundary cases",
        "meaning": (
            "Cases using corruption or scandal language but less clearly centered on "
            "political corruption by public officials."
        ),
    },
    {
        "higher_order_topic_id": "local_sectoral_corruption",
        "label": "Local or sectoral corruption",
        "short_label": "Local/sectoral cases",
        "meaning": (
            "Municipal, local-government, public-service, school, housing, charity, "
            "sport, public company, or other sector-specific corruption cases."
        ),
    },
    {
        "higher_order_topic_id": "electoral_party_finance_scandal",
        "label": "Electoral or party-finance scandal",
        "short_label": "Elections & finance",
        "meaning": (
            "Campaign finance, party funding, vote manipulation, electoral control, "
            "or corruption allegations organized around elections."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign GPT-labelled BERTopic topics to higher-order topics.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument(
        "--target-groups",
        type=int,
        default=12,
        help="Kept for backward compatibility; higher-order topics are fixed by the prompt.",
    )
    parser.add_argument("--min-groups", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--max-groups", type=int, default=16, help=argparse.SUPPRESS)
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


def compact_text(text: object, max_chars: int = 450) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def load_topic_table(bertopic_dir: Path):
    import pandas as pd

    labels_path = bertopic_dir / "topic_labels_llm.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Run 03_label_topics_with_llm.py first. Missing: {labels_path}")

    labels = pd.read_csv(labels_path)
    labels = labels[labels["Topic"].ne(-1)].copy()
    for column in ["llm_topic_label", "llm_topic_short_label", "llm_topic_summary"]:
        if column not in labels.columns:
            raise ValueError(f"Missing expected column in topic labels: {column}")

    keep_cols = [
        col
        for col in [
            "Topic",
            "Count",
            "Name",
            "llm_topic_label",
            "llm_topic_short_label",
            "llm_topic_summary",
            "llm_country_event_specific",
            "llm_cross_country_comparability",
            "llm_label_rationale",
            "llm_confidence",
        ]
        if col in labels.columns
    ]
    return labels[keep_cols].sort_values("Count", ascending=False)


def higher_order_topic_rows(prompt_version: str) -> dict[str, dict]:
    return {
        topic_group["higher_order_topic_id"]: {
            "llm_group_prompt_version": prompt_version,
            "topic_group_id": topic_group["higher_order_topic_id"],
            "topic_group_label": topic_group["label"],
            "topic_group_short_label": topic_group["short_label"],
            "topic_group_summary": topic_group["meaning"],
            "topic_group_cross_country_comparability": "",
            "topic_grouping_principle": topic_group["meaning"],
        }
        for topic_group in HIGHER_ORDER_TOPICS
    }


def build_prompt(topic_rows) -> str:
    topic_lines = []
    for row in topic_rows:
        topic_lines.append(
            "\n".join(
                [
                    f"Topic {int(row['Topic'])}",
                    f"size: {row.get('Count', '')}",
                    f"label: {row.get('llm_topic_label', '')}",
                    f"short_label: {row.get('llm_topic_short_label', '')}",
                    f"summary: {compact_text(row.get('llm_topic_summary', ''))}",
                    f"country_event_specific: {row.get('llm_country_event_specific', '')}",
                    f"cross_country_comparability: {row.get('llm_cross_country_comparability', '')}",
                ]
            )
        )
    topic_block = "\n\n".join(topic_lines)
    higher_order_topic_block = "\n".join(
        (
            f"- {topic_group['higher_order_topic_id']} | {topic_group['label']} | "
            f"{topic_group['meaning']} Chart label: {topic_group['short_label']}"
        )
        for topic_group in HIGHER_ORDER_TOPICS
    )

    return f"""
You are helping interpret inductively discovered BERTopic clusters from multilingual political-corruption news.

Task:
Assign each fine-grained topic to exactly one higher-order topic.

Important:
- These are higher-order topics: ways corruption is organized in the news coverage.
- They are not objective corruption-type labels.
- Use the topic labels and summaries to decide which higher-order topic best describes how the topic is narratively organized.
- Every topic must be assigned exactly once.
- Do not assign the same topic id more than once.
- If a topic could fit multiple higher-order topics, choose the dominant higher-order topic and mention the competing higher-order topic in the rationale.
- Keep the higher-order topic labels exactly as listed; do not invent new higher-order topic ids.

Higher-order topics:
{higher_order_topic_block}

Topics:
{topic_block}

Return valid JSON only with this structure:
{{
  "prompt_version": "{PROMPT_VERSION}",
  "higher_order_topic_notes": [
    {{
      "higher_order_topic_id": "one higher_order_topic_id from the higher-order topics",
      "higher_order_topic_summary_for_this_solution": "1-2 sentences on how this higher-order topic appears in these topics",
      "cross_country_comparability": "high" | "medium" | "low"
    }}
  ],
  "assignments": [
    {{
      "topic_id": 1,
      "higher_order_topic_id": "one higher_order_topic_id from the higher-order topics",
      "assignment_rationale": "brief reason this topic belongs in the selected higher-order topic"
    }}
  ]
}}
""".strip()


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")

    output_path = args.output or (args.bertopic_dir / "topic_groups_llm.csv")
    audit_path = output_path.with_name(output_path.stem + "_audit.json")
    topics = load_topic_table(args.bertopic_dir)

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    prompt = build_prompt(topics.to_dict("records"))
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw_response = response.choices[0].message.content
    parsed = extract_json(raw_response)

    groups = higher_order_topic_rows(parsed.get("prompt_version", PROMPT_VERSION))
    for note in parsed.get("higher_order_topic_notes", []):
        higher_order_topic_id = note.get("higher_order_topic_id")
        if higher_order_topic_id in groups:
            groups[higher_order_topic_id]["topic_group_summary"] = note.get(
                "higher_order_topic_summary_for_this_solution",
                groups[higher_order_topic_id]["topic_group_summary"],
            )
            groups[higher_order_topic_id]["topic_group_cross_country_comparability"] = note.get(
                "cross_country_comparability",
                "",
            )

    topic_to_group_rows = []
    for assignment in parsed.get("assignments", []):
        if not isinstance(assignment, dict) or "topic_id" not in assignment or "higher_order_topic_id" not in assignment:
            continue
        higher_order_topic_id = assignment["higher_order_topic_id"]
        if higher_order_topic_id not in groups:
            raise RuntimeError(f"LLM returned unknown higher_order_topic_id={higher_order_topic_id!r} for topic {assignment['topic_id']}.")
        topic_to_group_rows.append(
            {
                **groups[higher_order_topic_id],
                "Topic": int(assignment["topic_id"]),
                "topic_group_assignment_rationale": assignment.get("assignment_rationale", ""),
            }
        )

    mapping = pd.DataFrame(topic_to_group_rows)
    if mapping.empty:
        raise RuntimeError("LLM returned no topic-group assignments.")

    duplicate_assignments = mapping[mapping.duplicated("Topic", keep=False)].sort_values("Topic")
    if not duplicate_assignments.empty:
        duplicate_summary = (
            duplicate_assignments.groupby("Topic")["topic_group_short_label"]
            .apply(lambda values: ", ".join(values.astype(str)))
            .to_dict()
        )
        raise RuntimeError(f"LLM assigned some topics to multiple groups: {duplicate_summary}")

    merged = topics.merge(mapping, on="Topic", how="left")
    missing = merged[merged["topic_group_id"].fillna("").eq("")]
    if not missing.empty:
        raise RuntimeError(f"LLM did not assign all topics. Missing topic ids: {missing['Topic'].tolist()}")

    assigned_topic_ids = set(merged["Topic"].astype(int))
    group_rows = []
    for higher_order_topic_id, group_row in groups.items():
        topic_ids = sorted(
            int(row["Topic"])
            for row in topic_to_group_rows
            if row["topic_group_id"] == higher_order_topic_id and int(row["Topic"]) in assigned_topic_ids
        )
        if topic_ids:
            group_rows.append({**group_row, "topic_ids": json.dumps(topic_ids)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    summaries_path = output_path.with_name("topic_group_summaries_llm.csv")
    pd.DataFrame(group_rows).to_csv(summaries_path, index=False)
    audit_path.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "model": args.model,
                "temperature": 0,
                "higher_order_topics": HIGHER_ORDER_TOPICS,
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed": parsed,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    write_run_manifest(
        args.bertopic_dir,
        script_name=Path(__file__).name,
        args=args,
        inputs={"topic_labels": args.bertopic_dir / "topic_labels_llm.csv"},
        outputs={
            "topic_groups": output_path,
            "topic_group_summaries": summaries_path,
            "topic_group_audit": audit_path,
        },
        extra={
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "temperature": 0,
            "higher_order_topics": HIGHER_ORDER_TOPICS,
        },
        manifest_name="topic_groups_run_manifest.json",
    )

    print(f"Saved topic-group mapping: {output_path}", flush=True)
    print(f"Saved group summaries: {summaries_path}", flush=True)


if __name__ == "__main__":
    main()
