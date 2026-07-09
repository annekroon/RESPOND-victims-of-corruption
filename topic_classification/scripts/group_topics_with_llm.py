"""Cluster inductive BERTopic labels into higher-order topic groups with GPT.

This script reads a BERTopic output directory after `label_topics_with_llm.py`
has created `topic_labels_llm.csv`. It asks GPT to group the discovered topics
inductively, without using a predefined corruption-type taxonomy.
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


PROMPT_VERSION = "inductive_topic_groups_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group GPT-labelled BERTopic topics inductively.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--target-groups", type=int, default=12)
    parser.add_argument("--min-groups", type=int, default=8)
    parser.add_argument("--max-groups", type=int, default=16)
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
        raise FileNotFoundError(f"Run label_topics_with_llm.py first. Missing: {labels_path}")

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
    labels = labels[keep_cols].sort_values("Count", ascending=False)
    return labels


def build_prompt(topic_rows, target_groups: int, min_groups: int, max_groups: int) -> str:
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

    return f"""
You are helping organize inductively discovered BERTopic clusters from multilingual political-corruption news.

Task:
Group these fine-grained topics into a smaller set of higher-order topic groups.

Important:
- Do not apply a predefined corruption taxonomy.
- Induce the groups from the topic labels and summaries below.
- The grouping should preserve meaningful variation across countries and over time.
- Do not create groups that are only generic placeholders such as "corruption", "scandals", or "politics".
- Some topics may be country/person/event-specific; group them only when they share a recognizable narrative,
  institution, issue type, or coverage pattern.
- Aim for about {target_groups} groups, with a reasonable range of {min_groups}-{max_groups}.
- Every topic must be assigned to exactly one group.
- For each topic assignment, explain briefly why that topic belongs in that group.
- Do not list the same topic id in more than one group.

Topics:
{topic_block}

Return valid JSON only with this structure:
{{
  "prompt_version": "{PROMPT_VERSION}",
  "groups": [
    {{
      "group_id": "G01",
      "group_label": "5-10 word higher-order group label",
      "group_short_label": "2-5 word chart label",
      "group_summary": "1-2 sentence explanation of the common thread",
      "cross_country_comparability": "high" | "medium" | "low",
      "grouping_principle": "brief explanation of why these topics are grouped together",
      "topics": [
        {{
          "topic_id": 1,
          "assignment_rationale": "brief reason this topic belongs in the group"
        }}
      ]
    }}
  ]
}}
""".strip()


def normalize_topic_assignments(group: dict) -> list[dict]:
    """Support the current schema and older topic_ids-only responses."""
    if isinstance(group.get("topics"), list):
        assignments = []
        for assignment in group["topics"]:
            if isinstance(assignment, dict) and "topic_id" in assignment:
                assignments.append(
                    {
                        "topic_id": int(assignment["topic_id"]),
                        "assignment_rationale": assignment.get("assignment_rationale", ""),
                    }
                )
        return assignments

    assignments = []
    for topic_id in group.get("topic_ids", []):
        assignments.append({"topic_id": int(topic_id), "assignment_rationale": ""})
    return assignments


def main() -> None:
    args = parse_args()

    import pandas as pd
    from openai import OpenAI

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")

    output_path = args.output or (args.bertopic_dir / "topic_groups_llm.csv")
    topics = load_topic_table(args.bertopic_dir)

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": build_prompt(
                    topics.to_dict("records"),
                    target_groups=args.target_groups,
                    min_groups=args.min_groups,
                    max_groups=args.max_groups,
                ),
            }
        ],
        temperature=0,
    )
    parsed = extract_json(response.choices[0].message.content)

    topic_to_group_rows = []
    group_rows = []
    for group in parsed.get("groups", []):
        group_row = {
            "llm_group_prompt_version": parsed.get("prompt_version", PROMPT_VERSION),
            "topic_group_id": group.get("group_id", ""),
            "topic_group_label": group.get("group_label", ""),
            "topic_group_short_label": group.get("group_short_label", ""),
            "topic_group_summary": group.get("group_summary", ""),
            "topic_group_cross_country_comparability": group.get("cross_country_comparability", ""),
            "topic_grouping_principle": group.get("grouping_principle", ""),
        }
        assignments = normalize_topic_assignments(group)
        group_rows.append(
            {
                **group_row,
                "topic_ids": json.dumps([assignment["topic_id"] for assignment in assignments]),
            }
        )
        for assignment in assignments:
            topic_to_group_rows.append(
                {
                    **group_row,
                    "Topic": assignment["topic_id"],
                    "topic_group_assignment_rationale": assignment["assignment_rationale"],
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
        raise RuntimeError(
            "LLM assigned some topics to multiple groups. Rerun grouping or adjust group counts. "
            f"Duplicate assignments: {duplicate_summary}"
        )

    merged = topics.merge(mapping, on="Topic", how="left")
    missing = merged[merged["topic_group_id"].fillna("").eq("")]
    if not missing.empty:
        raise RuntimeError(f"LLM did not assign all topics. Missing topic ids: {missing['Topic'].tolist()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    pd.DataFrame(group_rows).to_csv(output_path.with_name("topic_group_summaries_llm.csv"), index=False)

    print(f"Saved topic-group mapping: {output_path}", flush=True)
    print(f"Saved group summaries: {output_path.with_name('topic_group_summaries_llm.csv')}", flush=True)


if __name__ == "__main__":
    main()
