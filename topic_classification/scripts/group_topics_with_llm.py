"""Assign inductive BERTopic topics to higher-level coverage frames with GPT.

This script reads a BERTopic output directory after `label_topics_with_llm.py`
has created `topic_labels_llm.csv`. It asks GPT to assign each fine-grained
topic to exactly one higher-level coverage frame, then writes an auditable
topic-to-frame mapping with assignment rationales.
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


PROMPT_VERSION = "coverage_frame_groups_v1"

COVERAGE_FRAMES = [
    {
        "frame_id": "individualized_elite_scandal",
        "label": "Individualized elite scandal",
        "short_label": "Elite scandals",
        "meaning": (
            "Corruption coverage centered on named politicians, leaders, trials, "
            "accusations, scandals, or personal misconduct."
        ),
    },
    {
        "frame_id": "systemic_institutional_corruption",
        "label": "Systemic institutional corruption",
        "short_label": "Systemic corruption",
        "meaning": (
            "Corruption represented as broader institutional dysfunction, state capture, "
            "governance crisis, anti-corruption politics, or abuse of power."
        ),
    },
    {
        "frame_id": "transnational_investigative_corruption",
        "label": "Transnational investigative corruption",
        "short_label": "Transnational probes",
        "meaning": (
            "Cross-border probes, international investigations, foreign-linked cases, "
            "offshore money, sanctions, or investigative journalism across borders."
        ),
    },
    {
        "frame_id": "boundary_or_nonpolitical_cases",
        "label": "Boundary or less clearly political cases",
        "short_label": "Boundary cases",
        "meaning": (
            "Cases using corruption or scandal language but less clearly centered on "
            "political corruption by public officials."
        ),
    },
    {
        "frame_id": "local_sectoral_corruption",
        "label": "Local or sectoral corruption",
        "short_label": "Local/sectoral cases",
        "meaning": (
            "Municipal, local-government, public-service, school, housing, charity, "
            "sport, public company, or other sector-specific corruption cases."
        ),
    },
    {
        "frame_id": "electoral_party_finance_scandal",
        "label": "Electoral or party-finance scandal",
        "short_label": "Elections & finance",
        "meaning": (
            "Campaign finance, party funding, vote manipulation, electoral control, "
            "or corruption allegations organized around elections."
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign GPT-labelled BERTopic topics to coverage frames.")
    parser.add_argument("--bertopic-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument(
        "--target-groups",
        type=int,
        default=12,
        help="Kept for backward compatibility; coverage frames are fixed by the prompt.",
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
    return labels[keep_cols].sort_values("Count", ascending=False)


def frame_rows(prompt_version: str) -> dict[str, dict]:
    return {
        frame["frame_id"]: {
            "llm_group_prompt_version": prompt_version,
            "topic_group_id": frame["frame_id"],
            "topic_group_label": frame["label"],
            "topic_group_short_label": frame["short_label"],
            "topic_group_summary": frame["meaning"],
            "topic_group_cross_country_comparability": "",
            "topic_grouping_principle": frame["meaning"],
        }
        for frame in COVERAGE_FRAMES
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
    frame_block = "\n".join(
        (
            f"- {frame['frame_id']} | {frame['label']} | "
            f"{frame['meaning']} Chart label: {frame['short_label']}"
        )
        for frame in COVERAGE_FRAMES
    )

    return f"""
You are helping interpret inductively discovered BERTopic clusters from multilingual political-corruption news.

Task:
Assign each fine-grained topic to exactly one higher-level coverage frame.

Important:
- These are coverage frames: ways corruption is organized in the news coverage.
- They are not objective corruption-type labels.
- Use the topic labels and summaries to decide which frame best describes how the topic is narratively organized.
- Every topic must be assigned exactly once.
- Do not assign the same topic id more than once.
- If a topic could fit multiple frames, choose the dominant frame and mention the competing frame in the rationale.
- Keep the frame labels exactly as listed; do not invent new frame ids.

Coverage frames:
{frame_block}

Topics:
{topic_block}

Return valid JSON only with this structure:
{{
  "prompt_version": "{PROMPT_VERSION}",
  "frame_notes": [
    {{
      "frame_id": "one frame_id from the coverage frames",
      "frame_summary_for_this_solution": "1-2 sentences on how this frame appears in these topics",
      "cross_country_comparability": "high" | "medium" | "low"
    }}
  ],
  "assignments": [
    {{
      "topic_id": 1,
      "frame_id": "one frame_id from the coverage frames",
      "assignment_rationale": "brief reason this topic belongs in the selected frame"
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
    topics = load_topic_table(args.bertopic_dir)

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": build_prompt(topics.to_dict("records"))}],
        temperature=0,
    )
    parsed = extract_json(response.choices[0].message.content)

    groups = frame_rows(parsed.get("prompt_version", PROMPT_VERSION))
    for note in parsed.get("frame_notes", []):
        frame_id = note.get("frame_id")
        if frame_id in groups:
            groups[frame_id]["topic_group_summary"] = note.get(
                "frame_summary_for_this_solution",
                groups[frame_id]["topic_group_summary"],
            )
            groups[frame_id]["topic_group_cross_country_comparability"] = note.get(
                "cross_country_comparability",
                "",
            )

    topic_to_group_rows = []
    for assignment in parsed.get("assignments", []):
        if not isinstance(assignment, dict) or "topic_id" not in assignment or "frame_id" not in assignment:
            continue
        frame_id = assignment["frame_id"]
        if frame_id not in groups:
            raise RuntimeError(f"LLM returned unknown frame_id={frame_id!r} for topic {assignment['topic_id']}.")
        topic_to_group_rows.append(
            {
                **groups[frame_id],
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
    for frame_id, group_row in groups.items():
        topic_ids = sorted(
            int(row["Topic"])
            for row in topic_to_group_rows
            if row["topic_group_id"] == frame_id and int(row["Topic"]) in assigned_topic_ids
        )
        if topic_ids:
            group_rows.append({**group_row, "topic_ids": json.dumps(topic_ids)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    pd.DataFrame(group_rows).to_csv(output_path.with_name("topic_group_summaries_llm.csv"), index=False)

    print(f"Saved topic-group mapping: {output_path}", flush=True)
    print(f"Saved group summaries: {output_path.with_name('topic_group_summaries_llm.csv')}", flush=True)


if __name__ == "__main__":
    main()
