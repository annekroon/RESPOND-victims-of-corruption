"""Prompt specifications for zero-shot content classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from content_codebook import CODEBOOK_MARKDOWN, codebook_for


@dataclass(frozen=True)
class ClassifierSpec:
    name: str
    prompt_version: str
    default_output_name: str
    result_columns: list[str]
    build_prompt: Callable[[str, dict], str]
    normalize_result: Callable[[dict], dict]


def _value(parsed: dict, key: str, default: str = ""):
    value = parsed.get(key, default)
    return default if value is None else value


def _confidence(parsed: dict):
    value = parsed.get("confidence", "")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric > 1:
        numeric = numeric / 100
    return max(0.0, min(1.0, numeric))


def base_article_block(article_text: str, metadata: dict) -> str:
    country = metadata.get("country", "")
    year = metadata.get("year", "")
    return f"""
publication_country: {country}
Publication year: {year}

article_text:
{article_text}
""".strip()


CONTENT_CODING_INSTRUCTIONS = CODEBOOK_MARKDOWN


VICTIM_CLASSIFIER_INSTRUCTIONS = f"""
{codebook_for("victim_visibility")}

## Machine-Coding Protocol

Apply the evidence gate before selecting a positive label. Evidence fields
must contain short verbatim quotations from the article, not interpretive
paraphrases or inferred consequences. If no exact quotation expresses the
harmed entity, harm, and corruption-to-harm link, code `no_victim`.

Classify concrete and institutional/societal victim presence independently,
then apply the codebook priority mechanically. Never return
`both_concrete_and_institutional` as a final label.
""".strip()


def build_victim_visibility_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{VICTIM_CLASSIFIER_INSTRUCTIONS}

Task: Extract the evidence first, classify concrete and institutional/societal
victim presence independently, and then assign victim_visibility. Do not return
a positive label unless the harm evidence, victim entity, and
corruption-to-harm link are all present as verbatim article quotations.

Return valid JSON only:
{{
  "harm_cause": "corruption_itself | scandal_investigation_or_response | unrelated | none | unclear",
  "harm_status": "realized_or_alleged_realized | possible_intended_or_future | none_or_unrelated | unclear",
  "victim_entity": "verbatim quotation identifying the harmed person, group, organization, institution, or public interest; otherwise none",
  "victim_inferred_from_beneficiary_or_favoritism": "yes | no | unclear",
  "harm_caused_by_case_handling_or_resource_constraints": "yes | no | unclear",
  "harmed_entity_or_public_interest_explicit": "yes | no | unclear",
  "explicit_harm_statement_present": "yes | no | unclear",
  "coercive_demand_or_pressure_made": "yes | no | unclear",
  "harm_evidence": "verbatim quotation describing the harm, or none",
  "corruption_harm_link_evidence": "verbatim quotation linking the harm to corruption, or none",
  "concrete_victim_visible": "yes | no | unclear",
  "institutional_societal_victim_visible": "yes | no | unclear",
  "victim_visibility": "no_victim | concrete_victim | institutional_societal_victim | unclear",
  "victim_visible": "yes | no | unclear",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def _normalized_yes_no(value: object) -> str:
    value = str(value or "").strip().lower()
    if value in {"yes", "true", "1"}:
        return "yes"
    if value in {"no", "false", "0"}:
        return "no"
    if value == "unclear":
        return "unclear"
    return ""


def _has_evidence(value: object) -> bool:
    value = str(value or "").strip().lower()
    return value not in {
        "",
        "none",
        "no evidence",
        "not stated",
        "not explicit",
        "n/a",
        "nan",
    }


def normalize_victim_visibility(parsed: dict) -> dict:
    harm_cause = str(_value(parsed, "harm_cause")).strip()
    harm_status = str(_value(parsed, "harm_status")).strip()
    inferred_from_favoritism = _normalized_yes_no(
        _value(parsed, "victim_inferred_from_beneficiary_or_favoritism")
    )
    case_handling_harm = _normalized_yes_no(
        _value(parsed, "harm_caused_by_case_handling_or_resource_constraints")
    )
    harmed_entity_explicit = _normalized_yes_no(
        _value(parsed, "harmed_entity_or_public_interest_explicit")
    )
    explicit_harm_statement = _normalized_yes_no(
        _value(parsed, "explicit_harm_statement_present")
    )
    coercive_demand = _normalized_yes_no(
        _value(parsed, "coercive_demand_or_pressure_made")
    )
    concrete = _normalized_yes_no(_value(parsed, "concrete_victim_visible"))
    institutional = _normalized_yes_no(
        _value(parsed, "institutional_societal_victim_visible")
    )
    harm_evidence = _value(parsed, "harm_evidence")
    corruption_harm_link_evidence = _value(
        parsed, "corruption_harm_link_evidence"
    )

    if concrete == "yes":
        derived_visibility = "concrete_victim"
    elif concrete == "no" and institutional == "yes":
        derived_visibility = "institutional_societal_victim"
    elif concrete == "no" and institutional == "no":
        derived_visibility = "no_victim"
    else:
        derived_visibility = ""

    if inferred_from_favoritism == "yes" or case_handling_harm == "yes":
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif harm_cause in {
        "scandal_investigation_or_response",
        "unrelated",
        "none",
    } or harmed_entity_explicit == "no" or not (
        _has_evidence(harm_evidence)
        and _has_evidence(corruption_harm_link_evidence)
    ):
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif harm_cause == "unclear" or harmed_entity_explicit == "unclear":
        visibility = "unclear"
    elif coercive_demand == "yes" and harm_cause == "corruption_itself":
        harm_status = "realized_or_alleged_realized"
        visibility = derived_visibility or str(
            _value(parsed, "victim_visibility")
        ).strip()
    elif explicit_harm_statement == "no":
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif explicit_harm_statement == "unclear":
        visibility = "unclear"
    elif harm_status in {"possible_intended_or_future", "none_or_unrelated"}:
        visibility = "no_victim"
        concrete = "no"
        institutional = "no"
    elif harm_status == "unclear":
        visibility = "unclear"
    elif derived_visibility:
        visibility = derived_visibility
    else:
        visibility = str(_value(parsed, "victim_visibility")).strip()

    allowed = {
        "no_victim",
        "concrete_victim",
        "institutional_societal_victim",
        "unclear",
    }
    if visibility not in allowed:
        visibility = "unclear"

    visible = _normalized_yes_no(_value(parsed, "victim_visible"))
    if visibility in {
        "concrete_victim",
        "institutional_societal_victim",
    }:
        visible = "yes"
    elif visibility == "no_victim":
        visible = "no"
    elif visibility == "unclear":
        visible = "unclear"

    return {
        "victim_visibility": visibility,
        "victim_visible": visible,
        "victim_harm_cause": harm_cause,
        "victim_harm_status": harm_status,
        "victim_entity": _value(parsed, "victim_entity"),
        "victim_inferred_from_beneficiary_or_favoritism": inferred_from_favoritism,
        "victim_harm_caused_by_case_handling_or_resource_constraints": case_handling_harm,
        "victim_harmed_entity_or_public_interest_explicit": harmed_entity_explicit,
        "victim_explicit_harm_statement_present": explicit_harm_statement,
        "victim_coercive_demand_or_pressure_made": coercive_demand,
        "victim_harm_evidence": harm_evidence,
        "victim_corruption_harm_link_evidence": corruption_harm_link_evidence,
        "concrete_victim_visible": concrete,
        "institutional_societal_victim_visible": institutional,
        "victim_reasoning_brief": _value(parsed, "reasoning_brief"),
        "victim_confidence": _confidence(parsed),
    }


def build_corruption_frame_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{codebook_for("corruption_frame")}

Task: Assign the corruption_frame label. Judge the article's dominant
explanation and emphasis. Require one short supporting quotation or close
paraphrase before selecting the label.

Return valid JSON only:
{{
  "corruption_frame": "individualized | systemic | other_or_mixed | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_corruption_frame(parsed: dict) -> dict:
    frame = str(_value(parsed, "corruption_frame")).strip()
    if frame not in {"individualized", "systemic", "other_or_mixed", "unclear"}:
        frame = "unclear"
    return {
        "corruption_frame": frame,
        "frame_evidence": _value(parsed, "evidence"),
        "frame_reasoning_brief": _value(parsed, "reasoning_brief"),
        "frame_confidence": _confidence(parsed),
    }


def build_abroad_case_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{codebook_for("case_location")}

Task: Assign the case_location label and derive abroad_case from it. Use the
publication country supplied below. Require one short supporting quotation or
close paraphrase before selecting the label.

Return valid JSON only:
{{
  "case_location": "domestic | abroad | unclear",
  "abroad_case": "yes | no | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_abroad_case(parsed: dict) -> dict:
    location = str(_value(parsed, "case_location")).strip()
    if location not in {"domestic", "abroad", "unclear"}:
        location = "unclear"
    abroad = {"abroad": "yes", "domestic": "no", "unclear": "unclear"}[location]
    return {
        "case_location": location,
        "abroad_case": abroad,
        "abroad_evidence": _value(parsed, "evidence"),
        "abroad_reasoning_brief": _value(parsed, "reasoning_brief"),
        "abroad_confidence": _confidence(parsed),
    }


def build_accused_actor_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
{codebook_for("accused_actor_visibility")}

Task: Assign the accused_actor_visibility label and derive accused_actor_visible
from it. First identify individual alleged participants and organizational
alleged participants separately. Require one short supporting quotation or close
paraphrase before selecting the label.

Return valid JSON only:
{{
  "accused_actor_visibility": "no_accused_actor | individual_actor | organizational_or_institutional_actor | both_individual_and_organizational | unclear",
  "accused_actor_visible": "yes | no | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_accused_actor(parsed: dict) -> dict:
    visibility = str(_value(parsed, "accused_actor_visibility")).strip()
    allowed = {
        "no_accused_actor",
        "individual_actor",
        "organizational_or_institutional_actor",
        "both_individual_and_organizational",
        "unclear",
    }
    if visibility not in allowed:
        visibility = "unclear"
    if visibility in {
        "individual_actor",
        "organizational_or_institutional_actor",
        "both_individual_and_organizational",
    }:
        visible = "yes"
    elif visibility == "no_accused_actor":
        visible = "no"
    else:
        visible = "unclear"
    return {
        "accused_actor_visibility": visibility,
        "accused_actor_visible": visible,
        "accused_evidence": _value(parsed, "evidence"),
        "accused_reasoning_brief": _value(parsed, "reasoning_brief"),
        "accused_confidence": _confidence(parsed),
    }


VICTIM_VISIBILITY = ClassifierSpec(
    name="victim_visibility",
    prompt_version="victim_visibility_zero_shot_v9",
    default_output_name="victim_visibility_labels.csv.gz",
    result_columns=[
        "victim_visibility",
        "victim_visible",
        "victim_harm_cause",
        "victim_harm_status",
        "victim_entity",
        "victim_entity_verbatim",
        "victim_inferred_from_beneficiary_or_favoritism",
        "victim_harm_caused_by_case_handling_or_resource_constraints",
        "victim_harmed_entity_or_public_interest_explicit",
        "victim_explicit_harm_statement_present",
        "victim_coercive_demand_or_pressure_made",
        "victim_harm_evidence",
        "victim_corruption_harm_link_evidence",
        "victim_harm_evidence_verbatim",
        "victim_corruption_harm_link_evidence_verbatim",
        "concrete_victim_visible",
        "institutional_societal_victim_visible",
        "victim_reasoning_brief",
        "victim_confidence",
    ],
    build_prompt=build_victim_visibility_prompt,
    normalize_result=normalize_victim_visibility,
)

CORRUPTION_FRAME = ClassifierSpec(
    name="corruption_frame",
    prompt_version="corruption_frame_zero_shot_v4",
    default_output_name="corruption_frame_labels.csv.gz",
    result_columns=[
        "corruption_frame",
        "frame_evidence",
        "frame_reasoning_brief",
        "frame_confidence",
    ],
    build_prompt=build_corruption_frame_prompt,
    normalize_result=normalize_corruption_frame,
)

ABROAD_CASE = ClassifierSpec(
    name="abroad_case",
    prompt_version="abroad_case_zero_shot_v4",
    default_output_name="abroad_case_labels.csv.gz",
    result_columns=[
        "case_location",
        "abroad_case",
        "abroad_evidence",
        "abroad_reasoning_brief",
        "abroad_confidence",
    ],
    build_prompt=build_abroad_case_prompt,
    normalize_result=normalize_abroad_case,
)

ACCUSED_ACTOR = ClassifierSpec(
    name="accused_actor",
    prompt_version="accused_actor_zero_shot_v7",
    default_output_name="accused_actor_labels.csv.gz",
    result_columns=[
        "accused_actor_visibility",
        "accused_actor_visible",
        "accused_evidence",
        "accused_reasoning_brief",
        "accused_confidence",
    ],
    build_prompt=build_accused_actor_prompt,
    normalize_result=normalize_accused_actor,
)
