"""Prompt specifications for zero-shot content classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


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
Publication country: {country}
Publication year: {year}

Article text:
{article_text}
""".strip()


def build_victim_visibility_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
Classify the article according to the victim visibility codebook.

Question: Who or what does the article explicitly represent as harmed by the
corruption being discussed?

Use only the article text. Code only harm that the article directly connects to
corruption. Do not code every person, organization, or public interest harmed
elsewhere in the story. First identify the corruption allegation or case, then
look for an explicit connection between that corruption and harm, then identify
who or what suffers that harm. Do not infer a victim merely from the type of
offense.

Categories:
- no_victim: No person, group, organization, institution, or public interest is
  explicitly described as suffering harm from the corruption.
- concrete_victim: An identifiable person, group, community, company,
  association, or other concrete entity is explicitly described as losing money,
  property, rights, opportunities, services, or otherwise suffering because of
  the corruption. A person explicitly subjected to corrupt extortion or a
  coercive bribe demand may also count.
- institutional_societal_victim: The article explicitly states that corruption
  harms democracy, the rule of law, public trust, institutional legitimacy, state
  capacity, public finances, society, the economy, development, or another broad
  public interest.
- unclear: The article is too incomplete, ambiguous, or translation-problematic
  to determine whether corruption-related harm is visible.

The following do not automatically establish victimhood: the mention of bribery,
fraud, embezzlement, money laundering, or tax evasion; the existence of public
money or public institutions; repayment of money to tax authorities; an
investigation, prosecution, conviction, fine, or confiscation; a bribe offer or
attempted bribe; political pressure that produces no described injury; harm
caused by an unrelated event in the same article; or the general assumption that
corruption harms taxpayers or society.

A clearly reported allegation can establish victim visibility even when the
accused denies it. Conviction or proof is not required. However, the alleged harm
must still be explicit. For this single-label variable, use concrete_victim when
both a concrete victim and broader institutional harm are explicitly visible.
Use institutional_societal_victim when only broad institutional or societal harm
is visible.

Return valid JSON only:
{{
  "victim_visibility": "no_victim | concrete_victim | institutional_societal_victim | unclear",
  "victim_visible": "yes | no | unclear",
  "evidence": "short quote or close paraphrase from the article",
  "reasoning_brief": "one short sentence",
  "confidence": 0.0
}}

{article_block}
""".strip()


def normalize_victim_visibility(parsed: dict) -> dict:
    visibility = _value(parsed, "victim_visibility")
    visible = _value(parsed, "victim_visible")
    if not visible and visibility in {"concrete_victim", "institutional_societal_victim"}:
        visible = "yes"
    elif not visible and visibility == "no_victim":
        visible = "no"
    return {
        "victim_visibility": visibility,
        "victim_visible": visible,
        "victim_evidence": _value(parsed, "evidence"),
        "victim_reasoning_brief": _value(parsed, "reasoning_brief"),
        "victim_confidence": _confidence(parsed),
    }


def build_corruption_frame_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
Classify the article according to how it frames political corruption.

Use only the article text. Code individualized when corruption is mainly framed
through named or clearly identifiable actors, accusations, investigations,
trials, resignations, sanctions, or elite scandals. Code systemic when
corruption is mainly framed as embedded in institutions, governance systems,
rule-of-law conflict, state capture, clientelism, democratic backsliding,
institutional decay, or recurring public-power abuse. Code other_or_mixed when
both frames are central, when the article is mainly procedural, technical,
local/sectoral, election-finance related, or when it does not fit clearly into
either category. Use unclear only if the text is too incomplete or ambiguous.

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
    return {
        "corruption_frame": _value(parsed, "corruption_frame"),
        "frame_evidence": _value(parsed, "evidence"),
        "frame_reasoning_brief": _value(parsed, "reasoning_brief"),
        "frame_confidence": _confidence(parsed),
    }


def build_abroad_case_prompt(article_text: str, metadata: dict) -> str:
    article_block = base_article_block(article_text, metadata)
    return f"""
Classify whether the article is mainly about a corruption case abroad.

Use only the article text and the publication country. Code domestic when the
case mainly concerns corruption in the publication country, including domestic,
national, local, municipal, sectoral, or public-service cases. Code abroad when
the case mainly concerns corruption in another country, foreign political
actors, foreign institutions, EU or international oversight of another country,
foreign bribery, offshore schemes, international sanctions, or cross-border
investigations centered outside the publication country. If EU funds or
international institutions are involved but the corrupt conduct mainly concerns
actors in the publication country, code domestic. Use unclear only if the text
is too incomplete or ambiguous.

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
    location = _value(parsed, "case_location")
    abroad = _value(parsed, "abroad_case")
    if not abroad and location == "abroad":
        abroad = "yes"
    elif not abroad and location == "domestic":
        abroad = "no"
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
Classify accused actor visibility in the political-corruption article.

Question: Is an actor represented as responsible for or participating in the
corruption discussed?

Use only the article text. Code alleged perpetrators or participants only.
Derivative ownership, control, connection, or sanctions do not count.

Categories:
- no_accused_actor: Corruption is discussed, but no alleged perpetrator or
  participant is identified.
- individual_actor: A person or officeholder is accused, investigated, charged,
  convicted, or explicitly alleged to have participated in the corruption.
- organizational_or_institutional_actor: An organization or institution is
  explicitly alleged to have participated in, enabled, financed, directed, or
  concealed the corruption.
- both_individual_and_organizational: Both individual and organizational alleged
  participants are visible.
- unclear: There is insufficient information to determine actor visibility.

Decision rule: Do not count an organization merely because it is owned by,
controlled by, connected to, or derivatively sanctioned through an accused
person. A sanction counts only when the article alleges the organization's own
participation in corruption. Such organizations can be recorded separately as
linked or sanctioned entities, but they do not affect accused actor visibility.

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
    visibility = _value(parsed, "accused_actor_visibility")
    visible = _value(parsed, "accused_actor_visible")
    if not visible and visibility in {
        "individual_actor",
        "organizational_or_institutional_actor",
        "both_individual_and_organizational",
    }:
        visible = "yes"
    elif not visible and visibility == "no_accused_actor":
        visible = "no"
    return {
        "accused_actor_visibility": visibility,
        "accused_actor_visible": visible,
        "accused_evidence": _value(parsed, "evidence"),
        "accused_reasoning_brief": _value(parsed, "reasoning_brief"),
        "accused_confidence": _confidence(parsed),
    }


VICTIM_VISIBILITY = ClassifierSpec(
    name="victim_visibility",
    prompt_version="victim_visibility_zero_shot_v2",
    default_output_name="victim_visibility_labels.csv.gz",
    result_columns=[
        "victim_visibility",
        "victim_visible",
        "victim_evidence",
        "victim_reasoning_brief",
        "victim_confidence",
    ],
    build_prompt=build_victim_visibility_prompt,
    normalize_result=normalize_victim_visibility,
)

CORRUPTION_FRAME = ClassifierSpec(
    name="corruption_frame",
    prompt_version="corruption_frame_zero_shot_v1",
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
    prompt_version="abroad_case_zero_shot_v1",
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
    prompt_version="accused_actor_zero_shot_v2",
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
