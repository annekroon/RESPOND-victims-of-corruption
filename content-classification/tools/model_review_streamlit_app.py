"""Streamlit interface for reviewing and adjudicating model content codes.

This app intentionally exposes model labels, evidence, and rationales. It is
separate from annotation_streamlit_app.py, which remains the blind human-coding
interface used for independent validation.
"""

from __future__ import annotations

import gzip
import html
import os
import re
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = PROJECT_ROOT / "content-classification" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from content_annotation_common import evidence_passages, safe_coder_id
from content_codebook import (
    CODEBOOK_PATH,
    CODEBOOK_SHA256,
    CODEBOOK_VERSION,
    codebook_preamble,
    codebook_section,
)
from model_review_common import (
    REVIEW_DECISIONS,
    VARIABLES,
    load_review_data,
    record_review,
    review_errors,
    review_output_path,
    reviewed_mask,
    save_review_data,
)


DEFAULT_VALIDATION_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification/validation_final"
)
DEFAULT_INPUT_PATH = (
    DEFAULT_VALIDATION_DIR / "content_validation_final_n500_english.csv.gz"
)
DEFAULT_MODEL_DIR = DEFAULT_VALIDATION_DIR / "gpt51_labels"

INPUT_PATH = Path(
    os.environ.get("CONTENT_MODEL_REVIEW_INPUT", DEFAULT_INPUT_PATH)
)
MODEL_DIR = Path(
    os.environ.get("CONTENT_MODEL_REVIEW_GPT_DIR", DEFAULT_MODEL_DIR)
)
OUTPUT_TEMPLATE = os.environ.get("CONTENT_MODEL_REVIEW_OUTPUT_TEMPLATE", "")
OUTPUT_PATH_ENV = os.environ.get("CONTENT_MODEL_REVIEW_OUTPUT", "")
APP_PASSWORD = os.environ.get("CONTENT_MODEL_REVIEW_PASSWORD", "")
DEFAULT_CODER_ID = os.environ.get("CONTENT_MODEL_REVIEW_CODER_ID", "")
DEFAULT_CODER_FIRST_NAME = os.environ.get(
    "CONTENT_MODEL_REVIEW_CODER_FIRST_NAME",
    DEFAULT_CODER_ID,
)

LABELS = {
    "victim_visibility": {
        "title": "Victim visibility",
        "question": "Who or what is explicitly harmed by the corruption?",
        "display": {
            "no_victim": "No corruption-related victim is explicit",
            "concrete_victim": "Concrete person, group, or entity",
            "institutional_societal_victim": "Institution, society, or public interest",
            "unclear": "Unclear from the text",
        },
    },
    "corruption_frame": {
        "title": "Corruption frame",
        "question": "How does the article primarily frame the corruption problem?",
        "display": {
            "individualized": "Individualized case or actor",
            "systemic": "Systemic governance pattern",
            "other_or_mixed": "Other, incidental, procedural, or mixed",
            "unclear": "Unclear from the text",
        },
    },
    "case_location": {
        "title": "Case location",
        "question": "Where is the main corruption case relative to the publication country?",
        "display": {
            "domestic": "Domestic",
            "abroad": "Abroad",
            "unclear": "No principal location is clear",
        },
    },
    "accused_actor_visibility": {
        "title": "Accused actor visibility",
        "question": "Which alleged corruption participants are visible?",
        "display": {
            "no_accused_actor": "No alleged participant is identified",
            "individual_actor": "Individual actor only",
            "organizational_or_institutional_actor": "Organization or institution only",
            "both_individual_and_organizational": "Both individual and organization",
            "unclear": "Unclear from the text",
        },
    },
}

VARIABLE_COLORS = {
    "victim_visibility": ("#8a6d00", "#fff0a8"),
    "corruption_frame": ("#24608a", "#cfe8ff"),
    "case_location": ("#6f4b8b", "#e5d6f2"),
    "accused_actor_visibility": ("#9a4365", "#f6cada"),
}

EVIDENCE_FIELDS = {
    "victim_visibility": [
        ("victim", "Victim entity", "victim_entity"),
        ("victim", "Harm", "victim_harm_evidence"),
        (
            "victim",
            "Corruption-to-harm link",
            "victim_corruption_harm_link_evidence",
        ),
    ],
    "corruption_frame": [("frame", "Frame", "frame_evidence")],
    "case_location": [("location", "Location", "abroad_evidence")],
    "accused_actor_visibility": [
        ("individual", "Individual actor", "accused_individual_evidence"),
        ("organization", "Organizational actor", "accused_organization_evidence"),
    ],
}

DECISION_DISPLAY = {
    "confirm": "Confirm the model label",
    "correct": "Correct the model label",
    "uncertain": "Cannot decide from this article",
}


st.set_page_config(
    page_title="RESPOND Model Review",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {max-width: 1580px; padding-top: 1.3rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #d9dee8;}
    .article-copy {white-space: pre-wrap; line-height: 1.65; font-size: 1rem; color: inherit;}
    .article-meta {color: inherit; opacity: .72; font-size: .9rem; margin: .25rem 0 .8rem;}
    .review-warning {border-left: 4px solid #8a6d00; padding: .7rem .9rem; background: #fff8db; color: #27231a; margin-bottom: 1rem;}
    .model-summary {border-left: 4px solid var(--accent); padding: .65rem .85rem; background: color-mix(in srgb, var(--soft) 38%, transparent); margin: .35rem 0 .75rem;}
    .model-label {font-weight: 750; color: var(--accent);}
    .section-key {display: inline-block; width: .82rem; height: .82rem; background: var(--soft); border: 2px solid var(--accent); margin-right: .42rem; vertical-align: -.06rem;}
    .option-reference {border-left: 3px solid var(--accent); background: color-mix(in srgb, var(--soft) 32%, transparent); padding: .35rem .55rem; margin: .2rem 0 .55rem; font-size: .83rem;}
    .evidence-quote {padding: .38rem .55rem; margin: .28rem 0; border-left: 3px solid currentColor; line-height: 1.45;}
    .evidence-victim {background: #fff0a8; color: #4c3d00; padding: 0 .08rem;}
    .evidence-frame {background: #cfe8ff; color: #183d59; padding: 0 .08rem;}
    .evidence-location {background: #e5d6f2; color: #4f3563; padding: 0 .08rem;}
    .evidence-individual {background: #f6cada; color: #6f3049; padding: 0 .08rem;}
    .evidence-organization {background: #cdebd6; color: #285c38; padding: 0 .08rem;}
    .saved-note {font-weight: 650; font-size: .9rem;}
    div[role="radiogroup"] label {padding-top: .14rem; padding-bottom: .14rem;}
    h1, h2, h3 {letter-spacing: 0;}
</style>
""",
    unsafe_allow_html=True,
)


def value(row: pd.Series, column: str, default: str = "") -> str:
    item = row.get(column, default)
    if item is None or pd.isna(item):
        return default
    return str(item)


def useful_evidence(raw: str) -> bool:
    return raw.strip().casefold() not in {
        "",
        "none",
        "no evidence",
        "not stated",
        "not explicit",
        "n/a",
        "nan",
    }


def model_evidence(row: pd.Series) -> list[tuple[str, str, str]]:
    evidence = []
    for variable, fields in EVIDENCE_FIELDS.items():
        for kind, label, source_column in fields:
            raw = value(row, f"model_{variable}_{source_column}")
            if useful_evidence(raw):
                evidence.append((kind, label, raw))
    return evidence


def render_article_text(
    text: str,
    evidence: list[tuple[str, str, str]],
) -> None:
    text = text or "No text is available for this view."
    matches = []
    for priority, (kind, _, raw_passages) in enumerate(evidence):
        for passage in evidence_passages(raw_passages):
            pattern = r"\s+".join(re.escape(token) for token in passage.split())
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                matches.append((match.start(), match.end(), priority, kind))

    accepted = []
    for start, end, priority, kind in sorted(
        matches,
        key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
    ):
        if any(
            start < previous_end and end > previous_start
            for previous_start, previous_end, _, _ in accepted
        ):
            continue
        accepted.append((start, end, priority, kind))

    parts = []
    cursor = 0
    for start, end, _, kind in sorted(accepted):
        parts.append(html.escape(text[cursor:start]))
        parts.append(
            f'<mark class="evidence-{kind}">{html.escape(text[start:end])}</mark>'
        )
        cursor = end
    parts.append(html.escape(text[cursor:]))
    st.markdown(
        f'<div class="article-copy">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def backup_file(data: pd.DataFrame, output_path: Path) -> tuple[bytes, str]:
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    if output_path.name.endswith(".gz"):
        return gzip.compress(csv_bytes), "application/gzip"
    return csv_bytes, "text/csv"


def authenticate() -> None:
    if not APP_PASSWORD or st.session_state.get("model_review_authenticated"):
        return
    st.title("RESPOND Model Review")
    st.write("Enter the review password to continue.")
    with st.form("model_review_login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        if password == APP_PASSWORD:
            st.session_state.model_review_authenticated = True
            st.rerun()
        st.error("The password is incorrect.")
    st.stop()


def establish_reviewer() -> None:
    if st.session_state.get("model_review_coder_id"):
        return
    if DEFAULT_CODER_ID and DEFAULT_CODER_FIRST_NAME:
        st.session_state.model_review_coder_first_name = (
            DEFAULT_CODER_FIRST_NAME.strip()
        )
        st.session_state.model_review_coder_id = safe_coder_id(DEFAULT_CODER_ID)
        st.session_state.model_review_session_id = uuid.uuid4().hex
        return
    st.sidebar.subheader("Reviewer")
    with st.sidebar.form("model_review_identity"):
        first_name = st.text_input(
            "First name",
            value=DEFAULT_CODER_FIRST_NAME,
            autocomplete="given-name",
        )
        coder_id = st.text_input(
            "Reviewer ID",
            value=DEFAULT_CODER_ID or first_name,
            help="Use the same ID to resume the same review file later.",
        )
        start = st.form_submit_button("Start or resume", type="primary")
    if start:
        if not first_name.strip():
            st.sidebar.error("Enter your first name.")
        else:
            st.session_state.model_review_coder_first_name = first_name.strip()
            st.session_state.model_review_coder_id = safe_coder_id(
                coder_id or first_name
            )
            st.session_state.model_review_session_id = uuid.uuid4().hex
            st.rerun()
    st.title("RESPOND Model Review")
    st.info("Enter your name in the sidebar to start or resume model review.")
    st.stop()


def initialize_data() -> None:
    output_path = review_output_path(
        input_path=INPUT_PATH,
        output_path=Path(OUTPUT_PATH_ENV) if OUTPUT_PATH_ENV else None,
        output_template=OUTPUT_TEMPLATE,
        coder_id=st.session_state.model_review_coder_id,
    )
    signature = (
        f"{INPUT_PATH.resolve()}::{MODEL_DIR.resolve()}::{output_path.resolve()}"
    )
    if st.session_state.get("model_review_data_signature") == signature:
        return
    data, provenance, model_paths, resumed = load_review_data(
        INPUT_PATH,
        MODEL_DIR,
        output_path,
    )
    st.session_state.model_review_data = data
    st.session_state.model_review_provenance = provenance
    st.session_state.model_review_model_paths = model_paths
    st.session_state.model_review_output_path = output_path
    st.session_state.model_review_data_signature = signature
    st.session_state.model_review_position = 0
    st.session_state.model_review_filter_signature = ""
    st.session_state.model_review_resumed = resumed


def filtered_indices(
    data: pd.DataFrame,
    *,
    country: str,
    status: str,
    query: str,
) -> list[int]:
    filtered = data
    if country != "All countries":
        filtered = filtered[filtered["country"].fillna("").astype(str).eq(country)]
    completed = reviewed_mask(filtered)
    if status == "To review":
        filtered = filtered.loc[~completed]
    elif status == "Completed":
        filtered = filtered.loc[completed]
    if query.strip():
        columns = [
            column
            for column in [
                "translated_text_en",
                "translated_text",
                "article_text",
                "uri",
                "source_uri",
                *[f"model_{variable}_label" for variable in VARIABLES],
            ]
            if column in filtered.columns
        ]
        searchable = (
            filtered[columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.casefold()
        )
        filtered = filtered.loc[
            searchable.str.contains(query.strip().casefold(), regex=False, na=False)
        ]
    return filtered.index.tolist()


def country_progress(data: pd.DataFrame) -> pd.DataFrame:
    table = data[["country"]].copy()
    table["completed"] = reviewed_mask(data).astype(int)
    result = (
        table.groupby("country", dropna=False)
        .agg(completed=("completed", "sum"), total=("completed", "size"))
        .reset_index()
    )
    result["progress"] = result.apply(
        lambda row: f"{int(row['completed'])} / {int(row['total'])}", axis=1
    )
    return result[["country", "progress"]]


def display_label(variable: str, label: str) -> str:
    return LABELS[variable]["display"].get(label, label)


def format_confidence(raw: str) -> str:
    try:
        value_float = float(raw)
    except (TypeError, ValueError):
        return raw or "Not recorded"
    return f"{value_float:.0%}" if value_float <= 1 else f"{value_float:.0f}%"


def show_model_summary(row: pd.Series, variable: str) -> None:
    foreground, background = VARIABLE_COLORS[variable]
    label = value(row, f"model_{variable}_label")
    specification = VARIABLES[variable]
    reasoning = value(
        row,
        f"model_{variable}_{specification['reasoning_column']}",
        "No model rationale was recorded.",
    )
    confidence = format_confidence(
        value(row, f"model_{variable}_{specification['confidence_column']}")
    )
    st.markdown(
        (
            f'<div class="model-summary" style="--accent:{foreground};--soft:{background}">'
            f'<span class="model-label">Model label: {html.escape(display_label(variable, label))} '
            f'({html.escape(label)})</span>'
            f'<br><b>Rationale:</b> {html.escape(reasoning)}'
            f'<br><b>Confidence:</b> {html.escape(confidence)}</div>'
        ),
        unsafe_allow_html=True,
    )
    for kind, evidence_label, source_column in EVIDENCE_FIELDS[variable]:
        raw = value(row, f"model_{variable}_{source_column}")
        if useful_evidence(raw):
            st.markdown(
                f"**{evidence_label} evidence**",
            )
            st.markdown(
                f'<div class="evidence-quote evidence-{kind}">{html.escape(raw)}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption(f"{evidence_label} evidence: none recorded")


def review_variable(row: pd.Series, variable: str) -> tuple[str | None, str, str]:
    foreground, background = VARIABLE_COLORS[variable]
    article_id = value(row, "article_id")
    existing_decision = value(row, f"model_review_{variable}_decision")
    existing_correction = value(
        row, f"model_review_{variable}_corrected_label"
    )
    existing_comment = value(row, f"model_review_{variable}_comment")

    st.markdown(
        (
            f'<h3 style="--accent:{foreground};--soft:{background}">'
            f'<span class="section-key"></span>{LABELS[variable]["title"]}</h3>'
        ),
        unsafe_allow_html=True,
    )
    st.caption(LABELS[variable]["question"])
    option_text = " | ".join(
        html.escape(label) for label in VARIABLES[variable]["labels"]
    )
    st.markdown(
        (
            f'<div class="option-reference" style="--accent:{foreground};--soft:{background}">'
            f"Valid labels: {option_text}</div>"
        ),
        unsafe_allow_html=True,
    )
    show_model_summary(row, variable)
    decision = st.radio(
        "Review decision",
        REVIEW_DECISIONS,
        index=(
            REVIEW_DECISIONS.index(existing_decision)
            if existing_decision in REVIEW_DECISIONS
            else None
        ),
        format_func=lambda option: DECISION_DISPLAY[option],
        key=f"model_review_decision_{variable}_{article_id}",
    )
    corrected_options = ["", *VARIABLES[variable]["labels"]]
    corrected = st.selectbox(
        "Correct label",
        corrected_options,
        index=(
            corrected_options.index(existing_correction)
            if existing_correction in corrected_options
            else 0
        ),
        format_func=lambda option: (
            "Select only when correcting"
            if not option
            else f"{display_label(variable, option)} ({option})"
        ),
        help="Used only when Review decision is Correct the model label.",
        key=f"model_review_correction_{variable}_{article_id}",
    )
    comment = st.text_area(
        "Reviewer comment or rationale",
        value=existing_comment,
        placeholder=(
            "Optional when confirming. Required when correcting or when the article "
            "does not permit a decision."
        ),
        height=78,
        key=f"model_review_comment_{variable}_{article_id}",
    )
    return decision, corrected, comment


def save_current(
    row_index: int,
    decisions: dict[str, str | None],
    corrected_labels: dict[str, str],
    comments: dict[str, str],
    overall_comment: str,
) -> bool:
    model_labels = {
        variable: value(
            st.session_state.model_review_data.loc[row_index],
            f"model_{variable}_label",
        )
        for variable in VARIABLES
    }
    normalized_decisions = {
        variable: decision or "" for variable, decision in decisions.items()
    }
    errors = review_errors(
        model_labels=model_labels,
        decisions=normalized_decisions,
        corrected_labels=corrected_labels,
        comments=comments,
    )
    if errors:
        st.error(" ".join(errors))
        return False
    record_review(
        st.session_state.model_review_data,
        row_index,
        decisions=normalized_decisions,
        corrected_labels=corrected_labels,
        comments=comments,
        overall_comment=overall_comment,
        coder_id=st.session_state.model_review_coder_id,
        coder_first_name=st.session_state.model_review_coder_first_name,
        session_id=st.session_state.model_review_session_id,
    )
    save_review_data(
        st.session_state.model_review_data,
        input_path=INPUT_PATH,
        model_paths=st.session_state.model_review_model_paths,
        output_path=st.session_state.model_review_output_path,
        coder_id=st.session_state.model_review_coder_id,
        coder_first_name=st.session_state.model_review_coder_first_name,
        session_id=st.session_state.model_review_session_id,
        provenance=st.session_state.model_review_provenance,
    )
    article_id = value(st.session_state.model_review_data.loc[row_index], "article_id")
    widget_prefixes = [
        "model_review_decision_",
        "model_review_correction_",
        "model_review_comment_",
        "model_review_overall_",
    ]
    for key in list(st.session_state):
        if key.endswith(article_id) and any(
            key.startswith(prefix) for prefix in widget_prefixes
        ):
            st.session_state.pop(key, None)
    st.session_state.model_review_saved_message = "Saved"
    return True


def show_codebook() -> None:
    st.subheader("Full coding guide")
    st.caption(f"Canonical {CODEBOOK_VERSION} | SHA-256 {CODEBOOK_SHA256[:12]}...")
    general, victim, frame, location, actor = st.tabs(
        ["General", "Victim", "Frame", "Location", "Accused actor"]
    )
    with general:
        st.markdown(codebook_preamble())
    with victim:
        st.markdown(codebook_section("victim_visibility"))
    with frame:
        st.markdown(codebook_section("corruption_frame"))
    with location:
        st.markdown(codebook_section("case_location"))
    with actor:
        st.markdown(codebook_section("accused_actor_visibility"))


authenticate()
establish_reviewer()

try:
    initialize_data()
except Exception as exc:
    st.title("RESPOND Model Review")
    st.error(f"The review files could not be opened: {exc}")
    st.stop()

data = st.session_state.model_review_data
total = len(data)
reviewed_total = int(reviewed_mask(data).sum())
countries = sorted(data["country"].dropna().astype(str).unique())
model_names = sorted(
    {
        value(data.iloc[0], f"model_{variable}_llm_model")
        for variable in VARIABLES
    }
)

st.sidebar.title("Model Review")
st.sidebar.caption(
    "Reviewer: "
    f"{st.session_state.model_review_coder_first_name} "
    f"({st.session_state.model_review_coder_id})"
)
st.sidebar.caption(f"Model: {', '.join(model_names)}")
st.sidebar.metric("Completed", f"{reviewed_total} / {total}")
st.sidebar.progress(reviewed_total / total if total else 0.0)

country = st.sidebar.selectbox(
    "Publication country",
    ["All countries", *countries],
    key="model_review_filter_country",
)
status = st.sidebar.radio(
    "Queue",
    ["To review", "All", "Completed"],
    horizontal=True,
    key="model_review_filter_status",
)
query = st.sidebar.text_input(
    "Find article or model label",
    placeholder="Search text, URI, source, or label",
    key="model_review_filter_query",
)

with st.sidebar.expander("Progress by country"):
    st.dataframe(country_progress(data), hide_index=True, use_container_width=True)

backup_data, backup_mime = backup_file(
    data,
    st.session_state.model_review_output_path,
)
st.sidebar.download_button(
    "Download review backup",
    data=backup_data,
    file_name=st.session_state.model_review_output_path.name,
    mime=backup_mime,
    use_container_width=True,
    disabled=not st.session_state.model_review_output_path.exists(),
)
with st.sidebar.expander("Files and provenance"):
    st.caption(f"Input: {INPUT_PATH}")
    st.caption(f"GPT outputs: {MODEL_DIR}")
    st.caption(f"Review output: {st.session_state.model_review_output_path}")
    st.caption(f"Codebook: {CODEBOOK_PATH}")
    st.caption(f"Version: {CODEBOOK_VERSION}")
    st.caption(f"SHA-256: {CODEBOOK_SHA256}")
if st.sidebar.button("Change reviewer", use_container_width=True):
    for key in [
        "model_review_coder_id",
        "model_review_coder_first_name",
        "model_review_session_id",
        "model_review_data",
        "model_review_data_signature",
        "model_review_position",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

indices = filtered_indices(data, country=country, status=status, query=query)
filter_signature = f"{country}|{status}|{query}"
if st.session_state.get("model_review_filter_signature") != filter_signature:
    st.session_state.model_review_position = 0
    st.session_state.model_review_filter_signature = filter_signature

if reviewed_total == total and total:
    st.success(
        "Model review complete. Every article has four review decisions. "
        "Select Completed or All to revisit and revise any item."
    )

if not indices:
    st.title("RESPOND Model Review")
    if status == "To review" and reviewed_total == total:
        st.subheader("You are done")
        st.write("All review decisions are saved. Switch to Completed to revise them.")
    else:
        st.info("No articles match the current country, queue, and search filters.")
    show_codebook()
    st.stop()

position = max(
    0,
    min(int(st.session_state.model_review_position), len(indices) - 1),
)
st.session_state.model_review_position = position
row_index = indices[position]
row = data.loc[row_index]

st.markdown(
    '<div class="review-warning"><b>Model-assisted review.</b> '
    "GPT labels, evidence, and rationales are visible here. Decisions from this "
    "app are adjudication data and must not be reported as independent blind "
    "human validation.</div>",
    unsafe_allow_html=True,
)

heading_left, heading_right = st.columns([4, 1])
with heading_left:
    st.title("RESPOND Model Review")
    st.markdown(
        f"**{html.escape(value(row, 'country'))}** &nbsp; "
        f"{html.escape(value(row, 'year'))} &nbsp; "
        f"Article {position + 1} of {len(indices)} in this view",
        unsafe_allow_html=True,
    )
with heading_right:
    st.metric("Overall progress", f"{reviewed_total}/{total}")

nav_previous, nav_status, nav_next = st.columns([1, 3, 1])
with nav_previous:
    if st.button("Previous", use_container_width=True, disabled=position == 0):
        st.session_state.model_review_position = position - 1
        st.rerun()
with nav_status:
    saved_message = st.session_state.pop("model_review_saved_message", "")
    if saved_message:
        st.markdown(
            '<p class="saved-note">Saved to the reviewer file.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Save before navigating away from changed review decisions.")
with nav_next:
    if st.button(
        "Next",
        use_container_width=True,
        disabled=position >= len(indices) - 1,
    ):
        st.session_state.model_review_position = position + 1
        st.rerun()

article_column, review_column = st.columns([1.45, 1], gap="large")
with article_column:
    st.subheader("Article with model evidence")
    st.markdown(
        '<div class="article-meta">'
        f"Publication country: <b>{html.escape(value(row, 'country'))}</b>"
        f" &nbsp; Source: {html.escape(value(row, 'source_uri') or value(row, 'source.uri'))}"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Victim evidence is yellow; frame evidence blue; location evidence purple; "
        "individual actor evidence pink; organizational actor evidence green. "
        "Only evidence found verbatim in the displayed text is highlighted."
    )
    translated = value(row, "translated_text_en") or value(row, "translated_text")
    original = value(row, "article_text")
    evidence = model_evidence(row)
    translation_tab, original_tab, compare_tab = st.tabs(
        ["English translation", "Original", "Compare"]
    )
    with translation_tab:
        with st.container(height=650, border=True):
            render_article_text(translated, evidence)
    with original_tab:
        with st.container(height=650, border=True):
            render_article_text(original, evidence)
    with compare_tab:
        translated_column, original_column = st.columns(2)
        with translated_column:
            st.caption("English translation")
            with st.container(height=580, border=True):
                render_article_text(translated, evidence)
        with original_column:
            st.caption("Original")
            with st.container(height=580, border=True):
                render_article_text(original, evidence)
    with st.expander("Article and model details"):
        st.write(f"Sample ID: {value(row, 'content_sample_id')}")
        st.write(f"Article ID: {value(row, 'article_id')}")
        st.write(f"URI: {value(row, 'uri')}")
        st.write(f"Classifier probability: {value(row, 'prob_political_corruption')}")
        for variable in VARIABLES:
            st.write(
                f"{LABELS[variable]['title']} prompt: "
                f"{value(row, f'model_{variable}_prompt_version')}"
            )

with review_column:
    st.subheader("Validate the model coding")
    st.caption(
        "Review each model decision against the article and codebook. A rationale "
        "is required for corrections and genuinely undecidable cases."
    )
    with st.form(f"model_review_form_{value(row, 'article_id')}"):
        decisions = {}
        corrected_labels = {}
        comments = {}
        for number, variable in enumerate(VARIABLES):
            if number:
                st.divider()
            decision, corrected, comment = review_variable(row, variable)
            decisions[variable] = decision
            corrected_labels[variable] = corrected
            comments[variable] = comment

        st.divider()
        overall_comment = st.text_area(
            "Overall article comment (optional)",
            value=value(row, "model_review_overall_comment"),
            placeholder="Record translation concerns or cross-variable observations.",
            height=90,
            key=f"model_review_overall_{value(row, 'article_id')}",
        )
        save_column, continue_column = st.columns([1, 1.5])
        save_only = save_column.form_submit_button("Save", use_container_width=True)
        save_and_continue = continue_column.form_submit_button(
            "Save and continue",
            type="primary",
            use_container_width=True,
        )

    if save_only or save_and_continue:
        if save_current(
            row_index,
            decisions,
            corrected_labels,
            comments,
            overall_comment,
        ):
            if save_and_continue:
                updated_indices = filtered_indices(
                    data,
                    country=country,
                    status=status,
                    query=query,
                )
                if status == "To review":
                    st.session_state.model_review_position = min(
                        position,
                        max(len(updated_indices) - 1, 0),
                    )
                else:
                    st.session_state.model_review_position = min(
                        position + 1,
                        max(len(updated_indices) - 1, 0),
                    )
            st.rerun()

st.divider()
show_codebook()
