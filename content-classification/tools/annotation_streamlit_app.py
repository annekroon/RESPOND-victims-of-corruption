"""Streamlit interface for article-level content annotation.

Run with environment variables documented in content-classification/README.md.
The app writes the same coder CSV and annotation manifest as the Flask app.
"""

from __future__ import annotations

import gzip
import html
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from content_annotation_common import (
    load_annotation_data,
    output_path_for_coder,
    record_annotation,
    reviewed_mask,
    safe_coder_id,
    save_annotation_data,
)


DEFAULT_VALIDATION_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification/validation_final"
)
DEFAULT_INPUT_PATH = (
    DEFAULT_VALIDATION_DIR / "content_validation_final_n500_english.csv.gz"
)

INPUT_PATH = Path(os.environ.get("CONTENT_ANNOTATION_INPUT", DEFAULT_INPUT_PATH))
OUTPUT_TEMPLATE = os.environ.get("CONTENT_ANNOTATION_OUTPUT_TEMPLATE", "")
OUTPUT_PATH_ENV = os.environ.get("CONTENT_ANNOTATION_OUTPUT", "")
APP_PASSWORD = os.environ.get("CONTENT_ANNOTATION_PASSWORD", "")
DEFAULT_CODER_ID = os.environ.get("CONTENT_ANNOTATION_CODER_ID", "")
DEFAULT_CODER_FIRST_NAME = os.environ.get(
    "CONTENT_ANNOTATION_CODER_FIRST_NAME",
    DEFAULT_CODER_ID,
)

LABELS = {
    "victim_visibility": {
        "question": "Who or what is explicitly harmed by the corruption?",
        "options": [
            "no_victim",
            "concrete_victim",
            "institutional_societal_victim",
            "unclear",
        ],
        "display": {
            "no_victim": "No corruption-related victim is explicit",
            "concrete_victim": "Concrete person, group, or entity",
            "institutional_societal_victim": "Institution, society, or public interest",
            "unclear": "Unclear from the text",
        },
        "help": (
            "Require explicit corruption-caused harm. Do not infer a victim from "
            "the offense, public money, an investigation, or the surrounding scandal."
        ),
    },
    "corruption_frame": {
        "question": "How does the article primarily frame the corruption problem?",
        "options": ["individualized", "systemic", "other_or_mixed", "unclear"],
        "display": {
            "individualized": "Individualized case or actor",
            "systemic": "Systemic governance pattern",
            "other_or_mixed": "Other, incidental, procedural, or mixed",
            "unclear": "Unclear from the text",
        },
        "help": (
            "Code the dominant explanation. Named actors do not automatically make "
            "a case individualized; multiple institutions do not automatically make it systemic."
        ),
    },
    "case_location": {
        "question": "Where is the main corruption case relative to the publication country?",
        "options": ["domestic", "abroad", "unclear"],
        "display": {
            "domestic": "Domestic",
            "abroad": "Abroad",
            "unclear": "No principal location is clear",
        },
        "help": (
            "Locate the corruption case, not the news agency, quoted speaker, or an "
            "unrelated event. Domestic misuse of EU or foreign money remains domestic."
        ),
    },
    "accused_actor_visibility": {
        "question": "Which alleged corruption participants are visible?",
        "options": [
            "no_accused_actor",
            "individual_actor",
            "organizational_or_institutional_actor",
            "both_individual_and_organizational",
            "unclear",
        ],
        "display": {
            "no_accused_actor": "No alleged participant is identified",
            "individual_actor": "Individual actor only",
            "organizational_or_institutional_actor": "Organization or institution only",
            "both_individual_and_organizational": "Both individual and organization",
            "unclear": "Unclear from the text",
        },
        "help": (
            "Count only alleged participants. An organization counts only when the "
            "article attributes corrupt conduct to the organization itself."
        ),
    },
}

HUMAN_COLUMNS = {
    "victim_visibility": "human_victim_visibility",
    "corruption_frame": "human_corruption_frame",
    "case_location": "human_case_location",
    "accused_actor_visibility": "human_accused_actor_visibility",
}


st.set_page_config(
    page_title="RESPOND Content Coding",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .block-container {max-width: 1540px; padding-top: 1.4rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #d9dee8;}
    .article-copy {white-space: pre-wrap; line-height: 1.62; font-size: 1rem; color: inherit;}
    .article-meta {color: inherit; opacity: .72; font-size: .9rem; margin: .25rem 0 .8rem;}
    .saved-note {color: inherit; font-weight: 650; font-size: .9rem;}
    div[role="radiogroup"] label {padding-top: .16rem; padding-bottom: .16rem;}
    h1, h2, h3 {letter-spacing: 0;}
</style>
""",
    unsafe_allow_html=True,
)


def value(row: pd.Series, column: str, default: str = "") -> str:
    item = row.get(column, default)
    if pd.isna(item):
        return default
    return str(item)


def render_article_text(text: str) -> None:
    safe_text = html.escape(text or "No text is available for this view.")
    st.markdown(
        f'<div class="article-copy">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def backup_file(data: pd.DataFrame, output_path: Path) -> tuple[bytes, str]:
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    if output_path.name.endswith(".gz"):
        return gzip.compress(csv_bytes), "application/gzip"
    return csv_bytes, "text/csv"


def authenticate() -> None:
    if not APP_PASSWORD or st.session_state.get("authenticated"):
        return
    st.title("RESPOND Content Coding")
    st.write("Enter the annotation password to continue.")
    with st.form("login"):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Continue", type="primary")
    if submitted:
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        st.error("The password is incorrect.")
    st.stop()


def establish_coder() -> None:
    if st.session_state.get("coder_id"):
        return
    st.sidebar.subheader("Coder")
    with st.sidebar.form("coder_identity"):
        first_name = st.text_input(
            "First name",
            value=DEFAULT_CODER_FIRST_NAME,
            autocomplete="given-name",
        )
        coder_id = st.text_input(
            "Coder ID",
            value=DEFAULT_CODER_ID or first_name,
            help="Used in the output filename. Use the same ID to resume later.",
        )
        start = st.form_submit_button("Start or resume", type="primary")
    if start:
        if not first_name.strip():
            st.sidebar.error("Enter your first name.")
        else:
            st.session_state.coder_first_name = first_name.strip()
            st.session_state.coder_id = safe_coder_id(coder_id or first_name)
            st.session_state.code_session_id = uuid.uuid4().hex
            st.rerun()
    st.title("RESPOND Content Coding")
    st.info("Enter your name in the sidebar to start or resume coding.")
    st.stop()


def initialize_data() -> None:
    output_path = output_path_for_coder(
        input_path=INPUT_PATH,
        output_path=Path(OUTPUT_PATH_ENV) if OUTPUT_PATH_ENV else None,
        output_template=OUTPUT_TEMPLATE,
        coder_id=st.session_state.coder_id,
    )
    signature = f"{INPUT_PATH.resolve()}::{output_path.resolve()}"
    if st.session_state.get("data_signature") == signature:
        return
    data, provenance, resumed = load_annotation_data(INPUT_PATH, output_path)
    st.session_state.data = data
    st.session_state.provenance = provenance
    st.session_state.output_path = output_path
    st.session_state.data_signature = signature
    st.session_state.position = 0
    st.session_state.filter_signature = ""
    st.session_state.resumed = resumed


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
    reviewed = reviewed_mask(filtered)
    if status == "To code":
        filtered = filtered.loc[~reviewed]
    elif status == "Completed":
        filtered = filtered.loc[reviewed]
    if query.strip():
        columns = [
            column
            for column in [
                "translated_text_en",
                "translated_text",
                "article_text",
                "uri",
                "source_uri",
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


def save_current(
    row_index: int,
    victim_visibility: str | None,
    corruption_frame: str | None,
    case_location: str | None,
    accused_actor_visibility: str | None,
    notes: str,
) -> bool:
    selections = [
        victim_visibility,
        corruption_frame,
        case_location,
        accused_actor_visibility,
    ]
    if any(not selection for selection in selections):
        st.error("Choose one answer for each of the four questions before saving.")
        return False
    record_annotation(
        st.session_state.data,
        row_index,
        victim_visibility=str(victim_visibility),
        corruption_frame=str(corruption_frame),
        case_location=str(case_location),
        accused_actor_visibility=str(accused_actor_visibility),
        notes=notes,
        coder_id=st.session_state.coder_id,
        coder_first_name=st.session_state.coder_first_name,
        code_session_id=st.session_state.code_session_id,
    )
    save_annotation_data(
        st.session_state.data,
        input_path=INPUT_PATH,
        output_path=st.session_state.output_path,
        coder_id=st.session_state.coder_id,
        coder_first_name=st.session_state.coder_first_name,
        code_session_id=st.session_state.code_session_id,
        provenance=st.session_state.provenance,
    )
    st.session_state.saved_message = "Saved"
    return True


def label_radio(variable: str, row: pd.Series):
    specification = LABELS[variable]
    existing = value(row, HUMAN_COLUMNS[variable])
    options = specification["options"]
    return st.radio(
        specification["question"],
        options,
        index=options.index(existing) if existing in options else None,
        format_func=lambda option: specification["display"][option],
        help=specification["help"],
        key=f"{variable}_{value(row, 'article_id')}",
    )


def show_codebook() -> None:
    st.subheader("Quick decision guide")
    with st.expander("Victim visibility"):
        st.markdown(
            """
**Require all three:** a victim or public interest, an explicit harm, and an
explicit connection between that harm and the corruption. Do not infer harm
from bribery, fraud, public money, prosecution, scandal, or a favored bidder.

- **Concrete:** a bounded person, group, company, association, or community
  explicitly loses money, rights, services, work, contracts, or opportunities,
  or faces a coercive corrupt demand.
- **Institutional/societal:** corruption explicitly harms public finances,
  public services, democracy, rule of law, public trust, state capacity, the
  economy, or another broad public interest.
- If both are explicit, choose **concrete**. Use **unclear** only when the text
  genuinely prevents a decision.
"""
        )
    with st.expander("Corruption frame"):
        st.markdown(
            """
- **Individualized:** a specific actor, allegation, investigation, trial, or scandal.
- **Systemic:** corruption is represented as an entrenched institutional or governance pattern.
- **Other/mixed:** corruption is incidental, procedural, technical, evenly mixed,
  or does not fit the individualized/systemic distinction.

Code the article's dominant explanation, not the number of people or institutions mentioned.
"""
        )
    with st.expander("Case location"):
        st.markdown(
            """
Locate the **main corruption case** and compare it with the publication country.
Do not use the news agency, quoted speaker, court-reporting location, or an
unrelated event. Domestic misuse of foreign or EU money remains domestic.
"""
        )
    with st.expander("Accused actor visibility"):
        st.markdown(
            """
Count actors explicitly represented as committing, attempting, assisting,
enabling, financing, directing, or concealing the corruption. Do not count
victims, witnesses, investigators, regulators, beneficiaries, employers,
associates, or owners unless they are separately alleged to have participated.

An organization counts only when the text independently attributes corrupt
participation to the organization itself. An accusation against an employee,
leader, owner, member, subsidiary, or associate does not automatically count
as organizational participation.
"""
        )


authenticate()
establish_coder()

try:
    initialize_data()
except Exception as exc:
    st.title("RESPOND Content Coding")
    st.error(f"The annotation files could not be opened: {exc}")
    st.stop()

data = st.session_state.data
total = len(data)
reviewed_total = int(reviewed_mask(data).sum())
countries = sorted(data["country"].dropna().astype(str).unique())

st.sidebar.title("Content Coding")
st.sidebar.caption(
    f"Coder: {st.session_state.coder_first_name} ({st.session_state.coder_id})"
)
st.sidebar.metric("Completed", f"{reviewed_total} / {total}")
st.sidebar.progress(reviewed_total / total if total else 0.0)

country = st.sidebar.selectbox(
    "Publication country",
    ["All countries", *countries],
    key="filter_country",
)
status = st.sidebar.radio(
    "Queue",
    ["To code", "All", "Completed"],
    horizontal=True,
    key="filter_status",
)
query = st.sidebar.text_input(
    "Find article",
    placeholder="Search text, URI, or source",
    key="filter_query",
)

with st.sidebar.expander("Progress by country"):
    st.dataframe(country_progress(data), hide_index=True, use_container_width=True)

backup_data, backup_mime = backup_file(data, st.session_state.output_path)
st.sidebar.download_button(
    "Download backup",
    data=backup_data,
    file_name=st.session_state.output_path.name,
    mime=backup_mime,
    use_container_width=True,
    disabled=not st.session_state.output_path.exists(),
)
with st.sidebar.expander("Files"):
    st.caption(f"Input: {INPUT_PATH}")
    st.caption(f"Output: {st.session_state.output_path}")
if st.sidebar.button("Change coder", use_container_width=True):
    for key in [
        "coder_id",
        "coder_first_name",
        "code_session_id",
        "data",
        "data_signature",
        "position",
    ]:
        st.session_state.pop(key, None)
    st.rerun()

indices = filtered_indices(data, country=country, status=status, query=query)
filter_signature = f"{country}|{status}|{query}"
if st.session_state.get("filter_signature") != filter_signature:
    st.session_state.position = 0
    st.session_state.filter_signature = filter_signature

if reviewed_total == total and total:
    st.success(
        "Coding complete. Every article has all four required labels. "
        "You can still select Completed or All in the sidebar and revise any article."
    )

if not indices:
    st.title("RESPOND Content Coding")
    if status == "To code" and reviewed_total == total:
        st.subheader("You are done")
        st.write("All annotations are saved. Switch the queue to Completed to review them.")
    else:
        st.info("No articles match the current country, queue, and search filters.")
    show_codebook()
    st.stop()

position = max(0, min(int(st.session_state.position), len(indices) - 1))
st.session_state.position = position
row_index = indices[position]
row = data.loc[row_index]

heading_left, heading_right = st.columns([4, 1])
with heading_left:
    st.title("RESPOND Content Coding")
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
        st.session_state.position = position - 1
        st.rerun()
with nav_status:
    saved_message = st.session_state.pop("saved_message", "")
    if saved_message:
        st.markdown('<p class="saved-note">Saved to the coder file.</p>', unsafe_allow_html=True)
    else:
        st.caption("Use Save and continue before navigating away from changed answers.")
with nav_next:
    if st.button(
        "Next",
        use_container_width=True,
        disabled=position >= len(indices) - 1,
    ):
        st.session_state.position = position + 1
        st.rerun()

article_column, coding_column = st.columns([1.55, 1], gap="large")
with article_column:
    st.subheader("Article")
    st.markdown(
        '<div class="article-meta">'
        f"Publication country: <b>{html.escape(value(row, 'country'))}</b>"
        f" &nbsp; Source: {html.escape(value(row, 'source_uri') or value(row, 'source.uri'))}"
        "</div>",
        unsafe_allow_html=True,
    )
    translated = value(row, "translated_text_en") or value(row, "translated_text")
    original = value(row, "article_text")
    translation_tab, original_tab, compare_tab = st.tabs(
        ["English translation", "Original", "Compare"]
    )
    with translation_tab:
        with st.container(height=600, border=True):
            render_article_text(translated)
    with original_tab:
        with st.container(height=600, border=True):
            render_article_text(original)
    with compare_tab:
        translated_column, original_column = st.columns(2)
        with translated_column:
            st.caption("English translation")
            with st.container(height=540, border=True):
                render_article_text(translated)
        with original_column:
            st.caption("Original")
            with st.container(height=540, border=True):
                render_article_text(original)
    with st.expander("Article details"):
        st.write(f"Sample ID: {value(row, 'content_sample_id')}")
        st.write(f"Article ID: {value(row, 'article_id')}")
        st.write(f"URI: {value(row, 'uri')}")
        st.write(f"Classifier probability: {value(row, 'prob_political_corruption')}")
        if value(row, "translation_notes"):
            st.write(f"Translation note: {value(row, 'translation_notes')}")

with coding_column:
    st.subheader("Your coding")
    st.caption(
        "Choose one answer for each question. Uncoded questions start blank; "
        "saved answers reappear when you review an article."
    )
    with st.form(f"coding_form_{value(row, 'article_id')}"):
        victim_visibility = label_radio("victim_visibility", row)
        st.divider()
        corruption_frame = label_radio("corruption_frame", row)
        st.divider()
        case_location = label_radio("case_location", row)
        st.caption(f"Publication country: {value(row, 'country')}")
        st.divider()
        accused_actor_visibility = label_radio("accused_actor_visibility", row)
        notes = st.text_area(
            "Notes (optional)",
            value=value(row, "human_notes"),
            placeholder=(
                "Record ambiguity, translation concerns, or the reason for a "
                "difficult decision."
            ),
            height=90,
        )
        save_column, continue_column = st.columns([1, 1.5])
        save_only = save_column.form_submit_button(
            "Save",
            use_container_width=True,
        )
        save_and_continue = continue_column.form_submit_button(
            "Save and continue",
            type="primary",
            use_container_width=True,
        )

    if save_only or save_and_continue:
        if save_current(
            row_index,
            victim_visibility,
            corruption_frame,
            case_location,
            accused_actor_visibility,
            notes,
        ):
            if save_and_continue:
                updated_indices = filtered_indices(
                    data,
                    country=country,
                    status=status,
                    query=query,
                )
                if status == "To code":
                    st.session_state.position = min(position, max(len(updated_indices) - 1, 0))
                else:
                    st.session_state.position = min(
                        position + 1,
                        max(len(updated_indices) - 1, 0),
                    )
            st.rerun()

st.divider()
show_codebook()
