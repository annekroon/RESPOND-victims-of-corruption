"""Streamlit interface for manually reviewing active-learning annotations.

Run on the remote machine with:
    streamlit run miscellaneous/annotation_interface.py --server.address 0.0.0.0 --server.port 8501
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_AL_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline/active_learning"
)
DEFAULT_INPUT_PATH = DEFAULT_AL_DIR / "active_learning_batch_with_llm_suggestions.csv"
DEFAULT_OUTPUT_PATH = DEFAULT_AL_DIR / "active_learning_batch_reviewed.csv"

INPUT_PATH = Path(os.environ.get("RESPOND_ANNOTATION_INPUT", DEFAULT_INPUT_PATH))
OUTPUT_PATH = Path(os.environ.get("RESPOND_ANNOTATION_OUTPUT", DEFAULT_OUTPUT_PATH))

FINAL_LABEL_OPTIONS = [
    "",
    "political corruption",
    "no political corruption",
    "unsure / revisit",
]

LLM_TO_HUMAN_LABEL = {
    "Yes": "political corruption",
    "No": "no political corruption",
    "Mentioned but not central": "no political corruption",
    "Unsure": "unsure / revisit",
}


st.set_page_config(
    page_title="RESPOND Annotation Review",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def atomic_write_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    dataframe.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_data() -> pd.DataFrame:
    if OUTPUT_PATH.exists():
        data = read_csv(OUTPUT_PATH)
        st.sidebar.success(f"Loaded reviewed file: {OUTPUT_PATH}")
    else:
        data = read_csv(INPUT_PATH)
        st.sidebar.info(f"Loaded input file: {INPUT_PATH}")

    for column in ["human_final_label", "human_notes"]:
        if column not in data.columns:
            data[column] = ""

    data["human_final_label"] = data["human_final_label"].fillna("")
    data["human_notes"] = data["human_notes"].fillna("")
    return data


def save_data() -> None:
    atomic_write_csv(st.session_state.data, OUTPUT_PATH)
    read_csv.clear()
    st.toast("Saved")


def reviewed_mask(data: pd.DataFrame) -> pd.Series:
    return data["human_final_label"].fillna("").astype(str).str.strip().ne("")


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data.copy()

    countries = sorted(filtered["country"].dropna().astype(str).unique()) if "country" in filtered.columns else []
    llm_labels = (
        sorted(filtered["llm_label_suggestion"].dropna().astype(str).unique())
        if "llm_label_suggestion" in filtered.columns
        else []
    )
    buckets = sorted(filtered["al_bucket"].dropna().astype(str).unique()) if "al_bucket" in filtered.columns else []

    status = st.sidebar.radio("Status", ["Unreviewed", "All", "Reviewed"], horizontal=True)
    st.session_state.current_status_filter = status
    selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)
    selected_llm_labels = st.sidebar.multiselect("LLM suggestions", llm_labels, default=llm_labels)
    selected_buckets = st.sidebar.multiselect("Active-learning buckets", buckets, default=buckets)

    if selected_countries and "country" in filtered.columns:
        filtered = filtered[filtered["country"].astype(str).isin(selected_countries)]
    if selected_llm_labels and "llm_label_suggestion" in filtered.columns:
        filtered = filtered[filtered["llm_label_suggestion"].astype(str).isin(selected_llm_labels)]
    if selected_buckets and "al_bucket" in filtered.columns:
        filtered = filtered[filtered["al_bucket"].astype(str).isin(selected_buckets)]

    is_reviewed = reviewed_mask(filtered)
    if status == "Unreviewed":
        filtered = filtered[~is_reviewed]
    elif status == "Reviewed":
        filtered = filtered[is_reviewed]

    return filtered


def value(row: pd.Series, column: str, default: str = "") -> str:
    item = row.get(column, default)
    if pd.isna(item):
        return default
    return str(item)


def format_prob(row: pd.Series) -> str:
    prob = row.get("prob_political_corruption", "")
    try:
        return f"{float(prob):.3f}"
    except (TypeError, ValueError):
        return ""


def update_row(row_index: int, label: str, notes: str) -> None:
    st.session_state.data.loc[row_index, "human_final_label"] = label
    st.session_state.data.loc[row_index, "human_notes"] = notes
    save_data()


if "data" not in st.session_state:
    st.session_state.data = load_data()
if "position" not in st.session_state:
    st.session_state.position = 0

data = st.session_state.data
total_rows = len(data)
reviewed_rows = int(reviewed_mask(data).sum())

st.sidebar.title("RESPOND Review")
st.sidebar.caption("Manual validation of LLM-assisted active-learning labels.")
st.sidebar.metric("Reviewed", f"{reviewed_rows:,} / {total_rows:,}")
st.sidebar.progress(reviewed_rows / total_rows if total_rows else 0)

if st.sidebar.button("Save Now", use_container_width=True):
    save_data()

st.sidebar.divider()
filtered = apply_filters(data)

if filtered.empty:
    st.title("RESPOND Annotation Review")
    st.warning("No rows match the current filters.")
    st.stop()

filtered_indices = filtered.index.tolist()
st.session_state.position = min(st.session_state.position, len(filtered_indices) - 1)
st.session_state.position = max(st.session_state.position, 0)

row_index = filtered_indices[st.session_state.position]
row = data.loc[row_index]

st.title("RESPOND Annotation Review")

meta_cols = st.columns([1, 1, 1, 1, 1])
meta_cols[0].metric("Filtered Row", f"{st.session_state.position + 1:,} / {len(filtered_indices):,}")
meta_cols[1].metric("Country", value(row, "country"))
meta_cols[2].metric("Year", value(row, "year"))
meta_cols[3].metric("Model Prob.", format_prob(row))
meta_cols[4].metric("LLM", value(row, "llm_label_suggestion"))

st.caption(
    f"URI: {value(row, 'uri')} | Source: {value(row, 'source_uri')} | "
    f"Bucket: {value(row, 'al_bucket')}"
)

nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
with nav_left:
    if st.button("Previous", use_container_width=True, disabled=st.session_state.position == 0):
        st.session_state.position -= 1
        st.rerun()
with nav_right:
    if st.button("Next", use_container_width=True, disabled=st.session_state.position >= len(filtered_indices) - 1):
        st.session_state.position += 1
        st.rerun()

st.divider()

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Translation")
    st.write(value(row, "translated_text", ""))

with right:
    st.subheader("Original")
    st.write(value(row, "article_text", ""))

st.divider()

llm_cols = st.columns([1, 1, 2])
llm_cols[0].metric("Suggestion", value(row, "llm_label_suggestion"))
llm_cols[1].metric("Confidence", value(row, "llm_confidence"))
with llm_cols[2]:
    st.markdown("**Evidence**")
    st.write(value(row, "llm_evidence"))

st.markdown("**LLM rationale**")
st.write(value(row, "llm_rationale"))

st.divider()

existing_label = value(row, "human_final_label")
if existing_label not in FINAL_LABEL_OPTIONS:
    existing_label = ""

default_label = LLM_TO_HUMAN_LABEL.get(value(row, "llm_label_suggestion"), "")

with st.form(key=f"annotation_form_{row_index}", clear_on_submit=False):
    st.subheader("Human Annotation")

    if not existing_label and default_label:
        st.caption(f"Suggested mapped label: {default_label}")

    final_label = st.radio(
        "Final label",
        FINAL_LABEL_OPTIONS,
        index=FINAL_LABEL_OPTIONS.index(existing_label),
        horizontal=True,
    )
    notes = st.text_area("Notes", value=value(row, "human_notes"), height=120)

    form_cols = st.columns([1, 1, 1, 2])
    save_clicked = form_cols[0].form_submit_button("Save", use_container_width=True)
    accept_clicked = form_cols[1].form_submit_button("Accept LLM", use_container_width=True)
    save_next_clicked = form_cols[2].form_submit_button("Save + Next", use_container_width=True)

if accept_clicked:
    if not default_label:
        st.warning("No mapped LLM label available for this row.")
    else:
        update_row(row_index, default_label, notes)
        st.rerun()

if save_clicked or save_next_clicked:
    update_row(row_index, final_label, notes)
    if (
        save_next_clicked
        and st.session_state.current_status_filter != "Unreviewed"
        and st.session_state.position < len(filtered_indices) - 1
    ):
        st.session_state.position += 1
    st.rerun()

st.sidebar.divider()
st.sidebar.download_button(
    "Download Reviewed CSV",
    data=st.session_state.data.to_csv(index=False),
    file_name=OUTPUT_PATH.name,
    mime="text/csv",
    use_container_width=True,
)

st.sidebar.caption(f"Input: {INPUT_PATH}")
st.sidebar.caption(f"Output: {OUTPUT_PATH}")
