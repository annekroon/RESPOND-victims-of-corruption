"""Flask app for manually annotating content-classification validation samples.

Run on a remote server with:

    CONTENT_ANNOTATION_INPUT=/path/to/content_validation_sample_100_per_country_english.csv.gz \
    CONTENT_ANNOTATION_OUTPUT_TEMPLATE=/path/to/content_validation_sample_100_per_country_{coder_id}.csv.gz \
    CONTENT_ANNOTATION_PASSWORD='choose-a-password' \
    flask --app content-classification/tools/annotation_flask_app.py run --host 0.0.0.0 --port 8502

For multiple external coders, prefer CONTENT_ANNOTATION_OUTPUT_TEMPLATE so each
coder writes to a distinct CSV while reading the same translated input file.
"""

from __future__ import annotations

import os
import re
import sys
from functools import wraps
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from flask import Flask, redirect, render_template_string, request, session, url_for


DEFAULT_VALIDATION_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification/validation"
)
DEFAULT_INPUT_PATH = DEFAULT_VALIDATION_DIR / "content_validation_sample_100_per_country_english.csv.gz"
DEFAULT_OUTPUT_PATH = DEFAULT_VALIDATION_DIR / "content_validation_sample_100_per_country_human_coded.csv.gz"

INPUT_PATH = Path(os.environ.get("CONTENT_ANNOTATION_INPUT", DEFAULT_INPUT_PATH))
OUTPUT_PATH = Path(os.environ.get("CONTENT_ANNOTATION_OUTPUT", DEFAULT_OUTPUT_PATH))
OUTPUT_TEMPLATE = os.environ.get("CONTENT_ANNOTATION_OUTPUT_TEMPLATE", "")
APP_PASSWORD = os.environ.get("CONTENT_ANNOTATION_PASSWORD", "")
SECRET_KEY = os.environ.get("CONTENT_ANNOTATION_SECRET_KEY", "dev-change-me")
CODER_ID = os.environ.get("CONTENT_ANNOTATION_CODER_ID", "coder")

VICTIM_OPTIONS = [
    "",
    "no_victim",
    "concrete_victim",
    "institutional_societal_victim",
    "unclear",
]
FRAME_OPTIONS = [
    "",
    "individualized",
    "systemic",
    "other_or_mixed",
    "unclear",
]
CASE_LOCATION_OPTIONS = [
    "",
    "domestic",
    "abroad",
    "unclear",
]
YES_NO_UNCLEAR_OPTIONS = ["", "yes", "no", "unclear"]
ACCUSED_OPTIONS = [
    "",
    "no_accused_actor",
    "individual_actor",
    "organizational_or_institutional_actor",
    "both_individual_and_organizational",
    "unclear",
]

HUMAN_COLUMNS = [
    "human_victim_visibility",
    "human_corruption_frame",
    "human_case_location",
    "human_abroad_case",
    "human_accused_actor_visibility",
    "human_accused_actor_visible",
    "human_notes",
    "human_coder_id",
    "human_coded_at",
]


app = Flask(__name__)
app.secret_key = SECRET_KEY


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer")


def write_csv_atomic(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    data.to_csv(tmp_path, index=False, compression="gzip" if path.name.endswith(".gz") else None)
    tmp_path.replace(path)


def ensure_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "article_id" not in data.columns:
        data["article_id"] = data["uri"].fillna("").astype(str) if "uri" in data.columns else ""
    missing_id = data["article_id"].fillna("").astype(str).str.strip().eq("")
    if missing_id.any():
        fallback = data.index.astype(str)
        if "country" in data.columns:
            fallback = data["country"].fillna("").astype(str) + "::" + fallback
        data.loc[missing_id, "article_id"] = fallback[missing_id]

    for column in HUMAN_COLUMNS:
        if column not in data.columns:
            data[column] = ""
        data[column] = data[column].fillna("")
    return data


def safe_coder_id(coder_id: str) -> str:
    coder_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", coder_id.strip())
    return coder_id or CODER_ID


def current_coder_id() -> str:
    return safe_coder_id(session.get("coder_id") or CODER_ID)


def output_path_for_coder(coder_id: str) -> Path:
    if OUTPUT_TEMPLATE:
        return Path(OUTPUT_TEMPLATE.format(coder_id=safe_coder_id(coder_id)))
    return OUTPUT_PATH


def load_data(coder_id: str) -> pd.DataFrame:
    output_path = output_path_for_coder(coder_id)
    path = output_path if output_path.exists() else INPUT_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    return ensure_columns(read_csv(path))


def save_data(data: pd.DataFrame, coder_id: str) -> None:
    write_csv_atomic(data, output_path_for_coder(coder_id))


def value(row: pd.Series, column: str, default: str = "") -> str:
    item = row.get(column, default)
    if pd.isna(item):
        return default
    return str(item)


def reviewed_mask(data: pd.DataFrame) -> pd.Series:
    required = [
        "human_victim_visibility",
        "human_corruption_frame",
        "human_case_location",
        "human_accused_actor_visibility",
    ]
    mask = pd.Series(True, index=data.index)
    for column in required:
        mask = mask & data[column].fillna("").astype(str).str.strip().ne("")
    return mask


def filtered_indices(data: pd.DataFrame) -> list[int]:
    filtered = data.copy()
    country = request.args.get("country", "")
    status = request.args.get("status", "unreviewed")
    query = request.args.get("q", "").strip().lower()

    if country:
        filtered = filtered[filtered["country"].astype(str).eq(country)]
    if status == "unreviewed":
        filtered = filtered[~reviewed_mask(filtered)]
    elif status == "reviewed":
        filtered = filtered[reviewed_mask(filtered)]
    if query:
        text_columns = [column for column in ["translated_text_en", "article_text", "uri"] if column in filtered.columns]
        text = filtered[text_columns].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[text.str.contains(query, regex=False, na=False)]

    return filtered.index.tolist()


def require_login(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if APP_PASSWORD and not session.get("authenticated"):
            return redirect(url_for("login", next=request.url))
        return func(*args, **kwargs)

    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        if request.form.get("password", "") == APP_PASSWORD:
            session["authenticated"] = True
            session["coder_id"] = safe_coder_id(request.form.get("coder_id", CODER_ID))
            return redirect(request.args.get("next") or url_for("index"))
        error = "Incorrect password."
    return render_template_string(LOGIN_TEMPLATE, error=error, default_coder_id=CODER_ID)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@require_login
def index():
    coder_id = current_coder_id()
    data = load_data(coder_id)
    indices = filtered_indices(data)
    pos = max(0, min(int(request.args.get("pos", 0)), max(len(indices) - 1, 0)))
    reviewed = int(reviewed_mask(data).sum())
    countries = sorted(data["country"].dropna().astype(str).unique()) if "country" in data.columns else []
    if not indices:
        return render_template_string(
            APP_TEMPLATE,
            no_rows=True,
            total=len(data),
            reviewed=reviewed,
            countries=countries,
            filters=current_filters(),
            coder_id=coder_id,
            output_path=output_path_for_coder(coder_id),
        )
    row_index = indices[pos]
    row = data.loc[row_index]
    translated_text = value(row, "translated_text_en") or value(row, "translated_text")
    return render_template_string(
        APP_TEMPLATE,
        no_rows=False,
        row=row,
        translated_text=translated_text,
        row_index=row_index,
        pos=pos,
        n_filtered=len(indices),
        total=len(data),
        reviewed=reviewed,
        countries=countries,
        filters=current_filters(),
        coder_id=coder_id,
        output_path=output_path_for_coder(coder_id),
        prev_url=nav_url(max(pos - 1, 0)),
        next_url=nav_url(min(pos + 1, len(indices) - 1)),
        save_url=url_for("save", row_index=row_index, **current_filters()),
        victim_options=VICTIM_OPTIONS,
        frame_options=FRAME_OPTIONS,
        case_location_options=CASE_LOCATION_OPTIONS,
        yes_no_unclear_options=YES_NO_UNCLEAR_OPTIONS,
        accused_options=ACCUSED_OPTIONS,
        value=value,
    )


@app.route("/save/<int:row_index>", methods=["POST"])
@require_login
def save(row_index: int):
    coder_id = current_coder_id()
    data = load_data(coder_id)
    if row_index not in data.index:
        return "Unknown row", 404

    import datetime as dt

    for column in HUMAN_COLUMNS:
        if column in {"human_coder_id", "human_coded_at"}:
            continue
        data.loc[row_index, column] = request.form.get(column, "")
    data.loc[row_index, "human_coder_id"] = request.form.get("human_coder_id", coder_id)
    data.loc[row_index, "human_coded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_data(data, coder_id)

    action = request.form.get("action", "save")
    pos = int(request.form.get("pos", 0))
    if action == "save_next":
        pos += 1
    return redirect(nav_url(pos))


def current_filters() -> dict:
    return {
        "country": request.args.get("country", ""),
        "status": request.args.get("status", "unreviewed"),
        "q": request.args.get("q", ""),
    }


def nav_url(pos: int) -> str:
    filters = current_filters()
    return url_for("index", pos=pos, **filters)


LOGIN_TEMPLATE = """
<!doctype html>
<title>RESPOND Annotation Login</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f5f7fb; color: #172033; }
main { margin: 8vh auto; max-width: 440px; background: white; border: 1px solid #d8deea; border-radius: 10px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(20, 35, 60, .08); }
input, button { width: 100%; box-sizing: border-box; padding: .75rem; margin-top: .45rem; font-size: 1rem; border: 1px solid #c9d2e3; border-radius: 7px; }
button { cursor: pointer; background: #244f86; color: white; border-color: #244f86; margin-top: 1rem; font-weight: 700; }
label { display: block; margin-top: .85rem; font-weight: 700; }
.hint { color: #5c6678; line-height: 1.45; }
.error { color: #a40000; font-weight: 700; }
</style>
<main>
  <h1>RESPOND Annotation</h1>
  <p class="hint">Log in with your coder ID. Your annotations are saved under that ID.</p>
  <form method="post">
    <label>Coder ID</label>
    <input name="coder_id" value="{{ default_coder_id }}" autocomplete="username" autofocus>
    <label>Password</label>
    <input type="password" name="password" autocomplete="current-password">
    <button type="submit">Log in</button>
  </form>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
</main>
"""


APP_TEMPLATE = """
<!doctype html>
<title>RESPOND Content Annotation</title>
<style>
:root {
  --border: #d7dde8;
  --muted: #637083;
  --bg: #f4f6fa;
  --ink: #172033;
  --accent: #244f86;
  --accent-soft: #e8f0fb;
  --warn: #fff4d8;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: var(--ink); background: var(--bg); }
header { position: sticky; top: 0; background: white; border-bottom: 1px solid var(--border); padding: .85rem 1.25rem; z-index: 2; box-shadow: 0 2px 12px rgba(20, 35, 60, .05); }
main { padding: 1rem 1.25rem 2rem; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.filters { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .75rem; }
select, input, textarea, button { font: inherit; padding: .55rem .65rem; border: 1px solid var(--border); border-radius: 7px; background: white; }
button, .button { cursor: pointer; background: var(--accent); color: white; border: 1px solid var(--accent); border-radius: 7px; font-weight: 700; padding: .55rem .65rem; text-decoration: none; display: inline-block; }
.ghost { color: var(--accent); background: white; border-color: var(--border); }
.meta { color: var(--muted); font-size: .9rem; }
.layout { display: grid; grid-template-columns: minmax(320px, 440px) minmax(0, 1fr); gap: 1rem; align-items: start; }
.panel { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; background: white; }
.codebook { position: sticky; top: 6.6rem; max-height: calc(100vh - 7.5rem); overflow: auto; }
.codebook h2, .panel h2 { margin: 0 0 .6rem; font-size: 1.15rem; }
.codebook h3 { margin: 1rem 0 .35rem; font-size: .98rem; }
.codebook p, .codebook li { line-height: 1.35; }
.codebook ul { padding-left: 1.1rem; margin: .35rem 0; }
.definition { border-top: 1px solid var(--border); padding-top: .65rem; margin-top: .65rem; }
.definition:first-of-type { border-top: 0; padding-top: 0; }
.tag { display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: var(--accent-soft); color: #163d6f; border-radius: 5px; padding: .08rem .28rem; font-size: .82rem; }
.hint { background: var(--warn); border: 1px solid #ead79a; border-radius: 8px; padding: .75rem; margin-bottom: .9rem; line-height: 1.4; }
.article-meta { display: flex; flex-wrap: wrap; gap: .45rem .9rem; margin-bottom: .75rem; }
.reader-toolbar { display: flex; gap: .45rem; flex-wrap: wrap; margin-bottom: .75rem; }
.reader-button { color: var(--accent); background: white; border-color: var(--border); }
.reader-button.active { background: var(--accent); color: white; border-color: var(--accent); }
.text-grid { display: grid; grid-template-columns: 1fr; gap: 1rem; }
.text-grid.side-by-side { grid-template-columns: 1fr 1fr; }
.text-panel[data-hidden="true"] { display: none; }
.text { white-space: pre-wrap; line-height: 1.52; max-height: 64vh; overflow: auto; border: 1px solid var(--border); border-radius: 8px; padding: .9rem; background: #fbfcfe; }
.form-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .75rem; }
.form-grid label { display: flex; flex-direction: column; gap: .3rem; font-weight: 700; }
.field-help { color: var(--muted); font-weight: 400; font-size: .86rem; line-height: 1.35; }
textarea { width: 100%; min-height: 96px; }
.actions { display: flex; gap: .5rem; margin-top: .75rem; flex-wrap: wrap; }
.progress { font-weight: 800; }
.savebar { position: sticky; bottom: 0; background: rgba(244, 246, 250, .96); border-top: 1px solid var(--border); padding: .75rem 0 0; }
@media (max-width: 1180px) {
  .layout, .text-grid.side-by-side, .form-grid { grid-template-columns: 1fr; }
  .codebook { position: static; max-height: none; }
}
</style>
<header>
  <div class="top">
    <div>
      <div class="progress">RESPOND content annotation</div>
      <div class="meta">Coder {{ coder_id }} · Reviewed {{ reviewed }} / {{ total }}{% if not no_rows %} · Filtered row {{ pos + 1 }} / {{ n_filtered }}{% endif %}</div>
    </div>
    <div><a href="{{ url_for('logout') }}">Log out</a></div>
  </div>
  <form class="filters" method="get" action="{{ url_for('index') }}">
    <input type="hidden" name="pos" value="0">
    <select name="status">
      {% for status in ["unreviewed", "all", "reviewed"] %}
      <option value="{{ status }}" {% if filters.status == status %}selected{% endif %}>{{ status }}</option>
      {% endfor %}
    </select>
    <select name="country">
      <option value="">all countries</option>
      {% for country in countries %}
      <option value="{{ country }}" {% if filters.country == country %}selected{% endif %}>{{ country }}</option>
      {% endfor %}
    </select>
    <input name="q" value="{{ filters.q }}" placeholder="search text or URI">
    <button type="submit">Filter</button>
  </form>
</header>
<main>
{% if no_rows %}
  <div class="panel">No rows match the current filters.</div>
{% else %}
  <div class="layout">
    <aside class="panel codebook">
      <h2>Codebook</h2>
      <div class="hint"><b>Code what is substantively present in the article.</b> Use the English translation by default. Check the original when wording, names, or ambiguity matter.</div>

      <div class="definition">
        <h3>Victim visibility</h3>
        <p>Who or what is described as harmed by corruption?</p>
        <ul>
          <li><span class="tag">no_victim</span> No harmed person, group, institution, or public interest is made visible.</li>
          <li><span class="tag">concrete_victim</span> Identifiable people or groups are harmed, such as citizens, voters, taxpayers, residents, patients, students, workers, firms, or communities.</li>
          <li><span class="tag">institutional_societal_victim</span> Harm is framed at the level of democracy, rule of law, public trust, state capacity, institutions, society, the economy, development, or EU accession.</li>
          <li><span class="tag">unclear</span> The article is too incomplete, ambiguous, or translation-problematic to decide.</li>
        </ul>
      </div>

      <div class="definition">
        <h3>Corruption frame</h3>
        <p>How is the corruption problem represented?</p>
        <ul>
          <li><span class="tag">individualized</span> Centered on named or identifiable actors, personal misconduct, accusations, trials, resignations, or scandal episodes.</li>
          <li><span class="tag">systemic</span> Centered on institutional dysfunction, state capture, clientelism, rule-of-law conflict, democratic backsliding, recurring abuse, or corruption as a governance pattern.</li>
          <li><span class="tag">other_or_mixed</span> Both frames are equally central, or the article is mainly procedural, sectoral, technical, election-finance-specific, or otherwise outside the two-way distinction.</li>
          <li><span class="tag">unclear</span> Not enough information to classify the frame.</li>
        </ul>
      </div>

      <div class="definition">
        <h3>Case location</h3>
        <p>Where is the corruption case primarily located?</p>
        <ul>
          <li><span class="tag">domestic</span> The case mainly concerns the publication country or domestic actors/institutions.</li>
          <li><span class="tag">abroad</span> The case mainly concerns another country, foreign actors, foreign institutions, offshore schemes, sanctions, or cross-border probes centered elsewhere.</li>
          <li><span class="tag">unclear</span> Location cannot be determined.</li>
        </ul>
        <p class="meta">EU funds misused domestically still count as domestic. For <span class="tag">abroad_case</span>, use yes only when the main case is abroad.</p>
      </div>

      <div class="definition">
        <h3>Accused actor visibility</h3>
        <p>Is a suspected or accused actor visible?</p>
        <ul>
          <li><span class="tag">no_accused_actor</span> Corruption is discussed generally but no accused actor is identified.</li>
          <li><span class="tag">individual_actor</span> A person or officeholder is accused, investigated, charged, convicted, or explicitly linked.</li>
          <li><span class="tag">organizational_or_institutional_actor</span> A party, company, agency, office, court, ministry, police unit, or other collective actor is implicated.</li>
          <li><span class="tag">both_individual_and_organizational</span> Both individual and collective accused actors are visible.</li>
          <li><span class="tag">unclear</span> Not enough information to decide.</li>
        </ul>
        <p class="meta">Conviction is not required. Allegation, investigation, charge, sanction, or strong linkage is enough.</p>
      </div>
    </aside>

    <section>
      <div class="panel">
        <div class="article-meta">
          <span><b>{{ value(row, "country") }}</b></span>
          <span>{{ value(row, "year") }}</span>
          <span>Article {{ pos + 1 }} / {{ n_filtered }}</span>
          <span>Output: {{ output_path.name }}</span>
        </div>
        <div class="meta">URI: {{ value(row, "uri") }}{% if value(row, "source_uri") %} · Source: {{ value(row, "source_uri") }}{% endif %}</div>
        <div class="actions">
          <a class="button ghost" href="{{ prev_url }}">Previous</a>
          <a class="button ghost" href="{{ next_url }}">Next</a>
        </div>
      </div>

      <section class="panel" style="margin-top: 1rem;">
        <h2>Article Text</h2>
        <div class="reader-toolbar" aria-label="Text view">
          <button class="reader-button active" type="button" data-view="translated">Translation</button>
          <button class="reader-button" type="button" data-view="original">Original</button>
          <button class="reader-button" type="button" data-view="both">Side by side</button>
        </div>
        <div id="text-grid" class="text-grid">
          <div id="translated-panel" class="text-panel">
            <h3>English Translation</h3>
            <div class="text">{{ translated_text or "No translation available for this row." }}</div>
            {% if value(row, "translation_notes") %}<p class="meta">{{ value(row, "translation_notes") }}</p>{% endif %}
          </div>
          <div id="original-panel" class="text-panel" data-hidden="true">
            <h3>Original Article</h3>
            <div class="text">{{ value(row, "article_text") }}</div>
          </div>
        </div>
      </section>

      <form method="post" action="{{ save_url }}" style="margin-top: 1rem;">
        <input type="hidden" name="pos" value="{{ pos }}">
        <section class="panel">
          <h2>Human Codes</h2>
          <div class="form-grid">
            <label>Victim visibility
              <span class="field-help">Code the visibility of harmed people, groups, institutions, or public interests.</span>
              <select name="human_victim_visibility" required>
                {% for option in victim_options %}
                <option value="{{ option }}" {% if value(row, "human_victim_visibility") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
                {% endfor %}
              </select>
            </label>
            <label>Corruption frame
              <span class="field-help">Choose whether corruption is mainly individualized, systemic, mixed/other, or unclear.</span>
              <select name="human_corruption_frame" required>
                {% for option in frame_options %}
                <option value="{{ option }}" {% if value(row, "human_corruption_frame") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
                {% endfor %}
              </select>
            </label>
            <label>Case location
              <span class="field-help">Domestic if centered in the publication country; abroad if centered elsewhere.</span>
              <select name="human_case_location" required>
                {% for option in case_location_options %}
                <option value="{{ option }}" {% if value(row, "human_case_location") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
                {% endfor %}
              </select>
            </label>
            <label>Abroad case
              <span class="field-help">Yes only if the main corruption case is abroad.</span>
              <select name="human_abroad_case">
                {% for option in yes_no_unclear_options %}
                <option value="{{ option }}" {% if value(row, "human_abroad_case") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
                {% endfor %}
              </select>
            </label>
            <label>Accused actor visibility
              <span class="field-help">Who is visibly accused, investigated, sanctioned, or linked?</span>
              <select name="human_accused_actor_visibility" required>
                {% for option in accused_options %}
                <option value="{{ option }}" {% if value(row, "human_accused_actor_visibility") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
                {% endfor %}
              </select>
            </label>
            <label>Accused actor visible
              <span class="field-help">Binary summary: is any accused actor visible?</span>
              <select name="human_accused_actor_visible">
                {% for option in yes_no_unclear_options %}
                <option value="{{ option }}" {% if value(row, "human_accused_actor_visible") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
                {% endfor %}
              </select>
            </label>
          </div>
          <label style="display:block; margin-top:.75rem; font-weight:700;">Notes
            <textarea name="human_notes" placeholder="Optional: record ambiguity, translation issues, or why a difficult choice was made.">{{ value(row, "human_notes") }}</textarea>
          </label>
          <label style="display:block; margin-top:.75rem; font-weight:700;">Coder ID
            <input name="human_coder_id" value="{{ value(row, 'human_coder_id', coder_id) }}">
          </label>
          <div class="savebar">
            <div class="actions">
              <button type="submit" name="action" value="save">Save</button>
              <button type="submit" name="action" value="save_next">Save + Next</button>
            </div>
          </div>
        </section>
      </form>
    </section>
  </div>
{% endif %}
</main>
<script>
const buttons = document.querySelectorAll(".reader-button");
const grid = document.getElementById("text-grid");
const translated = document.getElementById("translated-panel");
const original = document.getElementById("original-panel");
buttons.forEach((button) => {
  button.addEventListener("click", () => {
    buttons.forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    const view = button.dataset.view;
    if (view === "translated") {
      grid.classList.remove("side-by-side");
      translated.dataset.hidden = "false";
      original.dataset.hidden = "true";
    } else if (view === "original") {
      grid.classList.remove("side-by-side");
      translated.dataset.hidden = "true";
      original.dataset.hidden = "false";
    } else {
      grid.classList.add("side-by-side");
      translated.dataset.hidden = "false";
      original.dataset.hidden = "false";
    }
  });
});
</script>
"""


if __name__ == "__main__":
    app.run(host=os.environ.get("CONTENT_ANNOTATION_HOST", "127.0.0.1"), port=int(os.environ.get("CONTENT_ANNOTATION_PORT", "8502")))
