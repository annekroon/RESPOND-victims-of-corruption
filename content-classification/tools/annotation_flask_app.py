"""Flask app for manually annotating content-classification validation samples.

Run on a remote server with:

    CONTENT_ANNOTATION_INPUT=/path/to/content_validation_sample_100_per_country_english.csv.gz \
    CONTENT_ANNOTATION_OUTPUT=/path/to/content_validation_sample_100_per_country_human_coded.csv.gz \
    CONTENT_ANNOTATION_PASSWORD='choose-a-password' \
    flask --app content-classification/tools/annotation_flask_app.py run --host 0.0.0.0 --port 8502

For multiple external coders, give each coder a distinct output file to avoid
simultaneous writes to the same CSV.
"""

from __future__ import annotations

import os
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


def load_data() -> pd.DataFrame:
    path = OUTPUT_PATH if OUTPUT_PATH.exists() else INPUT_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    return ensure_columns(read_csv(path))


def save_data(data: pd.DataFrame) -> None:
    write_csv_atomic(data, OUTPUT_PATH)


DATA = load_data()


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
            return redirect(request.args.get("next") or url_for("index"))
        error = "Incorrect password."
    return render_template_string(LOGIN_TEMPLATE, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@require_login
def index():
    indices = filtered_indices(DATA)
    pos = max(0, min(int(request.args.get("pos", 0)), max(len(indices) - 1, 0)))
    reviewed = int(reviewed_mask(DATA).sum())
    countries = sorted(DATA["country"].dropna().astype(str).unique()) if "country" in DATA.columns else []
    if not indices:
        return render_template_string(
            APP_TEMPLATE,
            no_rows=True,
            total=len(DATA),
            reviewed=reviewed,
            countries=countries,
            filters=current_filters(),
        )
    row_index = indices[pos]
    row = DATA.loc[row_index]
    return render_template_string(
        APP_TEMPLATE,
        no_rows=False,
        row=row,
        row_index=row_index,
        pos=pos,
        n_filtered=len(indices),
        total=len(DATA),
        reviewed=reviewed,
        countries=countries,
        filters=current_filters(),
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
    if row_index not in DATA.index:
        return "Unknown row", 404

    import datetime as dt

    for column in HUMAN_COLUMNS:
        if column in {"human_coder_id", "human_coded_at"}:
            continue
        DATA.loc[row_index, column] = request.form.get(column, "")
    DATA.loc[row_index, "human_coder_id"] = request.form.get("human_coder_id", CODER_ID)
    DATA.loc[row_index, "human_coded_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    save_data(DATA)

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
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 3rem auto; max-width: 420px; }
input, button { width: 100%; padding: .75rem; margin-top: .5rem; font-size: 1rem; }
.error { color: #a40000; }
</style>
<h1>RESPOND Annotation</h1>
<form method="post">
  <label>Password</label>
  <input type="password" name="password" autofocus>
  <button type="submit">Log in</button>
</form>
{% if error %}<p class="error">{{ error }}</p>{% endif %}
"""


APP_TEMPLATE = """
<!doctype html>
<title>RESPOND Content Annotation</title>
<style>
:root { --border: #d7d7d7; --muted: #666; --bg: #f7f7f7; --accent: #1f5f8b; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1b1b1b; }
header { position: sticky; top: 0; background: white; border-bottom: 1px solid var(--border); padding: .75rem 1rem; z-index: 2; }
main { padding: 1rem; }
.top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
.filters { display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .75rem; }
select, input, textarea, button { font: inherit; padding: .45rem; border: 1px solid var(--border); border-radius: 6px; background: white; }
button { cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); }
.ghost { color: var(--accent); background: white; }
.meta { color: var(--muted); font-size: .9rem; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; align-items: start; }
.panel { border: 1px solid var(--border); border-radius: 8px; padding: 1rem; background: white; }
.text { white-space: pre-wrap; line-height: 1.45; max-height: 58vh; overflow: auto; }
.codebook { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; margin-bottom: 1rem; }
.codebook .panel { background: var(--bg); font-size: .92rem; }
.codebook h3 { margin-top: 0; font-size: 1rem; }
.codebook ul { padding-left: 1.1rem; }
.form-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .75rem; }
.form-grid label { display: flex; flex-direction: column; gap: .3rem; font-weight: 600; }
textarea { width: 100%; min-height: 90px; box-sizing: border-box; }
.actions { display: flex; gap: .5rem; margin-top: .75rem; }
.progress { font-weight: 700; }
@media (max-width: 1100px) { .grid, .codebook, .form-grid { grid-template-columns: 1fr; } }
</style>
<header>
  <div class="top">
    <div>
      <div class="progress">RESPOND content annotation</div>
      <div class="meta">Reviewed {{ reviewed }} / {{ total }}{% if not no_rows %} · Filtered row {{ pos + 1 }} / {{ n_filtered }}{% endif %}</div>
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
  <section class="codebook">
    <div class="panel">
      <h3>Victim Visibility</h3>
      <ul>
        <li><b>no_victim:</b> corruption/scandal only; no harmed party.</li>
        <li><b>concrete_victim:</b> harmed people, groups, firms, voters, taxpayers, residents, patients, students, workers, communities.</li>
        <li><b>institutional_societal_victim:</b> harm to democracy, rule of law, public trust, state, institutions, society, development, EU accession, economy.</li>
        <li><b>unclear:</b> too incomplete or ambiguous.</li>
      </ul>
    </div>
    <div class="panel">
      <h3>Corruption Frame</h3>
      <ul>
        <li><b>individualized:</b> named/identifiable actors, allegations, trials, scandals, resignations.</li>
        <li><b>systemic:</b> institutions, state capture, rule of law, clientelism, democratic backsliding, recurring abuse.</li>
        <li><b>other_or_mixed:</b> both central, procedural, technical, local/sectoral, election-finance, unclear fit.</li>
      </ul>
    </div>
    <div class="panel">
      <h3>Domestic / Abroad</h3>
      <ul>
        <li><b>domestic:</b> case mainly concerns publication country.</li>
        <li><b>abroad:</b> case mainly concerns another country, foreign actors, offshore schemes, sanctions, cross-border probes centered elsewhere.</li>
        <li>EU funds in domestic misuse still count as domestic.</li>
      </ul>
    </div>
    <div class="panel">
      <h3>Accused Actor</h3>
      <ul>
        <li>Code visible if a person, organization, company, party, institution, officeholder, or group is accused/investigated/charged/linked.</li>
        <li>Conviction is not required.</li>
        <li><b>no_accused_actor:</b> corruption discussed generally but no actor identified.</li>
      </ul>
    </div>
  </section>

  <div class="meta">
    <b>{{ value(row, "country") }}</b> · {{ value(row, "year") }} · URI: {{ value(row, "uri") }} · Source: {{ value(row, "source_uri") }}
  </div>
  <div class="actions">
    <a href="{{ prev_url }}"><button class="ghost">Previous</button></a>
    <a href="{{ next_url }}"><button class="ghost">Next</button></a>
  </div>

  <section class="grid" style="margin-top: 1rem;">
    <div class="panel">
      <h2>English Translation</h2>
      <div class="text">{{ value(row, "translated_text_en", value(row, "translated_text", "")) }}</div>
      <p class="meta">{{ value(row, "translation_notes") }}</p>
    </div>
    <div class="panel">
      <h2>Original Article</h2>
      <div class="text">{{ value(row, "article_text") }}</div>
    </div>
  </section>

  <form method="post" action="{{ save_url }}" style="margin-top: 1rem;">
    <input type="hidden" name="pos" value="{{ pos }}">
    <section class="panel">
      <h2>Human Codes</h2>
      <div class="form-grid">
        <label>Victim visibility
          <select name="human_victim_visibility">
            {% for option in victim_options %}
            <option value="{{ option }}" {% if value(row, "human_victim_visibility") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Corruption frame
          <select name="human_corruption_frame">
            {% for option in frame_options %}
            <option value="{{ option }}" {% if value(row, "human_corruption_frame") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Case location
          <select name="human_case_location">
            {% for option in case_location_options %}
            <option value="{{ option }}" {% if value(row, "human_case_location") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Abroad case
          <select name="human_abroad_case">
            {% for option in yes_no_unclear_options %}
            <option value="{{ option }}" {% if value(row, "human_abroad_case") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Accused actor visibility
          <select name="human_accused_actor_visibility">
            {% for option in accused_options %}
            <option value="{{ option }}" {% if value(row, "human_accused_actor_visibility") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
            {% endfor %}
          </select>
        </label>
        <label>Accused actor visible
          <select name="human_accused_actor_visible">
            {% for option in yes_no_unclear_options %}
            <option value="{{ option }}" {% if value(row, "human_accused_actor_visible") == option %}selected{% endif %}>{{ option or "choose..." }}</option>
            {% endfor %}
          </select>
        </label>
      </div>
      <label style="display:block; margin-top:.75rem; font-weight:600;">Notes
        <textarea name="human_notes">{{ value(row, "human_notes") }}</textarea>
      </label>
      <label style="display:block; margin-top:.75rem; font-weight:600;">Coder ID
        <input name="human_coder_id" value="{{ value(row, 'human_coder_id', 'coder') }}">
      </label>
      <div class="actions">
        <button type="submit" name="action" value="save">Save</button>
        <button type="submit" name="action" value="save_next">Save + Next</button>
      </div>
    </section>
  </form>
{% endif %}
</main>
"""


if __name__ == "__main__":
    app.run(host=os.environ.get("CONTENT_ANNOTATION_HOST", "127.0.0.1"), port=int(os.environ.get("CONTENT_ANNOTATION_PORT", "8502")))
