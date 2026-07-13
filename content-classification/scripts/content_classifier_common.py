"""Shared runner for article-level GPT content classifiers."""

from __future__ import annotations

import argparse
import io
import json
import posixpath
import re
import sys
import time
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import ALL_COUNTRIES, LLMPROXY_API_KEY, LLMPROXY_BASE_URL, LLMPROXY_MODEL, RD_BASE_DIR
from content_prompts import ClassifierSpec


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_CLASSIFIED_DIR = DEFAULT_PIPELINE_DIR / "silver_classifier" / "classified_country_files"
DEFAULT_OUTPUT_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "content_classification"
)
DEFAULT_RD_CLASSIFIED_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "derived_data",
    "political_classifier",
    "classifier_outputs",
    "classified_country_files",
)

KEEP_COLUMNS = {
    "uri",
    "country",
    "date_parsed",
    "dateTime",
    "date",
    "published_at",
    "year",
    "month",
    "week",
    "source_uri",
    "source.uri",
    "word_count",
    "prob_political_corruption",
    "pred_political_corruption",
    "article_text",
    "combined_text",
    "title",
    "body",
}

METADATA_COLUMNS = [
    "article_id",
    "uri",
    "country",
    "year",
    "month",
    "week",
    "source_uri",
    "source.uri",
    "word_count",
    "prob_political_corruption",
    "pred_political_corruption",
]


def parse_common_args(description: str, default_output_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source",
        choices=["classified", "classified-webdav", "csv"],
        default="classified",
        help="Read local classified country files, archived WebDAV files, or one CSV file.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Input CSV/CSV.GZ when --source csv.")
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument("--classified-rd-dir", default=DEFAULT_RD_CLASSIFIED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES)
    parser.add_argument("--model", default=LLMPROXY_MODEL)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N unfinished rows.")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Reprocess rows with a non-empty llm_error in the existing output.",
    )
    parser.add_argument(
        "--keep-non-political",
        action="store_true",
        help="Do not filter to pred_political_corruption == 1.",
    )
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        help="Optional random sample of unfinished rows, useful for pilot checks.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.set_defaults(default_output_name=default_output_name)
    return parser.parse_args()


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def choose_text(data):
    if "article_text" in data.columns:
        return data["article_text"]
    if "combined_text" in data.columns:
        return data["combined_text"]
    if {"title", "body"}.issubset(data.columns):
        return data["title"].fillna("").astype(str) + "\n" + data["body"].fillna("").astype(str)
    raise ValueError("Input data must contain article_text, combined_text, or title/body columns.")


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


def compression_for_path(path: Path):
    return "gzip" if path.name.endswith(".gz") else None


def read_csv(path: Path, **kwargs):
    import pandas as pd

    return pd.read_csv(path, compression="gzip" if path.name.endswith(".gz") else "infer", **kwargs)


def write_csv_atomic(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    data.to_csv(tmp_path, index=False, compression=compression_for_path(output_path))
    tmp_path.replace(output_path)


def append_audit_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def country_path(args: argparse.Namespace, country: str) -> Path:
    return args.classified_dir / f"{country}_classified.csv.gz"


def country_rd_path(args: argparse.Namespace, country: str) -> str:
    return posixpath.join(args.classified_rd_dir, f"{country}_classified.csv.gz")


def read_country_file(args: argparse.Namespace, country: str):
    import pandas as pd

    if args.source == "classified-webdav":
        from rd_utils import webdav_download_bytes

        rd_path = country_rd_path(args, country)
        data = webdav_download_bytes(rd_path)
        return pd.read_csv(
            io.BytesIO(data),
            compression="gzip" if rd_path.endswith(".gz") else "infer",
            usecols=lambda column: column in KEEP_COLUMNS,
        )

    path = country_path(args, country)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, usecols=lambda column: column in KEEP_COLUMNS)


def prepare_frame(data, country: str | None, args: argparse.Namespace):
    import pandas as pd

    data = data.copy()
    if country is not None:
        data["country"] = country
    elif "country" not in data.columns:
        data["country"] = ""

    data["article_text"] = choose_text(data).map(normalize_text)
    data = data[data["article_text"].str.split().str.len().fillna(0).ge(args.min_words)].copy()

    if not args.keep_non_political and "pred_political_corruption" in data.columns:
        data = data[pd.to_numeric(data["pred_political_corruption"], errors="coerce").eq(1)].copy()

    if "year" not in data.columns:
        date_column = next(
            (column for column in ["date_parsed", "dateTime", "date", "published_at"] if column in data.columns),
            None,
        )
        if date_column:
            data["date_parsed"] = pd.to_datetime(data[date_column], errors="coerce", utc=True)
            data["year"] = data["date_parsed"].dt.year
    data["year"] = pd.to_numeric(data.get("year", ""), errors="coerce")

    if "month" not in data.columns:
        date_column = next(
            (column for column in ["date_parsed", "dateTime", "date", "published_at"] if column in data.columns),
            None,
        )
        if date_column:
            date_values = pd.to_datetime(data[date_column], errors="coerce", utc=True)
            data["month"] = date_values.dt.month

    if "source_uri" not in data.columns and "source.uri" in data.columns:
        data["source_uri"] = data["source.uri"]

    if "uri" not in data.columns:
        data["uri"] = ""
    data["article_id"] = data["uri"].fillna("").astype(str)
    missing_id = data["article_id"].str.strip().eq("")
    if missing_id.any():
        fallback = (
            data["country"].astype(str)
            + "::"
            + data["year"].fillna("").astype(str)
            + "::"
            + data.index.astype(str)
        )
        data.loc[missing_id, "article_id"] = fallback[missing_id]

    return data


def load_input(args: argparse.Namespace):
    import pandas as pd

    if args.source == "csv":
        if args.input is None:
            raise ValueError("--input is required when --source csv.")
        data = read_csv(args.input, usecols=lambda column: column in KEEP_COLUMNS)
        return prepare_frame(data, None, args)

    frames = []
    for country in args.countries:
        print(f"Loading {country}", flush=True)
        frame = prepare_frame(read_country_file(args, country), country, args)
        frames.append(frame)
        print(f"{country}: {len(frame):,} political-corruption rows after filters", flush=True)

    if not frames:
        raise FileNotFoundError("No classified country files loaded.")
    return pd.concat(frames, ignore_index=True)


def existing_done_ids(existing, retry_errors: bool) -> set[str]:
    if existing.empty or "article_id" not in existing.columns:
        return set()
    if retry_errors and "llm_error" in existing.columns:
        ok = existing[existing["llm_error"].fillna("").astype(str).str.strip().eq("")]
        return set(ok["article_id"].dropna().astype(str))
    return set(existing["article_id"].dropna().astype(str))


def classify_article(client, spec: ClassifierSpec, row: dict, model: str, max_chars: int) -> tuple[dict, str, str]:
    article_text = normalize_text(row.get("article_text", ""))[:max_chars]
    prompt = spec.build_prompt(article_text, row)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = response.choices[0].message.content
    return spec.normalize_result(extract_json(raw)), prompt, raw


def base_output_row(row: dict, spec: ClassifierSpec) -> dict:
    out = {column: row.get(column, "") for column in METADATA_COLUMNS if column in row}
    out["classifier_name"] = spec.name
    out["prompt_version"] = spec.prompt_version
    return out


def write_checkpoint(existing, new_rows: list[dict], output_path: Path) -> None:
    import pandas as pd

    checkpoint = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    write_csv_atomic(checkpoint, output_path)
    print(f"Saved checkpoint: {output_path} ({len(checkpoint):,} rows)", flush=True)


def select_unfinished(data, done_ids: set[str], args: argparse.Namespace):
    unfinished = data[~data["article_id"].astype(str).isin(done_ids)].copy()
    if args.random_sample is not None:
        unfinished = unfinished.sample(n=min(args.random_sample, len(unfinished)), random_state=args.random_state)
    if args.limit is not None:
        unfinished = unfinished.head(args.limit).copy()
    return unfinished


def run_classifier(spec: ClassifierSpec) -> None:
    args = parse_common_args(
        description=f"Classify political-corruption articles for {spec.name}.",
        default_output_name=spec.default_output_name,
    )

    import pandas as pd
    from openai import OpenAI
    from tqdm.auto import tqdm

    if not LLMPROXY_API_KEY:
        raise RuntimeError("Set LLMPROXY_API_KEY in your environment or config_local.py.")

    output_path = args.output or (args.output_dir / spec.default_output_name)
    audit_path = output_path.with_name(output_path.stem + "_audit.jsonl")

    client = OpenAI(api_key=LLMPROXY_API_KEY, base_url=LLMPROXY_BASE_URL)
    data = load_input(args)
    print(f"Loaded {len(data):,} eligible rows.", flush=True)

    if output_path.exists():
        existing = read_csv(output_path)
        done_ids = existing_done_ids(existing, args.retry_errors)
        print(f"Resuming from {output_path}; already done: {len(done_ids):,}", flush=True)
    else:
        existing = pd.DataFrame()
        done_ids = set()

    unfinished = select_unfinished(data, done_ids, args)
    print(f"Rows to process now: {len(unfinished):,}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Model: {args.model}", flush=True)

    new_rows: list[dict] = []
    for _, row in tqdm(unfinished.iterrows(), total=len(unfinished), desc=f"LLM {spec.name}"):
        row_dict = row.to_dict()
        out = base_output_row(row_dict, spec)

        for attempt in range(1, args.retries + 1):
            try:
                parsed, prompt, raw_response = classify_article(
                    client=client,
                    spec=spec,
                    row=row_dict,
                    model=args.model,
                    max_chars=args.max_chars,
                )
                out.update(parsed)
                out["llm_error"] = ""
                append_audit_record(
                    audit_path,
                    {
                        "article_id": out.get("article_id", ""),
                        "classifier_name": spec.name,
                        "prompt_version": spec.prompt_version,
                        "prompt": prompt,
                        "raw_response": raw_response,
                    },
                )
                break
            except Exception as exc:
                out["llm_error"] = repr(exc)
                print(
                    f"Error on article_id={out.get('article_id', '')} "
                    f"attempt {attempt}/{args.retries}: {exc!r}",
                    flush=True,
                )
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)

        new_rows.append(out)

        if len(new_rows) % args.save_every == 0:
            write_checkpoint(existing, new_rows, output_path)
            time.sleep(args.sleep)

    write_checkpoint(existing, new_rows, output_path)
    print("Done.", flush=True)
