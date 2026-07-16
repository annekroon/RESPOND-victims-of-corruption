"""Classify uncertain source rows using live-web evidence plus an LLM.

This is a review aid, not an automatic source-workbook updater. It reads the
LLM source-assessment workbook, selects knowledge-gap sources by default,
collects web evidence with DuckDuckGo HTML search and optional homepage fetches,
then asks the configured LLM to make a provisional C1/C2/C3 decision based only
on the collected evidence.

Outputs are checkpointed, resumable, and stored separately from the authoritative
manual workbook.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except Exception:  # pragma: no cover
    config = None


DEFAULT_PIPELINE_DIR = Path(
    "/home/akroon/data/1t_storage/RESPOND-victims-of-corruption/"
    "political_corruption_pipeline"
)
DEFAULT_INPUT = (
    DEFAULT_PIPELINE_DIR
    / "source_inclusion"
    / "political_corruption_all_sources_llm_assessed.xlsx"
)
DEFAULT_OUTPUT = (
    DEFAULT_PIPELINE_DIR
    / "source_inclusion"
    / "source_unknown_web_llm_decisions.xlsx"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_PIPELINE_DIR
    / "source_inclusion"
    / "source_unknown_web_llm_decisions_checkpoint.csv"
)

VALID_CRITERIA = {"yes", "no", "unclear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use web evidence plus an LLM to provisionally decide uncertain source rows."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sheet", default="llm_assessed_sources")
    parser.add_argument("--model", default=os.environ.get("UVA_LLM_MODEL") or getattr(config, "LLMPROXY_MODEL", "gpt-5.1"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-articles", type=int, default=0)
    parser.add_argument("--countries", nargs="+", default=None)
    parser.add_argument("--source-domains", nargs="+", default=None)
    parser.add_argument("--top-results", type=int, default=5)
    parser.add_argument("--max-evidence-chars", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--no-homepage", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    return parser.parse_args()


def get_llm_credentials() -> tuple[str, str]:
    api_key = os.environ.get("UVA_LLM_API_KEY") or os.environ.get("LLMPROXY_API_KEY")
    base_url = os.environ.get("UVA_LLM_BASE_URL") or os.environ.get("LLMPROXY_BASE_URL")
    if config is not None:
        api_key = api_key or getattr(config, "LLMPROXY_API_KEY", None)
        base_url = base_url or getattr(config, "LLMPROXY_BASE_URL", None)
    if not api_key:
        raise RuntimeError("Set UVA_LLM_API_KEY/LLMPROXY_API_KEY or config_local.py LLMPROXY_API_KEY.")
    if not base_url:
        raise RuntimeError("Set UVA_LLM_BASE_URL/LLMPROXY_BASE_URL or config_local.py LLMPROXY_BASE_URL.")
    return api_key, base_url


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def normalize_domain(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^https?://", "", text)
    return text.split("/", 1)[0]


def source_key(country: str, source: str) -> str:
    return f"{country.strip()}||{normalize_domain(source)}"


def select_sources(data, args: argparse.Namespace):
    import pandas as pd

    selected = data.copy()
    selected["source_clean"] = selected["source_clean"].map(normalize_domain)
    selected["political_corruption_articles"] = pd.to_numeric(
        selected.get("political_corruption_articles", 0),
        errors="coerce",
    ).fillna(0)

    if args.source_domains:
        domains = {normalize_domain(domain) for domain in args.source_domains}
        selected = selected[selected["source_clean"].isin(domains)].copy()
    else:
        llm_decision = selected.get("llm_decision", "").fillna("").astype(str).str.lower()
        llm_knowledge = selected.get("llm_knowledge_status", "").fillna("").astype(str).str.lower()
        selected = selected[
            llm_decision.eq("review") | llm_knowledge.isin(["unknown", "inferred"])
        ].copy()

    if args.countries:
        countries = {country.strip() for country in args.countries}
        selected = selected[selected["country"].astype(str).str.strip().isin(countries)].copy()
    if args.min_articles:
        selected = selected[selected["political_corruption_articles"].ge(args.min_articles)].copy()

    selected = selected.sort_values(
        ["political_corruption_articles", "country", "source_clean"],
        ascending=[False, True, True],
    )
    if args.limit is not None:
        selected = selected.head(args.limit).copy()
    return selected


def html_to_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    except Exception:
        text = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
        text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()


def parse_duckduckgo_results(html: str, limit: int) -> list[dict[str, str]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for result in soup.select(".result")[:limit]:
        link = result.select_one(".result__a")
        snippet = result.select_one(".result__snippet")
        if not link:
            continue
        rows.append(
            {
                "title": link.get_text(" ", strip=True),
                "url": link.get("href", ""),
                "snippet": snippet.get_text(" ", strip=True) if snippet else "",
            }
        )
    return rows


def search_web(session, query: str, timeout: float, limit: int) -> list[dict[str, str]]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_duckduckgo_results(response.text, limit)


def fetch_homepage(session, source: str, timeout: float) -> dict[str, str]:
    for url in [f"https://{source}", f"http://{source}"]:
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            return {
                "homepage_url": url,
                "homepage_final_url": str(response.url),
                "homepage_text": html_to_text(response.text)[:2500],
                "homepage_error": "",
            }
        except Exception as exc:
            last_error = repr(exc)
    return {"homepage_url": f"https://{source}", "homepage_final_url": "", "homepage_text": "", "homepage_error": last_error}


def collect_evidence(session, source: str, country: str, args: argparse.Namespace) -> tuple[str, list[str], str]:
    queries = [
        f'"{source}" "{country}" news outlet',
        f'"{source}" journalism newsroom {country}',
        f'"{source}" about editorial {country}',
    ]
    evidence_parts = []
    urls = []
    errors = []
    for query in queries:
        try:
            results = search_web(session, query, args.timeout, args.top_results)
            evidence_parts.append(f"Search query: {query}")
            for index, result in enumerate(results, start=1):
                url = result.get("url", "")
                if url:
                    urls.append(url)
                evidence_parts.append(
                    f"- Result {index}: {result.get('title','')} | {url} | {result.get('snippet','')}"
                )
        except Exception as exc:
            errors.append(f"{query}: {exc!r}")
    if not args.no_homepage:
        homepage = fetch_homepage(session, source, args.timeout)
        if homepage.get("homepage_final_url"):
            urls.append(homepage["homepage_final_url"])
        evidence_parts.append(
            "Homepage fetch: "
            f"{homepage.get('homepage_url','')} -> {homepage.get('homepage_final_url','')}\n"
            f"{homepage.get('homepage_text','')}"
        )
        if homepage.get("homepage_error"):
            errors.append(f"homepage: {homepage['homepage_error']}")
    return "\n\n".join(evidence_parts)[: args.max_evidence_chars], sorted(set(urls)), "; ".join(errors)


SYSTEM_PROMPT = """You classify source-domain/country pairs for a comparative news corpus using only the provided web evidence.

Criteria:
C1 domestic base: the outlet or edition is based in the assigned country or operates as a distinct domestic newsroom there.
C2 editorial activity: the outlet publishes original reporting or exercises identifiable editorial curation.
C3 journalistic primary function: the source is primarily journalistic, including newspapers, broadcasters, digital newsrooms, professional news agencies/wire services, and editorial specialist publications.

Do not exclude outlets merely because they are partisan, tabloid, digital-native, regional/local, publicly funded, specialist, or a news agency. Exclude automated aggregators, press-release hosts, official repositories, document archives, data platforms with incidental news, or foreign/international outlets without a domestic edition for the assigned country.

If the evidence is insufficient, use unclear/review. Return valid JSON only."""


def build_prompt(source: str, country: str, evidence: str) -> str:
    return f"""Input:
{{
  "source_domain": "{source}",
  "assigned_country": "{country}"
}}

Web evidence:
{evidence}

Return this JSON:
{{
  "c1": "yes | no | unclear",
  "c2": "yes | no | unclear",
  "c3": "yes | no | unclear",
  "decision": "include | exclude | review",
  "reason": "one concise sentence",
  "confidence": 0.0
}}"""


def extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?", "", text.strip()).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            return json.loads(match.group(0))
        raise


def derive_decision(c1: str, c2: str, c3: str) -> str:
    values = {c1, c2, c3}
    if "no" in values:
        return "exclude"
    if values == {"yes"}:
        return "include"
    return "review"


def normalize_result(parsed: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ["c1", "c2", "c3"]:
        value = str(parsed.get(key, "unclear")).strip().lower()
        out[f"web_llm_{key}"] = value if value in VALID_CRITERIA else "unclear"
    out["web_llm_decision"] = derive_decision(out["web_llm_c1"], out["web_llm_c2"], out["web_llm_c3"])
    out["web_llm_include"] = {"include": "Yes", "exclude": "No", "review": "Review"}[out["web_llm_decision"]]
    try:
        confidence = float(parsed.get("confidence", 0.0))
        if 1.0 < confidence <= 100.0:
            confidence /= 100.0
    except (TypeError, ValueError):
        confidence = 0.0
    out["web_llm_confidence"] = min(1.0, max(0.0, confidence))
    out["web_llm_reason"] = normalize_text(parsed.get("reason", ""))
    return out


def classify_with_llm(client, model: str, source: str, country: str, evidence: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(source, country, evidence)},
        ],
        temperature=0,
    )
    return normalize_result(extract_json(response.choices[0].message.content))


def load_checkpoint(path: Path, overwrite: bool, retry_errors: bool) -> tuple[list[dict[str, Any]], set[str]]:
    import pandas as pd

    if overwrite or not path.exists():
        return [], set()
    data = pd.read_csv(path)
    if retry_errors:
        data = data[data["web_llm_error"].fillna("").astype(str).str.strip().eq("")]
    rows = data.to_dict("records")
    return rows, set(data["source_key"].astype(str))


def write_outputs(rows: list[dict[str, Any]], checkpoint: Path, output: Path) -> None:
    import pandas as pd

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    tmp_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    data.to_csv(tmp_checkpoint, index=False)
    tmp_checkpoint.replace(checkpoint)

    tmp_output = output.with_name(output.stem + ".tmp" + output.suffix)
    with pd.ExcelWriter(tmp_output, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="web_llm_decisions", index=False)
        if not data.empty:
            summary = (
                data.groupby(["country", "web_llm_decision", "web_llm_include"], dropna=False)
                .size()
                .reset_index(name="sources")
            )
            summary.to_excel(writer, sheet_name="summary", index=False)
    tmp_output.replace(output)
    print(f"Saved checkpoint: {checkpoint} ({len(rows):,} rows)", flush=True)
    print(f"Saved workbook:   {output}", flush=True)


def main() -> None:
    args = parse_args()

    import pandas as pd
    import requests
    from openai import OpenAI
    from tqdm.auto import tqdm

    api_key, base_url = get_llm_credentials()
    source_data = pd.read_excel(args.input, sheet_name=args.sheet)
    selected = select_sources(source_data, args)
    rows, done = load_checkpoint(args.checkpoint, args.overwrite, args.retry_errors)

    selected["source_key"] = selected.apply(
        lambda row: source_key(normalize_text(row.get("country", "")), normalize_domain(row.get("source_clean", ""))),
        axis=1,
    )
    selected = selected[~selected["source_key"].isin(done)].copy()

    print(f"Input:      {args.input}", flush=True)
    print(f"Output:     {args.output}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Model:      {args.model}", flush=True)
    print(f"Rows to process now: {len(selected):,}", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 RESPOND source review"})
    client = OpenAI(api_key=api_key, base_url=base_url)

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Web + LLM source decisions"):
        country = normalize_text(row.get("country", ""))
        source = normalize_domain(row.get("source_clean", ""))
        key = source_key(country, source)
        out = {
            "source_key": key,
            "country": country,
            "source_clean": source,
            "political_corruption_articles": row.get("political_corruption_articles", ""),
            "manual_decision": row.get("main_sample_decision", ""),
            "conventional_journalism": row.get("conventional_journalism", ""),
            "model_memory_decision": row.get("llm_decision", ""),
            "model_memory_reason": row.get("llm_reason", ""),
            "web_llm_model": args.model,
            "web_llm_checked_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            evidence, urls, search_errors = collect_evidence(session, source, country, args)
            out["web_evidence_urls"] = "\n".join(urls[:20])
            out["web_evidence_excerpt"] = evidence
            out["web_search_error"] = search_errors
            out.update(classify_with_llm(client, args.model, source, country, evidence))
            out["web_llm_error"] = ""
        except Exception as exc:
            out.update(
                {
                    "web_llm_c1": "unclear",
                    "web_llm_c2": "unclear",
                    "web_llm_c3": "unclear",
                    "web_llm_decision": "review",
                    "web_llm_include": "Review",
                    "web_llm_confidence": 0.0,
                    "web_llm_reason": "Web evidence collection or LLM call failed.",
                    "web_evidence_urls": out.get("web_evidence_urls", ""),
                    "web_evidence_excerpt": out.get("web_evidence_excerpt", ""),
                    "web_search_error": out.get("web_search_error", ""),
                    "web_llm_error": repr(exc),
                }
            )
        rows.append(out)
        if len(rows) % args.save_every == 0:
            write_outputs(rows, args.checkpoint, args.output)
        time.sleep(args.sleep)

    write_outputs(rows, args.checkpoint, args.output)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
