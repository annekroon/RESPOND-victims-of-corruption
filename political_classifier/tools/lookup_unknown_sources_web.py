"""Collect live-web evidence for source rows the LLM could not assess.

This is a manual-review aid. It does not change source-inclusion decisions and
does not classify outlets by itself. It reads the LLM source-assessment workbook,
selects rows where the model had insufficient knowledge, searches the web for
each source/country pair, optionally fetches the source homepage, and writes the
evidence to an Excel workbook.

The output is evidence, not ground truth. Search results can vary over time, so
the script stores query strings, result URLs, snippets, timestamps, and fetch
errors for reproducibility.

Example:
    python3 political_classifier/tools/lookup_unknown_sources_web.py \
      --limit 100 \
      --min-articles 50
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse


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
    / "source_unknown_web_lookup_evidence.xlsx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Look up sources with LLM knowledge gaps and save web evidence for "
            "manual source review."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sheet",
        default="llm_assessed_sources",
        help="Input workbook sheet containing LLM assessment columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of source rows to look up.",
    )
    parser.add_argument(
        "--min-articles",
        type=int,
        default=0,
        help="Only look up sources with at least this many political-corruption articles.",
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=None,
        help="Optional country subset.",
    )
    parser.add_argument(
        "--source-domains",
        nargs="+",
        default=None,
        help="Optional explicit source_clean domains to look up.",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=5,
        help="Maximum search results to keep per source.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Delay between source lookups.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--no-homepage",
        action="store_true",
        help="Skip direct homepage metadata fetch.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def normalize_domain(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^https?://", "", text)
    text = text.split("/", 1)[0]
    return text


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


def make_queries(source: str, country: str) -> list[str]:
    return [
        f'"{source}" "{country}" news outlet',
        f'"{source}" journalism newsroom {country}',
        f'"{source}" about editorial {country}',
    ]


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
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        results = []
        for result in soup.select(".result")[:limit]:
            link = result.select_one(".result__a")
            snippet = result.select_one(".result__snippet")
            if not link:
                continue
            results.append(
                {
                    "title": link.get_text(" ", strip=True),
                    "url": link.get("href", ""),
                    "snippet": snippet.get_text(" ", strip=True) if snippet else "",
                }
            )
        return results
    except Exception:
        results = []
        for match in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.I | re.S,
        ):
            title = re.sub(r"<[^>]+>", " ", match.group(2))
            results.append(
                {
                    "title": re.sub(r"\s+", " ", title).strip(),
                    "url": match.group(1),
                    "snippet": "",
                }
            )
            if len(results) >= limit:
                break
        return results


def search_duckduckgo(session, query: str, timeout: float, limit: int) -> tuple[list[dict[str, str]], str]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_duckduckgo_results(response.text, limit), url


def fetch_homepage(session, source: str, timeout: float) -> dict[str, Any]:
    candidates = [f"https://{source}", f"http://{source}"]
    last_error = ""
    for url in candidates:
        try:
            response = session.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            text = html_to_text(response.text)
            parsed = urlparse(str(response.url))
            return {
                "homepage_url": url,
                "homepage_final_url": str(response.url),
                "homepage_domain": parsed.netloc,
                "homepage_status": response.status_code,
                "homepage_title_or_text": text[:1200],
                "homepage_error": "",
            }
        except Exception as exc:
            last_error = repr(exc)
    return {
        "homepage_url": candidates[0],
        "homepage_final_url": "",
        "homepage_domain": "",
        "homepage_status": "",
        "homepage_title_or_text": "",
        "homepage_error": last_error,
    }


def main() -> None:
    args = parse_args()

    import pandas as pd
    import requests
    from tqdm.auto import tqdm

    data = pd.read_excel(args.input, sheet_name=args.sheet)
    if "country" not in data.columns or "source_clean" not in data.columns:
        raise ValueError("Input must contain country and source_clean columns.")

    selected = select_sources(data, args)
    print(f"Sources selected for web lookup: {len(selected):,}", flush=True)
    print(f"Input:  {args.input}", flush=True)
    print(f"Output: {args.output}", flush=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 source-review-research-script "
                "(RESPOND victims-of-corruption project)"
            )
        }
    )

    lookup_rows = []
    search_rows = []
    homepage_rows = []
    checked_at = datetime.now(timezone.utc).isoformat()

    for _, row in tqdm(selected.iterrows(), total=len(selected), desc="Looking up sources"):
        country = normalize_text(row.get("country", ""))
        source = normalize_domain(row.get("source_clean", ""))
        queries = make_queries(source, country)

        source_search_count = 0
        source_search_error = ""
        for query in queries:
            try:
                results, search_url = search_duckduckgo(session, query, args.timeout, args.top_results)
                for rank, result in enumerate(results, start=1):
                    source_search_count += 1
                    search_rows.append(
                        {
                            "country": country,
                            "source_clean": source,
                            "query": query,
                            "search_url": search_url,
                            "rank": rank,
                            "result_title": result.get("title", ""),
                            "result_url": result.get("url", ""),
                            "result_snippet": result.get("snippet", ""),
                            "checked_at": checked_at,
                        }
                    )
            except Exception as exc:
                source_search_error = repr(exc)

        homepage = {}
        if not args.no_homepage:
            homepage = fetch_homepage(session, source, args.timeout)
            homepage.update({"country": country, "source_clean": source, "checked_at": checked_at})
            homepage_rows.append(homepage)

        lookup_rows.append(
            {
                "country": country,
                "source_clean": source,
                "political_corruption_articles": row.get("political_corruption_articles", ""),
                "manual_decision": row.get("main_sample_decision", ""),
                "conventional_journalism": row.get("conventional_journalism", ""),
                "llm_decision": row.get("llm_decision", ""),
                "llm_knowledge_status": row.get("llm_knowledge_status", ""),
                "llm_comparison": row.get("llm_comparison", ""),
                "llm_reason": row.get("llm_reason", ""),
                "search_results_found": source_search_count,
                "search_error": source_search_error,
                "homepage_error": homepage.get("homepage_error", ""),
                "checked_at": checked_at,
            }
        )
        time.sleep(args.sleep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        pd.DataFrame(lookup_rows).to_excel(writer, sheet_name="web_lookup_sources", index=False)
        pd.DataFrame(search_rows).to_excel(writer, sheet_name="search_results", index=False)
        pd.DataFrame(homepage_rows).to_excel(writer, sheet_name="homepage_snapshots", index=False)
        summary = (
            pd.DataFrame(lookup_rows)
            .groupby(["country", "llm_decision", "llm_knowledge_status"], dropna=False)
            .size()
            .reset_index(name="sources")
        )
        summary.to_excel(writer, sheet_name="summary", index=False)

    print(f"Saved web lookup evidence: {args.output}", flush=True)


if __name__ == "__main__":
    main()
