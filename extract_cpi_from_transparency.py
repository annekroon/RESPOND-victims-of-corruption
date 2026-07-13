"""Extract CPI country-year scores from Transparency International result files.

This is the preferred CPI extractor for the project. It visits the official
Transparency International CPI year pages, discovers the linked "Full Results"
spreadsheet/archive, reads the country rows, and writes the same tidy output
schema as `extract_cpi_from_pdfs.py`.

Example
-------
    python3 extract_cpi_from_transparency.py \
      --years 2018 2019 2020 2021 2022 2023 2024 2025 \
      --output output/cpi_country_year_scores.csv
"""

from __future__ import annotations

import argparse
import html
import io
import mimetypes
import posixpath
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALL_COUNTRIES


DEFAULT_YEARS = list(range(2018, 2026))
DEFAULT_OUTPUT = Path("output/cpi_country_year_scores.csv")
PROJECT_COUNTRIES = {country.replace("_", " ") for country in ALL_COUNTRIES}
PAGE_URL_TEMPLATE = "https://www.transparency.org/en/cpi/{year}/index"
KNOWN_RESULT_URLS = {
    2018: [
        "https://images.transparencycdn.org/images/CPI2018_Full-Results_1801.xlsx",
    ],
    2019: [
        "https://images.transparencycdn.org/images/CPI2019-1.xlsx",
    ],
    2020: [
        "https://images.transparencycdn.org/images/CPI_FULL_DATA_2021-01-27-162209.zip",
    ],
    2021: [
        "https://images.transparencycdn.org/images/CPI-2021-Full-Data-Set.zip",
    ],
    2022: [
        "https://images.transparencycdn.org/images/CPI2022_GlobalResultsTrends.xlsx",
    ],
    2023: [
        "https://images.transparencycdn.org/images/CPI2023_Global_Results__Trends.xlsx",
    ],
    2024: [
        "https://images.transparencycdn.org/images/CPI2024-Results-and-trends.xlsx",
    ],
    2025: [
        "https://files.transparencycdn.org/images/CPI2025_Results.xlsx",
    ],
}


@dataclass(frozen=True)
class DownloadedFile:
    year: int
    url: str
    name: str
    data: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract country-year CPI scores from official Transparency "
            "International CPI full-results files."
        )
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=DEFAULT_YEARS,
        help="CPI years to extract.",
    )
    parser.add_argument(
        "--country-scope",
        choices=["selected", "all"],
        default="selected",
        help="Keep only project countries by default. Use 'all' for every parsed row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV path for extracted country-year CPI scores.",
    )
    parser.add_argument(
        "--log-output",
        type=Path,
        default=None,
        help="Optional CSV path for per-year extraction diagnostics.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("output/cpi_transparency_cache"),
        help="Directory for downloaded Transparency International result files.",
    )
    parser.add_argument(
        "--allow-partial-years",
        action="store_true",
        help="Keep selected-country years even when not all project countries are found.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    text = html.unescape(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_country(value: object) -> str:
    text = clean_text(value)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s*\[[^\]]*\]\s*$", "", text)
    text = text.replace("*", "").replace("†", "").strip(" -–—:")
    text = re.sub(r"\s+", " ", text)
    return normalize_country_name(text)


def normalize_country_name(country: str) -> str:
    country = country.replace("_", " ").strip()
    replacements = {
        "UK": "United Kingdom",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "Czech Republic": "Czechia",
        "Korea, South": "South Korea",
        "Türkiye": "Turkey",
    }
    return replacements.get(country, country)


def keep_country(country: str, country_scope: str) -> bool:
    if not country or len(country) < 3:
        return False
    if re.search(r"[:@<>]|www\.|http|fax|phone", country, flags=re.IGNORECASE):
        return False
    if country_scope == "selected":
        return country in PROJECT_COUNTRIES
    return bool(re.search(r"[A-Za-z]", country))


def parse_number(value: object) -> float | None:
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def number_as_int(value: float | None) -> int | None:
    if value is None:
        return None
    return int(round(value))


def make_session():
    import requests

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "RESPOND-victims-of-corruption CPI extractor "
                "(academic reproducibility script)"
            )
        }
    )
    return session


def discover_result_urls(session, year: int) -> list[str]:
    page_url = PAGE_URL_TEMPLATE.format(year=year)
    response = session.get(page_url, timeout=60)
    response.raise_for_status()
    page = response.text

    hrefs = re.findall(r"""href=["']([^"']+)["']""", page)
    direct_urls = re.findall(r"""https?:\\?/\\?/[^"'\\\s<>]+""", page)
    links = []
    for raw in [*hrefs, *direct_urls]:
        link = html.unescape(raw).replace("\\/", "/")
        links.append(urljoin(page_url, link))

    candidates = []
    for link in sorted(set(links)):
        parsed = urlparse(link)
        path = parsed.path.lower()
        if not path.endswith((".xlsx", ".xls", ".csv", ".zip")):
            continue
        haystack = f"{path} {parsed.query}".lower()
        if "cpi" not in haystack:
            continue
        score = 0
        if "full" in haystack:
            score += 4
        if "result" in haystack or "results" in haystack:
            score += 4
        if str(year) in haystack:
            score += 2
        if "source" in haystack or "method" in haystack:
            score -= 3
        candidates.append((score, link))

    candidates.sort(key=lambda item: item[0], reverse=True)

    urls: list[str] = []
    for link in [*KNOWN_RESULT_URLS.get(year, []), *[link for _, link in candidates]]:
        if link not in urls:
            urls.append(link)
    return urls


def download_file(session, year: int, url: str, cache_dir: Path) -> DownloadedFile:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    name = Path(parsed.path).name or f"cpi_{year}_download"
    cache_path = cache_dir / name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        data = cache_path.read_bytes()
        if not data[:200].lstrip().lower().startswith(b"<!doctype html"):
            return DownloadedFile(year=year, url=url, name=name, data=data)

    response = session.get(url, timeout=120)
    response.raise_for_status()
    data = response.content
    final_name = Path(urlparse(response.url).path).name
    if final_name and final_name != name:
        name = final_name
        cache_path = cache_dir / name
    cache_path.write_bytes(data)
    return DownloadedFile(year=year, url=url, name=name, data=data)


def read_downloaded_tables(download: DownloadedFile) -> list[tuple[str, object]]:
    import pandas as pd

    name = download.name.lower()
    data = io.BytesIO(download.data)
    tables: list[tuple[str, object]] = []

    if name.endswith(".zip"):
        with zipfile.ZipFile(data) as archive:
            for member in archive.namelist():
                lower = member.lower()
                if not lower.endswith((".xlsx", ".xlsm", ".xls", ".csv")):
                    continue
                member_data = archive.read(member)
                nested = DownloadedFile(
                    year=download.year,
                    url=f"{download.url}#{member}",
                    name=Path(member).name,
                    data=member_data,
                )
                tables.extend(read_downloaded_tables(nested))
        return tables

    if name.endswith(".csv"):
        table = pd.read_csv(data, header=None, sep=None, engine="python")
        return [(download.name, table)]

    if name.endswith((".xlsx", ".xlsm")):
        try:
            sheets = pd.read_excel(data, sheet_name=None, header=None, engine="openpyxl")
            tables = [(f"{download.name}:{sheet_name}", table) for sheet_name, table in sheets.items()]
        except Exception:
            tables = []
        xml_tables = read_xlsx_tables_with_xml(download)
        return [*tables, *xml_tables]

    if name.endswith(".xls"):
        sheets = pd.read_excel(data, sheet_name=None, header=None)
        return [(f"{download.name}:{sheet_name}", table) for sheet_name, table in sheets.items()]

    content_type = mimetypes.guess_type(download.name)[0]
    raise ValueError(f"Unsupported CPI result file type: {download.name} ({content_type})")


def xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if xml_local_name(child.tag) == name]


def xml_descendants(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if xml_local_name(child.tag) == name]


def excel_column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    text_nodes = [node.text or "" for node in xml_descendants(cell, "t")]
    if cell.attrib.get("t") == "inlineStr" or text_nodes:
        return "".join(text_nodes)

    values = xml_children(cell, "v")
    if not values:
        return ""

    value = values[0].text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    return value


def read_xlsx_tables_with_xml(download: DownloadedFile) -> list[tuple[str, list[list[str]]]]:
    tables: list[tuple[str, list[list[str]]]] = []
    with zipfile.ZipFile(io.BytesIO(download.data)) as workbook:
        names = set(workbook.namelist())
        if "xl/workbook.xml" not in names or "xl/_rels/workbook.xml.rels" not in names:
            return tables

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            for item in xml_descendants(shared_root, "si"):
                shared_strings.append("".join(node.text or "" for node in xml_descendants(item, "t")))

        rel_root = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
        relationship_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in xml_children(rel_root, "Relationship")
            if "Id" in rel.attrib and "Target" in rel.attrib
        }

        workbook_root = ET.fromstring(workbook.read("xl/workbook.xml"))
        for sheet in xml_descendants(workbook_root, "sheet"):
            relationship_id = next(
                (
                    value
                    for key, value in sheet.attrib.items()
                    if key.endswith("}id") or key == "id"
                ),
                None,
            )
            if relationship_id is None or relationship_id not in relationship_targets:
                continue

            target = relationship_targets[relationship_id]
            sheet_path = posixpath.normpath(posixpath.join("xl", target.lstrip("/")))
            if sheet_path not in names:
                continue

            sheet_root = ET.fromstring(workbook.read(sheet_path))
            rows: list[list[str]] = []
            for row in xml_descendants(sheet_root, "row"):
                values: list[str] = []
                for cell in xml_children(row, "c"):
                    column = excel_column_index(cell.attrib.get("r", "A"))
                    while len(values) <= column:
                        values.append("")
                    values[column] = xlsx_cell_value(cell, shared_strings)
                rows.append(values)

            tables.append((f"{download.name}:{sheet.attrib.get('name', sheet_path)}:xml", rows))
    return tables


def table_shape(table) -> tuple[int, int]:
    if hasattr(table, "shape"):
        return int(table.shape[0]), int(table.shape[1])
    row_count = len(table)
    column_count = max((len(row) for row in table), default=0)
    return row_count, column_count


def table_row(table, row_index: int) -> list[object]:
    if hasattr(table, "iloc"):
        return list(table.iloc[row_index])
    return list(table[row_index])


def table_cell(table, row_index: int, column_index: int) -> object:
    if hasattr(table, "iat"):
        return table.iat[row_index, column_index]
    row = table[row_index]
    return row[column_index] if column_index < len(row) else ""


def row_country(row: Iterable[object], country_scope: str) -> tuple[int, str] | None:
    for idx, value in enumerate(row):
        country = clean_country(value)
        if keep_country(country, country_scope):
            return idx, country
    return None


def header_labels(table, row_index: int) -> list[str]:
    labels: list[str] = []
    start = max(0, row_index - 8)
    _, column_count = table_shape(table)
    for col in range(column_count):
        bits = []
        for prior_row in range(start, row_index):
            value = clean_text(table_cell(table, prior_row, col))
            if not value:
                continue
            if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                continue
            bits.append(value)
        labels.append(" ".join(bits).lower())
    return labels


def pick_labelled_number(
    row,
    labels: list[str],
    terms: tuple[str, ...],
    *,
    year: int | None = None,
) -> int | None:
    if year is not None:
        year_text = str(year)
        for col, label in enumerate(labels):
            if year_text not in label:
                continue
            if not all(term in label for term in terms):
                continue
            value = number_as_int(parse_number(row[col]))
            if value is not None:
                return value

    for col, label in enumerate(labels):
        if not all(term in label for term in terms):
            continue
        value = number_as_int(parse_number(row[col]))
        if value is not None:
            return value
    return None


def fallback_score_rank(row, country_col: int) -> tuple[int | None, int | None]:
    numeric = []
    for col, value in enumerate(row):
        if col == country_col:
            continue
        parsed = number_as_int(parse_number(value))
        if parsed is not None:
            numeric.append((col, parsed))

    score = None
    rank = None
    for _, value in numeric:
        if score is None and 0 <= value <= 100:
            score = value
            continue
        if rank is None and 1 <= value <= 220:
            rank = value
    return score, rank


def parse_table_rows(
    table_name: str,
    table,
    *,
    year: int,
    source_url: str,
    country_scope: str,
) -> list[dict]:
    rows: list[dict] = []
    row_count, _ = table_shape(table)
    for row_index in range(row_count):
        row = table_row(table, row_index)
        country_match = row_country(row, country_scope)
        if country_match is None:
            continue

        country_col, country = country_match
        labels = header_labels(table, row_index)
        score = (
            pick_labelled_number(row, labels, ("cpi", "score"), year=year)
            or pick_labelled_number(row, labels, ("score",), year=year)
            or pick_labelled_number(row, labels, ("cpi", "score"))
            or pick_labelled_number(row, labels, ("score",))
        )
        rank = pick_labelled_number(row, labels, ("rank",), year=year) or pick_labelled_number(
            row, labels, ("rank",)
        )
        if score is None:
            score, fallback_rank = fallback_score_rank(row, country_col)
            rank = rank or fallback_rank

        if score is None or not (0 <= score <= 100):
            continue

        rows.append(
            {
                "year": year,
                "country": country,
                "cpi_score": score,
                "cpi_rank": rank,
                "source_url": source_url,
                "source_file": table_name,
                "extraction_method": "transparency_full_results",
            }
        )
    return rows


def extract_year(session, year: int, cache_dir: Path, country_scope: str) -> tuple[list[dict], dict]:
    urls = discover_result_urls(session, year)
    errors: list[str] = []
    for url in urls:
        try:
            download = download_file(session, year, url, cache_dir)
            tables = read_downloaded_tables(download)
            rows: list[dict] = []
            for table_name, table in tables:
                rows.extend(
                    parse_table_rows(
                        table_name,
                        table,
                        year=year,
                        source_url=url,
                        country_scope=country_scope,
                    )
                )
            rows = tidy_rows(rows)
            if rows:
                return rows, {
                    "year": year,
                    "status": "ok",
                    "rows_extracted": len(rows),
                    "source_url": url,
                    "candidate_urls": len(urls),
                    "error": "",
                }
        except Exception as exc:  # noqa: BLE001 - log and try the next official file.
            errors.append(f"{url}: {type(exc).__name__}: {exc}")

    return [], {
        "year": year,
        "status": "no_rows",
        "rows_extracted": 0,
        "source_url": urls[0] if urls else "",
        "candidate_urls": len(urls),
        "error": " | ".join(errors),
    }


def tidy_rows(rows: list[dict]) -> list[dict]:
    seen = set()
    tidy = []
    for row in sorted(
        rows,
        key=lambda item: (
            item["year"],
            item["country"],
            source_priority(item["source_file"]),
            item["source_file"],
        ),
    ):
        key = (row["year"], row["country"])
        if key in seen:
            continue
        seen.add(key)
        tidy.append(row)
    return tidy


def source_priority(source_file: str) -> int:
    source = source_file.lower()
    if any(term in source for term in ["change", "significant", "regional"]):
        return 4
    if "trend" in source:
        return 3
    if "historical" in source:
        return 2
    if "timeseries" in source or "time series" in source:
        return 1
    return 0


def drop_incomplete_selected_years(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_year: dict[int, list[dict]] = {}
    for row in rows:
        by_year.setdefault(int(row["year"]), []).append(row)

    kept: list[dict] = []
    dropped: list[dict] = []
    for year, year_rows in sorted(by_year.items()):
        countries = {row["country"] for row in year_rows}
        missing = sorted(PROJECT_COUNTRIES - countries)
        if missing:
            dropped.append(
                {
                    "year": year,
                    "countries_extracted": len(countries),
                    "countries_expected": len(PROJECT_COUNTRIES),
                    "missing_countries": "; ".join(missing),
                }
            )
        else:
            kept.extend(year_rows)
    return kept, dropped


def main() -> None:
    args = parse_args()

    import pandas as pd

    session = make_session()
    all_rows: list[dict] = []
    logs: list[dict] = []
    for year in args.years:
        rows, log = extract_year(session, year, args.cache_dir, args.country_scope)
        all_rows.extend(rows)
        logs.append(log)
        print(
            f"{year}: {log['rows_extracted']} rows "
            f"({log['status']}, {log.get('candidate_urls', 0)} candidate files)",
            flush=True,
        )

    all_rows = tidy_rows(all_rows)
    dropped_years: list[dict] = []
    if args.country_scope == "selected" and not args.allow_partial_years:
        all_rows, dropped_years = drop_incomplete_selected_years(all_rows)
        for item in dropped_years:
            print(
                "Dropped incomplete selected-country CPI year "
                f"{item['year']}: {item['countries_extracted']}/"
                f"{item['countries_expected']} countries extracted; missing "
                f"{item['missing_countries']}",
                flush=True,
            )

    if not all_rows:
        raise RuntimeError(
            "No CPI country-year rows were extracted. Check the log output and "
            "download cache, or run with --allow-partial-years for diagnostics."
        )

    output = pd.DataFrame(all_rows).sort_values(["year", "country"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    log_path = args.log_output or args.output.with_name(args.output.stem + "_extraction_log.csv")
    pd.DataFrame(logs).to_csv(log_path, index=False)

    print(f"Saved CPI country-year scores: {args.output} ({len(output):,} rows)", flush=True)
    print(f"Saved extraction log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
