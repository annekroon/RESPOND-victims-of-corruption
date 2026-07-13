"""Extract country-year Corruption Perceptions Index scores from CPI PDFs.

The default source is the project Research Drive/SURF folder:

    ASCOR-FMG-5580-RESPOND-news-data (Projectfolder)/
      victims-of-corruption-paper/CPI/

The script can also parse a local PDF folder. It writes a tidy CSV with one row
per country-year score and a small extraction log. The parser is intentionally
auditable: it prefers tables extracted with pdfplumber and falls back to simple
text-line patterns when no usable table is found.

Examples
--------
Parse PDFs from Research Drive/WebDAV:

    python3 extract_cpi_from_pdfs.py \
      --source webdav \
      --output output/cpi_country_year_scores.csv

Parse already downloaded PDFs:

    python3 extract_cpi_from_pdfs.py \
      --source local \
      --local-dir /path/to/CPI \
      --output output/cpi_country_year_scores.csv
"""

from __future__ import annotations

import argparse
import io
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ALL_COUNTRIES, RD_BASE_DIR


DEFAULT_RD_CPI_DIR = posixpath.join(
    RD_BASE_DIR,
    "victims-of-corruption-paper",
    "CPI",
)
DEFAULT_OUTPUT = Path("output/cpi_country_year_scores.csv")
PROJECT_COUNTRIES = {country.replace("_", " ") for country in ALL_COUNTRIES}


@dataclass(frozen=True)
class PdfSource:
    name: str
    data: bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract yearly country-level CPI scores from CPI PDFs."
    )
    parser.add_argument(
        "--source",
        choices=["webdav", "local"],
        default="webdav",
        help="Read PDFs from Research Drive/WebDAV or from a local directory.",
    )
    parser.add_argument(
        "--rd-dir",
        default=DEFAULT_RD_CPI_DIR,
        help="Research Drive directory containing CPI PDF files.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory containing CPI PDF files when --source local.",
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
        help="Optional CSV path for per-PDF extraction diagnostics.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional local directory where WebDAV PDFs are cached before parsing.",
    )
    parser.add_argument(
        "--min-rows-per-pdf",
        type=int,
        default=5,
        help="Warn when fewer than this many country rows are extracted from a PDF.",
    )
    parser.add_argument(
        "--country-scope",
        choices=["selected", "all"],
        default="selected",
        help=(
            "Keep only project countries by default. Use 'all' to keep every "
            "country/territory row that can be parsed."
        ),
    )
    return parser.parse_args()


def clean_cell(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_country(value: object) -> str:
    text = clean_cell(value)
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s*\[[^\]]*\]\s*$", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    text = text.replace("*", "").replace("†", "").strip(" -–—:")
    text = re.sub(r"\s+", " ", text)
    return text


def parse_int(value: object) -> int | None:
    text = clean_cell(value)
    if not text:
        return None
    match = re.search(r"\d{1,3}", text.replace(",", ""))
    if not match:
        return None
    return int(match.group(0))


def parse_year_from_name(name: str) -> int | None:
    patterns = [
        r"(?:^|[_\-\s])CPI[_\-\s]?(?P<year>(?:19|20)\d{2})",
        r"(?P<year>(?:19|20)\d{2})[_\-\s]?CPI(?:[_\-\s]|$)",
        r"Report[_\-\s]?CPI(?P<year>(?:19|20)\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group("year"))

    years = [int(match) for match in re.findall(r"(?:19|20)\d{2}", name)]
    plausible = [year for year in years if 1995 <= year <= 2100]
    return plausible[0] if plausible else None


def normalize_country_name(country: object) -> str:
    country = clean_country(country).replace("_", " ")
    replacements = {
        "UK": "United Kingdom",
        "United States of America": "United States",
        "Czech Republic": "Czechia",
        "Korea, South": "South Korea",
    }
    return replacements.get(country, country)


def keep_country(country: str, country_scope: str) -> bool:
    country = normalize_country_name(country)
    if not country or len(country) < 3:
        return False
    if re.search(r"[:@<>]|www\.|http|fax|phone", country, flags=re.IGNORECASE):
        return False
    if country_scope == "selected":
        return country in PROJECT_COUNTRIES
    return bool(re.search(r"[A-Za-z]", country))


def normalized_header(cells: Iterable[object]) -> list[str]:
    return [clean_cell(cell).lower() for cell in cells]


def find_header_columns(row: list[object]) -> tuple[int | None, int | None, int | None]:
    header = normalized_header(row)
    country_col = None
    score_col = None
    rank_col = None
    for idx, cell in enumerate(header):
        if any(term in cell for term in ["country", "territory", "countries"]):
            country_col = idx
        if "score" in cell or "cpi" in cell:
            score_col = idx
        if "rank" in cell:
            rank_col = idx
    return country_col, score_col, rank_col


def row_from_columns(
    row: list[object],
    *,
    country_col: int,
    score_col: int,
    rank_col: int | None,
    year: int,
    pdf_name: str,
    page_number: int,
    method: str,
    country_scope: str,
) -> dict | None:
    if country_col >= len(row) or score_col >= len(row):
        return None

    country = normalize_country_name(row[country_col])
    score = parse_int(row[score_col])
    rank = parse_int(row[rank_col]) if rank_col is not None and rank_col < len(row) else None

    if not country or score is None or not (0 <= score <= 100):
        return None
    if not keep_country(country, country_scope):
        return None
    if country.lower() in {"country", "country/territory", "countries"}:
        return None

    return {
        "year": year,
        "country": country,
        "cpi_score": score,
        "cpi_rank": rank,
        "source_pdf": pdf_name,
        "source_page": page_number,
        "extraction_method": method,
    }


def infer_row_without_header(
    row: list[object],
    *,
    year: int,
    pdf_name: str,
    page_number: int,
    country_scope: str,
) -> dict | None:
    cells = [clean_cell(cell) for cell in row]
    if len([cell for cell in cells if cell]) < 2:
        return None

    numeric = [(idx, parse_int(cell)) for idx, cell in enumerate(cells)]
    numeric = [(idx, value) for idx, value in numeric if value is not None]
    if not numeric:
        return None

    score_candidates = [(idx, value) for idx, value in numeric if 0 <= value <= 100]
    if not score_candidates:
        return None

    text_candidates = [
        (idx, clean_country(cell))
        for idx, cell in enumerate(cells)
        if clean_country(cell) and not re.fullmatch(r"[\d\s,./-]+", clean_cell(cell))
    ]
    if not text_candidates:
        return None

    country_col, country = max(text_candidates, key=lambda item: len(item[1]))
    country = normalize_country_name(country)
    score_col, score = score_candidates[-1]
    rank_values = [value for idx, value in numeric if idx != score_col and 1 <= value <= 200]
    rank = rank_values[0] if rank_values else None

    if country_col == score_col or not country or score is None:
        return None
    if not keep_country(country, country_scope):
        return None

    return {
        "year": year,
        "country": country,
        "cpi_score": score,
        "cpi_rank": rank,
        "source_pdf": pdf_name,
        "source_page": page_number,
        "extraction_method": "pdfplumber_table_inferred",
    }


def extract_rows_with_pdfplumber(source: PdfSource, year: int, country_scope: str) -> list[dict]:
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError(
            "pdfplumber is required for table extraction. Install it with "
            "`python3 -m pip install pdfplumber`, or use an environment that has it."
        ) from exc

    rows: list[dict] = []
    with pdfplumber.open(io.BytesIO(source.data)) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            for table in tables:
                country_col = score_col = rank_col = None
                table_rows: list[dict] = []
                inferred_rows: list[dict] = []
                for raw_row in table:
                    if not raw_row:
                        continue
                    header_cols = find_header_columns(raw_row)
                    if header_cols[0] is not None and header_cols[1] is not None:
                        country_col, score_col, rank_col = header_cols
                        continue

                    if country_col is not None and score_col is not None:
                        parsed = row_from_columns(
                            raw_row,
                            country_col=country_col,
                            score_col=score_col,
                            rank_col=rank_col,
                            year=year,
                            pdf_name=source.name,
                            page_number=page_index,
                            method="pdfplumber_table",
                            country_scope=country_scope,
                        )
                        if parsed:
                            table_rows.append(parsed)
                    else:
                        parsed = infer_row_without_header(
                            raw_row,
                            year=year,
                            pdf_name=source.name,
                            page_number=page_index,
                            country_scope=country_scope,
                        )
                        if parsed:
                            inferred_rows.append(parsed)
                rows.extend(table_rows or inferred_rows)
    return rows


def extract_rows_with_text(source: PdfSource, year: int, country_scope: str) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install pypdf or PyPDF2 for text fallback extraction.") from exc

    reader = PdfReader(io.BytesIO(source.data))
    rows: list[dict] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for line in text.splitlines():
            line = clean_cell(line)
            if not line:
                continue

            parsed = None
            patterns = [
                r"^(?P<country>.+?)\s+(?P<rank>\d{1,3})\s+(?P<score>\d{1,3})(?:\s|$)",
                r"^(?P<rank>\d{1,3})\s+(?P<country>.+?)\s+(?P<score>\d{1,3})(?:\s|$)",
            ]
            for pattern in patterns:
                match = re.match(pattern, line)
                if not match:
                    continue

                country = normalize_country_name(match.group("country"))
                score = int(match.group("score"))
                rank = int(match.group("rank"))
                if (
                    keep_country(country, country_scope)
                    and 0 <= score <= 100
                    and 1 <= rank <= 200
                ):
                    parsed = {
                        "year": year,
                        "country": country,
                        "cpi_score": score,
                        "cpi_rank": rank,
                        "source_pdf": source.name,
                        "source_page": page_index,
                        "extraction_method": "pypdf_text_regex",
                    }
                    break
            if parsed is None:
                continue
            rows.append(parsed)
    return rows


def list_webdav_pdfs(rd_dir: str) -> list[PdfSource]:
    from rd_utils import webdav_download_bytes, webdav_list

    names = [name for name in webdav_list(rd_dir) if name.lower().endswith(".pdf")]
    sources = []
    for name in sorted(names):
        rd_path = posixpath.join(rd_dir, name)
        sources.append(PdfSource(name=name, data=webdav_download_bytes(rd_path)))
    return sources


def list_local_pdfs(local_dir: Path) -> list[PdfSource]:
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)
    paths = sorted(local_dir.glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDF files found in {local_dir}")
    return [PdfSource(name=path.name, data=path.read_bytes()) for path in paths]


def cache_sources(sources: list[PdfSource], cache_dir: Path | None) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        (cache_dir / source.name).write_bytes(source.data)


def extract_pdf(source: PdfSource, min_rows: int, country_scope: str) -> tuple[list[dict], dict]:
    year = parse_year_from_name(source.name)
    if year is None:
        return [], {
            "source_pdf": source.name,
            "year": None,
            "rows_extracted": 0,
            "status": "skipped_no_year_in_filename",
        }

    rows = extract_rows_with_pdfplumber(source, year, country_scope)
    method = "pdfplumber_table"
    if len(rows) < min_rows:
        text_rows = extract_rows_with_text(source, year, country_scope)
        if len(text_rows) > len(rows):
            rows = text_rows
            method = "pypdf_text_regex"

    status = "ok" if len(rows) >= min_rows else "few_rows"
    return rows, {
        "source_pdf": source.name,
        "year": year,
        "rows_extracted": len(rows),
        "status": status,
        "primary_method": method,
    }


def tidy_rows(rows: list[dict]) -> pd.DataFrame:
    import pandas as pd

    if not rows:
        return pd.DataFrame(
            columns=[
                "year",
                "country",
                "cpi_score",
                "cpi_rank",
                "source_pdf",
                "source_page",
                "extraction_method",
            ]
        )
    data = pd.DataFrame(rows)
    data = data.drop_duplicates(subset=["year", "country", "cpi_score", "cpi_rank"])
    data = data.sort_values(["year", "country", "source_pdf", "source_page"]).reset_index(drop=True)
    return data


def main() -> None:
    args = parse_args()

    import pandas as pd

    if args.source == "webdav":
        sources = list_webdav_pdfs(args.rd_dir)
    else:
        if args.local_dir is None:
            raise ValueError("--local-dir is required when --source local.")
        sources = list_local_pdfs(args.local_dir)

    if not sources:
        raise FileNotFoundError("No CPI PDFs found.")

    cache_sources(sources, args.cache_dir)

    all_rows: list[dict] = []
    logs: list[dict] = []
    for source in sources:
        rows, log = extract_pdf(source, args.min_rows_per_pdf, args.country_scope)
        all_rows.extend(rows)
        logs.append(log)
        print(
            f"{source.name}: {log['rows_extracted']} rows "
            f"({log['status']}, {log.get('primary_method', 'n/a')})",
            flush=True,
        )

    output = tidy_rows(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    log_path = args.log_output or args.output.with_name(args.output.stem + "_extraction_log.csv")
    pd.DataFrame(logs).to_csv(log_path, index=False)

    print(f"Saved CPI country-year scores: {args.output} ({len(output):,} rows)", flush=True)
    print(f"Saved extraction log: {log_path}", flush=True)


if __name__ == "__main__":
    main()
