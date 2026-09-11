"""Load the canonical corruption-content codebook and expose provenance metadata."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


CODEBOOK_PATH = Path(__file__).resolve().parents[1] / "CODEBOOK.md"
_CODEBOOK_BYTES = CODEBOOK_PATH.read_bytes()
CODEBOOK_MARKDOWN = _CODEBOOK_BYTES.decode("utf-8").strip()

_VERSION_MATCH = re.search(
    r"^Version:\s*`?([^`\s]+)`?\s*$",
    CODEBOOK_MARKDOWN,
    flags=re.MULTILINE,
)
if _VERSION_MATCH is None:
    raise RuntimeError(f"Codebook version is missing from {CODEBOOK_PATH}")

CODEBOOK_VERSION = _VERSION_MATCH.group(1)
CODEBOOK_SHA256 = hashlib.sha256(_CODEBOOK_BYTES).hexdigest()

SECTION_HEADINGS = {
    "victim_visibility": "1. Victim Visibility",
    "corruption_frame": "2. Corruption Frame",
    "case_location": "3. Case Location",
    "accused_actor_visibility": "4. Accused Actor Visibility",
}


def _section(heading: str) -> str:
    marker = f"## {heading}"
    start = CODEBOOK_MARKDOWN.find(marker)
    if start < 0:
        raise RuntimeError(f"Missing codebook section: {heading}")
    end = CODEBOOK_MARKDOWN.find("\n## ", start + len(marker))
    return CODEBOOK_MARKDOWN[start : end if end >= 0 else None].strip()


def codebook_preamble() -> str:
    """Return the title and general rules shared by all variables."""
    first_variable = f"## {SECTION_HEADINGS['victim_visibility']}"
    return CODEBOOK_MARKDOWN.split(first_variable, maxsplit=1)[0].strip()


def codebook_section(variable: str) -> str:
    """Return one complete variable section from the canonical codebook."""
    try:
        heading = SECTION_HEADINGS[variable]
    except KeyError as exc:
        raise KeyError(f"Unknown content variable: {variable}") from exc
    return _section(heading)


def codebook_for(variable: str) -> str:
    """Return the general rules and one variable section for an LLM prompt."""
    return f"{codebook_preamble()}\n\n{codebook_section(variable)}"
