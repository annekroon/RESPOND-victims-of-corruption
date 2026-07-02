"""Low-level Research Drive WebDAV helpers."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from urllib.parse import quote
from urllib.parse import unquote

import requests

import config


def _get_session() -> tuple[requests.Session, str]:
    app_password = getattr(config, "APP_PASSWORD", None)
    user = getattr(config, "USER", None)
    base_url = getattr(config, "BASE_URL", None)

    missing = [
        name
        for name, value in {
            "BASE_URL or RD_BASE_URL": base_url,
            "USER or RD_USER": user,
            "APP_PASSWORD or RD_PASS": app_password,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Research Drive WebDAV setting(s): "
            + ", ".join(missing)
            + ". Add them to config_local.py or set environment variables."
        )

    session = requests.Session()
    session.auth = (user, app_password)
    base = f"{base_url.rstrip('/')}/{quote(user, safe='')}"
    return session, base


def _enc_path(rel_path: str) -> str:
    parts = [p for p in rel_path.strip("/").split("/") if p]
    return "/".join(quote(p, safe="") for p in parts)


def webdav_mkdirs(rel_dir: str) -> None:
    """Recursively create folders on Research Drive."""
    session, base = _get_session()
    parts = [p for p in rel_dir.strip("/").split("/") if p]
    url = base

    for part in parts:
        url = f"{url}/{quote(part, safe='')}"
        response = session.request("MKCOL", url)
        if response.status_code in (201, 405):
            continue
        response.raise_for_status()


def webdav_upload_bytes(rel_path: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    """Upload bytes to Research Drive at rel_path."""
    session, base = _get_session()
    url = f"{base}/{_enc_path(rel_path)}"
    response = session.put(url, data=data, headers={"Content-Type": content_type})
    response.raise_for_status()


def webdav_download_bytes(rel_path: str) -> bytes:
    """Download a Research Drive file into memory."""
    session, base = _get_session()
    url = f"{base}/{_enc_path(rel_path)}"
    response = session.get(url, stream=True)
    response.raise_for_status()

    buf = io.BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def webdav_list(rel_dir: str) -> list[str]:
    """List direct children in a Research Drive directory."""
    session, base = _get_session()
    encoded_path = _enc_path(rel_dir)
    url = f"{base}/{encoded_path}".rstrip("/") + "/"
    response = session.request("PROPFIND", url, headers={"Depth": "1"})
    response.raise_for_status()

    root = ET.fromstring(response.content)
    hrefs = []
    for response_el in root.findall("{DAV:}response"):
        href_el = response_el.find("{DAV:}href")
        if href_el is not None and href_el.text:
            hrefs.append(unquote(href_el.text.rstrip("/").split("/")[-1]))

    return [href for href in hrefs if href and href != rel_dir.rstrip("/").split("/")[-1]]
