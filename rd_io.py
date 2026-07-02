"""Research Drive WebDAV I/O helpers for RESPOND data."""

from __future__ import annotations

import io
import json
import posixpath

import pandas as pd

from config import RD_BASE_DIR
from rd_utils import webdav_download_bytes, webdav_list, webdav_mkdirs, webdav_upload_bytes

DATA_DIR = RD_BASE_DIR
ANNOTATION_DIR = f"{RD_BASE_DIR}/annotations"
OUT_DIR = f"{RD_BASE_DIR}/analysis_out"


def rd_join(*parts: str) -> str:
    """Join Research Drive paths with POSIX semantics."""
    clean = [p.strip("/ ") for p in parts if p]
    return posixpath.join(*clean)


def rd_parent(path: str) -> str:
    """Return the parent folder of a Research Drive path."""
    return posixpath.dirname(path.rstrip("/"))


def rd_download_bytes(path: str) -> bytes:
    return webdav_download_bytes(path)


def rd_list_dir(path: str) -> list[str]:
    return webdav_list(path)


def rd_upload_bytes(path: str, data: bytes, content_type: str | None = None) -> None:
    webdav_mkdirs(rd_parent(path))
    webdav_upload_bytes(path, data, content_type or "application/octet-stream")


def rd_read_text(path: str, encoding: str = "utf-8") -> str:
    return rd_download_bytes(path).decode(encoding)


def rd_write_text(path: str, text: str, encoding: str = "utf-8") -> None:
    rd_upload_bytes(path, text.encode(encoding), "text/plain; charset=utf-8")


def rd_read_json(path: str, encoding: str = "utf-8"):
    return json.loads(rd_read_text(path, encoding))


def rd_write_json(path: str, obj, encoding: str = "utf-8", indent: int = 2) -> None:
    rd_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent), encoding)


def rd_read_csv_df(path: str, **kwargs) -> pd.DataFrame:
    data = rd_download_bytes(path)
    return pd.read_csv(io.BytesIO(data), **kwargs)


def rd_write_csv_df(df: pd.DataFrame, path: str, index: bool = False, **kwargs) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=index, **kwargs)
    rd_upload_bytes(path, buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8")


def rd_read_parquet_df(path: str, **kwargs) -> pd.DataFrame:
    data = rd_download_bytes(path)
    return pd.read_parquet(io.BytesIO(data), **kwargs)


def rd_write_parquet_df(df: pd.DataFrame, path: str, **kwargs) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, **kwargs)
    rd_upload_bytes(path, buf.getvalue(), "application/octet-stream")
