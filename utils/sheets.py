from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def _parse_worksheet_names(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _get_service_account_info() -> Optional[dict]:
    # Streamlit secrets may contain either of these keys.
    try:
        if "gcp_service_account" in st.secrets:
            return dict(st.secrets["gcp_service_account"])
        if "google_service_account_json" in st.secrets:
            return dict(st.secrets["google_service_account_json"])
    except StreamlitSecretNotFoundError:
        return None
    return None


def _get_client():
    import gspread
    from google.oauth2.service_account import Credentials

    info = _get_service_account_info()
    if not info:
        return None

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    return gspread.authorize(creds)


def _dedupe_headers(headers: list[str]) -> list[str]:
    seen = {}
    out = []
    for h in headers:
        key = (h or "").strip()
        if not key:
            key = "column"
        count = seen.get(key, 0)
        seen[key] = count + 1
        if count == 0:
            out.append(key)
        else:
            out.append(f"{key}_{count+1}")
    return out


def _looks_like_year_header(row: list[str]) -> bool:
    year_hits = 0
    for cell in row:
        if isinstance(cell, str) and cell.isdigit() and len(cell) == 4:
            year_hits += 1
    return year_hits >= 2


def _worksheet_to_df(ws) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()

    header_row_idx = 0
    if len(values) > 1:
        row0 = values[0]
        row1 = values[1]
        if ("SUM" in " ".join(row0) or _looks_like_year_header(row1)) and _looks_like_year_header(row1):
            header_row_idx = 1

    headers = _dedupe_headers(values[header_row_idx])
    rows = values[header_row_idx + 1 :]
    if not rows:
        return pd.DataFrame(columns=headers)
    return pd.DataFrame(rows, columns=headers)


def _load_sheet_records(sheet_id: str, worksheets: Iterable[str]) -> pd.DataFrame:
    client = _get_client()
    if client is None:
        return pd.DataFrame()

    try:
        book = client.open_by_key(sheet_id)
    except Exception:
        return pd.DataFrame()

    frames = []
    if worksheets:
        for name in worksheets:
            ws = book.worksheet(name)
            df = _worksheet_to_df(ws)
            if not df.empty:
                df["worksheet"] = name
                frames.append(df)
    else:
        ws = book.get_worksheet(0)
        df = _worksheet_to_df(ws)
        if not df.empty:
            df["worksheet"] = ws.title
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=300)
def list_worksheets(sheet_id: str) -> List[str]:
    client = _get_client()
    if client is None:
        return []
    try:
        book = client.open_by_key(sheet_id)
        return [ws.title for ws in book.worksheets()]
    except Exception:
        return []


@st.cache_data(ttl=300)
def _cached_load(sheet_id: str, worksheets_key: str) -> pd.DataFrame:
    worksheets = _parse_worksheet_names(worksheets_key)
    return _load_sheet_records(sheet_id, worksheets)


def _worksheets_key_from_value(value: Optional[Sequence[str] | str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ",".join([v for v in value if v])


def load_data(refresh: bool = False, worksheets: Optional[Sequence[str] | str] = None) -> pd.DataFrame:
    try:
        if "SHEET_ID" not in st.secrets:
            return pd.DataFrame()
        sheet_id = str(st.secrets["SHEET_ID"])
        worksheets_key = str(st.secrets.get("WORKSHEET_NAMES", "") or "")
    except StreamlitSecretNotFoundError:
        return pd.DataFrame()

    override_key = _worksheets_key_from_value(worksheets)
    if override_key.strip():
        worksheets_key = override_key

    if refresh:
        _cached_load.clear()

    return _cached_load(sheet_id, worksheets_key)
