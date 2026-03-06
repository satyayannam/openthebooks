from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import pandas as pd


EXPECTED_COLUMNS = [
    "Agency",
    "Account",
    "Project",
    "Recipient",
    "Location",
    "Amount",
    "House Requestor",
    "Senate Requestor",
    "Origin",
]


STATE_TO_ABBREV: Dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

ABBREV_TO_STATE: Dict[str, str] = {v: k.title() for k, v in STATE_TO_ABBREV.items()}


def _normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _canonical_column_name(normalized: str) -> str | None:
    mapping = {
        "agency": "Agency",
        "account": "Account",
        "project": "Project",
        "recipient": "Recipient",
        "location": "Location",
        "amount": "Amount",
        "house_requestor": "House Requestor",
        "house_requestors": "House Requestor",
        "house_requester": "House Requestor",
        "house_requesters": "House Requestor",
        "senate_requestor": "Senate Requestor",
        "senate_requestors": "Senate Requestor",
        "senate_requester": "Senate Requestor",
        "senate_requesters": "Senate Requestor",
        "origin": "Origin",
    }
    if normalized in mapping:
        return mapping[normalized]

    if normalized.startswith("house_") and "request" in normalized:
        return "House Requestor"
    if normalized.startswith("senate_") and "request" in normalized:
        return "Senate Requestor"
    return None


def _clean_text_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})


def parse_amount_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    is_negative = s.str.match(r"^\(.*\)$", na=False)

    s = s.str.replace("$", "", regex=False)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("(", "", regex=False)
    s = s.str.replace(")", "", regex=False)
    s = s.str.replace(r"[^0-9.\-]", "", regex=True)

    out = pd.to_numeric(s, errors="coerce")
    out.loc[is_negative & out.notna()] = -out.loc[is_negative & out.notna()].abs()
    return out


def derive_city_state(location: str) -> Tuple[str, str, str]:
    if not isinstance(location, str):
        return "Unknown", "Unknown", "Unknown"

    raw = location.strip()
    if not raw:
        return "Unknown", "Unknown", "Unknown"

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    city = parts[0] if parts else "Unknown"

    state_abbrev = "Unknown"
    for part in reversed(parts):
        upper = part.upper()
        lower = part.lower()
        if upper in ABBREV_TO_STATE:
            state_abbrev = upper
            break
        if lower in STATE_TO_ABBREV:
            state_abbrev = STATE_TO_ABBREV[lower]
            break

    if state_abbrev == "Unknown":
        lower_raw = raw.lower()
        for state_name, abbrev in STATE_TO_ABBREV.items():
            if re.search(rf"\b{re.escape(state_name)}\b", lower_raw):
                state_abbrev = abbrev
                break

    state_name = ABBREV_TO_STATE.get(state_abbrev, "Unknown")
    return city or "Unknown", state_abbrev, state_name


def normalize_earmarks_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map: Dict[str, str] = {}
    for col in out.columns:
        canonical = _canonical_column_name(_normalize_col(col))
        if canonical and canonical not in rename_map.values():
            rename_map[col] = canonical
    out = out.rename(columns=rename_map)

    for col in EXPECTED_COLUMNS:
        if col not in out.columns:
            out[col] = None

    # Keep a predictable column order for downstream rendering.
    passthrough = [c for c in out.columns if c not in EXPECTED_COLUMNS]
    return out[EXPECTED_COLUMNS + passthrough]


def prepare_earmarks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = normalize_earmarks_columns(df)
    out["Amount"] = parse_amount_series(out["Amount"])

    _clean_text_columns(
        out,
        [
            "Agency",
            "Account",
            "Project",
            "Recipient",
            "Location",
            "House Requestor",
            "Senate Requestor",
            "Origin",
        ],
    )

    location_parts = out["Location"].apply(derive_city_state)
    out["derived_city"] = location_parts.apply(lambda v: v[0])
    out["derived_state"] = location_parts.apply(lambda v: v[1])
    out["derived_state_name"] = location_parts.apply(lambda v: v[2])

    out["keyword_blob"] = (
        out["Project"].astype(str).str.lower()
        + " "
        + out["Recipient"].astype(str).str.lower()
        + " "
        + out["Location"].astype(str).str.lower()
    )

    out["has_amount"] = out["Amount"].notna()
    return out


def top_share(values: pd.Series, top_n: int = 10) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    total = float(numeric.sum())
    if total <= 0:
        return 0.0
    top_total = float(numeric.nlargest(top_n).sum())
    return top_total / total


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
