from __future__ import annotations

import io
from typing import Dict, Iterable, Optional

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def normalize_header(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip(),
        errors="coerce",
    ).fillna(0.0)


def _to_fips(series: pd.Series, width: int) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str.zfill(width)
        .replace({("0" * width): ""})
    )


def _pick_first(columns: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        key = normalize_header(cand)
        if key in columns:
            return columns[key]
    return None


def _get_series(df: pd.DataFrame, column: str) -> pd.Series:
    data = df[column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def _resolve_positional_mapping(df_raw: pd.DataFrame) -> Dict[str, Optional[str]]:
    # Fallback for headerless sheet tabs where first data row became headers.
    data_cols = [c for c in df_raw.columns if normalize_header(c) != "worksheet"]
    # Expected IRS order (subset):
    # 0 STATEFIPS, 1 STATE, 2 COUNTYFIPS, 3 COUNTYNAME, 4 agi_stub, 5 N1, 14 N2,
    # 21 A00100, 25 A00200, 27 A00300, 31 A00600
    if len(data_cols) <= 31:
        return {}
    return {
        "state_fips_col": data_cols[0],
        "state_col": data_cols[1],
        "county_fips_col": data_cols[2],
        "county_col": data_cols[3],
        "agi_stub_col": data_cols[4],
        "n1_col": data_cols[5],
        "n2_col": data_cols[14],
        "agi_col": data_cols[21],
        "wages_col": data_cols[25],
        "interest_col": data_cols[27],
        "dividends_col": data_cols[31],
    }


def prepare_tax_stats_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    cmap = {normalize_header(c): c for c in df_raw.columns}

    state_fips_col = _pick_first(cmap, ["statefips", "state_fips"])
    state_col = _pick_first(cmap, ["state", "stateabbr", "state_abbr"])
    county_fips_col = _pick_first(cmap, ["countyfips", "county_fips"])
    county_col = _pick_first(cmap, ["countyname", "county", "county_name"])
    agi_stub_col = _pick_first(cmap, ["agi_stub", "agistub"])
    n1_col = _pick_first(cmap, ["n1", "returns", "numberofreturns"])
    n2_col = _pick_first(cmap, ["n2", "exemptions", "personalexemptions"])
    agi_col = _pick_first(cmap, ["a00100", "agi", "adjustedgrossincome"])
    wages_col = _pick_first(cmap, ["a00200", "wages", "wagesandsalaries"])
    dividends_col = _pick_first(cmap, ["a00600", "dividends", "dividendsbeforeexclusion"])
    interest_col = _pick_first(cmap, ["a00300", "interest", "interestreceived"])

    needed = [state_fips_col, state_col, county_fips_col, county_col, n1_col, n2_col, agi_col, wages_col, dividends_col, interest_col]
    if any(col is None for col in needed):
        pos = _resolve_positional_mapping(df_raw)
        if pos:
            state_fips_col = pos["state_fips_col"]
            state_col = pos["state_col"]
            county_fips_col = pos["county_fips_col"]
            county_col = pos["county_col"]
            agi_stub_col = pos["agi_stub_col"]
            n1_col = pos["n1_col"]
            n2_col = pos["n2_col"]
            agi_col = pos["agi_col"]
            wages_col = pos["wages_col"]
            interest_col = pos["interest_col"]
            dividends_col = pos["dividends_col"]
        else:
            return pd.DataFrame()

    df = pd.DataFrame(
        {
            "state_fips": _to_fips(_get_series(df_raw, state_fips_col), 2),
            "state": _get_series(df_raw, state_col).astype(str).str.strip(),
            "county_fips_local": _to_fips(_get_series(df_raw, county_fips_col), 3),
            "county": _get_series(df_raw, county_col).astype(str).str.strip(),
            "agi_stub": _to_numeric(_get_series(df_raw, agi_stub_col)) if agi_stub_col else 0.0,
            "number_of_returns": _to_numeric(_get_series(df_raw, n1_col)),
            "personal_exemptions": _to_numeric(_get_series(df_raw, n2_col)),
            "adjusted_gross_income": _to_numeric(_get_series(df_raw, agi_col)),
            "wages_and_salaries": _to_numeric(_get_series(df_raw, wages_col)),
            "dividends": _to_numeric(_get_series(df_raw, dividends_col)),
            "interest_received": _to_numeric(_get_series(df_raw, interest_col)),
        }
    )

    # Keep county rows; remove state summary rows like countyfips=000.
    state_num = pd.to_numeric(df["state_fips"], errors="coerce")
    county_num = pd.to_numeric(df["county_fips_local"], errors="coerce")
    df = df[
        state_num.between(1, 56, inclusive="both")
        & county_num.between(1, 999, inclusive="both")
    ].copy()

    df["county_fips"] = df["state_fips"] + df["county_fips_local"]
    df["county_label"] = (df["county"] + ", " + df["state"]).str.strip(", ")
    return df


def _apply_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    returns = out["number_of_returns"].replace({0: pd.NA})
    agi = out["adjusted_gross_income"].replace({0: pd.NA})
    out["avg_income_per_return"] = out["adjusted_gross_income"] / returns
    out["avg_wages_per_return"] = out["wages_and_salaries"] / returns
    out["exemptions_per_return"] = out["personal_exemptions"] / returns
    out["investment_income"] = out["dividends"] + out["interest_received"]
    out["investment_income_share"] = out["investment_income"] / agi
    out["wage_income_share"] = out["wages_and_salaries"] / agi
    return out.fillna(0.0)


def build_county_summary(df_prepared: pd.DataFrame) -> pd.DataFrame:
    if df_prepared.empty:
        return pd.DataFrame()
    county = (
        df_prepared.groupby(["state_fips", "state", "county_fips", "county", "county_label"], as_index=False)
        .agg(
            number_of_returns=("number_of_returns", "sum"),
            personal_exemptions=("personal_exemptions", "sum"),
            adjusted_gross_income=("adjusted_gross_income", "sum"),
            wages_and_salaries=("wages_and_salaries", "sum"),
            dividends=("dividends", "sum"),
            interest_received=("interest_received", "sum"),
        )
        .sort_values("adjusted_gross_income", ascending=False)
    )
    return _apply_derived_metrics(county).reset_index(drop=True)


def build_state_summary(county_summary: pd.DataFrame) -> pd.DataFrame:
    if county_summary.empty:
        return pd.DataFrame()
    state = (
        county_summary.groupby(["state_fips", "state"], as_index=False)
        .agg(
            county_count=("county_fips", "nunique"),
            number_of_returns=("number_of_returns", "sum"),
            personal_exemptions=("personal_exemptions", "sum"),
            adjusted_gross_income=("adjusted_gross_income", "sum"),
            wages_and_salaries=("wages_and_salaries", "sum"),
            dividends=("dividends", "sum"),
            interest_received=("interest_received", "sum"),
        )
        .sort_values("adjusted_gross_income", ascending=False)
    )
    return _apply_derived_metrics(state).reset_index(drop=True)
