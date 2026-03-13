from __future__ import annotations

import io
import re
from typing import Dict, Iterable, Optional

import pandas as pd


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _clean_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _get_series(df: pd.DataFrame, column: str) -> pd.Series:
    data = df[column]
    if isinstance(data, pd.DataFrame):
        return data.iloc[:, 0]
    return data


def _pick_first(columns: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        key = normalize_header(cand)
        if key in columns:
            return columns[key]
    return None


def _resolve_mapping(df_raw: pd.DataFrame, direction: str) -> Dict[str, Optional[str]]:
    col_map = {normalize_header(c): c for c in df_raw.columns}

    returns_col = _pick_first(
        col_map,
        ["n1", "returns", "returncount", "numreturns", "numberofreturns", "inboundreturns", "outboundreturns"],
    )
    people_col = _pick_first(
        col_map,
        ["n2", "exemptions", "people", "personcount", "migrants", "inboundpeople", "outboundpeople"],
    )
    agi_col = _pick_first(
        col_map,
        [
            "agi",
            "aggagi",
            "adjustedgrossincome",
            "agiamount",
            "inboundagi",
            "outboundagi",
            "agiamount",
        ],
    )

    explicit_origin_state_fips = _pick_first(col_map, ["origin_state_fips", "from_state_fips", "statefips_from"])
    explicit_origin_county_fips = _pick_first(
        col_map, ["origin_county_fips", "from_county_fips", "countyfips_from"]
    )
    explicit_origin_state = _pick_first(col_map, ["origin_state", "from_state"])
    explicit_origin_county = _pick_first(col_map, ["origin_county", "from_county", "origin_county_name", "from_county_name"])

    explicit_destination_state_fips = _pick_first(
        col_map, ["destination_state_fips", "to_state_fips", "statefips_to"]
    )
    explicit_destination_county_fips = _pick_first(
        col_map, ["destination_county_fips", "to_county_fips", "countyfips_to"]
    )
    explicit_destination_state = _pick_first(col_map, ["destination_state", "to_state"])
    explicit_destination_county = _pick_first(
        col_map, ["destination_county", "to_county", "destination_county_name", "to_county_name"]
    )

    if all(
        [
            explicit_origin_state_fips,
            explicit_origin_county_fips,
            explicit_destination_state_fips,
            explicit_destination_county_fips,
        ]
    ):
        return {
            "origin_state_fips": explicit_origin_state_fips,
            "origin_county_fips": explicit_origin_county_fips,
            "origin_state": explicit_origin_state,
            "origin_county": explicit_origin_county,
            "destination_state_fips": explicit_destination_state_fips,
            "destination_county_fips": explicit_destination_county_fips,
            "destination_state": explicit_destination_state,
            "destination_county": explicit_destination_county,
            "returns": returns_col,
            "people": people_col,
            "agi": agi_col,
        }

    # IRS county files use y1 as origin and y2 as destination for both
    # inflow and outflow variants; the worksheet perspective differs, not
    # the coordinate semantics.
    origin_prefix = "y1"
    destination_prefix = "y2"

    return {
        "origin_state_fips": _pick_first(col_map, [f"{origin_prefix}_statefips", f"{origin_prefix}statefips"]),
        "origin_county_fips": _pick_first(col_map, [f"{origin_prefix}_countyfips", f"{origin_prefix}countyfips"]),
        "origin_state": _pick_first(col_map, [f"{origin_prefix}_state", f"{origin_prefix}state"]),
        "origin_county": _pick_first(
            col_map,
            [
                f"{origin_prefix}_countyname",
                f"{origin_prefix}countyname",
                f"{origin_prefix}_county",
                f"{origin_prefix}county",
            ],
        ),
        "destination_state_fips": _pick_first(
            col_map, [f"{destination_prefix}_statefips", f"{destination_prefix}statefips"]
        ),
        "destination_county_fips": _pick_first(
            col_map, [f"{destination_prefix}_countyfips", f"{destination_prefix}countyfips"]
        ),
        "destination_state": _pick_first(col_map, [f"{destination_prefix}_state", f"{destination_prefix}state"]),
        "destination_county": _pick_first(
            col_map,
            [
                f"{destination_prefix}_countyname",
                f"{destination_prefix}countyname",
                f"{destination_prefix}_county",
                f"{destination_prefix}county",
            ],
        ),
        "returns": returns_col,
        "people": people_col,
        "agi": agi_col,
    }


def _resolve_positional_mapping(df_raw: pd.DataFrame, direction: str) -> Dict[str, Optional[str]]:
    # Fallback for headerless sheets where first data row became headers.
    data_cols = [c for c in df_raw.columns if normalize_header(c) != "worksheet"]
    if len(data_cols) < 9:
        return {}

    c0, c1, c2, c3, c4, c5, c6, c7, c8 = data_cols[:9]
    if direction == "inflow":
        # inflow order: y2_statefips,y2_countyfips,y1_statefips,y1_countyfips,y1_state,y1_countyname,n1,n2,agi
        return {
            "origin_state_fips": c2,
            "origin_county_fips": c3,
            "origin_state": c4,
            "origin_county": c5,
            "destination_state_fips": c0,
            "destination_county_fips": c1,
            "destination_state": None,
            "destination_county": None,
            "returns": c6,
            "people": c7,
            "agi": c8,
        }

    # outflow order: y1_statefips,y1_countyfips,y2_statefips,y2_countyfips,y2_state,y2_countyname,n1,n2,agi
    return {
        "origin_state_fips": c0,
        "origin_county_fips": c1,
        "origin_state": None,
        "origin_county": None,
        "destination_state_fips": c2,
        "destination_county_fips": c3,
        "destination_state": c4,
        "destination_county": c5,
        "returns": c6,
        "people": c7,
        "agi": c8,
    }


def prepare_migration_dataframe(df_raw: pd.DataFrame, direction: str) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    mapping = _resolve_mapping(df_raw, direction=direction)
    needed = [
        mapping["origin_state_fips"],
        mapping["origin_county_fips"],
        mapping["destination_state_fips"],
        mapping["destination_county_fips"],
        mapping["returns"],
    ]
    if any(col is None for col in needed):
        mapping = _resolve_positional_mapping(df_raw, direction=direction)
        needed = [
            mapping.get("origin_state_fips"),
            mapping.get("origin_county_fips"),
            mapping.get("destination_state_fips"),
            mapping.get("destination_county_fips"),
            mapping.get("returns"),
        ]
    if any(col is None for col in needed):
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "origin_state_fips": _to_fips(_get_series(df_raw, mapping["origin_state_fips"]), 2),
            "origin_county_fips_local": _to_fips(_get_series(df_raw, mapping["origin_county_fips"]), 3),
            "destination_state_fips": _to_fips(_get_series(df_raw, mapping["destination_state_fips"]), 2),
            "destination_county_fips_local": _to_fips(_get_series(df_raw, mapping["destination_county_fips"]), 3),
            "returns": _to_numeric(_get_series(df_raw, mapping["returns"])),
        }
    )

    if mapping["people"]:
        df["people"] = _to_numeric(_get_series(df_raw, mapping["people"]))
    else:
        df["people"] = 0.0

    if mapping["agi"]:
        df["agi"] = _to_numeric(_get_series(df_raw, mapping["agi"]))
    else:
        df["agi"] = 0.0

    if mapping["origin_state"]:
        df["origin_state"] = _get_series(df_raw, mapping["origin_state"]).apply(_clean_string)
    else:
        df["origin_state"] = ""
    if mapping["origin_county"]:
        df["origin_county"] = _get_series(df_raw, mapping["origin_county"]).apply(_clean_string)
    else:
        df["origin_county"] = ""

    if mapping["destination_state"]:
        df["destination_state"] = _get_series(df_raw, mapping["destination_state"]).apply(_clean_string)
    else:
        df["destination_state"] = ""
    if mapping["destination_county"]:
        df["destination_county"] = _get_series(df_raw, mapping["destination_county"]).apply(_clean_string)
    else:
        df["destination_county"] = ""

    df["origin_county_fips"] = df["origin_state_fips"] + df["origin_county_fips_local"]
    df["destination_county_fips"] = df["destination_state_fips"] + df["destination_county_fips_local"]

    df["origin_label"] = (
        df["origin_county"].fillna("").astype(str).str.strip() + ", " + df["origin_state"].fillna("").astype(str).str.strip()
    ).str.strip(", ")
    df["destination_label"] = (
        df["destination_county"].fillna("").astype(str).str.strip()
        + ", "
        + df["destination_state"].fillna("").astype(str).str.strip()
    ).str.strip(", ")

    # Keep only county-to-county migration rows (exclude totals, non-migrants, foreign/special buckets).
    state_min, state_max = 1, 56
    origin_state_num = pd.to_numeric(df["origin_state_fips"], errors="coerce")
    destination_state_num = pd.to_numeric(df["destination_state_fips"], errors="coerce")
    origin_county_num = pd.to_numeric(df["origin_county_fips_local"], errors="coerce")
    destination_county_num = pd.to_numeric(df["destination_county_fips_local"], errors="coerce")

    df = df[
        (df["origin_county_fips"].str.len() == 5)
        & (df["destination_county_fips"].str.len() == 5)
        & origin_state_num.between(state_min, state_max, inclusive="both")
        & destination_state_num.between(state_min, state_max, inclusive="both")
        & origin_county_num.between(1, 999, inclusive="both")
        & destination_county_num.between(1, 999, inclusive="both")
        & (df["origin_county_fips"] != df["destination_county_fips"])
        & (df["returns"] >= 0)
    ].copy()

    return df


def build_county_summary(inflow_df: pd.DataFrame, outflow_df: pd.DataFrame) -> pd.DataFrame:
    county_names_inflow = inflow_df[["origin_county_fips", "origin_state", "origin_county", "origin_label"]].rename(
        columns={
            "origin_county_fips": "county_fips",
            "origin_state": "state_name",
            "origin_county": "county_name",
            "origin_label": "county_label_name",
        }
    )
    county_names_outflow = outflow_df[
        ["destination_county_fips", "destination_state", "destination_county", "destination_label"]
    ].rename(
        columns={
            "destination_county_fips": "county_fips",
            "destination_state": "state_name",
            "destination_county": "county_name",
            "destination_label": "county_label_name",
        }
    )
    county_name_lookup = pd.concat([county_names_inflow, county_names_outflow], ignore_index=True)
    county_name_lookup["state_name"] = county_name_lookup["state_name"].fillna("").astype(str).str.strip()
    county_name_lookup["county_name"] = county_name_lookup["county_name"].fillna("").astype(str).str.strip()
    county_name_lookup["county_label_name"] = county_name_lookup["county_label_name"].fillna("").astype(str).str.strip()
    county_name_lookup = county_name_lookup[
        (county_name_lookup["state_name"] != "") & (county_name_lookup["county_name"] != "")
    ].copy()
    county_name_lookup = county_name_lookup.drop_duplicates(subset=["county_fips"], keep="first")

    inflow_summary = (
        inflow_df.groupby(["destination_county_fips", "destination_state_fips"], as_index=False)
        .agg(inbound_returns=("returns", "sum"), inbound_people=("people", "sum"), inbound_agi=("agi", "sum"))
        .rename(
            columns={
                "destination_county_fips": "county_fips",
                "destination_state_fips": "state_fips",
            }
        )
    )

    outflow_summary = (
        outflow_df.groupby(["origin_county_fips", "origin_state_fips"], as_index=False)
        .agg(outbound_returns=("returns", "sum"), outbound_people=("people", "sum"), outbound_agi=("agi", "sum"))
        .rename(
            columns={
                "origin_county_fips": "county_fips",
                "origin_state_fips": "state_fips",
            }
        )
    )

    county = inflow_summary.merge(outflow_summary, on=["county_fips", "state_fips"], how="outer")
    county = county.merge(county_name_lookup, on="county_fips", how="left")
    county["state"] = county["state_name"].fillna("")
    county["county"] = county["county_name"].fillna("")
    county["county_label"] = county["county_label_name"].fillna("")
    for col in [
        "inbound_returns",
        "inbound_people",
        "inbound_agi",
        "outbound_returns",
        "outbound_people",
        "outbound_agi",
    ]:
        if col not in county.columns:
            county[col] = 0.0
        county[col] = county[col].fillna(0.0)

    county["net_returns"] = county["inbound_returns"] - county["outbound_returns"]
    county["net_people"] = county["inbound_people"] - county["outbound_people"]
    county["net_agi"] = county["inbound_agi"] - county["outbound_agi"]
    county["migration_volume"] = county["inbound_returns"] + county["outbound_returns"]
    county["agi_moved"] = county["inbound_agi"] + county["outbound_agi"]
    county["county_label"] = county["county_label"].replace("", pd.NA).fillna(county["county"] + ", " + county["state"])
    county = county.drop(columns=["state_name", "county_name", "county_label_name"])

    return county.sort_values("migration_volume", ascending=False).reset_index(drop=True)


def build_state_summary(county_summary: pd.DataFrame) -> pd.DataFrame:
    if county_summary.empty:
        return pd.DataFrame()
    state = (
        county_summary.groupby(["state_fips", "state"], as_index=False)
        .agg(
            inbound_returns=("inbound_returns", "sum"),
            inbound_people=("inbound_people", "sum"),
            inbound_agi=("inbound_agi", "sum"),
            outbound_returns=("outbound_returns", "sum"),
            outbound_people=("outbound_people", "sum"),
            outbound_agi=("outbound_agi", "sum"),
            county_count=("county_fips", "nunique"),
        )
        .sort_values("inbound_returns", ascending=False)
    )
    state["net_returns"] = state["inbound_returns"] - state["outbound_returns"]
    state["net_people"] = state["inbound_people"] - state["outbound_people"]
    state["net_agi"] = state["inbound_agi"] - state["outbound_agi"]
    state["migration_volume"] = state["inbound_returns"] + state["outbound_returns"]
    state["agi_moved"] = state["inbound_agi"] + state["outbound_agi"]
    return state.reset_index(drop=True)
