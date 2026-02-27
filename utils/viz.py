from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


STATE_ABBREV: Dict[str, str] = {
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


def _latest_year(df: pd.DataFrame) -> Optional[int]:
    if "year" not in df.columns:
        return None
    years = pd.to_numeric(df["year"], errors="coerce").dropna()
    if years.empty:
        return None
    return int(years.max())


def kpi_row(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No data for KPI summary.")
        return

    latest = _latest_year(df)
    df2 = df.copy()
    if latest:
        df2 = df2[df2["year"] == latest]

    df2 = df2.dropna(subset=["value"])
    if df2.empty:
        st.info("No data for KPI summary.")
        return

    avg = df2["value"].mean()
    max_row = df2.loc[df2["value"].idxmax()]
    min_row = df2.loc[df2["value"].idxmin()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average", f"{avg:,.2f}")
    c2.metric("Highest", f"{max_row['state']} - {max_row['value']:,.2f}")
    c3.metric("Lowest", f"{min_row['state']} - {min_row['value']:,.2f}")


def time_series(df: pd.DataFrame, metric: Optional[str]) -> px.line:
    df2 = df.copy()
    if metric:
        df2 = df2[df2["metric"] == metric]

    if "year" not in df2.columns or df2["year"].isna().all():
        df2["year"] = "All"

    if df2["state"].nunique() <= 10:
        fig = px.line(df2, x="year", y="value", color="state", markers=True, title="Trend by state")
    else:
        avg = df2.groupby("year", as_index=False)["value"].mean()
        fig = px.line(avg, x="year", y="value", markers=True, title="Average trend")

    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig


def ranking_bar(df: pd.DataFrame, metric: Optional[str]) -> px.bar:
    df2 = df.copy()
    if metric:
        df2 = df2[df2["metric"] == metric]

    latest = _latest_year(df2)
    if latest:
        df2 = df2[df2["year"] == latest]

    df2 = df2.dropna(subset=["value"])
    df2 = df2.sort_values("value", ascending=False).head(15)

    fig = px.bar(df2, x="value", y="state", orientation="h", title="Top states")
    fig.update_layout(xaxis_title="", yaxis_title="")
    return fig


def choropleth_map(df: pd.DataFrame, metric: Optional[str]) -> px.choropleth:
    df2 = df.copy()
    if metric:
        df2 = df2[df2["metric"] == metric]

    latest = _latest_year(df2)
    if latest:
        df2 = df2[df2["year"] == latest]

    df2 = df2.dropna(subset=["value"])
    df2 = df2[["state", "value"]].copy()

    def to_abbrev(s: str) -> Optional[str]:
        if not isinstance(s, str):
            return None
        s_clean = s.strip().lower()
        if len(s_clean) == 2:
            return s_clean.upper()
        return STATE_ABBREV.get(s_clean)

    df2["state_abbrev"] = df2["state"].apply(to_abbrev)
    missing = df2["state_abbrev"].isna().mean()
    if missing > 0.4:
        st.warning("Map needs state abbreviations. Please add a State_Abbrev column or use abbreviations.")
        return px.scatter_geo()

    fig = px.choropleth(
        df2,
        locations="state_abbrev",
        locationmode="USA-states",
        scope="usa",
        color="value",
        hover_name="state",
        title="Latest values by state",
    )
    fig.update_layout(margin={"r": 0, "l": 0, "t": 40, "b": 0})
    return fig
