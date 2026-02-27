from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class Schema:
    state: Optional[str]
    year: Optional[str]
    metric: Optional[str]
    value: Optional[str]
    domain: Optional[str]
    format: str


def _find_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _guess_year_column(columns: List[str], df: pd.DataFrame) -> Optional[str]:
    year_col = _find_column(columns, ["year", "fy", "fiscal year"])
    if year_col:
        return year_col
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna().astype(int)
            if values.empty:
                continue
            if values.between(1900, 2100).mean() > 0.8:
                return col
    return None


def infer_schema(df: pd.DataFrame) -> Schema:
    cols = [c.strip() for c in df.columns]
    df = df.copy()
    df.columns = cols

    state_col = _find_column(cols, ["state", "state name", "state_name", "location", "geo"])
    year_col = _guess_year_column(cols, df)
    metric_col = _find_column(cols, ["metric", "indicator", "measure", "series"])
    value_col = _find_column(cols, ["value", "amount", "total", "count"])
    domain_col = _find_column(cols, ["domain", "category", "group", "area", "topic", "worksheet"])

    if metric_col and value_col:
        fmt = "long"
    else:
        fmt = "wide"

    return Schema(
        state=state_col,
        year=year_col,
        metric=metric_col,
        value=value_col,
        domain=domain_col,
        format=fmt,
    )


def normalize_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Schema]:
    if df.empty:
        return df, infer_schema(df)

    schema = infer_schema(df)
    df2 = df.copy()
    df2.columns = [c.strip() for c in df2.columns]

    if schema.format == "long" and schema.metric and schema.value:
        long_df = df2.rename(
            columns={
                schema.state or "": "state",
                schema.year or "": "year",
                schema.metric: "metric",
                schema.value: "value",
                schema.domain or "": "domain",
            }
        )
        if "state" not in long_df.columns:
            long_df["state"] = "All"
        if "year" not in long_df.columns:
            long_df["year"] = np.nan
        if "domain" not in long_df.columns:
            long_df["domain"] = "All"
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        return long_df, schema

    # Wide format: melt numeric columns into metric/value
    schema.metric = "metric"
    schema.value = "value"
    schema.format = "wide"

    if schema.state and schema.state in df2.columns:
        id_vars = [schema.state]
    else:
        id_vars = []
        df2["state"] = "All"
        schema.state = "state"
        id_vars = ["state"]

    if schema.year and schema.year in df2.columns:
        id_vars.append(schema.year)

    if schema.domain and schema.domain in df2.columns:
        id_vars.append(schema.domain)

    numeric_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    if schema.year in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != schema.year]

    if not numeric_cols:
        candidate_cols = [c for c in df2.columns if c not in id_vars]
        for col in candidate_cols:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
        numeric_cols = candidate_cols

    long_df = df2.melt(id_vars=id_vars, value_vars=numeric_cols, var_name="metric", value_name="value")

    long_df = long_df.rename(columns={schema.state: "state"})
    if schema.year and schema.year in long_df.columns:
        long_df = long_df.rename(columns={schema.year: "year"})
    else:
        long_df["year"] = np.nan

    if schema.domain and schema.domain in long_df.columns:
        long_df = long_df.rename(columns={schema.domain: "domain"})
    else:
        long_df["domain"] = "All"

    return long_df, schema


def render_global_filters() -> bool:
    st.sidebar.header("Global Filters")

    refresh = st.sidebar.button("Refresh data")

    view_modes = [
        "Raw",
        "Indexed (First Year=100)",
        "Z-Score",
        "Gap(A-B)",
        "Ratio(A/B)",
    ]
    current = st.session_state.get("view_mode", "Raw")
    view_mode = st.sidebar.selectbox("View mode", view_modes, index=view_modes.index(current))
    st.session_state["view_mode"] = view_mode

    return refresh


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    long_df, _schema = normalize_dataframe(df)

    long_df = long_df.dropna(subset=["value"])

    # Metrics selector
    metrics = sorted([m for m in long_df["metric"].dropna().unique().tolist()])
    st.session_state["metrics"] = metrics
    if metrics:
        current_metric = st.session_state.get("metric_selector", metrics[0])
        if current_metric not in metrics:
            current_metric = metrics[0]
        st.session_state["metric_selector"] = st.sidebar.selectbox("Metric", metrics, index=metrics.index(current_metric))

    # State filter
    states = sorted([s for s in long_df["state"].dropna().unique().tolist()])
    if states:
        default_states = st.session_state.get("state_filter", states)
        selected_states = st.sidebar.multiselect("States", states, default=default_states)
        st.session_state["state_filter"] = selected_states
        long_df = long_df[long_df["state"].isin(selected_states)]

    # Domain filter
    domains = sorted([d for d in long_df["domain"].dropna().unique().tolist()])
    if len(domains) > 1:
        default_domains = st.session_state.get("domain_filter", domains)
        selected_domains = st.sidebar.multiselect("Domains", domains, default=default_domains)
        st.session_state["domain_filter"] = selected_domains
        long_df = long_df[long_df["domain"].isin(selected_domains)]

    # Year filter
    if "year" in long_df.columns:
        years = long_df["year"].dropna()
        if not years.empty:
            years = pd.to_numeric(years, errors="coerce").dropna()
            if not years.empty:
                y_min = int(years.min())
                y_max = int(years.max())
                default_years = st.session_state.get("year_filter", (y_min, y_max))
                year_range = st.sidebar.slider("Year range", y_min, y_max, default_years)
                st.session_state["year_filter"] = year_range
                long_df = long_df[(long_df["year"] >= year_range[0]) & (long_df["year"] <= year_range[1])]

    return long_df
