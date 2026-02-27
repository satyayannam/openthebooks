from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.viz import STATE_ABBREV
from utils.sheets import list_worksheets, load_data


REQUIRED_COLS = {
    "state": ["State", "state"],
    "hpi": ["HPI_Growth", "HPI Growth", "HPI_GROWTH"],
    "income": ["Income_Growth", "Income Growth", "INCOME_GROWTH", "Income_Grow"],
    "gap": ["Affordability_Gap", "Affordability Gap", "GAP"],
    "rank": ["Affordability_Gap_Rank", "Gap_Rank", "Rank"],
    "bucket": ["Gap_Bucket", "Gap Bucket", "Bucket"],
}


def _pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    candidates = []
    for name in names:
        key = name.lower()
        if key in cols:
            candidates.append(cols[key])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the column with the most non-empty values.
    best = None
    best_count = -1
    for col in candidates:
        series = df[col].replace("", np.nan)
        count = series.notna().sum()
        if count > best_count:
            best_count = count
            best = col
    return best


def detect_housing_schema(df: pd.DataFrame) -> Dict[str, str]:
    schema: Dict[str, str] = {}
    for key, names in REQUIRED_COLS.items():
        col = _pick_col(df, names)
        if col:
            schema[key] = col
    return schema


def _coerce_ratio(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.replace("%", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    if s.dropna().median() > 1.5:
        s = s / 100.0
    return s


def _prep(df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2.rename(
        columns={
            schema.get("state", ""): "state",
            schema.get("hpi", ""): "hpi_growth",
            schema.get("income", ""): "income_growth",
            schema.get("gap", ""): "gap",
            schema.get("rank", ""): "gap_rank",
            schema.get("bucket", ""): "gap_bucket",
        }
    )

    def _coalesce_dup(col: str) -> None:
        if col not in df2.columns:
            return
        cols = df2.loc[:, df2.columns == col]
        if cols.shape[1] <= 1:
            return
        combined = cols.replace("", np.nan).bfill(axis=1).iloc[:, 0]
        df2.drop(columns=cols.columns, inplace=True)
        df2[col] = combined

    for col in ["state", "hpi_growth", "income_growth", "gap", "gap_rank", "gap_bucket"]:
        _coalesce_dup(col)

    for col in ["hpi_growth", "income_growth", "gap"]:
        if col in df2.columns:
            df2[col] = _coerce_ratio(df2[col])

    if "gap_rank" in df2.columns:
        df2["gap_rank"] = pd.to_numeric(df2["gap_rank"], errors="coerce")

    if "gap_bucket" not in df2.columns:
        def bucket(g):
            if pd.isna(g):
                return None
            if g < 0.30:
                return "Lower (<30%)"
            if g < 0.50:
                return "Moderate (30-49%)"
            if g < 0.70:
                return "High (50-69%)"
            return "Severe (70%+)"
        df2["gap_bucket"] = df2["gap"].apply(bucket)

    return df2


def _state_abbrev(state: str) -> str | None:
    if not isinstance(state, str):
        return None
    s = state.strip()
    if len(s) == 2:
        return s.upper()
    return STATE_ABBREV.get(s.lower())


def _normalize_state(state: str) -> str | None:
    abbr = _state_abbrev(state)
    if abbr:
        return abbr
    return state.strip() if isinstance(state, str) else None


def _year_columns(columns: list[str]) -> list[str]:
    years = []
    for col in columns:
        if isinstance(col, str) and col.isdigit() and len(col) == 4:
            years.append(col)
    return years


def _pivot_to_long(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if df.empty:
        return df

    cols = list(df.columns)
    state_col = None
    for cand in ["State", "state", "STATE"]:
        if cand in cols:
            state_col = cand
            break
    if state_col is None:
        state_col = cols[0]

    year_cols = _year_columns(cols)
    if not year_cols and {"Year", "year"}.intersection(set(cols)):
        # Already long format
        year_col = "Year" if "Year" in cols else "year"
        val_col = value_name if value_name in cols else None
        if val_col is None:
            candidates = [c for c in cols if c not in {state_col, year_col}]
            if not candidates:
                return pd.DataFrame()
            val_col = candidates[0]
        long_df = df[[state_col, year_col, val_col]].copy()
        long_df = long_df.rename(columns={state_col: "state", year_col: "year", val_col: value_name})
        long_df["state"] = long_df["state"].replace("", np.nan).ffill()
        long_df["state"] = long_df["state"].apply(_normalize_state)
        long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
        long_df[value_name] = _coerce_ratio(long_df[value_name])
        return long_df

    if not year_cols:
        return pd.DataFrame()

    long_df = df.melt(id_vars=[state_col], value_vars=year_cols, var_name="year", value_name=value_name)
    long_df = long_df.rename(columns={state_col: "state"})
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df[value_name] = _coerce_ratio(long_df[value_name])
    long_df["state"] = long_df["state"].replace("", np.nan).ffill()
    long_df["state"] = long_df["state"].apply(_normalize_state)
    return long_df


def _palette(name: str) -> list[str]:
    palettes = {
        "Ocean": ["#7dd3fc", "#22d3ee", "#14b8a6", "#0ea5e9", "#38bdf8"],
        "Sunset": ["#fbbf24", "#f97316", "#ef4444", "#f43f5e", "#fb7185"],
        "Mono": ["#e5e7eb", "#9ca3af", "#6b7280", "#4b5563", "#374151"],
    }
    return palettes.get(name, palettes["Ocean"])


def render_housing_dashboard(df: pd.DataFrame) -> None:
    schema = detect_housing_schema(df)
    needed = {"state", "hpi", "income", "gap"}
    if "State" in df.columns:
        state_series = df["State"].replace("", np.nan)
        if state_series.notna().any():
            schema["state"] = "State"
    if not needed.issubset(schema.keys()):
        st.info("Housing dashboard needs State, HPI_Growth, Income_Growth, and Affordability_Gap columns.")
        return

    data = _prep(df, schema)
    if "worksheet" in data.columns:
        state_series = data["state"].replace("", np.nan)
        gap_series = data["gap"].replace("", np.nan)
        scored = (
            data.assign(
                _state_ok=state_series.notna(),
                _gap_ok=gap_series.notna(),
            )
            .groupby("worksheet", dropna=False)[["_state_ok", "_gap_ok"]]
            .sum()
        )
        scored["score"] = scored["_state_ok"] + scored["_gap_ok"]
        if not scored.empty:
            best_ws = scored.sort_values("score", ascending=False).index[0]
            data = data[data["worksheet"] == best_ws].copy()
    data = data.dropna(subset=["state", "gap"])
    if data.empty:
        st.warning("No rows with valid State and Affordability_Gap values after cleaning.")
        return

    st.sidebar.header("Dashboard")
    with st.sidebar.expander("Sections", expanded=True):
        show_compare = st.checkbox("Comparative analysis", value=True)
        show_ranking = st.checkbox("Gap ranking", value=True)
        show_map = st.checkbox("Map", value=True)
        show_table = st.checkbox("Data table", value=True)
        show_yearly = st.checkbox("Year-wise analysis", value=True)

    with st.sidebar.expander("Style", expanded=False):
        palette_name = st.selectbox("Color palette", ["Ocean", "Sunset", "Mono"], index=0)
        top_n = st.slider("Top N states", min_value=5, max_value=30, value=15, step=1)

    palette = _palette(palette_name)

    st.subheader("Overview")

    avg_gap = data["gap"].mean()
    med_gap = data["gap"].median()
    top_row = data.loc[data["gap"].idxmax()]
    low_row = data.loc[data["gap"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average gap", f"{avg_gap:.1%}")
    c2.metric("Median gap", f"{med_gap:.1%}")
    c3.metric("Worst gap", f"{top_row['state']} - {top_row['gap']:.1%}")
    c4.metric("Best gap", f"{low_row['state']} - {low_row['gap']:.1%}")

    if show_compare:
        st.markdown("### Comparative analysis")
        left, right = st.columns([1.3, 1.0])

        with left:
            melted = data.melt(
                id_vars=["state", "gap", "gap_bucket"],
                value_vars=["hpi_growth", "income_growth"],
                var_name="metric",
                value_name="growth",
            )
            label_map = {"hpi_growth": "Housing price growth", "income_growth": "Income growth"}
            melted["metric"] = melted["metric"].map(label_map)
            fig = px.bar(
                melted.sort_values("gap", ascending=False),
                x="state",
                y="growth",
                color="metric",
                barmode="group",
                title="Housing vs Income growth by state",
                hover_data={"gap": ":.2%", "gap_bucket": True},
                color_discrete_sequence=palette,
            )
            fig.update_yaxes(tickformat=".0%")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.scatter(
                data,
                x="income_growth",
                y="hpi_growth",
                color="gap_bucket",
                hover_name="state",
                title="Income vs Housing growth",
                color_discrete_sequence=palette,
            )
            fig.update_xaxes(tickformat=".0%", title="Income growth")
            fig.update_yaxes(tickformat=".0%", title="Housing growth")
            st.plotly_chart(fig, use_container_width=True)

    if show_ranking:
        st.markdown("### Gap ranking")
        left2, right2 = st.columns([1.0, 1.0])

        with left2:
            ranked = data.sort_values("gap", ascending=False).head(top_n)
            fig = px.bar(
                ranked,
                x="gap",
                y="state",
                orientation="h",
                color="gap_bucket",
                title="Affordability gap (higher is worse)",
                color_discrete_sequence=palette,
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

        with right2:
            bucket_counts = data["gap_bucket"].value_counts().reset_index()
            bucket_counts.columns = ["gap_bucket", "count"]
            fig = px.bar(
                bucket_counts,
                x="gap_bucket",
                y="count",
                title="States by severity bucket",
                color_discrete_sequence=palette,
            )
            fig.update_layout(xaxis_title="", yaxis_title="States")
            st.plotly_chart(fig, use_container_width=True)

    if show_map:
        st.markdown("### Map (latest values)")
        df_map = data.copy()
        df_map["state_abbrev"] = df_map["state"].apply(_state_abbrev)
        if df_map["state_abbrev"].isna().mean() < 0.4:
            fig = px.choropleth(
                df_map,
                locations="state_abbrev",
                locationmode="USA-states",
                scope="usa",
                color="gap",
                hover_name="state",
                hover_data={"gap": ":.1%"},
                title="Affordability gap by state",
                color_continuous_scale=px.colors.sequential.Tealgrn,
            )
            fig.update_coloraxes(colorbar_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map needs state abbreviations (FL, ID, etc.). Add a State_Abbrev column if you have full names.")

    if show_yearly:
        st.markdown("### Year-wise analysis")
        try:
            sheet_id = str(st.secrets.get("SHEET_ID", ""))
        except Exception:
            sheet_id = ""

        if sheet_id:
            available = list_worksheets(sheet_id)
        else:
            available = []

        default_hpi = "Pivot Table 2" if "Pivot Table 2" in available else (available[0] if available else "")
        default_income = "Pivot Table 3" if "Pivot Table 3" in available else (available[1] if len(available) > 1 else default_hpi)

        hpi_sheet = st.selectbox("HPI yearly sheet", available, index=available.index(default_hpi) if default_hpi in available else 0)
        income_sheet = st.selectbox("Income yearly sheet", available, index=available.index(default_income) if default_income in available else 0)

        if hpi_sheet and income_sheet:
            df_hpi = load_data(worksheets=hpi_sheet)
            df_income = load_data(worksheets=income_sheet)
            long_hpi = _pivot_to_long(df_hpi, "hpi")
            long_income = _pivot_to_long(df_income, "income")

            if not long_hpi.empty and not long_income.empty:
                yearly = pd.merge(long_hpi, long_income, on=["state", "year"], how="inner")
                yearly = yearly[yearly["state"].str.lower().ne("grand total")]
                yearly = yearly[yearly["year"].notna()]
                yearly["gap"] = yearly["hpi"] - yearly["income"]

                years = sorted(yearly["year"].dropna().unique().tolist())
                if years:
                    year_choice = st.selectbox("Year", years, index=len(years) - 1)
                    metric_choice = st.selectbox("Rank by", ["gap", "hpi", "income"], index=0)

                    yr_df = yearly[yearly["year"] == year_choice].copy()
                    yr_df = yr_df.sort_values(metric_choice, ascending=False).head(top_n)

                    fig = px.bar(
                        yr_df,
                        x=metric_choice,
                        y="state",
                        orientation="h",
                        title=f"Top {top_n} states by {metric_choice.upper()} in {int(year_choice)}",
                        color_discrete_sequence=palette,
                    )
                    fig.update_xaxes(tickformat=".0%")
                    fig.update_layout(xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

                    state_pick = st.multiselect("Compare states", sorted(yearly["state"].unique()), default=sorted(yearly["state"].unique())[:5])
                    if state_pick:
                        trend = yearly[yearly["state"].isin(state_pick)].copy()
                        fig = px.line(
                            trend,
                            x="year",
                            y="gap",
                            color="state",
                            title="Gap trend over time",
                            markers=True,
                            color_discrete_sequence=palette,
                        )
                        fig.update_yaxes(tickformat=".0%")
                        fig.update_layout(xaxis_title="", yaxis_title="")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Yearly sheets must have a State column and year columns like 2015, 2016, 2017.")
        else:
            st.info("Select yearly sheets from the dropdowns.")

    if show_table:
        st.markdown("### Data table")
        st.dataframe(data.sort_values("gap", ascending=False), use_container_width=True)
