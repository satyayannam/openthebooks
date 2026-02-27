# OpenTheBooks Streamlit App
#
# README (Quick Start)
# 1) Secrets setup (Streamlit)
#    st.secrets["gcp_service_account"] or st.secrets["google_service_account_json"]
#    st.secrets["SHEET_ID"]
#    Optional: st.secrets["WORKSHEET_NAMES"] = "housing,income,health"
#
# 2) Local run
#    pip install streamlit pandas numpy plotly gspread google-auth statsmodels
#    streamlit run app.py
#
# 3) Streamlit Cloud
#    Add secrets in the Streamlit Cloud UI (same keys as above)
#    Deploy from your repo, set main file to app.py

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.theme import apply_theme
from utils.sheets import load_data
from utils.housing import detect_housing_schema, render_housing_dashboard


apply_theme()

st.title("OpenTheBooks")
st.caption("Transparency. Accountability. Public Trust.")
st.markdown("**In data we trust.**")

worksheet_list = []
try:
    worksheet_list = [
        w.strip()
        for w in str(st.secrets.get("WORKSHEET_NAMES", "") or "").split(",")
        if w.strip()
    ]
except Exception:
    worksheet_list = []

looker_sheet = "Looker_Source"
if worksheet_list and looker_sheet not in worksheet_list:
    st.sidebar.warning(f"Primary source '{looker_sheet}' not listed in WORKSHEET_NAMES.")

refresh = st.sidebar.button("Refresh data")
st.sidebar.caption(f"Primary dashboard source: {looker_sheet}")

df_looker = load_data(refresh=refresh, worksheets=[looker_sheet])
if df_looker.empty:
    st.warning("No data found. Check your Sheet ID, permissions, and worksheet names.")
    st.stop()

def _to_ratio(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("%", "", regex=False).str.strip()
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return s
    if s.dropna().median() > 1.5:
        s = s / 100.0
    return s


def _briefing(df: pd.DataFrame) -> tuple[dict, list[str], pd.DataFrame]:
    schema = detect_housing_schema(df)
    needed = {"state", "hpi", "income", "gap"}
    if not needed.issubset(schema.keys()):
        return {}, ["Schema mismatch in Looker_Source. Check expected column names."], pd.DataFrame()

    clean = pd.DataFrame(
        {
            "state": df[schema["state"]],
            "hpi_growth": _to_ratio(df[schema["hpi"]]),
            "income_growth": _to_ratio(df[schema["income"]]),
            "gap": _to_ratio(df[schema["gap"]]),
        }
    )
    clean = clean.dropna(subset=["state", "gap"])
    clean = clean[clean["state"].astype(str).str.lower().ne("grand total")]
    if clean.empty:
        return {}, ["Looker_Source loaded, but no valid state and gap rows were found."], clean

    top = clean.loc[clean["gap"].idxmax()]
    bottom = clean.loc[clean["gap"].idxmin()]
    severe_count = int((clean["gap"] >= 0.70).sum())
    high_count = int((clean["gap"] >= 0.50).sum())
    negative_count = int((clean["gap"] < 0).sum())

    stats = {
        "rows": int(len(clean)),
        "avg_gap": float(clean["gap"].mean()),
        "median_gap": float(clean["gap"].median()),
        "avg_hpi": float(clean["hpi_growth"].mean()),
        "avg_income": float(clean["income_growth"].mean()),
        "worst_state": str(top["state"]),
        "worst_gap": float(top["gap"]),
        "best_state": str(bottom["state"]),
        "best_gap": float(bottom["gap"]),
    }

    notes = [
        f"Highest pressure state: {stats['worst_state']} ({stats['worst_gap']:.1%} gap).",
        f"Lowest pressure state: {stats['best_state']} ({stats['best_gap']:.1%} gap).",
        f"States at or above 50% gap: {high_count}; severe states (70%+): {severe_count}.",
    ]
    if negative_count > 0:
        notes.append(f"{negative_count} state(s) show negative affordability gap values.")
    notes.append(
        f"Average housing growth {stats['avg_hpi']:.1%} vs income growth {stats['avg_income']:.1%}."
    )
    return stats, notes, clean


brief_stats, brief_notes, brief_df = _briefing(df_looker)

st.markdown("### Executive Brief")
if brief_stats:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Average Gap", f"{brief_stats['avg_gap']:.1%}")
    k2.metric("Median Gap", f"{brief_stats['median_gap']:.1%}")
    k3.metric("Worst State", f"{brief_stats['worst_state']} ({brief_stats['worst_gap']:.1%})")
    k4.metric("Best State", f"{brief_stats['best_state']} ({brief_stats['best_gap']:.1%})")
for note in brief_notes:
    st.markdown(f"- {note}")
st.caption(f"Live source: {looker_sheet} | Loaded rows: {len(df_looker)}")

tabs = st.tabs(["Dashboard", "HPI Pivot", "Income Pivot", "Looker Data", "Debug"])

with tabs[0]:
    st.subheader("Housing Affordability Dashboard")
    st.caption("Interactive visuals sourced only from Looker_Source.")
    housing_schema = detect_housing_schema(df_looker)
    if {"state", "hpi", "income", "gap"}.issubset(housing_schema.keys()):
        render_housing_dashboard(df_looker)
    else:
        st.error("Looker_Source schema mismatch. Expected State/HPI_Growth/Income_Growth/Affordability_Gap.")
        st.write({"detected_schema": housing_schema})
        st.dataframe(df_looker.head(20), use_container_width=True)


def _render_pivot_tab(sheet_name: str, value_label: str) -> None:
    df_pivot = load_data(refresh=refresh, worksheets=[sheet_name])
    if df_pivot.empty:
        st.info(f"No data found for {sheet_name}.")
        return

    st.write({"rows": len(df_pivot), "columns": len(df_pivot.columns)})
    st.dataframe(df_pivot.head(50), use_container_width=True)

    state_col = "State" if "State" in df_pivot.columns else ("state" if "state" in df_pivot.columns else None)
    year_col = "Year" if "Year" in df_pivot.columns else ("year" if "year" in df_pivot.columns else None)
    long_df = pd.DataFrame()
    metric_candidates = []

    if state_col and year_col:
        tmp = df_pivot.copy()
        tmp[year_col] = pd.to_numeric(tmp[year_col], errors="coerce")
        ignore_cols = {state_col, year_col, "worksheet"}
        for col in tmp.columns:
            if col in ignore_cols:
                continue
            vals = pd.to_numeric(tmp[col], errors="coerce")
            if vals.notna().sum() > 0:
                metric_candidates.append(col)
                tmp[col] = vals
        if metric_candidates:
            metric_col = st.selectbox(
                f"Metric ({sheet_name})",
                metric_candidates,
                index=0,
                key=f"{sheet_name}_metric",
            )
            long_df = tmp[[state_col, year_col, metric_col]].rename(
                columns={state_col: "state", year_col: "year", metric_col: "value"}
            )
    else:
        year_cols = [c for c in df_pivot.columns if isinstance(c, str) and c.isdigit() and len(c) == 4]
        if state_col and year_cols:
            long_df = df_pivot.melt(id_vars=[state_col], value_vars=year_cols, var_name="year", value_name="value")
            long_df = long_df.rename(columns={state_col: "state"})
            long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
            long_df["value"] = (
                long_df["value"].astype(str).str.replace("%", "", regex=False).str.strip()
            )
            long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    if long_df.empty:
        st.caption("No year-wise chart available for this table.")
        return

    long_df = long_df[long_df["state"].astype(str).str.lower().ne("grand total")]
    long_df = long_df.dropna(subset=["state", "year", "value"])
    if long_df.empty:
        st.caption("No plottable year-wise values in this table.")
        return

    is_ratio = long_df["value"].median() < 2 and long_df["value"].max() <= 5
    yfmt = ".0%" if is_ratio else ",.2f"

    st.markdown("#### Pivot Analytics")
    state_options = sorted(long_df["state"].astype(str).unique().tolist())
    default_states = state_options[:8]
    selected_states = st.multiselect(
        f"States ({sheet_name})",
        state_options,
        default=default_states,
        key=f"{sheet_name}_states",
    )
    chart_df = long_df[long_df["state"].isin(selected_states)] if selected_states else long_df

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(
            chart_df.sort_values("year"),
            x="year",
            y="value",
            color="state",
            markers=True,
            title=f"{sheet_name}: trend by state",
        )
        fig.update_yaxes(tickformat=yfmt)
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        latest_year = int(chart_df["year"].max())
        latest = chart_df[chart_df["year"] == latest_year].sort_values("value", ascending=False).head(15)
        fig = px.bar(
            latest,
            x="value",
            y="state",
            orientation="h",
            title=f"{sheet_name}: top states in {latest_year}",
        )
        fig.update_xaxes(tickformat=yfmt)
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.box(
            chart_df,
            x="year",
            y="value",
            points=False,
            title=f"{sheet_name}: yearly distribution",
        )
        fig.update_yaxes(tickformat=yfmt)
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        heat = chart_df.pivot_table(index="state", columns="year", values="value", aggfunc="mean")
        if not heat.empty:
            fig = px.imshow(
                heat.sort_index(),
                aspect="auto",
                title=f"{sheet_name}: state-year heatmap",
                labels={"x": "year", "y": "state", "color": "value"},
            )
            st.plotly_chart(fig, use_container_width=True)


with tabs[1]:
    st.subheader("Pivot Table 2 (HPI History)")
    st.caption("Use this for state-wise yearly housing price growth trends.")
    _render_pivot_tab("Pivot Table 2", "hpi")

with tabs[2]:
    st.subheader("Pivot Table 3 (Income History)")
    st.caption("Use this for state-wise yearly income growth trends.")
    _render_pivot_tab("Pivot Table 3", "income")

with tabs[3]:
    st.subheader("Looker Source Data")
    st.caption("Raw base table feeding the main dashboard visuals.")
    st.write({"rows": len(df_looker), "columns": len(df_looker.columns)})
    st.dataframe(df_looker, use_container_width=True)

with tabs[4]:
    st.subheader("Debug")
    st.caption("Diagnostics for schema detection and source loading.")
    st.write(
        {
            "worksheet_names_secret": str(st.secrets.get("WORKSHEET_NAMES", "") or ""),
            "primary_source": looker_sheet,
            "looker_rows": len(df_looker),
            "looker_columns": list(df_looker.columns),
            "detected_housing_schema": detect_housing_schema(df_looker),
            "brief_rows_used": int(len(brief_df)),
        }
    )

st.markdown("---")
st.caption("In data we trust. Sources: see Data Explorer.")
