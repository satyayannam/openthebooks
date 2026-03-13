from __future__ import annotations

import json
from typing import Iterable, Optional
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.sheets import list_worksheets, load_data
from utils.tax_stats import build_county_summary, build_state_summary, prepare_tax_stats_dataframe, to_csv_bytes
from utils.theme import apply_theme


WORKSHEET_CANDIDATES = [
    "irs_tax",
    "IRS_TAX",
    "22incyallagi",
    "irs_tax_statistics_by_county",
    "irs_tax_stats",
    "tax_stats_county",
]

PRIMARY_COLOR = "#2C7FB8"
WAGE_COLOR = "#2E8B57"
INVESTMENT_COLOR = "#8E44AD"
COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"


def _matches_tokens(name: str, tokens: Iterable[str]) -> bool:
    text = str(name).strip().lower()
    return all(t in text for t in tokens)


def _resolve_tax_worksheet(available: list[str]) -> list[str]:
    token_hits = [
        w
        for w in available
        if _matches_tokens(w, ["22", "agi"])
        or _matches_tokens(w, ["tax", "county"])
        or _matches_tokens(w, ["irs", "tax"])
    ]
    ordered = token_hits + WORKSHEET_CANDIDATES
    out = []
    seen = set()
    for name in ordered:
        key = str(name).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(name)
    return out


@st.cache_data(ttl=300)
def _load_with_candidates(refresh: bool, worksheet_names: tuple[str, ...]) -> tuple[pd.DataFrame, Optional[str]]:
    if not worksheet_names:
        return pd.DataFrame(), None
    for idx, ws_name in enumerate(worksheet_names):
        try:
            df = load_data(refresh=refresh and idx == 0, worksheets=[ws_name])
            if not df.empty:
                return df, ws_name
        except Exception:
            continue
    return pd.DataFrame(), None


def _top_bar(df: pd.DataFrame, metric: str, title: str, n: int = 15, subtitle: str = "") -> None:
    if df.empty:
        st.info("No data available for this view.")
        return
    top = df.sort_values(metric, ascending=False).head(n)
    fig = px.bar(
        top.sort_values(metric, ascending=True),
        x=metric,
        y="county_label" if "county_label" in top.columns else "state",
        orientation="h",
        title=title,
        color_discrete_sequence=[PRIMARY_COLOR],
    )
    fig.update_layout(xaxis_title="", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    if subtitle:
        st.caption(subtitle)


@st.cache_data(ttl=3600)
def _load_county_geojson() -> Optional[dict]:
    try:
        with urlopen(COUNTY_GEOJSON_URL, timeout=15) as resp:
            return json.load(resp)
    except Exception:
        return None


apply_theme()

st.title("IRS Tax Statistics by County")
st.caption("County-level IRS return and income structure analysis from the connected Google Sheet.")

refresh = st.sidebar.button("Refresh data")
sheet_id = str(st.secrets.get("SHEET_ID", "") or "")
available_worksheets = list_worksheets(sheet_id) if sheet_id else []
worksheet_order = _resolve_tax_worksheet(available_worksheets)

df_raw, source_ws = _load_with_candidates(refresh=refresh, worksheet_names=tuple(worksheet_order))
if df_raw.empty:
    st.warning(
        "Unable to load IRS tax statistics worksheet from Google Sheets. "
        "Expected a tab similar to '22incyallagi'."
    )
    st.write({"available_worksheets": available_worksheets, "candidates_tried": worksheet_order})
    st.stop()

prepared = prepare_tax_stats_dataframe(df_raw)
if prepared.empty:
    st.warning("IRS worksheet loaded but schema mapping failed.")
    st.write({"source_worksheet": source_ws, "columns": list(df_raw.columns)})
    st.stop()

county_summary = build_county_summary(prepared)
state_summary = build_state_summary(county_summary)

st.caption(f"Source worksheet: {source_ws} | Raw rows loaded: {len(df_raw):,}")

st.sidebar.header("Filters")
all_states = sorted(county_summary["state"].dropna().astype(str).unique().tolist())
selected_states = st.sidebar.multiselect("State", all_states, default=all_states)
county_base = county_summary[county_summary["state"].isin(selected_states)] if selected_states else county_summary.iloc[0:0].copy()

county_search = st.sidebar.text_input("County search", "").strip().lower()
if county_search:
    county_base = county_base[county_base["county_label"].str.lower().str.contains(county_search, na=False)]

county_options = sorted(county_base["county_label"].dropna().astype(str).unique().tolist())
selected_counties = st.sidebar.multiselect("County", county_options)
if selected_counties:
    county_base = county_base[county_base["county_label"].isin(selected_counties)]

max_returns = float(county_summary["number_of_returns"].max()) if not county_summary.empty else 0.0
min_returns = st.sidebar.slider("Minimum returns", 0.0, max_returns, 0.0)

max_agi = float(county_summary["adjusted_gross_income"].max()) if not county_summary.empty else 0.0
min_agi = st.sidebar.slider("Minimum AGI", 0.0, max_agi, 0.0)

max_avg_income = float(county_summary["avg_income_per_return"].max()) if not county_summary.empty else 0.0
min_avg_income = st.sidebar.slider("Minimum avg income per return", 0.0, max_avg_income, 0.0)

county_filtered = county_base[
    (county_base["number_of_returns"] >= min_returns)
    & (county_base["adjusted_gross_income"] >= min_agi)
    & (county_base["avg_income_per_return"] >= min_avg_income)
].copy()
state_filtered = state_summary[state_summary["state"].isin(selected_states)].copy()

tabs = st.tabs(
    ["Overview + KPIs", "County Analysis", "State Analysis", "Distributions", "County Map", "Drilldown", "Downloads"]
)

with tabs[0]:
    st.subheader("Income Overview")
    st.markdown(
        """
This IRS dataset reports county-level tax return and income information.
- Number of tax returns is an approximate count of households.
- Personal exemptions are a rough population proxy.
- Adjusted Gross Income (AGI) is total reported income.
- Wages and salaries represent labor income.
- Dividends + interest represent investment-related income.
"""
    )
    st.info(
        "Use this page to compare income structure with government employee counts/spending in later analysis."
    )
    st.markdown(
        """
**What this page helps us analyze**
- Where income is concentrated.
- Where labor income dominates vs investment income dominates.
- Which counties have large household bases.
- Which counties appear unusually high-income relative to size.
- How these patterns can later be compared with government employee counts or spending.
"""
    )

    kpi = county_filtered if not county_filtered.empty else county_summary
    total_investment = float((kpi["dividends"] + kpi["interest_received"]).sum())
    avg_income_across_filtered = float(kpi["adjusted_gross_income"].sum() / kpi["number_of_returns"].sum()) if kpi["number_of_returns"].sum() else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c6, c7, c8, c9, c10 = st.columns(5)
    c1.metric("Number of tax returns", f"{kpi['number_of_returns'].sum():,.0f}")
    c2.metric("Personal exemptions", f"{kpi['personal_exemptions'].sum():,.0f}")
    c3.metric("Total reported income (AGI)", f"{kpi['adjusted_gross_income'].sum():,.0f}")
    c4.metric("Total wages", f"{kpi['wages_and_salaries'].sum():,.0f}")
    c5.metric("Total dividends", f"{kpi['dividends'].sum():,.0f}")
    c6.metric("Total interest", f"{kpi['interest_received'].sum():,.0f}")
    c7.metric("Total investment income", f"{total_investment:,.0f}")
    c8.metric("Counties covered", f"{kpi['county_fips'].nunique():,}")
    c9.metric("States covered", f"{kpi['state'].nunique():,}")
    c10.metric("Average income per return", f"{avg_income_across_filtered:,.2f}")
    st.caption("Dollar-based values use the units provided by the IRS source worksheet.")
    top_agi_row = county_filtered.sort_values("adjusted_gross_income", ascending=False).head(1)
    if not top_agi_row.empty:
        st.success(
            f"Key takeaway: {top_agi_row.iloc[0]['county_label']} has the highest total reported income "
            "in the current filtered view."
        )

with tabs[1]:
    st.subheader("Counties with Highest Income")
    metric_options = {
        "Total AGI": "adjusted_gross_income",
        "Avg income per return": "avg_income_per_return",
        "Wages": "wages_and_salaries",
        "Dividends": "dividends",
        "Interest": "interest_received",
        "Investment income share": "investment_income_share",
        "Wage income share": "wage_income_share",
        "Number of returns": "number_of_returns",
        "Exemptions": "personal_exemptions",
    }
    selected_metric_label = st.selectbox("County ranking metric", list(metric_options.keys()), index=0)
    selected_metric = metric_options[selected_metric_label]

    _top_bar(
        county_filtered,
        selected_metric,
        f"Counties with the highest {selected_metric_label.lower()}",
        n=20,
        subtitle="Counties are sorted from highest to lowest for quick ranking.",
    )
    st.info(
        "Interpretation: high AGI can reflect scale, while high average income per return reflects concentration of income per filing unit."
    )
    with st.expander("View filtered county summary table", expanded=False):
        st.dataframe(
            county_filtered[
                [
                    "county_label",
                    "number_of_returns",
                    "personal_exemptions",
                    "adjusted_gross_income",
                    "wages_and_salaries",
                    "dividends",
                    "interest_received",
                    "avg_income_per_return",
                    "avg_wages_per_return",
                    "exemptions_per_return",
                    "investment_income",
                    "investment_income_share",
                    "wage_income_share",
                ]
            ].sort_values(selected_metric, ascending=False),
            use_container_width=True,
        )

with tabs[2]:
    st.subheader("State Income Rankings")
    if state_filtered.empty:
        st.info("No states available after filters.")
    else:
        s1, s2 = st.columns(2)
        with s1:
            _top_bar(
                state_filtered.rename(columns={"state": "county_label"}),
                "adjusted_gross_income",
                "States with the highest total reported income (AGI)",
                n=20,
                subtitle="This chart highlights where overall income is most concentrated.",
            )
        with s2:
            _top_bar(
                state_filtered.rename(columns={"state": "county_label"}),
                "avg_income_per_return",
                "States with the highest average income per return",
                n=20,
            )
        s3, s4 = st.columns(2)
        with s3:
            _top_bar(
                state_filtered.rename(columns={"state": "county_label"}),
                "wages_and_salaries",
                "States with the highest wage and salary income",
                n=20,
            )
        with s4:
            _top_bar(
                state_filtered.rename(columns={"state": "county_label"}),
                "investment_income",
                "States with the highest investment-related income",
                n=20,
            )
        s5, s6 = st.columns(2)
        with s5:
            _top_bar(
                state_filtered.rename(columns={"state": "county_label"}),
                "number_of_returns",
                "States with the largest household base (tax returns)",
                n=20,
            )
        with s6:
            _top_bar(
                state_filtered.rename(columns={"state": "county_label"}),
                "personal_exemptions",
                "States with the largest population proxy (exemptions)",
                n=20,
            )
        st.info("High wage share suggests labor-driven income structure; high investment share suggests capital-income concentration.")
        leader_state = state_filtered.sort_values("adjusted_gross_income", ascending=False).head(1)
        if not leader_state.empty:
            st.success(f"Key takeaway: {leader_state.iloc[0]['state']} leads state-level AGI in this filtered view.")

with tabs[3]:
    st.subheader("Distribution and Relationship Analysis")
    use_log = st.checkbox("Use log scale for scatter x/y axes", value=True)
    d1, d2 = st.columns(2)
    with d1:
        fig = px.histogram(
            county_filtered,
            x="adjusted_gross_income",
            nbins=40,
            title="How Total Reported Income Is Distributed Across Counties",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This shows whether income is broadly spread out or concentrated in a small number of counties.")
    with d2:
        fig = px.histogram(
            county_filtered,
            x="avg_income_per_return",
            nbins=40,
            title="How Average Income per Return Varies Across Counties",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("This highlights counties where income per return is unusually high or low.")

    r1, r2 = st.columns(2)
    with r1:
        fig = px.scatter(
            county_filtered,
            x="number_of_returns",
            y="adjusted_gross_income",
            hover_name="county_label",
            title="Do Counties with More Tax Returns Also Have More Total Income?",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        if use_log:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each dot is a county. Moving up and right indicates larger income scale and household base.")
    with r2:
        fig = px.scatter(
            county_filtered,
            x="wages_and_salaries",
            y="adjusted_gross_income",
            hover_name="county_label",
            title="How Wage Income Relates to Total Reported Income",
            color_discrete_sequence=[WAGE_COLOR],
        )
        if use_log:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Counties above the general pattern may rely more on non-wage income sources.")

    r3, r4 = st.columns(2)
    with r3:
        fig = px.scatter(
            county_filtered,
            x="investment_income_share",
            y="avg_income_per_return",
            hover_name="county_label",
            title="Do Higher-Income Counties Depend More on Investment Income?",
            color_discrete_sequence=[INVESTMENT_COLOR],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher investment-income share often indicates stronger capital-income presence.")
    with r4:
        fig = px.scatter(
            county_filtered,
            x="number_of_returns",
            y="personal_exemptions",
            hover_name="county_label",
            title="Tax Returns vs Exemptions (Household vs Population Proxy)",
            color_discrete_sequence=[PRIMARY_COLOR],
        )
        if use_log:
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("A strong relationship is expected because both measures track county size.")
    st.success("Key takeaway: total income and average income tell different stories about county wealth and scale.")

with tabs[4]:
    st.subheader("County Map View")
    map_metric_label = st.selectbox(
        "County map metric",
        [
            "Total reported income (AGI)",
            "Average income per return",
            "Number of tax returns",
            "Wages and salaries",
            "Investment income share",
        ],
        index=0,
    )
    map_metric = {
        "Total reported income (AGI)": "adjusted_gross_income",
        "Average income per return": "avg_income_per_return",
        "Number of tax returns": "number_of_returns",
        "Wages and salaries": "wages_and_salaries",
        "Investment income share": "investment_income_share",
    }[map_metric_label]

    map_df = county_filtered.copy()
    map_df["county_fips"] = (
        map_df["county_fips"].astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(5)
    )
    map_df = map_df[map_df["county_fips"].str.fullmatch(r"\d{5}", na=False)].copy()
    if map_df.empty:
        st.info("No counties available for the current filters.")
    else:
        county_geojson = _load_county_geojson()
        geo_ids = set()
        if county_geojson and isinstance(county_geojson.get("features"), list):
            geo_ids = {str(f.get("id", "")).zfill(5) for f in county_geojson["features"]}

        matched = map_df[map_df["county_fips"].isin(geo_ids)] if geo_ids else pd.DataFrame()

        try:
            if county_geojson is None or matched.empty:
                raise ValueError("County geojson unavailable or no matching FIPS IDs.")
            fig = px.choropleth_mapbox(
                matched,
                geojson=county_geojson,
                locations="county_fips",
                featureidkey="id",
                color=map_metric,
                color_continuous_scale="Blues",
                hover_name="county_label",
                title=f"U.S. County Map: {map_metric_label}",
                mapbox_style="carto-positron",
                center={"lat": 37.8, "lon": -96.0},
                zoom=3,
                opacity=0.7,
            )
            fig.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0}, height=700)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "This map highlights where the selected tax metric is strongest. Darker counties indicate higher values."
            )
            top_county = matched.sort_values(map_metric, ascending=False).head(1)
            if not top_county.empty:
                st.success(f"Key takeaway: {top_county.iloc[0]['county_label']} is currently highest on this map metric.")
        except Exception:
            st.info("County map could not be rendered here. Showing a state-level fallback map.")
            state_map = (
                map_df.groupby("state", as_index=False)[map_metric].sum()
                if map_metric != "avg_income_per_return"
                else map_df.groupby("state", as_index=False)[map_metric].mean()
            )
            state_map["state_abbrev"] = state_map["state"].astype(str).str.upper()
            fig = px.choropleth(
                state_map,
                locations="state_abbrev",
                locationmode="USA-states",
                scope="usa",
                color=map_metric,
                title=f"State Map Fallback: {map_metric_label}",
                color_continuous_scale="Blues",
            )
            fig.update_layout(margin={"l": 0, "r": 0, "t": 50, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

with tabs[5]:
    st.subheader("County or State Drilldown")
    geography = st.radio("Drilldown type", ["County", "State"], horizontal=True)
    if geography == "County":
        options = sorted(county_filtered["county_label"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Select county", options) if options else None
        if selected:
            row = county_filtered[county_filtered["county_label"] == selected].iloc[0]
            comp_state = state_summary[state_summary["state"] == row["state"]]
            state_avg_income = float(comp_state["avg_income_per_return"].iloc[0]) if not comp_state.empty else 0.0
            national_avg_income = float(county_summary["adjusted_gross_income"].sum() / county_summary["number_of_returns"].sum())
            left, right = st.columns(2)
            with left:
                st.markdown("**Core totals**")
                st.write(
                    {
                        "number_of_tax_returns": float(row["number_of_returns"]),
                        "personal_exemptions": float(row["personal_exemptions"]),
                        "total_reported_income_agi": float(row["adjusted_gross_income"]),
                        "wages_and_salaries": float(row["wages_and_salaries"]),
                        "dividends": float(row["dividends"]),
                        "interest_received": float(row["interest_received"]),
                    }
                )
            with right:
                st.markdown("**Derived ratios**")
                st.write(
                    {
                        "avg_income_per_return": float(row["avg_income_per_return"]),
                        "avg_wages_per_return": float(row["avg_wages_per_return"]),
                        "exemptions_per_return": float(row["exemptions_per_return"]),
                        "investment_income_share": float(row["investment_income_share"]),
                        "wage_income_share": float(row["wage_income_share"]),
                        "state_avg_income_per_return": state_avg_income,
                        "national_avg_income_per_return": national_avg_income,
                    }
                )
            structure_df = pd.DataFrame(
                {
                    "component": ["wages_and_salaries", "dividends", "interest_received"],
                    "value": [row["wages_and_salaries"], row["dividends"], row["interest_received"]],
                }
            )
            fig = px.pie(structure_df, names="component", values="value", title="Income structure breakdown")
            st.plotly_chart(fig, use_container_width=True)
    else:
        options = sorted(state_filtered["state"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Select state", options) if options else None
        if selected:
            row = state_filtered[state_filtered["state"] == selected].iloc[0]
            st.write(
                {
                    "returns": float(row["number_of_returns"]),
                    "exemptions": float(row["personal_exemptions"]),
                    "agi": float(row["adjusted_gross_income"]),
                    "wages": float(row["wages_and_salaries"]),
                    "investment_income": float(row["investment_income"]),
                    "avg_income_per_return": float(row["avg_income_per_return"]),
                    "investment_income_share": float(row["investment_income_share"]),
                    "wage_income_share": float(row["wage_income_share"]),
                }
            )

with tabs[6]:
    st.subheader("Downloadable Outputs")
    st.download_button(
        "Download county summary (CSV)",
        data=to_csv_bytes(county_filtered),
        file_name="irs_tax_stats_county_summary_filtered.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download state summary (CSV)",
        data=to_csv_bytes(state_filtered),
        file_name="irs_tax_stats_state_summary_filtered.csv",
        mime="text/csv",
    )

    top_agi = county_filtered.sort_values("adjusted_gross_income", ascending=False).head(100)
    top_avg_income = county_filtered.sort_values("avg_income_per_return", ascending=False).head(100)
    st.download_button(
        "Download top AGI counties (CSV)",
        data=to_csv_bytes(top_agi),
        file_name="irs_tax_stats_top_agi_counties.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download top avg income counties (CSV)",
        data=to_csv_bytes(top_avg_income),
        file_name="irs_tax_stats_top_avg_income_counties.csv",
        mime="text/csv",
    )

    detail_options = sorted(county_summary["county_label"].dropna().astype(str).unique().tolist())
    detail_county = st.selectbox("County for selected detail download", detail_options) if detail_options else None
    if detail_county:
        detail = county_summary[county_summary["county_label"] == detail_county].copy()
        st.dataframe(detail, use_container_width=True)
        st.download_button(
            "Download selected county detail (CSV)",
            data=to_csv_bytes(detail),
            file_name="irs_tax_stats_selected_county_detail.csv",
            mime="text/csv",
        )
