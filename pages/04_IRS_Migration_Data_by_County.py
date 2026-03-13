from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.migration import build_county_summary, build_state_summary, prepare_migration_dataframe, to_csv_bytes
from utils.sheets import list_worksheets, load_data
from utils.theme import apply_theme
from utils.viz import STATE_ABBREV


INFLOW_CANDIDATES = [
    "countyinflow2122",
    "countyinflow2122",
    "county_inflow_2122",
    "irs_county_inflow_2122",
    "irs_county_inflow",
    "inflow2122",
]
OUTFLOW_CANDIDATES = [
    "countyoutflow2122",
    "county_outflow_2122",
    "irs_county_outflow_2122",
    "irs_county_outflow",
    "outflow2122",
]

GAIN_COLOR = "#2E8B57"
LOSS_COLOR = "#C0392B"
NEUTRAL_COLOR = "#2C7FB8"


def _matches_tokens(name: str, tokens: Iterable[str]) -> bool:
    text = str(name).strip().lower()
    return all(t in text for t in tokens)


def _resolve_worksheet_name(kind: str, available: list[str]) -> list[str]:
    if kind == "inflow":
        tokens = ["county", "inflow"]
        candidates = INFLOW_CANDIDATES
    else:
        tokens = ["county", "outflow"]
        candidates = OUTFLOW_CANDIDATES

    token_hits = [w for w in available if _matches_tokens(w, tokens)]
    ordered = token_hits + candidates
    seen = set()
    out = []
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
        df = load_data(refresh=refresh and idx == 0, worksheets=[ws_name])
        if not df.empty:
            return df, ws_name
    return pd.DataFrame(), None


def _state_to_abbrev(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) == 2:
        return text.upper()
    return STATE_ABBREV.get(text.lower())


def _render_rankings(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ascending: bool = False,
    n: int = 15,
    show_table: bool = False,
    subtitle: str = "",
) -> None:
    if df.empty or metric not in df.columns:
        st.info(f"No data available for {title.lower()}.")
        return
    top_df = df.sort_values(metric, ascending=ascending).head(n).copy()
    plot_df = top_df.sort_values(metric, ascending=True).copy()
    if "net" in metric:
        plot_df["direction"] = plot_df[metric].apply(lambda v: "Net gain" if v >= 0 else "Net loss")
        fig = px.bar(
            plot_df,
            x=metric,
            y="county_label",
            orientation="h",
            title=title,
            color="direction",
            color_discrete_map={"Net gain": GAIN_COLOR, "Net loss": LOSS_COLOR},
        )
    else:
        fig = px.bar(
            plot_df,
            x=metric,
            y="county_label",
            orientation="h",
            title=title,
            color_discrete_sequence=[NEUTRAL_COLOR],
        )
    fig.update_layout(xaxis_title="", yaxis_title="", margin={"l": 0, "r": 0, "t": 50, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    if subtitle:
        st.caption(subtitle)
    if show_table:
        st.dataframe(
            top_df[["county_label", "state", metric]].reset_index(drop=True),
            use_container_width=True,
        )


apply_theme()

st.title("IRS Migration Data by County")
st.caption("County and state migration patterns from IRS county-to-county inflow/outflow data in Google Sheets.")

refresh = st.sidebar.button("Refresh data")

sheet_id = str(st.secrets.get("SHEET_ID", "") or "")
available_worksheets = list_worksheets(sheet_id) if sheet_id else []
inflow_order = _resolve_worksheet_name("inflow", available_worksheets)
outflow_order = _resolve_worksheet_name("outflow", available_worksheets)

inflow_raw, inflow_ws = _load_with_candidates(refresh=refresh, worksheet_names=tuple(inflow_order))
outflow_raw, outflow_ws = _load_with_candidates(refresh=refresh, worksheet_names=tuple(outflow_order))

if inflow_raw.empty or outflow_raw.empty:
    st.warning(
        "Unable to load IRS county inflow/outflow worksheets from the connected Google Sheet. "
        "Expected worksheet names similar to countyinflow2122 / countyoutflow2122."
    )
    st.write(
        {
            "available_worksheets": available_worksheets,
            "inflow_candidates_tried": inflow_order,
            "outflow_candidates_tried": outflow_order,
        }
    )
    st.stop()

inflow_df = prepare_migration_dataframe(inflow_raw, direction="inflow")
outflow_df = prepare_migration_dataframe(outflow_raw, direction="outflow")

if inflow_df.empty or outflow_df.empty:
    st.warning("Migration worksheets loaded but schema mapping failed. Verify IRS inflow/outflow column names.")
    st.write(
        {
            "inflow_columns": list(inflow_raw.columns),
            "outflow_columns": list(outflow_raw.columns),
            "inflow_worksheet": inflow_ws,
            "outflow_worksheet": outflow_ws,
        }
    )
    st.stop()

primary_county = build_county_summary(inflow_df, outflow_df)
county_candidates = {
    "inflow->outflow (primary)": primary_county,
    "inflow-only matrix fallback": build_county_summary(inflow_df, inflow_df),
    "outflow-only matrix fallback": build_county_summary(outflow_df, outflow_df),
    "swapped pairing fallback": build_county_summary(outflow_df, inflow_df),
}


def _dispersion_score(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["net_returns"].abs().sum() + df["net_people"].abs().sum() + df["net_agi"].abs().sum())


best_label, county_summary = max(county_candidates.items(), key=lambda kv: _dispersion_score(kv[1]))
state_summary = build_state_summary(county_summary)

st.caption(
    f"Sources: inflow='{inflow_ws}', outflow='{outflow_ws}' | "
    f"rows loaded: inflow={len(inflow_raw):,}, outflow={len(outflow_raw):,}"
)
if best_label != "inflow->outflow (primary)":
    st.caption(
        f"Auto-selected migration pairing mode: {best_label}. "
        "Open Debug details below if you want technical mapping diagnostics."
    )
    with st.expander("Debug mapping details", expanded=False):
        scores = {name: _dispersion_score(df) for name, df in county_candidates.items()}
        st.write(
            {
                "selected_mode": best_label,
                "dispersion_scores": scores,
                "inflow_columns": list(inflow_raw.columns),
                "outflow_columns": list(outflow_raw.columns),
            }
        )

st.sidebar.header("Filters")
all_states = sorted([s for s in county_summary["state"].dropna().astype(str).unique().tolist() if s.strip()])
selected_states = st.sidebar.multiselect("State", all_states, default=all_states)

county_base = county_summary[county_summary["state"].isin(selected_states)] if selected_states else county_summary.iloc[0:0].copy()

county_search = st.sidebar.text_input("County search", "").strip().lower()
if county_search:
    county_base = county_base[county_base["county_label"].str.lower().str.contains(county_search, na=False)]

all_counties = sorted(county_base["county_label"].dropna().astype(str).unique().tolist())
selected_counties = st.sidebar.multiselect("County", all_counties)
if selected_counties:
    county_base = county_base[county_base["county_label"].isin(selected_counties)]

max_volume = float(county_summary["migration_volume"].max()) if not county_summary.empty else 0.0
min_volume = st.sidebar.slider("Minimum migration volume (returns)", 0.0, max_volume, 0.0)

max_agi_moved = float(county_summary["agi_moved"].max()) if not county_summary.empty else 0.0
min_agi = st.sidebar.slider("Minimum AGI moved", 0.0, max_agi_moved, 0.0)

geography_type = st.sidebar.selectbox("Selected geography type", ["County", "State"], index=0)

county_filtered = county_base[
    (county_base["migration_volume"] >= float(min_volume)) & (county_base["agi_moved"] >= float(min_agi))
].copy()
state_filtered = state_summary[state_summary["state"].isin(selected_states)].copy()

tabs = st.tabs(["Overview + KPIs", "County Analysis", "State Analysis", "Flow Drilldown", "Maps + Visuals", "Downloads"])

with tabs[0]:
    st.subheader("Overview")
    st.markdown(
        """
IRS county migration tables track tax-return moves between origin and destination counties.
`Inflow` means tax filers/people/income moving into a county, `outflow` means leaving it, and `net` is inflow minus outflow.
This helps identify where residents and income are concentrating before joining this view with government employee counts.
"""
    )
    st.info(
        "Interpretation note: Positive net values indicate counties or states gaining people/income; negative values indicate net loss."
    )
    st.markdown(
        """
**What this page helps us analyze**
- Which counties are gaining residents and which are losing residents.
- Where adjusted gross income is moving in and out.
- Whether migration patterns can later be compared with government employment concentration.
- How population movement may align with government-heavy regions.
"""
    )

    kpi = county_filtered if not county_filtered.empty else county_summary
    st.subheader("Big Picture Metrics")
    k1, k2, k3, k4 = st.columns(4)
    k5, k6, k7, k8 = st.columns(4)
    k9, k10, k11 = st.columns(3)
    k1.metric("People moving in (tax returns)", f"{kpi['inbound_returns'].sum():,.0f}")
    k2.metric("People moving out (tax returns)", f"{kpi['outbound_returns'].sum():,.0f}")
    k3.metric("Total net returns", f"{kpi['net_returns'].sum():,.0f}")
    k4.metric("Counties covered", f"{kpi['county_fips'].nunique():,}")
    k5.metric("People moving in (exemptions)", f"{kpi['inbound_people'].sum():,.0f}")
    k6.metric("People moving out (exemptions)", f"{kpi['outbound_people'].sum():,.0f}")
    k7.metric("Total net people", f"{kpi['net_people'].sum():,.0f}")
    k8.metric("States covered", f"{kpi['state'].nunique():,}")
    k9.metric("Income moving in (AGI)", f"{kpi['inbound_agi'].sum():,.0f}")
    k10.metric("Income moving out (AGI)", f"{kpi['outbound_agi'].sum():,.0f}")
    k11.metric("Total net AGI", f"{kpi['net_agi'].sum():,.0f}")
    st.caption("AGI means Adjusted Gross Income (total reported income). Values use IRS source units.")
    top_gain = county_filtered.sort_values("net_people", ascending=False).head(1)
    top_loss = county_filtered.sort_values("net_people", ascending=True).head(1)
    if not top_gain.empty and not top_loss.empty:
        st.success(
            f"Key takeaway: {top_gain.iloc[0]['county_label']} is among the strongest population gainers, "
            f"while {top_loss.iloc[0]['county_label']} is among the largest population losers in the filtered view."
        )

    if (
        len(selected_states) == len(all_states)
        and abs(float(kpi["net_returns"].sum())) < 0.5
        and abs(float(kpi["net_people"].sum())) < 0.5
        and abs(float(kpi["net_agi"].sum())) < 0.5
    ):
        st.info(
            "National-scope check: total inbound equals total outbound, so total net values are expected to be near zero. "
            "Use county/state net rankings to see where gains and losses occur."
        )

with tabs[1]:
    st.subheader("Counties Gaining or Losing Residents")
    county_metric_group = st.radio("County metric group", ["Returns", "People", "AGI"], horizontal=True)
    show_county_tables = st.checkbox("Show county ranking tables", value=False)

    if county_metric_group == "Returns":
        m_in, m_out, m_net = "inbound_returns", "outbound_returns", "net_returns"
    elif county_metric_group == "People":
        m_in, m_out, m_net = "inbound_people", "outbound_people", "net_people"
    else:
        m_in, m_out, m_net = "inbound_agi", "outbound_agi", "net_agi"

    c1, c2 = st.columns(2)
    with c1:
        _render_rankings(
            county_filtered,
            m_in,
            f"Counties receiving the most inbound {county_metric_group.lower()}",
            show_table=show_county_tables,
            subtitle="This shows where the largest incoming population/income flows are concentrated.",
        )
    with c2:
        _render_rankings(
            county_filtered,
            m_out,
            f"Counties with the highest outbound {county_metric_group.lower()}",
            show_table=show_county_tables,
            subtitle="This shows where the largest outflows are occurring.",
        )

    c3, c4 = st.columns(2)
    with c3:
        _render_rankings(
            county_filtered,
            m_net,
            f"Counties gaining the most ({county_metric_group.lower()})",
            show_table=show_county_tables,
            subtitle="Green bars indicate net gains; red bars indicate net losses.",
        )
    with c4:
        _render_rankings(
            county_filtered,
            m_net,
            f"Counties losing the most ({county_metric_group.lower()})",
            ascending=True,
            show_table=show_county_tables,
            subtitle="These counties have the largest negative net movement in the selected metric.",
        )

    with st.expander("View filtered county summary table", expanded=False):
        st.dataframe(
            county_filtered[
                [
                    "county_label",
                    "state",
                    "inbound_returns",
                    "outbound_returns",
                    "net_returns",
                    "inbound_people",
                    "outbound_people",
                    "net_people",
                    "inbound_agi",
                    "outbound_agi",
                    "net_agi",
                ]
            ].sort_values("net_returns", ascending=False),
            use_container_width=True,
        )
    st.info("Key takeaway: high inbound AGI suggests a strengthening local tax base; high outbound AGI suggests income leakage.")

with tabs[2]:
    st.subheader("State-Level Migration Patterns")
    if state_filtered.empty:
        st.info("No states available after current filters.")
    else:
        state_metric_group = st.radio("State metric group", ["Returns", "People", "AGI"], horizontal=True)
        if state_metric_group == "Returns":
            sm_in, sm_out, sm_net = "inbound_returns", "outbound_returns", "net_returns"
            sm_label = "migration"
        elif state_metric_group == "People":
            sm_in, sm_out, sm_net = "inbound_people", "outbound_people", "net_people"
            sm_label = "people movement"
        else:
            sm_in, sm_out, sm_net = "inbound_agi", "outbound_agi", "net_agi"
            sm_label = "AGI movement"

        s1, s2 = st.columns(2)
        with s1:
            inbound_states = state_filtered.sort_values(sm_in, ascending=False).head(20)
            fig = px.bar(
                inbound_states.sort_values(sm_in, ascending=True),
                x=sm_in,
                y="state",
                orientation="h",
                title=f"State rankings by inbound {sm_label}",
            )
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("States at the top are receiving the largest inflows in the selected metric.")
        with s2:
            outbound_states = state_filtered.sort_values(sm_out, ascending=False).head(20)
            fig = px.bar(
                outbound_states.sort_values(sm_out, ascending=True),
                x=sm_out,
                y="state",
                orientation="h",
                title=f"State rankings by outbound {sm_label}",
            )
            fig.update_layout(xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("States at the top are seeing the largest outflows in the selected metric.")

        net_states = state_filtered.sort_values(sm_net, ascending=False).head(20)
        fig = px.bar(
            net_states.sort_values(sm_net, ascending=True),
            x=sm_net,
            y="state",
            orientation="h",
            title=f"State rankings by net {sm_label}",
        )
        fig.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive values mean net gains; negative values mean net losses.")
        if not net_states.empty:
            st.success(
                f"Key takeaway: {net_states.iloc[0]['state']} leads net {sm_label} gains in the current filter."
            )

        with st.expander("View filtered state summary table", expanded=False):
            st.dataframe(state_filtered.sort_values("net_returns", ascending=False), use_container_width=True)

with tabs[3]:
    st.subheader("Where People and Income Are Coming From")
    st.caption("Inspect where returns/people/AGI are coming from and going to for a selected county or state.")

    if geography_type == "County":
        drill_options = sorted(county_filtered["county_label"].dropna().astype(str).unique().tolist())
        selected_geo = st.selectbox("Select county", drill_options) if drill_options else None
        if selected_geo:
            selected_fips = str(county_summary.loc[county_summary["county_label"] == selected_geo, "county_fips"].iloc[0])
            incoming = (
                inflow_df[inflow_df["destination_county_fips"] == selected_fips]
                .groupby(["origin_county_fips", "origin_label"], as_index=False)
                .agg(inbound_returns=("returns", "sum"), inbound_people=("people", "sum"), inbound_agi=("agi", "sum"))
                .sort_values("inbound_returns", ascending=False)
            )
            outgoing = (
                outflow_df[outflow_df["origin_county_fips"] == selected_fips]
                .groupby(["destination_county_fips", "destination_label"], as_index=False)
                .agg(outbound_returns=("returns", "sum"), outbound_people=("people", "sum"), outbound_agi=("agi", "sum"))
                .sort_values("outbound_returns", ascending=False)
            )
        else:
            incoming = pd.DataFrame()
            outgoing = pd.DataFrame()
    else:
        drill_options = sorted(state_filtered["state"].dropna().astype(str).unique().tolist())
        selected_geo = st.selectbox("Select state", drill_options) if drill_options else None
        if selected_geo:
            incoming = (
                inflow_df[inflow_df["destination_state"] == selected_geo]
                .groupby(["origin_state"], as_index=False)
                .agg(inbound_returns=("returns", "sum"), inbound_people=("people", "sum"), inbound_agi=("agi", "sum"))
                .rename(columns={"origin_state": "origin_label"})
                .sort_values("inbound_returns", ascending=False)
            )
            outgoing = (
                outflow_df[outflow_df["origin_state"] == selected_geo]
                .groupby(["destination_state"], as_index=False)
                .agg(outbound_returns=("returns", "sum"), outbound_people=("people", "sum"), outbound_agi=("agi", "sum"))
                .rename(columns={"destination_state": "destination_label"})
                .sort_values("outbound_returns", ascending=False)
            )
        else:
            incoming = pd.DataFrame()
            outgoing = pd.DataFrame()

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**Top origin geographies sending people in**")
        st.dataframe(incoming.head(25), use_container_width=True)
    with d2:
        st.markdown("**Top destination geographies receiving people out**")
        st.dataframe(outgoing.head(25), use_container_width=True)

    d3, d4 = st.columns(2)
    with d3:
        st.markdown("**Top incoming AGI sources**")
        if not incoming.empty and "inbound_agi" in incoming.columns:
            st.dataframe(incoming.sort_values("inbound_agi", ascending=False).head(25), use_container_width=True)
    with d4:
        st.markdown("**Top outgoing AGI destinations**")
        if not outgoing.empty and "outbound_agi" in outgoing.columns:
            st.dataframe(outgoing.sort_values("outbound_agi", ascending=False).head(25), use_container_width=True)

with tabs[4]:
    st.subheader("Map View and Comparison Charts")
    map_metric = st.selectbox(
        "Map metric",
        ["net_returns", "net_people", "net_agi", "inbound_returns", "outbound_returns"],
        index=1,
    )

    map_df = state_filtered.copy()
    map_df["state_abbrev"] = map_df["state"].apply(_state_to_abbrev)
    map_df = map_df.dropna(subset=["state_abbrev"])
    if not map_df.empty:
        fig_map = px.choropleth(
            map_df,
            locations="state_abbrev",
            locationmode="USA-states",
            scope="usa",
            color=map_metric,
            hover_name="state",
            title=f"State choropleth: {map_metric}",
        )
        fig_map.update_layout(margin={"r": 0, "l": 0, "t": 45, "b": 0})
        st.plotly_chart(fig_map, use_container_width=True)
        st.caption("Map reading tip: darker shades indicate higher values for the selected metric.")
    else:
        st.info("State names are not mappable to USPS abbreviations for a choropleth.")

    with st.expander("Additional visuals", expanded=False):
        v1, v2 = st.columns(2)
        with v1:
            if not state_filtered.empty:
                top_net = state_filtered.sort_values("net_people", ascending=False).head(20)
                fig = px.bar(
                    top_net.sort_values("net_people", ascending=True),
                    x="net_people",
                    y="state",
                    orientation="h",
                    title="State bar chart: net migration (people)",
                )
                fig.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
        with v2:
            if not county_filtered.empty:
                top_bottom = pd.concat(
                    [
                        county_filtered.sort_values("net_people", ascending=False).head(10),
                        county_filtered.sort_values("net_people", ascending=True).head(10),
                    ],
                    ignore_index=True,
                )
                fig = px.bar(
                    top_bottom.sort_values("net_people", ascending=True),
                    x="net_people",
                    y="county_label",
                    orientation="h",
                    title="County top/bottom rankings by net people",
                )
                fig.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

        v3, v4 = st.columns(2)
        with v3:
            if not county_filtered.empty:
                top_agi_move = county_filtered.sort_values("net_agi", ascending=False).head(20)
                fig = px.bar(
                    top_agi_move.sort_values("net_agi", ascending=True),
                    x="net_agi",
                    y="county_label",
                    orientation="h",
                    title="AGI movement rankings (county net AGI)",
                )
                fig.update_layout(xaxis_title="", yaxis_title="")
                st.plotly_chart(fig, use_container_width=True)
        with v4:
            if not state_filtered.empty:
                fig = px.scatter(
                    state_filtered,
                    x="inbound_returns",
                    y="outbound_returns",
                    size="migration_volume",
                    color="net_returns",
                    hover_name="state",
                    title="Inbound vs outbound comparison (states)",
                )
                fig.update_layout(xaxis_title="Inbound returns", yaxis_title="Outbound returns")
                st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Pairing note: this migration baseline can be joined later with government employee counts to test whether government-heavy regions are net gainers or net losers."
    )

with tabs[5]:
    st.subheader("Downloadable Outputs")
    st.caption("Download filtered summaries and detailed county flow tables for external analysis.")

    st.download_button(
        "Download county summary (CSV)",
        data=to_csv_bytes(county_filtered),
        file_name="irs_migration_county_summary_filtered.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download state summary (CSV)",
        data=to_csv_bytes(state_filtered),
        file_name="irs_migration_state_summary_filtered.csv",
        mime="text/csv",
    )

    detail_county_options = sorted(county_summary["county_label"].dropna().astype(str).unique().tolist())
    detail_county = st.selectbox("County for detailed flow download", detail_county_options) if detail_county_options else None

    if detail_county:
        detail_fips = str(county_summary.loc[county_summary["county_label"] == detail_county, "county_fips"].iloc[0])
        inflow_detail = inflow_df[inflow_df["destination_county_fips"] == detail_fips].copy()
        outflow_detail = outflow_df[outflow_df["origin_county_fips"] == detail_fips].copy()

        st.markdown("**Selected county inflow detail**")
        st.dataframe(inflow_detail.head(200), use_container_width=True)
        st.download_button(
            "Download selected county inflow detail (CSV)",
            data=to_csv_bytes(inflow_detail),
            file_name="irs_selected_county_inflow_detail.csv",
            mime="text/csv",
        )

        st.markdown("**Selected county outflow detail**")
        st.dataframe(outflow_detail.head(200), use_container_width=True)
        st.download_button(
            "Download selected county outflow detail (CSV)",
            data=to_csv_bytes(outflow_detail),
            file_name="irs_selected_county_outflow_detail.csv",
            mime="text/csv",
        )
