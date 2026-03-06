from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.earmarks import prepare_earmarks_dataframe, to_csv_bytes, top_share
from utils.sheets import load_data
from utils.theme import apply_theme


SOURCE_WORKSHEET = "Sheet7"


@st.cache_data(ttl=300)
def _prepare_cached(df_raw: pd.DataFrame) -> pd.DataFrame:
    return prepare_earmarks_dataframe(df_raw)


def _safe_options(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return []
    return sorted(v for v in df[column].dropna().astype(str).unique().tolist() if v)


def _top_amount_table(df: pd.DataFrame, group_col: str, n: int = 15) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame(columns=[group_col, "total_amount", "project_count"])
    out = (
        df.dropna(subset=["Amount"])
        .groupby(group_col, dropna=False, as_index=False)
        .agg(total_amount=("Amount", "sum"), project_count=("Project", "count"))
        .sort_values("total_amount", ascending=False)
        .head(n)
    )
    out[group_col] = out[group_col].fillna("Unknown")
    return out


def _render_top_bar(df: pd.DataFrame, group_col: str, title: str, n: int = 15) -> pd.DataFrame:
    top_df = _top_amount_table(df, group_col, n=n)
    if top_df.empty:
        st.info(f"No data available for {title.lower()}.")
        return top_df

    fig = px.bar(
        top_df.sort_values("total_amount", ascending=True),
        x="total_amount",
        y=group_col,
        orientation="h",
        title=title,
        text_auto=".2s",
    )
    fig.update_layout(xaxis_title="", yaxis_title="", margin={"l": 0, "r": 0, "t": 50, "b": 0})
    st.plotly_chart(fig, use_container_width=True)
    return top_df


def _build_analysis_points(df: pd.DataFrame) -> list[str]:
    if df.empty or df["Amount"].dropna().empty:
        return ["Not enough amount data to generate analysis points."]

    amount_df = df.dropna(subset=["Amount"]).copy()
    total = float(amount_df["Amount"].sum())

    agency = _top_amount_table(amount_df, "Agency", n=1)
    recipient = _top_amount_table(amount_df, "Recipient", n=1)
    house = _top_amount_table(amount_df, "House Requestor", n=10)
    senate = _top_amount_table(amount_df, "Senate Requestor", n=10)
    states = _top_amount_table(amount_df, "derived_state", n=1)

    origin = amount_df.groupby("Origin", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
    house_total = float(origin.loc[origin["Origin"].str.lower().str.contains("house", na=False), "Amount"].sum())
    senate_total = float(origin.loc[origin["Origin"].str.lower().str.contains("senate", na=False), "Amount"].sum())
    dominant_origin = "Mixed/Unclear"
    if house_total > senate_total:
        dominant_origin = "House"
    elif senate_total > house_total:
        dominant_origin = "Senate"

    top_project_share = top_share(amount_df["Amount"], top_n=10)
    requestor_conc = max(top_share(house["total_amount"], top_n=10), top_share(senate["total_amount"], top_n=10))

    points = []
    if not agency.empty:
        points.append(
            f"Spending is led by {agency.iloc[0]['Agency']}, with ${agency.iloc[0]['total_amount']:,.0f} "
            f"({agency.iloc[0]['total_amount'] / total:.1%} of filtered total)."
        )
    if not recipient.empty:
        points.append(
            f"Top recipient is {recipient.iloc[0]['Recipient']} at ${recipient.iloc[0]['total_amount']:,.0f} "
            f"across {int(recipient.iloc[0]['project_count'])} project(s)."
        )
    points.append(
        f"Funding concentration is {'high' if top_project_share >= 0.5 else 'moderate'}: "
        f"top 10 projects account for {top_project_share:.1%} of total dollars."
    )
    points.append(
        f"Requestor concentration (top 10 combined share proxy) is {requestor_conc:.1%}, "
        "indicating how concentrated approvals appear."
    )
    points.append(f"Origin comparison suggests {dominant_origin} has the larger share in this filtered view.")
    if not states.empty:
        points.append(f"Top derived state is {states.iloc[0]['derived_state']} with ${states.iloc[0]['total_amount']:,.0f}.")
    return points


apply_theme()

st.title("FY26 Enacted Earmarks")
st.caption("Standalone earmarks analysis sourced from the integrated Google Sheet tab.")

refresh = st.sidebar.button("Refresh data")
st.sidebar.caption(f"Data source worksheet: {SOURCE_WORKSHEET}")

df_raw = load_data(refresh=refresh, worksheets=[SOURCE_WORKSHEET])
if df_raw.empty:
    st.warning(
        f"No rows were loaded from worksheet '{SOURCE_WORKSHEET}'. "
        "Check worksheet name and sharing permissions."
    )
    st.stop()

df = _prepare_cached(df_raw)
if df.empty:
    st.warning("Earmarks dataset loaded but no rows are available after normalization.")
    st.stop()

st.sidebar.header("Filters")

agency_sel = st.sidebar.multiselect("Agency", _safe_options(df, "Agency"))
account_sel = st.sidebar.multiselect("Account", _safe_options(df, "Account"))
origin_sel = st.sidebar.multiselect("Origin", _safe_options(df, "Origin"))
state_sel = st.sidebar.multiselect("Derived state", _safe_options(df, "derived_state"))
house_sel = st.sidebar.multiselect("House Requestor", _safe_options(df, "House Requestor"))
senate_sel = st.sidebar.multiselect("Senate Requestor", _safe_options(df, "Senate Requestor"))
recipient_sel = st.sidebar.multiselect("Recipient", _safe_options(df, "Recipient"))
keyword = st.sidebar.text_input("Keyword (Project / Recipient / Location)", "").strip().lower()

amount_non_null = df["Amount"].dropna()
if not amount_non_null.empty:
    amount_min = float(amount_non_null.min())
    amount_max = float(amount_non_null.max())
    if math.isclose(amount_min, amount_max):
        amount_range = (amount_min, amount_max)
        st.sidebar.caption(f"Amount filter locked at ${amount_min:,.2f}")
    else:
        amount_range = st.sidebar.slider(
            "Amount range",
            min_value=amount_min,
            max_value=amount_max,
            value=(amount_min, amount_max),
        )
else:
    amount_range = (0.0, 0.0)

filtered = df.copy()
if agency_sel:
    filtered = filtered[filtered["Agency"].isin(agency_sel)]
if account_sel:
    filtered = filtered[filtered["Account"].isin(account_sel)]
if origin_sel:
    filtered = filtered[filtered["Origin"].isin(origin_sel)]
if state_sel:
    filtered = filtered[filtered["derived_state"].isin(state_sel)]
if house_sel:
    filtered = filtered[filtered["House Requestor"].isin(house_sel)]
if senate_sel:
    filtered = filtered[filtered["Senate Requestor"].isin(senate_sel)]
if recipient_sel:
    filtered = filtered[filtered["Recipient"].isin(recipient_sel)]
if keyword:
    filtered = filtered[filtered["keyword_blob"].str.contains(keyword, na=False)]
if not amount_non_null.empty:
    filtered = filtered[
        (filtered["Amount"].isna())
        | ((filtered["Amount"] >= float(amount_range[0])) & (filtered["Amount"] <= float(amount_range[1])))
    ]

st.caption(
    f"Rows loaded: {len(df_raw):,} | Rows after cleaning: {len(df):,} | Rows after filters: {len(filtered):,}"
)

metric_df = filtered.dropna(subset=["Amount"])

st.subheader("KPI Summary")
k1, k2, k3, k4 = st.columns(4)
k5, k6, k7, k8 = st.columns(4)

k1.metric("Total earmark amount", f"${float(metric_df['Amount'].sum()) if not metric_df.empty else 0.0:,.0f}")
k2.metric("Total number of projects", f"{int(len(filtered)):,}")
k3.metric("Average earmark amount", f"${float(metric_df['Amount'].mean()) if not metric_df.empty else 0.0:,.0f}")
k4.metric("Median earmark amount", f"${float(metric_df['Amount'].median()) if not metric_df.empty else 0.0:,.0f}")
k5.metric("Unique recipients", f"{filtered['Recipient'].nunique():,}")
k6.metric("Unique House requestors", f"{filtered['House Requestor'].nunique():,}")
k7.metric("Unique Senate requestors", f"{filtered['Senate Requestor'].nunique():,}")
k8.metric("Unique agencies", f"{filtered['Agency'].nunique():,}")

st.subheader("Core Visualizations")
v1, v2 = st.columns(2)
with v1:
    _render_top_bar(metric_df, "Agency", "Top agencies by total earmark amount")
with v2:
    _render_top_bar(metric_df, "Account", "Top accounts by total earmark amount")

v3, v4 = st.columns(2)
with v3:
    _render_top_bar(metric_df, "Recipient", "Top recipients by total earmark amount")
with v4:
    _render_top_bar(metric_df, "House Requestor", "Top House requestors by total earmark amount")

v5, v6 = st.columns(2)
with v5:
    _render_top_bar(metric_df, "Senate Requestor", "Top Senate requestors by total earmark amount")
with v6:
    origin_df = (
        metric_df.groupby("Origin", as_index=False)["Amount"].sum().sort_values("Amount", ascending=False)
        if not metric_df.empty
        else pd.DataFrame(columns=["Origin", "Amount"])
    )
    if origin_df.empty:
        st.info("No origin data available.")
    else:
        fig_origin = px.pie(origin_df, names="Origin", values="Amount", title="Origin breakdown (House vs Senate)")
        st.plotly_chart(fig_origin, use_container_width=True)

v7, v8 = st.columns(2)
with v7:
    state_amount = _top_amount_table(metric_df, "derived_state", n=20)
    if state_amount.empty:
        st.info("No state distribution available.")
    else:
        fig_state = px.bar(
            state_amount.sort_values("total_amount", ascending=True),
            x="total_amount",
            y="derived_state",
            orientation="h",
            title="Derived state distribution by total amount",
            text_auto=".2s",
        )
        fig_state.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_state, use_container_width=True)

with v8:
    largest_projects = metric_df.sort_values("Amount", ascending=False).head(20)
    if largest_projects.empty:
        st.info("No project amount data available.")
    else:
        fig_projects = px.bar(
            largest_projects.iloc[::-1],
            x="Amount",
            y="Project",
            orientation="h",
            title="Largest individual earmark projects",
        )
        fig_projects.update_layout(xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_projects, use_container_width=True)

v9, v10 = st.columns(2)
with v9:
    if metric_df.empty:
        st.info("No amount data for histogram.")
    else:
        fig_hist = px.histogram(metric_df, x="Amount", nbins=40, title="Distribution histogram of earmark amounts")
        fig_hist.update_layout(xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

with v10:
    if metric_df.empty:
        st.info("No amount data for treemap.")
    else:
        fig_tree = px.treemap(
            metric_df,
            path=["Agency", "Account", "Recipient"],
            values="Amount",
            title="Treemap: Agency -> Account -> Recipient",
        )
        st.plotly_chart(fig_tree, use_container_width=True)

st.markdown("**Repeated recipients receiving multiple earmarks**")
repeated_recipients = (
    metric_df.groupby("Recipient", as_index=False)
    .agg(project_count=("Project", "count"), total_amount=("Amount", "sum"), avg_amount=("Amount", "mean"))
    .query("project_count > 1")
    .sort_values(["project_count", "total_amount"], ascending=[False, False])
)
st.dataframe(repeated_recipients, use_container_width=True)

st.markdown("**Projects with highest amounts**")
projects_highest = metric_df.sort_values("Amount", ascending=False).head(50)[
    ["Project", "Recipient", "Agency", "Account", "derived_state", "Amount", "House Requestor", "Senate Requestor", "Origin"]
]
st.dataframe(projects_highest, use_container_width=True)

st.subheader("Analysis")
for point in _build_analysis_points(filtered):
    st.markdown(f"- {point}")

st.subheader("What we can do with this data")
st.markdown(
    f"""
- **Current insight coverage:** recipient concentration, requestor concentration, agency/account allocation, origin mix, and state/location distribution from {len(filtered):,} filtered rows.
- **Join for macro context:** BEA GDP by state/metro to test whether earmark intensity tracks output growth.
- **Join for labor impact:** BLS employment/unemployment (state and metro) to evaluate whether earmarks align with labor-market stress.
- **Join for entrepreneurship effects:** Census business formation (BFS) to test whether funding aligns with or predicts startup momentum.
- **Per-capita and equity views:** join Census population to build dollars-per-resident and recipient-normalized comparisons.
- **Crowding-out extension:** compare federal earmark intensity to private capital formation proxies to study substitution/complementarity.
"""
)

st.subheader("Data Tables")
full_display_cols = [
    "Agency",
    "Account",
    "Project",
    "Recipient",
    "Location",
    "derived_city",
    "derived_state",
    "Amount",
    "House Requestor",
    "Senate Requestor",
    "Origin",
]
full_table = filtered[full_display_cols].copy()
st.markdown("**Full filtered dataset**")
st.dataframe(full_table, use_container_width=True)
st.download_button(
    "Download full filtered dataset (CSV)",
    data=to_csv_bytes(full_table),
    file_name="fy26_enacted_earmarks_filtered.csv",
    mime="text/csv",
)

top_recipients_table = _top_amount_table(metric_df, "Recipient", n=50)
st.markdown("**Top recipients**")
st.dataframe(top_recipients_table, use_container_width=True)
st.download_button(
    "Download top recipients (CSV)",
    data=to_csv_bytes(top_recipients_table),
    file_name="fy26_top_recipients.csv",
    mime="text/csv",
)

top_house = _top_amount_table(metric_df, "House Requestor", n=50).rename(
    columns={"House Requestor": "requestor", "total_amount": "house_total_amount", "project_count": "house_projects"}
)
top_senate = _top_amount_table(metric_df, "Senate Requestor", n=50).rename(
    columns={"Senate Requestor": "requestor", "total_amount": "senate_total_amount", "project_count": "senate_projects"}
)
top_requestors_table = top_house.merge(top_senate, on="requestor", how="outer")
if not top_requestors_table.empty:
    for col in ["house_total_amount", "house_projects", "senate_total_amount", "senate_projects"]:
        if col in top_requestors_table.columns:
            top_requestors_table[col] = top_requestors_table[col].fillna(0)
    top_requestors_table["requestor"] = top_requestors_table["requestor"].replace("", "Unknown").fillna("Unknown")
st.markdown("**Top requestors (House + Senate)**")
st.dataframe(top_requestors_table, use_container_width=True)
st.download_button(
    "Download top requestors (CSV)",
    data=to_csv_bytes(top_requestors_table),
    file_name="fy26_top_requestors.csv",
    mime="text/csv",
)

largest_projects_table = metric_df.sort_values("Amount", ascending=False).head(100)[
    ["Project", "Recipient", "Agency", "Account", "derived_state", "Amount", "House Requestor", "Senate Requestor", "Origin"]
]
st.markdown("**Largest projects**")
st.dataframe(largest_projects_table, use_container_width=True)
st.download_button(
    "Download largest projects (CSV)",
    data=to_csv_bytes(largest_projects_table),
    file_name="fy26_largest_projects.csv",
    mime="text/csv",
)

state_summary = _top_amount_table(metric_df, "derived_state", n=500)
st.markdown("**State summary**")
st.dataframe(state_summary, use_container_width=True)
st.download_button(
    "Download state summary (CSV)",
    data=to_csv_bytes(state_summary),
    file_name="fy26_state_summary.csv",
    mime="text/csv",
)

st.subheader("Quality Checks")
unknown_state_count = int((filtered["derived_state"] == "Unknown").sum())
st.caption(
    f"Rows with unknown derived state: {unknown_state_count:,}. "
    "These are intentionally labeled 'Unknown' when state extraction from Location is not possible."
)
