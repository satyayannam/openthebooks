import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Housing Affordability Dashboard (2015–2024)",
    layout="wide",
)

# -----------------------------
# Helpers
# -----------------------------
def pct(x):
    # x is in ratio form (0.42) -> "42.0%"
    return f"{x*100:.1f}%"

def load_data(file) -> pd.DataFrame:
    # Try reading first sheet by default
    df = pd.read_excel(file, engine="openpyxl")
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    return df

def detect_columns(df: pd.DataFrame):
    """
    Tries to auto-detect columns from common names.
    You can override in sidebar.
    """
    candidates = {
        "state": ["State", "state", "STATE"],
        "hpi_growth": ["HPI_Growth", "HPI Growth", "HPI_GROWTH", "home_price_growth", "Housing_Growth"],
        "income_growth": ["Income_Growth", "Income Growth", "INCOME_GROWTH", "median_income_growth"],
        "gap": ["Affordability_Gap", "Affordability Gap", "GAP", "gap"],
        "gap_bucket": ["Gap_Bucket", "Bucket", "Severity", "gap_bucket"],
        "year": ["Year", "year", "YEAR"],
        "hpi_value": ["HPI", "HPI_Value", "HPI value", "HousePriceIndex"],
        "income_value": ["Median_Income", "median_income", "Median Income", "Income"],
    }

    found = {}
    for key, opts in candidates.items():
        for c in opts:
            if c in df.columns:
                found[key] = c
                break
        if key not in found:
            found[key] = None
    return found

def compute_missing_fields(df: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """
    If gap is missing and growth fields exist, compute it.
    If gap_bucket missing, compute based on thresholds.
    """
    df2 = df.copy()

    # Convert growth columns from % strings if needed
    def coerce_ratio(series: pd.Series):
        # handles "42%" or 42 or 0.42
        s = series.copy()
        if s.dtype == object:
            s = s.astype(str).str.replace("%", "", regex=False).str.strip()
            s = pd.to_numeric(s, errors="coerce")
        else:
            s = pd.to_numeric(s, errors="coerce")

        # Heuristic:
        # if values look like 42, treat as percent -> /100
        # if values look like 0.42 keep
        if s.dropna().median() and s.dropna().median() > 1.5:
            s = s / 100.0
        return s

    if cols["hpi_growth"]:
        df2[cols["hpi_growth"]] = coerce_ratio(df2[cols["hpi_growth"]])
    if cols["income_growth"]:
        df2[cols["income_growth"]] = coerce_ratio(df2[cols["income_growth"]])
    if cols["gap"] and cols["gap"] in df2.columns:
        df2[cols["gap"]] = coerce_ratio(df2[cols["gap"]])

    # Compute gap if missing
    if (cols["gap"] is None) and cols["hpi_growth"] and cols["income_growth"]:
        df2["Affordability_Gap"] = df2[cols["hpi_growth"]] - df2[cols["income_growth"]]
        cols["gap"] = "Affordability_Gap"

    # Compute bucket if missing
    if cols["gap_bucket"] is None and cols["gap"]:
        def bucket(g):
            if pd.isna(g):
                return None
            if g < 0.30:
                return "Lower (<30%)"
            if g < 0.50:
                return "Moderate (30–49%)"
            if g < 0.70:
                return "High (50–69%)"
            return "Severe (70%+)"
        df2["Gap_Bucket"] = df2[cols["gap"]].apply(bucket)
        cols["gap_bucket"] = "Gap_Bucket"

    return df2, cols

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

uploaded = st.sidebar.file_uploader("Upload your Excel (Housing_Affordibility.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Upload your Excel file to start.")
    st.stop()

df_raw = load_data(uploaded)
auto_cols = detect_columns(df_raw)

with st.sidebar.expander("Column mapping (only change if needed)", expanded=False):
    state_col = st.selectbox("State column", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["state"]))
    hpi_growth_col = st.selectbox("HPI Growth column", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["hpi_growth"]))
    income_growth_col = st.selectbox("Income Growth column", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["income_growth"]))
    gap_col = st.selectbox("Affordability Gap column (optional)", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["gap"]))
    bucket_col = st.selectbox("Gap Bucket column (optional)", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["gap_bucket"]))
    year_col = st.selectbox("Year column (optional, for trends)", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["year"]))
    hpi_val_col = st.selectbox("HPI value column (optional, for trends)", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["hpi_value"]))
    inc_val_col = st.selectbox("Median income value column (optional, for trends)", options=[None] + list(df_raw.columns), index=( [None]+list(df_raw.columns) ).index(auto_cols["income_value"]))

cols = {
    "state": state_col,
    "hpi_growth": hpi_growth_col,
    "income_growth": income_growth_col,
    "gap": gap_col,
    "gap_bucket": bucket_col,
    "year": year_col,
    "hpi_value": hpi_val_col,
    "income_value": inc_val_col,
}

# Validate minimal requirements
if not cols["state"]:
    st.error("I need a State column to build the dashboard.")
    st.stop()
if not (cols["gap"] or (cols["hpi_growth"] and cols["income_growth"])):
    st.error("I need either Affordability Gap, or both HPI Growth and Income Growth.")
    st.stop()

df, cols = compute_missing_fields(df_raw, cols)

# -----------------------------
# Filters
# -----------------------------
states = sorted(df[cols["state"]].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("States", options=states, default=states)

bucket_filter = st.sidebar.multiselect(
    "Severity buckets",
    options=sorted(df[cols["gap_bucket"]].dropna().unique().tolist()),
    default=sorted(df[cols["gap_bucket"]].dropna().unique().tolist()),
)

df_f = df[df[cols["state"]].isin(selected_states)].copy()
df_f = df_f[df_f[cols["gap_bucket"]].isin(bucket_filter)].copy()

# -----------------------------
# Header
# -----------------------------
st.title("Housing Affordability Dashboard (2015–2024)")
st.caption("Core metric: **Affordability Gap = Housing price growth − Median household income growth**")

# -----------------------------
# KPI row
# -----------------------------
gap_series = df_f[cols["gap"]].dropna()

if gap_series.empty:
    st.warning("No data after filters. Please broaden your selection.")
    st.stop()

avg_gap = gap_series.mean()
max_idx = gap_series.idxmax()
min_idx = gap_series.idxmin()

max_state = df_f.loc[max_idx, cols["state"]]
min_state = df_f.loc[min_idx, cols["state"]]

c1, c2, c3 = st.columns(3)
c1.metric("Average affordability gap", pct(avg_gap))
c2.metric("Highest gap (worst)", f"{max_state} — {pct(df_f.loc[max_idx, cols['gap']])}")
c3.metric("Lowest gap (best)", f"{min_state} — {pct(df_f.loc[min_idx, cols['gap']])}")

st.divider()

# -----------------------------
# Main visuals (Row 1)
# -----------------------------
left, right = st.columns([1.25, 1.0])

with left:
    # Growth comparison bar (state-level)
    if cols["hpi_growth"] and cols["income_growth"]:
        df_bar = df_f[[cols["state"], cols["hpi_growth"], cols["income_growth"], cols["gap"], cols["gap_bucket"]]].copy()
        df_bar = df_bar.dropna(subset=[cols["hpi_growth"], cols["income_growth"], cols["gap"]])
        df_bar = df_bar.sort_values(cols["gap"], ascending=False)

        # reshape for grouped bars
        melted = df_bar.melt(
            id_vars=[cols["state"], cols["gap"], cols["gap_bucket"]],
            value_vars=[cols["hpi_growth"], cols["income_growth"]],
            var_name="Metric",
            value_name="Growth",
        )
        # friendly labels
        metric_labels = {
            cols["hpi_growth"]: "Housing price growth (HPI)",
            cols["income_growth"]: "Median income growth",
        }
        melted["Metric"] = melted["Metric"].map(metric_labels).fillna(melted["Metric"])

        fig = px.bar(
            melted,
            x=cols["state"],
            y="Growth",
            color="Metric",
            barmode="group",
            title="10-year growth comparison by state",
            hover_data={cols["gap"]: ":.3f", cols["gap_bucket"]: True},
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(xaxis_title="", yaxis_title="Growth (2015→2024)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Growth comparison bar requires both HPI Growth and Income Growth columns.")

with right:
    # Map (requires full state names)
    df_map = df_f[[cols["state"], cols["gap"], cols["gap_bucket"]]].dropna().copy()
    fig_map = px.choropleth(
        df_map,
        locations=cols["state"],
        locationmode="USA-states",  # if your states are abbreviations; otherwise use "USA-states" requires abbreviations
        scope="usa",
        color=cols["gap"],
        hover_name=cols["state"],
        hover_data={cols["gap"]: ":.3f", cols["gap_bucket"]: True},
        title="Affordability gap by state (map)",
    )

    # If you have FULL state names (e.g., "Florida"), Plotly wants abbreviations for locationmode="USA-states".
    # Quick fallback: use Plotly's built-in name resolution via 'locations' + 'locationmode="USA-states"' won't work with full names.
    # So we detect and switch to geojson-free 'px.choropleth' using 'locations' won't resolve full names reliably.
    # We'll provide a safer alternative: map with abbreviations if you have them; otherwise show a table note.

    # Detect full names vs abbreviations
    sample = df_map[cols["state"]].iloc[0]
    if isinstance(sample, str) and len(sample.strip()) > 2:
        st.warning("Map needs **state abbreviations** (FL, ID, etc.) in Plotly. Your data looks like full names. "
                   "Quick fix: add a 'State_Abbrev' column and select it as State in the mapping panel.")
        st.dataframe(df_map.sort_values(cols["gap"], ascending=False), use_container_width=True)
    else:
        fig_map.update_coloraxes(colorbar_tickformat=".0%")
        st.plotly_chart(fig_map, use_container_width=True)

st.divider()

# -----------------------------
# Main visuals (Row 2)
# -----------------------------
left2, right2 = st.columns([1.0, 1.0])

with left2:
    df_rank = df_f[[cols["state"], cols["gap"], cols["gap_bucket"]]].dropna().copy()
    df_rank = df_rank.sort_values(cols["gap"], ascending=False)

    fig_rank = px.bar(
        df_rank,
        x=cols["gap"],
        y=cols["state"],
        orientation="h",
        color=cols["gap_bucket"],
        title="Affordability gap ranking (highest = worst)",
        hover_data={cols["gap"]: ":.3f"},
    )
    fig_rank.update_xaxes(tickformat=".0%")
    fig_rank.update_layout(xaxis_title="Affordability gap", yaxis_title="")
    st.plotly_chart(fig_rank, use_container_width=True)

with right2:
    if cols["hpi_growth"] and cols["income_growth"]:
        df_scatter = df_f[[cols["state"], cols["hpi_growth"], cols["income_growth"], cols["gap"], cols["gap_bucket"]]].dropna().copy()
        fig_scatter = px.scatter(
            df_scatter,
            x=cols["income_growth"],
            y=cols["hpi_growth"],
            color=cols["gap_bucket"],
            hover_name=cols["state"],
            title="Income growth vs Housing price growth (scatter)",
        )
        fig_scatter.update_xaxes(tickformat=".0%", title="Income growth (2015→2024)")
        fig_scatter.update_yaxes(tickformat=".0%", title="Housing price growth (2015→2024)")
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Scatter requires both HPI Growth and Income Growth columns.")

st.divider()

# -----------------------------
# Optional: Trends (if year + values exist)
# -----------------------------
st.subheader("Optional: State trends over time")

if cols["year"] and (cols["hpi_value"] or cols["hpi_growth"]) and (cols["income_value"] or cols["income_growth"]):
    st.caption("If your file includes yearly rows, select a state to view trends.")
    drill_state = st.selectbox("Pick a state for trend view", options=states)

    df_t = df[df[cols["state"]] == drill_state].copy()
    if cols["year"] in df_t.columns:
        df_t = df_t.dropna(subset=[cols["year"]]).sort_values(cols["year"])

        # Trend of values if available; else show growth by year if those exist.
        chart_cols = []
        labels = {}

        if cols["hpi_value"] and cols["hpi_value"] in df_t.columns:
            chart_cols.append(cols["hpi_value"])
            labels[cols["hpi_value"]] = "HPI (index)"
        if cols["income_value"] and cols["income_value"] in df_t.columns:
            chart_cols.append(cols["income_value"])
            labels[cols["income_value"]] = "Median income ($)"

        if chart_cols:
            df_line = df_t[[cols["year"]] + chart_cols].copy()
            df_line = df_line.melt(cols["year"], var_name="Metric", value_name="Value")
            df_line["Metric"] = df_line["Metric"].map(labels).fillna(df_line["Metric"])

            fig_line = px.line(
                df_line,
                x=cols["year"],
                y="Value",
                color="Metric",
                title=f"{drill_state}: HPI and Income over time",
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Trend view needs Year + either HPI value or Income value columns.")
else:
    st.info("Trend section requires Year column plus raw yearly HPI/Income values (optional).")

st.divider()

# -----------------------------
# Data table
# -----------------------------
with st.expander("Show filtered data table"):
    st.dataframe(df_f, use_container_width=True)

st.caption("Tip: Use the sidebar filters to narrow states and severity buckets. Export your filtered table if needed.")