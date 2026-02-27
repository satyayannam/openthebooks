import streamlit as st

from utils.sheets import list_worksheets, load_data
from utils.filters import normalize_dataframe
from utils.theme import apply_theme


apply_theme()

st.title("Data Explorer")
st.caption("Inspect raw and normalized data from the connected Google Sheet.")

refresh = st.sidebar.button("Refresh data")

worksheet_choice = None
try:
    sheet_id = str(st.secrets.get("SHEET_ID", ""))
except Exception:
    sheet_id = ""

if sheet_id:
    available = list_worksheets(sheet_id)
    if available and not st.secrets.get("WORKSHEET_NAMES"):
        worksheet_choice = st.sidebar.selectbox("Worksheet", available, index=0)

df_raw = load_data(refresh=refresh, worksheets=worksheet_choice)

if df_raw.empty:
    st.warning("No data found. Check your Sheet ID, permissions, and worksheet names.")
    st.stop()

st.subheader("Raw data")
st.dataframe(df_raw, use_container_width=True)

st.subheader("Normalized (long) data")
df_long, _ = normalize_dataframe(df_raw)
st.dataframe(df_long, use_container_width=True)

st.caption("Tip: Keep sheet headers consistent to improve auto-detection.")
