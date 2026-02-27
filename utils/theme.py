import streamlit as st
import plotly.express as px
import plotly.io as pio


def apply_theme() -> None:
    st.set_page_config(
        page_title="OpenTheBooks",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pio.templates["otb_dark"] = {
        "layout": {
            "paper_bgcolor": "#0f1117",
            "plot_bgcolor": "#0f1117",
            "font": {"color": "#e6e6e6"},
            "xaxis": {"gridcolor": "#232a36", "zerolinecolor": "#232a36"},
            "yaxis": {"gridcolor": "#232a36", "zerolinecolor": "#232a36"},
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
            "colorway": ["#7dd3fc", "#22d3ee", "#fbbf24", "#f97316", "#f43f5e"],
        }
    }
    px.defaults.template = "otb_dark"

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        body, .stApp { background-color: #0b0e14; color: #e6e6e6; }
        .st-emotion-cache-18ni7ap, .st-emotion-cache-1d391kg { background-color: #0b0e14; }
        h1, h2, h3 { letter-spacing: -0.5px; }
        .stMetric { background: #111827; padding: 0.75rem; border-radius: 0.6rem; border: 1px solid #1f2937; }
        .stMetric label, .stMetric div { color: #e5e7eb !important; }
        .stCaption { color: #9ca3af; }
        .stMarkdown { color: #e6e6e6; }
        .stButton > button { background: #111827; color: #e5e7eb; border: 1px solid #1f2937; }
        .stButton > button:hover { border-color: #38bdf8; color: #f8fafc; }
        .st-emotion-cache-1v0mbdj { padding-top: 0.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
