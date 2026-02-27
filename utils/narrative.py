from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def narrative_overview(df: pd.DataFrame, metric: str | None) -> Dict[str, List[str] | str]:
    if df.empty or not metric:
        return {"what": "No data available.", "facts": [], "why": ""}

    dfm = df[df["metric"] == metric].copy()
    if dfm.empty:
        return {"what": "No data available.", "facts": [], "why": ""}

    latest_year = None
    if "year" in dfm.columns:
        years = pd.to_numeric(dfm["year"], errors="coerce").dropna()
        if not years.empty:
            latest_year = int(years.max())
            dfm = dfm[dfm["year"] == latest_year]

    dfm = dfm.dropna(subset=["value"])
    if dfm.empty:
        return {"what": "No data available.", "facts": [], "why": ""}

    avg = dfm["value"].mean()
    med = dfm["value"].median()
    top = dfm.loc[dfm["value"].idxmax()]
    bottom = dfm.loc[dfm["value"].idxmin()]

    facts = [
        f"Average {metric}: {avg:,.2f}",
        f"Median {metric}: {med:,.2f}",
        f"Highest: {top['state']} at {top['value']:,.2f}",
        f"Lowest: {bottom['state']} at {bottom['value']:,.2f}",
    ]

    what = f"{metric} across {dfm['state'].nunique()} states"
    if latest_year:
        what += f" (latest year: {latest_year})."
    else:
        what += "."

    why = "Large gaps between states often reflect differences in policy, funding, or economic conditions."

    return {"what": what, "facts": facts, "why": why}
