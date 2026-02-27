from __future__ import annotations

import numpy as np
import pandas as pd


def apply_view_mode(df: pd.DataFrame, view_mode: str) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df.copy()

    if view_mode == "Raw":
        return df2

    if view_mode == "Z-Score":
        df2["value"] = df2.groupby("metric")["value"].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else np.nan)
        )
        return df2

    if view_mode == "Indexed (First Year=100)":
        if "year" not in df2.columns:
            return df2

        group_cols = ["state", "metric"]
        if "domain" in df2.columns:
            group_cols.append("domain")

        df2 = df2.sort_values("year")

        def _index_series(s: pd.Series) -> pd.Series:
            first = s.iloc[0]
            if pd.isna(first) or first == 0:
                return s * np.nan
            return (s / first) * 100.0

        df2["value"] = df2.groupby(group_cols, dropna=False)["value"].transform(_index_series)
        return df2

    return df2


def gap_ratio(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    mode: str,
    match_domain: bool = False,
) -> pd.DataFrame:
    if df.empty:
        return df

    cols = ["state", "year", "domain", "metric", "value"]
    df2 = df[cols].copy()

    pivot_cols = ["state", "year"]
    if match_domain:
        pivot_cols.append("domain")

    wide = df2.pivot_table(index=pivot_cols, columns="metric", values="value", aggfunc="mean")
    if metric_a not in wide.columns or metric_b not in wide.columns:
        return df.iloc[0:0].copy()

    if mode == "Gap(A-B)":
        values = wide[metric_a] - wide[metric_b]
        label = f"{metric_a} - {metric_b}"
    else:
        values = wide[metric_a] / wide[metric_b].replace({0: np.nan})
        label = f"{metric_a} / {metric_b}"

    out = values.reset_index()
    out["metric"] = label
    out["value"] = out[0]
    out = out.drop(columns=[0])

    if "domain" not in out.columns:
        out["domain"] = "All"

    return out[["state", "year", "domain", "metric", "value"]]
