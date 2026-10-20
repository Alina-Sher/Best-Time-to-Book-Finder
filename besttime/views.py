"""View-models: the exact tables the UI displays and the charts plot.

Keeping these pandas-only (no Plotly/Streamlit) means the numbers behind every
chart and table can be unit-tested, and guarantees charts and tables are built
from the same analysis result.
"""
from __future__ import annotations

import pandas as pd

from .analysis import AnalysisResult
from .formatting import format_fare, format_pct


def chart_frame(result: AnalysisResult) -> pd.DataFrame:
    """One row per booking window with data, ordered far -> near departure.

    Windows with zero observations are omitted (never plotted as zero).
    """
    if result.windows.empty:
        return pd.DataFrame()
    w = result.windows[result.windows["n_itineraries"] > 0].copy()
    w = w.sort_values("lead_min", ascending=False).reset_index(drop=True)
    best = result.recommendation.best_window if result.recommendation else None
    w["is_best"] = w["window"] == best
    return w


def display_table(result: AnalysisResult) -> pd.DataFrame:
    """Formatted comparison table (strings) built from the same frame."""
    frame = chart_frame(result)
    if frame.empty:
        return pd.DataFrame()
    cur = result.currency
    base = result.recommendation.comparison_window if result.recommendation else None
    vs_col = f"vs {base}" if base else "vs baseline"
    rows = []
    for _, r in frame.sort_values("lead_min").iterrows():
        rows.append({
            "Booking window": r["window"] + ("  ★ recommended" if r["is_best"] else ""),
            "Itineraries (n)": int(r["n_itineraries"]),
            "Raw snapshots": int(r["n_snapshots"]),
            f"Median ({cur})": format_fare(r["median"], cur).replace(f"{cur} ", ""),
            f"Mean ({cur})": format_fare(r["mean"], cur).replace(f"{cur} ", ""),
            "Typical range (25th–75th pct)": (
                f"{r['p25']:,.0f} – {r['p75']:,.0f}"
            ),
            vs_col: "baseline" if r["window"] == base else format_pct(r["vs_baseline_pct"]),
            "Sample size": "OK" if r["eligible"] else f"Too small (< {result.config.min_samples_per_window})",
        })
    return pd.DataFrame(rows)
