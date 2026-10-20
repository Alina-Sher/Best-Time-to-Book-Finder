"""Plotly figures. They only consume frames from besttime.views.chart_frame, so
charts always reflect the same filtered, analysed data as the results table.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

TEAL = "#0F766E"
BLUE = "#1D4ED8"
GREY = "#94A3B8"
INK = "#0F172A"
X_TITLE = "Booking window (days before departure)"


def _base_layout(fig: go.Figure, y_title: str, order: list, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=X_TITLE,
        yaxis_title=y_title,
        legend=dict(orientation="h", yanchor="top", y=-0.28, x=0),
        font=dict(color=INK),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(categoryorder="array", categoryarray=order)
    fig.update_yaxes(tickformat=",")
    return fig


def line_chart(frame: pd.DataFrame, currency: str) -> go.Figure:
    """Median fare across windows (far -> near departure) with the IQR band."""
    fig = go.Figure()
    order = list(frame["window"])
    ok = frame[frame["eligible"]]
    low = frame[~frame["eligible"]]
    if len(ok):
        fig.add_trace(go.Scatter(x=ok["window"], y=ok["p75"], mode="lines",
                                 line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=ok["window"], y=ok["p25"], mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(15,118,110,0.15)",
                                 name="Middle 50% of fares (IQR)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=ok["window"], y=ok["median"], mode="lines+markers", name="Median fare",
            line=dict(color=TEAL, width=3), marker=dict(size=9, color=TEAL),
            customdata=ok[["n_itineraries"]].to_numpy(),
            hovertemplate="%{x}<br>Median: " + currency + " %{y:,.0f}<br>Itineraries: %{customdata[0]}<extra></extra>"))
    if len(low):
        fig.add_trace(go.Scatter(
            x=low["window"], y=low["median"], mode="markers", name="Sample too small",
            marker=dict(size=10, color="white", line=dict(color=GREY, width=2)),
            customdata=low[["n_itineraries"]].to_numpy(),
            hovertemplate="%{x}<br>Median: " + currency + " %{y:,.0f}<br>Itineraries: %{customdata[0]} (too few)<extra></extra>"))
    return _base_layout(fig, f"Median fare ({currency})", order)


def bar_chart(frame: pd.DataFrame, currency: str) -> go.Figure:
    """Median fare per window; recommended window highlighted."""
    colors = [TEAL if b else (BLUE if e else "#CBD5E1") for b, e in zip(frame["is_best"], frame["eligible"])]
    fig = go.Figure(go.Bar(
        x=frame["window"], y=frame["median"], marker_color=colors,
        text=[f"{v:,.0f}" for v in frame["median"]], textposition="outside",
        customdata=frame[["n_itineraries", "mean"]].to_numpy(),
        hovertemplate="%{x}<br>Median: " + currency + " %{y:,.0f}<br>Mean: " + currency
                      + " %{customdata[1]:,.0f}<br>Itineraries: %{customdata[0]}<extra></extra>",
        showlegend=False))
    fig.update_yaxes(rangemode="tozero")
    return _base_layout(fig, f"Median fare ({currency})", list(frame["window"]))


def sample_chart(frame: pd.DataFrame, min_required: int) -> go.Figure:
    """Distinct itineraries per window vs the minimum needed for a recommendation."""
    colors = [BLUE if e else "#CBD5E1" for e in frame["eligible"]]
    fig = go.Figure(go.Bar(
        x=frame["window"], y=frame["n_itineraries"], marker_color=colors,
        text=frame["n_itineraries"], textposition="outside",
        hovertemplate="%{x}<br>Itineraries: %{y}<extra></extra>", showlegend=False))
    fig.add_hline(y=min_required, line_dash="dash", line_color=TEAL,
                  annotation_text=f"Minimum for a recommendation ({min_required})",
                  annotation_position="top left")
    fig.update_yaxes(rangemode="tozero")
    return _base_layout(fig, "Distinct itineraries (n)", list(frame["window"]))
