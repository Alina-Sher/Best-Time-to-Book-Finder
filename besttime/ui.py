"""Streamlit UI components. No analysis happens here - only presentation."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from . import charts, views
from .analysis import CAVEATS, INSUFFICIENT_MESSAGE, STATUS_EMPTY, STATUS_INSUFFICIENT, STATUS_OK, AnalysisResult, explain
from .config import AIRPORT_NAMES, DEFAULT_CONFIG
from .formatting import format_fare, format_pct, month_name, month_year
from .loader import Dataset, dataset_coverage
from .metadata import DEMO_LABEL
from .cleaning import EXCLUSION_RULES

CSS = """
<style>
:root { --teal:#0F766E; --blue:#1D4ED8; --navy:#0B2A4A; --ink:#0F172A; --muted:#475569; --line:#E2E8F0; }
.block-container { padding-top: 1.2rem; max-width: 1180px; }
.tp-hero { background: linear-gradient(120deg, #0B2A4A 0%, #0F5C7A 55%, #0F766E 100%);
  color:#fff; padding: 22px 26px; border-radius: 16px; margin-bottom: 14px; }
.tp-brand { font-weight: 800; letter-spacing:.2px; font-size: 1.05rem; opacity:.95; }
.tp-brand span { color:#7DD3C8; }
.tp-hero h1 { color:#fff; font-size: 2rem; line-height:1.15; margin:.35rem 0 .35rem 0; padding:0; }
.tp-hero p { color:#DCEBF3; margin:0; font-size:1rem; max-width: 760px; }
.tp-banner { background:#FFF4E0; border:1px solid #E0A800; border-left:6px solid #E0A800;
  color:#5C3D00; padding:12px 16px; border-radius:10px; margin: 6px 0 14px 0; font-size:.95rem; }
.tp-banner strong { letter-spacing:.3px; }
.tp-card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:18px 20px; margin: 8px 0; }
.tp-rec { border-left:6px solid var(--teal); background:#F0FDFA; }
.tp-rec .tp-eyebrow { color:var(--teal); font-weight:700; font-size:.8rem; text-transform:uppercase; letter-spacing:.6px; }
.tp-rec .tp-big { color:var(--navy); font-size:1.9rem; font-weight:800; line-height:1.2; margin:.2rem 0; }
.tp-rec .tp-sub { color:var(--muted); font-size:1rem; }
.tp-badge { display:inline-block; padding:3px 10px; border-radius:999px; font-weight:700; font-size:.82rem; }
.tp-b-strong { background:#DCFCE7; color:#14532D; } .tp-b-moderate { background:#DBEAFE; color:#1E3A8A; }
.tp-b-limited { background:#FEF3C7; color:#78350F; }
.tp-tag-demo { display:inline-block; background:#FFF4E0; color:#5C3D00; border:1px solid #E0A800;
  padding:2px 9px; border-radius:6px; font-size:.78rem; font-weight:700; margin-left:8px; }
.tp-muted { color:var(--muted); font-size:.92rem; }
.tp-footer { border-top:1px solid var(--line); margin-top:28px; padding-top:14px; color:var(--muted); font-size:.85rem; }
div[data-testid="stForm"] { border:1px solid var(--line); border-radius:14px; background:#fff; padding:14px 16px; }
@media (max-width: 640px) {
  .tp-hero { padding:16px; } .tp-hero h1 { font-size:1.5rem; }
  .tp-rec .tp-big { font-size:1.4rem; }
}
</style>
"""


def esc(value) -> str:
    return html.escape(str(value))


def airport_label(code: str) -> str:
    name = AIRPORT_NAMES.get(code)
    return f"{name} ({code})" if name else code


# ---------------------------------------------------------------- layout
def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="tp-hero">
          <div class="tp-brand">Travel<span>.pk</span> &nbsp;·&nbsp; Fare Insights</div>
          <h1>Best Time to Book Finder</h1>
          <p>For your route and travel month, see how far in advance fares have historically been
          lowest — with the evidence, sample sizes and limits shown.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_banner(meta) -> None:
    if meta.is_synthetic:
        st.markdown(
            f'<div class="tp-banner" role="alert"><strong>{DEMO_LABEL}.</strong> '
            f"{esc(meta.disclosure.partition('. ')[2] or meta.disclosure)}</div>",
            unsafe_allow_html=True,
        )


def render_footer() -> None:
    st.markdown(
        """
        <div class="tp-footer">
          <strong>Travel.pk · Best Time to Book Finder</strong> — an analytics prototype built for TechAbout.
          Historical fare patterns are not a guarantee of future prices. Fares shown are observations, not
          bookable offers. Not an official Travel.pk booking tool.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error(title: str, detail: str = "") -> None:
    st.error(f"**{title}**" + (f"\n\n{detail}" if detail else ""))


# ---------------------------------------------------------------- states
def render_initial_state(ds: Dataset) -> None:
    cov = dataset_coverage(ds.df)
    routes = ", ".join(f"{r.origin}→{r.destination}" for r in cov["routes"].itertuples())
    st.markdown(
        f"""
        <div class="tp-card">
          <strong>Choose a route and travel month, then press “Find Best Time to Book”.</strong>
          <p class="tp-muted" style="margin:.4rem 0 0 0;">
            The app compares fares seen 0–90 days before departure, grouped into booking windows, and
            recommends a window only when enough data supports it.<br>
            Routes in this dataset: <b>{esc(routes)}</b> · Departures: <b>{esc(cov['dep_start'].strftime('%b %Y'))}
            – {esc(cov['dep_end'].strftime('%b %Y'))}</b> · Currency: <b>{esc(ds.meta.currency)}</b>
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_selection_problem(check, ds: Dataset) -> None:
    if check.code == "same_airport":
        st.warning(f"**Invalid route.** {check.message}")
    elif check.code == "unknown_route":
        cov = dataset_coverage(ds.df)
        routes = "\n".join(f"- {airport_label(r.origin)} → {airport_label(r.destination)}" for r in cov["routes"].itertuples())
        st.warning(f"**Route not in dataset.** {check.message}\n\nAvailable routes:\n\n{routes}")
    elif check.code == "no_data":
        months = ", ".join(month_year(y, m) for y, m in check.months_with_data)
        st.info(f"**No data for this month.** {check.message}\n\nMonths with data for this route: {months}.")


# ---------------------------------------------------------------- results
def _badge(level: str) -> str:
    return f'<span class="tp-badge tp-b-{level.lower()}">{esc(level)} evidence</span>'


def render_result(result: AnalysisResult, ds: Dataset) -> None:
    cur = result.currency
    st.markdown(f"### {esc(result.route_label)} · {esc(result.period_label)}")

    if result.status == STATUS_EMPTY:
        st.info(f"**No data to analyse.** {result.message}")
        return

    if result.status == STATUS_INSUFFICIENT:
        st.warning(f"**{INSUFFICIENT_MESSAGE}.**\n\n{result.message.partition('. ')[2] or result.message}")
        _render_coverage_charts(result)
        return

    rec = result.recommendation
    tag = f'<span class="tp-tag-demo">{DEMO_LABEL}</span>' if ds.meta.is_synthetic else ""
    st.markdown(
        f"""
        <div class="tp-card tp-rec">
          <div class="tp-eyebrow">Recommended booking window {tag}</div>
          <div class="tp-big">Book {esc(rec.best_window)} before departure</div>
          <div class="tp-sub">Typical fare in that window (median):
            <b>{esc(format_fare(rec.best_median, cur))}</b> · based on {rec.best_n} itineraries ·
            {_badge(rec.evidence_level)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median fare, recommended window", format_fare(rec.best_median, cur))
    c2.metric(f"Median fare, {rec.comparison_window} (baseline)", format_fare(rec.comparison_median, cur))
    if rec.saving_abs is not None:
        c3.metric("Past difference vs baseline", format_fare(rec.saving_abs, cur),
                  delta=f"-{abs(rec.saving_pct) * 100:.0f}%", delta_color="off")
    else:
        c3.metric("Past difference vs baseline", "None")
    c4.metric("Itineraries analysed", f"{result.n_itineraries}")

    st.markdown("**Why this window?**")
    for line in explain(result):
        st.markdown(f"- {line}")
    level_rule = (
        f"Evidence quality is *{rec.evidence_level}*: one point for ≥ {result.config.strong_sample_size} "
        "itineraries in the recommended window, one for a bootstrap interval that does not overlap the "
        "next-cheapest window (2 = Strong, 1 = Moderate, 0 = Limited)."
    )
    st.caption(level_rule + " Met: " + (" ".join(rec.evidence_points) if rec.evidence_points else "none."))
    if ds.meta.is_synthetic:
        st.warning(f"**{DEMO_LABEL}** — this result illustrates the method on simulated fares and says "
                   "nothing about real-world booking behaviour.")

    st.markdown("#### Fare by booking window")
    frame = views.chart_frame(result)
    st.plotly_chart(charts.line_chart(frame, cur))
    left, right = st.columns(2)
    with left:
        st.markdown("**Median fare by window**")
        st.plotly_chart(charts.bar_chart(frame, cur))
    with right:
        st.markdown("**Data coverage (sample size)**")
        st.plotly_chart(charts.sample_chart(frame, result.config.min_samples_per_window))
    _render_table(result)

    with st.expander("Limitations of this recommendation", expanded=False):
        for c in CAVEATS:
            st.markdown(f"- {c}")
        if result.n_rows_outside_windows:
            st.markdown(f"- {result.n_rows_outside_windows} observation(s) for this selection fell outside "
                        "all booking windows and were not analysed.")


def _render_coverage_charts(result: AnalysisResult) -> None:
    frame = views.chart_frame(result)
    if frame.empty:
        return
    st.markdown("**Data coverage (sample size)**")
    st.plotly_chart(charts.sample_chart(frame, result.config.min_samples_per_window))
    _render_table(result)


def _render_table(result: AnalysisResult) -> None:
    st.markdown("#### Comparison table")
    st.dataframe(views.display_table(result), hide_index=True)
    st.caption(
        f"Median/mean in {result.currency}. 'Itineraries' = distinct route + carrier + departure-date "
        "combinations observed in the window; repeated snapshots of one itinerary are collapsed to their "
        "median so they are not double-counted. The baseline is the closest-to-departure window with enough data."
    )


# ---------------------------------------------------------------- methodology
def render_methodology(ds: Dataset) -> None:
    meta, cfg, cov = ds.meta, DEFAULT_CONFIG, dataset_coverage(ds.df)
    st.markdown("---")
    st.header("Data sources and methodology")

    st.subheader("Dataset")
    kind = ("**SIMULATED / DEMO** — not real airline data" if meta.is_synthetic
            else "**Historical fare observations** (real data)")
    facts = {
        "Dataset": meta.name,
        "Data type": kind.replace("**", ""),
        "Provider / original source URL": meta.source_url or "None — generated locally (see description)",
        "Source description": meta.source_description,
        "Retrieval / generation date": meta.retrieved_date or "Not stated",
        "Geographic & route coverage": meta.geographic_coverage,
        "Observation period": (f"{cov['obs_start']:%d %b %Y} – {cov['obs_end']:%d %b %Y}" if cov["obs_start"] is not None else "—"),
        "Departure (travel) period": (f"{cov['dep_start']:%d %b %Y} – {cov['dep_end']:%d %b %Y}" if cov["dep_start"] is not None else "—"),
        "Currency / units": f"{meta.currency}, whole currency units, one-way fares",
        "Rows loaded → analysed": f"{ds.report.rows_in:,} → {ds.report.rows_out:,}",
        "Licence": meta.license or "Not stated",
    }
    st.dataframe(pd.DataFrame({"Item": list(facts), "Value": list(facts.values())}), hide_index=True)
    if meta.observation_semantics:
        st.markdown(f"**What a row means:** {meta.observation_semantics}")
    if meta.provenance:
        st.markdown(f"**Provenance:** {meta.provenance}")

    st.subheader("Routes in the dataset")
    routes = cov["routes"].copy()
    if not routes.empty:
        routes["route"] = routes["origin"] + " → " + routes["destination"]
        routes["departures"] = (routes["first_departure"].dt.strftime("%b %Y") + " – "
                                + routes["last_departure"].dt.strftime("%b %Y"))
        st.dataframe(routes[["route", "observations", "itineraries", "departures"]], hide_index=True)
    st.caption("Only routes present in the data are offered; no route coverage is assumed.")

    st.subheader("Field definitions")
    if meta.field_definitions:
        st.dataframe(pd.DataFrame({"Field": list(meta.field_definitions), "Definition": list(meta.field_definitions.values())}),
                     hide_index=True)

    st.subheader("How the recommendation is calculated")
    windows = ", ".join(f"{lo}–{hi}" for lo, hi in cfg.windows)
    st.markdown(
        f"""
1. **Filter** to the chosen origin, destination and departure month/year.
2. **Lead time** = departure date − fare observation date, in days.
3. **Group** observations into booking windows ({windows} days before departure; configurable in `besttime/config.py`).
4. **Collapse repeats:** several snapshots of the same itinerary in one window count once (their median), so rows are not treated as independent.
5. **Statistics** per window: itinerary count, median, mean, standard deviation, 25th–75th percentile range, and a {int(cfg.ci_level * 100)}% bootstrap interval for the median ({cfg.bootstrap_resamples:,} resamples, fixed seed).
6. **Eligibility:** a window needs at least **{cfg.min_samples_per_window}** itineraries, and at least **{cfg.min_eligible_windows}** windows must qualify; otherwise the app says *"{INSUFFICIENT_MESSAGE}"*.
7. **Recommendation:** the eligible window with the lowest median fare (ties go to the longer lead time).
8. **Comparison:** differences are reported only against the closest-to-departure eligible window (the "baseline").
9. **Evidence quality:** Strong / Moderate / Limited from two rules — ≥ {cfg.strong_sample_size} itineraries in the recommended window, and a non-overlapping bootstrap interval versus the next-cheapest window. This is a rule-based label, not a probability.
10. **No look-ahead:** each fare is used only as the price seen on its own observation date.
"""
    )

    st.subheader("Cleaning and exclusion rules")
    st.markdown("Rows are excluded, never silently repaired. No currency conversion is applied: rows in a currency "
                "other than the dataset currency are excluded. Missing carrier is labelled UNKNOWN; missing dates, "
                "currency, route or fare exclude the row.")
    st.dataframe(ds.report.as_table(), hide_index=True)

    st.subheader("Limitations")
    for item in meta.limitations:
        st.markdown(f"- {item}")
    for item in CAVEATS:
        st.markdown(f"- {item}")
    st.markdown("- The analysis does not claim a best day of the week, a universal advance-booking rule, or any future price movement.")
