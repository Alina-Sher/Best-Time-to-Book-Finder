"""Booking-window analysis and recommendation logic.

Method (all thresholds live in AnalysisConfig)
----------------------------------------------
1. Filter clean observations to the chosen origin, destination and departure
   month/year.
2. lead_days = departure_date - observation_date (computed during cleaning).
3. Assign each observation to a booking window; observations outside every
   window (e.g. > 90 days) are counted but not analysed.
4. Repeated snapshots of the same itinerary inside one window are collapsed to
   their median, so the unit of analysis is "one itinerary in one window", not
   "one row". Sample sizes therefore count distinct itineraries.
5. For each window compute n, median, mean, standard deviation and the
   inter-quartile range; for eligible windows also a bootstrap CI of the median.
6. A window is eligible if it has >= min_samples_per_window itineraries. If fewer
   than min_eligible_windows windows are eligible, no recommendation is made.
7. The recommended window is the eligible window with the lowest median fare.
8. The comparison (baseline) window is the eligible window closest to
   departure ("last-minute"). Differences are reported only against it.
9. Evidence quality (Strong / Moderate / Limited) = number of these that hold:
   (a) the recommended window has >= strong_sample_size itineraries;
   (b) its bootstrap CI does not overlap the next-cheapest eligible window's CI.

No look-ahead: each observation is the fare seen on its own observation date;
nothing observed later is used to describe an earlier decision point.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AnalysisConfig
from .formatting import format_fare, format_pct, month_name, month_year

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient"
STATUS_EMPTY = "empty"

INSUFFICIENT_MESSAGE = "Not enough data for a reliable recommendation"


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
def window_label(lo: int, hi: int) -> str:
    return f"{lo}–{hi} days"


def window_labels(config: AnalysisConfig = DEFAULT_CONFIG) -> list:
    return [window_label(lo, hi) for lo, hi in config.windows]


def assign_window(lead_days: pd.Series, config: AnalysisConfig = DEFAULT_CONFIG) -> pd.Series:
    """Window label per observation (NaN when outside every window)."""
    edges = [config.windows[0][0] - 1] + [hi for _, hi in config.windows]
    cut = pd.cut(lead_days, bins=edges, labels=window_labels(config), right=True)
    return cut.astype(object)


def window_for_lead(days: int, config: AnalysisConfig = DEFAULT_CONFIG):
    """Window label for a single lead time, or None."""
    for lo, hi in config.windows:
        if lo <= days <= hi:
            return window_label(lo, hi)
    return None


# --------------------------------------------------------------------------
# Selection checks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SelectionCheck:
    code: str  # ok | same_airport | unknown_route | no_data
    message: str = ""
    months_with_data: tuple = ()  # (year, month) pairs, for no_data


def filter_route_month(df: pd.DataFrame, origin: str, destination: str, year: int, month: int) -> pd.DataFrame:
    o, d = origin.strip().upper(), destination.strip().upper()
    dep = df["departure_date"]
    mask = (df["origin"] == o) & (df["destination"] == d) & (dep.dt.year == year) & (dep.dt.month == month)
    return df[mask]


def check_selection(df: pd.DataFrame, origin: str, destination: str, year: int, month: int) -> SelectionCheck:
    o, d = origin.strip().upper(), destination.strip().upper()
    if o == d:
        return SelectionCheck("same_airport", "Departure and destination must be different airports.")
    route = df[(df["origin"] == o) & (df["destination"] == d)]
    if route.empty:
        return SelectionCheck(
            "unknown_route",
            f"The dataset has no fare observations for {o} → {d}. "
            "Choose one of the routes the dataset covers.",
        )
    if filter_route_month(df, o, d, year, month).empty:
        months = (
            route["departure_date"].dt.to_period("M").drop_duplicates().sort_values()
        )
        pairs = tuple((p.year, p.month) for p in months)
        return SelectionCheck(
            "no_data",
            f"No fare observations for {o} → {d} departing in {month_year(year, month)}.",
            pairs,
        )
    return SelectionCheck("ok")


# --------------------------------------------------------------------------
# Result objects
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Recommendation:
    best_window: str
    best_median: float
    best_mean: float
    best_n: int
    comparison_window: str | None
    comparison_median: float | None
    saving_abs: float | None
    saving_pct: float | None  # fraction of the comparison median (0.12 = 12 %)
    runner_up_window: str | None
    runner_up_median: float | None
    ci_separated: bool | None
    evidence_level: str
    evidence_points: tuple = ()


@dataclass
class AnalysisResult:
    status: str
    origin: str
    destination: str
    year: int
    month: int
    currency: str
    config: AnalysisConfig
    n_rows_selected: int = 0
    n_rows_outside_windows: int = 0
    n_itineraries: int = 0
    windows: pd.DataFrame = field(default_factory=pd.DataFrame)
    recommendation: Recommendation | None = None
    message: str = ""

    @property
    def route_label(self) -> str:
        return f"{self.origin} → {self.destination}"

    @property
    def period_label(self) -> str:
        return month_year(self.year, self.month)


WINDOW_COLUMNS = [
    "window", "lead_min", "lead_max", "n_itineraries", "n_snapshots",
    "median", "mean", "std", "p25", "p75", "ci_low", "ci_high", "eligible",
    "vs_baseline_pct",
]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _bootstrap_median_ci(values: np.ndarray, rng: np.random.Generator, config: AnalysisConfig):
    draws = rng.choice(values, size=(config.bootstrap_resamples, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    alpha = (1 - config.ci_level) / 2
    lo, hi = np.quantile(medians, [alpha, 1 - alpha])
    return float(lo), float(hi)


def compute_window_stats(units: pd.DataFrame, config: AnalysisConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Per-window statistics from itinerary-level units.

    `units` has columns: window, fare (one row per itinerary per window) and
    n_snapshots (raw rows collapsed into that unit).
    """
    rng = np.random.default_rng(config.bootstrap_seed)
    rows = []
    for (lo, hi), label in zip(config.windows, window_labels(config)):
        u = units[units["window"] == label]
        fares = u["fare"].to_numpy(dtype=float)
        n = len(fares)
        row = {
            "window": label, "lead_min": lo, "lead_max": hi,
            "n_itineraries": n, "n_snapshots": int(u["n_snapshots"].sum()) if n else 0,
            "median": np.nan, "mean": np.nan, "std": np.nan, "p25": np.nan, "p75": np.nan,
            "ci_low": np.nan, "ci_high": np.nan,
            "eligible": n >= config.min_samples_per_window,
            "vs_baseline_pct": np.nan,
        }
        if n:
            row["median"] = float(np.median(fares))
            row["mean"] = float(np.mean(fares))
            row["std"] = float(np.std(fares, ddof=1)) if n > 1 else np.nan
            row["p25"], row["p75"] = (float(q) for q in np.quantile(fares, [0.25, 0.75]))
            if row["eligible"]:
                row["ci_low"], row["ci_high"] = _bootstrap_median_ci(fares, rng, config)
        rows.append(row)
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def _evidence(best: pd.Series, runner: pd.Series, config: AnalysisConfig):
    points = []
    if best["n_itineraries"] >= config.strong_sample_size:
        points.append(f"Recommended window has {int(best['n_itineraries'])} itineraries "
                      f"(≥ {config.strong_sample_size}).")
    separated = bool(best["ci_high"] < runner["ci_low"])
    if separated:
        points.append(
            f"The {int(config.ci_level * 100)}% bootstrap interval for its median does not overlap "
            f"the next-cheapest window ({runner['window']})."
        )
    level = {2: "Strong", 1: "Moderate", 0: "Limited"}[len(points)]
    return level, tuple(points), separated


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def analyze_booking_window(
    df: pd.DataFrame,
    origin: str,
    destination: str,
    year: int,
    month: int,
    currency: str,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> AnalysisResult:
    o, d = origin.strip().upper(), destination.strip().upper()
    result = AnalysisResult(status=STATUS_EMPTY, origin=o, destination=d, year=int(year),
                            month=int(month), currency=currency, config=config)

    sel = filter_route_month(df, o, d, year, month) if len(df) else df
    result.n_rows_selected = len(sel)
    if sel.empty:
        result.message = f"No fare observations for {result.route_label} departing in {result.period_label}."
        return result

    sel = sel.assign(window=assign_window(sel["lead_days"], config))
    in_win = sel[sel["window"].notna()]
    result.n_rows_outside_windows = len(sel) - len(in_win)
    if in_win.empty:
        result.message = (
            f"All {len(sel)} observations for {result.route_label} in {result.period_label} "
            f"fall outside the configured booking windows."
        )
        return result

    units = (
        in_win.groupby(["itinerary_id", "window"], observed=True)["fare"]
        .agg(fare="median", n_snapshots="size")
        .reset_index()
    )
    result.n_itineraries = int(in_win["itinerary_id"].nunique())
    stats = compute_window_stats(units, config)
    result.windows = stats

    eligible = stats[stats["eligible"]]
    if len(eligible) < config.min_eligible_windows:
        result.status = STATUS_INSUFFICIENT
        result.message = (
            f"{INSUFFICIENT_MESSAGE}. Only {len(eligible)} of {len(stats)} booking windows have at "
            f"least {config.min_samples_per_window} itineraries for {result.route_label} in "
            f"{result.period_label}; at least {config.min_eligible_windows} are needed to compare."
        )
        return result

    # Lowest median wins; ties go to the earlier (longer lead time) window.
    ranked = eligible.sort_values(["median", "lead_min"], ascending=[True, False])
    best, runner = ranked.iloc[0], ranked.iloc[1]
    baseline = eligible.sort_values("lead_min").iloc[0]  # last-minute baseline

    saving_abs = saving_pct = None
    if best["window"] != baseline["window"]:
        saving_abs = float(baseline["median"] - best["median"])
        saving_pct = saving_abs / float(baseline["median"])

    level, points, separated = _evidence(best, runner, config)

    stats.loc[stats["eligible"], "vs_baseline_pct"] = (
        stats.loc[stats["eligible"], "median"] / float(baseline["median"]) - 1
    )

    result.status = STATUS_OK
    result.recommendation = Recommendation(
        best_window=str(best["window"]), best_median=float(best["median"]),
        best_mean=float(best["mean"]), best_n=int(best["n_itineraries"]),
        comparison_window=str(baseline["window"]), comparison_median=float(baseline["median"]),
        saving_abs=saving_abs, saving_pct=saving_pct,
        runner_up_window=str(runner["window"]), runner_up_median=float(runner["median"]),
        ci_separated=separated, evidence_level=level, evidence_points=points,
    )
    return result


# --------------------------------------------------------------------------
# Plain-language explanation (shared by UI and tests)
# --------------------------------------------------------------------------
def explain(result: AnalysisResult) -> list:
    """Sentences explaining how the recommendation follows from the data."""
    if result.status != STATUS_OK or result.recommendation is None:
        return [result.message]
    r, c, cfg = result.recommendation, result.currency, result.config
    lines = [
        f"Among {result.n_itineraries} itineraries departing {result.route_label} in "
        f"{result.period_label}, fares seen {r.best_window} before departure had the lowest "
        f"median: {format_fare(r.best_median, c)} across {r.best_n} itineraries.",
        f"Only windows with at least {cfg.min_samples_per_window} itineraries were considered "
        f"({int(result.windows['eligible'].sum())} of {len(result.windows)} qualified).",
    ]
    if r.runner_up_window:
        gap = r.runner_up_median / r.best_median - 1
        lines.append(
            f"The next-cheapest window, {r.runner_up_window}, had a median of "
            f"{format_fare(r.runner_up_median, c)} ({format_pct(gap)} vs the recommended window)."
        )
    if r.saving_abs is not None:
        lines.append(
            f"Compared with the closest-to-departure qualifying window ({r.comparison_window}, median "
            f"{format_fare(r.comparison_median, c)}), the recommended window's median was "
            f"{format_fare(r.saving_abs, c)} lower ({abs(r.saving_pct) * 100:.0f}%). "
            "This is a difference in past observed medians, not a guaranteed saving."
        )
    else:
        lines.append(
            f"The closest-to-departure qualifying window ({r.comparison_window}) was itself the "
            "cheapest by median, so no saving against it is claimed."
        )
    return lines


CAVEATS = (
    "Historical association, not a forecast: past fare patterns may not repeat.",
    "Itineraries on nearby dates and carriers share demand conditions, so the effective "
    "sample is smaller than the itinerary count suggests.",
    "Windows may contain different itineraries; differences can reflect which flights were "
    "observed, not only when they were booked.",
    "Fares are observations, not bookable offers; taxes, baggage and availability may differ.",
)
