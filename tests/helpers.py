"""Builders for small, hand-checkable fare datasets."""
from __future__ import annotations

import pandas as pd

from besttime.cleaning import clean_fares
from besttime.config import AnalysisConfig

SMALL_CFG = AnalysisConfig(min_samples_per_window=5, min_eligible_windows=3, strong_sample_size=100)


def raw_rows(rows: list) -> pd.DataFrame:
    """rows: list of dicts; unspecified fields get valid defaults. All values become text,
    like a CSV read with dtype=str."""
    base = {"obs_id": None, "origin": "LHE", "destination": "KHI", "observation_date": "2025-03-13",
            "departure_date": "2025-03-15", "fare": "1000", "currency": "PKR", "carrier": "C1"}
    out = []
    for i, r in enumerate(rows, start=1):
        row = {**base, **r}
        if row["obs_id"] is None:
            row["obs_id"] = f"T{i:04d}"
        out.append({k: (None if v is None else str(v)) for k, v in row.items()})
    return pd.DataFrame(out, dtype=object)


def snapshot(carrier: str, lead: int, fare, dep: str = "2025-03-15", **extra) -> dict:
    obs = (pd.Timestamp(dep) - pd.Timedelta(days=lead)).strftime("%Y-%m-%d")
    return {"carrier": carrier, "observation_date": obs, "departure_date": dep, "fare": fare, **extra}


def clean(rows: list, config: AnalysisConfig = SMALL_CFG):
    return clean_fares(raw_rows(rows), "PKR", config)


def manual_dataset() -> list:
    """Five itineraries (carriers C1..C5, one departure date) each seen in three windows.

    window 0-3   (lead 2):  500 520 540 560 580 -> median 540, mean 540
    window 8-14  (lead 10): 300 320 340 360 380 -> median 340, mean 340
    window 31-60 (lead 40): 250 260 270 280 290 -> median 270, mean 270
    """
    rows = []
    for i in range(5):
        c = f"C{i + 1}"
        rows += [snapshot(c, 2, 500 + 20 * i), snapshot(c, 10, 300 + 20 * i), snapshot(c, 40, 250 + 10 * i)]
    return rows
