"""Generate the SIMULATED demo dataset (data/demo_fares.csv + metadata).

!!! Every number produced here is random and invented. !!!
The fares are NOT real airline data and the booking-window patterns are
assumptions built into the simulation below. The output exists only to
demonstrate the app's methodology until a real dataset is supplied.

Usage:  python data/generate_demo_data.py
The output is deterministic (fixed seed), so the committed CSV is reproducible.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20250101
OUT_DIR = Path(__file__).resolve().parent
YEAR = 2025
WINDOWS = ((0, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90))
SNAPSHOTS_PER_ITINERARY = 10  # one in every window + extras

# Illustrative assumptions only: base fare (PKR) and the lead time (days) at
# which the simulated fare curve is lowest.
ROUTES = {
    ("LHE", "KHI"): {"base": 24000, "sweet": 35},
    ("KHI", "LHE"): {"base": 24500, "sweet": 28},
    ("LHE", "ISB"): {"base": 15500, "sweet": 18},
    ("ISB", "LHE"): {"base": 15000, "sweet": 21},
    ("KHI", "ISB"): {"base": 25500, "sweet": 32},
    ("ISB", "KHI"): {"base": 26000, "sweet": 40},
}
# Generic placeholder carriers - NOT real airlines.
CARRIERS = {"Carrier A": 1.00, "Carrier B": 0.93}
SEASONALITY = {1: .95, 2: .92, 3: 1.0, 4: 1.10, 5: 1.05, 6: 1.12,
               7: 1.08, 8: 1.0, 9: .97, 10: 1.0, 11: 1.03, 12: 1.15}
DOW = {0: 1.0, 1: .94, 2: .94, 3: 1.0, 4: 1.08, 5: 1.02, 6: 1.08}

# Deliberately thin / missing cells so the app's edge-case states can be seen.
SPARSE_CELL = ("ISB", "KHI", 2)   # route + month with only a handful of itineraries
MISSING_CELL = ("KHI", "ISB", 1)  # route + month with no departures at all


def lead_factor(lead: np.ndarray, sweet: int) -> np.ndarray:
    last_minute = 1 + 0.60 * np.exp(-lead / 4.0)
    curvature = 1 + 0.10 * ((lead - sweet) / 45.0) ** 2
    return last_minute * curvature


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (origin, dest), p in ROUTES.items():
        for day in pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31"):
            if (origin, dest, day.month) == MISSING_CELL:
                continue
            for ci, (carrier, c_mult) in enumerate(CARRIERS.items()):
                if (origin, dest, day.month) == SPARSE_CELL and (ci != 0 or day.day % 4 != 0):
                    continue
                level = (p["base"] * SEASONALITY[day.month] * DOW[day.dayofweek] * c_mult
                         * rng.lognormal(0.0, 0.06))
                leads = {int(rng.integers(lo, hi + 1)) for lo, hi in WINDOWS}
                while len(leads) < SNAPSHOTS_PER_ITINERARY:
                    leads.add(int(rng.integers(0, 91)))
                lead_arr = np.array(sorted(leads), dtype=float)
                fares = level * lead_factor(lead_arr, p["sweet"]) * rng.lognormal(0.0, 0.05, len(lead_arr))
                for lead, fare in zip(lead_arr.astype(int), fares):
                    rows.append((origin, dest, carrier, "ONE_WAY",
                                 (day - pd.Timedelta(days=int(lead))).date().isoformat(),
                                 day.date().isoformat(), int(round(fare, -1)), "PKR", "SIMULATED"))
    df = pd.DataFrame(rows, columns=["origin", "destination", "carrier", "trip_type",
                                     "observation_date", "departure_date", "fare", "currency", "source"])
    df = df.sort_values(["origin", "destination", "departure_date", "carrier", "observation_date"]).reset_index(drop=True)
    df.insert(0, "obs_id", [f"SIM-{i:06d}" for i in range(1, len(df) + 1)])
    return df


def metadata(df: pd.DataFrame) -> dict:
    return {
        "name": "Travel.pk Best-Time-to-Book DEMO dataset (SIMULATED - not real airline data)",
        "is_synthetic": True,
        "currency": "PKR",
        "source_url": None,
        "source_description": (
            "Generated locally by data/generate_demo_data.py using a seeded random simulation. "
            "No external provider, API or website was used."
        ),
        "retrieved_date": f"Generated on {dt.date.today().isoformat()}",
        "geographic_coverage": (
            "Pakistan domestic, one-way: LHE→KHI, KHI→LHE, LHE→ISB, ISB→LHE, KHI→ISB, ISB→KHI "
            "(these routes exist only as simulated rows)."
        ),
        "observation_semantics": (
            "Each row is one simulated fare snapshot for one simulated itinerary (route + carrier + "
            "departure date), as if observed on observation_date. Each itinerary has "
            f"{SNAPSHOTS_PER_ITINERARY} snapshots at different lead times."
        ),
        "field_definitions": {
            "obs_id": "Unique observation identifier (SIM-nnnnnn).",
            "origin": "Departure airport IATA code.",
            "destination": "Arrival airport IATA code.",
            "carrier": "Generic placeholder carrier name (not a real airline).",
            "trip_type": "ONE_WAY.",
            "observation_date": "Date the simulated fare was 'seen' (ISO-8601).",
            "departure_date": "Date of the simulated flight (ISO-8601).",
            "fare": "Simulated one-way fare in whole PKR.",
            "currency": "ISO-4217 code; all rows are PKR.",
            "source": "Always SIMULATED here.",
        },
        "provenance": (
            "All fares are random draws from formulas whose parameters (base fare, seasonality, "
            "weekday effect, last-minute premium, route-specific cheapest lead time, noise) are "
            "illustrative assumptions by the developer. They are not estimates of real fares, "
            "airlines or traveller behaviour. Carrier names are generic placeholders."
        ),
        "limitations": [
            "Booking-window patterns in this dataset are produced by the simulation's own assumptions, "
            "so any recommendation demonstrates the method only.",
            f"Departures cover calendar year {YEAR} only; observations start up to 90 days earlier.",
            "ISB→KHI in February and KHI→ISB in January are deliberately thin/missing to exercise "
            "the app's insufficient-data and empty-data states.",
            "Every itinerary has snapshots in every booking window by construction, unlike real scrapes.",
        ],
        "license": "Generated data, free to use and replace.",
    }


def main() -> None:
    df = build()
    df.to_csv(OUT_DIR / "demo_fares.csv", index=False)
    (OUT_DIR / "demo_fares.metadata.json").write_text(
        json.dumps(metadata(df), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(df):,} simulated rows to demo_fares.csv")


if __name__ == "__main__":
    main()
