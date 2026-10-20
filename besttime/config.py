"""Central, explicit configuration for the analysis.

Everything that influences a recommendation lives here so it can be inspected,
documented in the UI and overridden in tests.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location. Override with the TRAVELPK_FARE_DATA environment variable.
# A dataset "foo.csv" must be accompanied by "foo.metadata.json" (see README).
DEFAULT_DATA_PATH = BASE_DIR / "data" / "demo_fares.csv"
DATA_PATH_ENV_VAR = "TRAVELPK_FARE_DATA"


def resolve_data_path() -> Path:
    override = os.environ.get(DATA_PATH_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_DATA_PATH


# Display names only. The app lists just the airports present in the dataset.
AIRPORT_NAMES = {
    "LHE": "Lahore",
    "KHI": "Karachi",
    "ISB": "Islamabad",
    "PEW": "Peshawar",
    "MUX": "Multan",
    "UET": "Quetta",
    "SKT": "Sialkot",
    "LYP": "Faisalabad",
}


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters of the booking-window analysis."""

    # Inclusive (min_days, max_days) before departure. Windows must be sorted
    # and contiguous (next min == previous max + 1).
    windows: tuple = ((0, 3), (4, 7), (8, 14), (15, 30), (31, 60), (61, 90))

    # A window is only eligible for a recommendation with at least this many
    # distinct itineraries (not raw rows).
    min_samples_per_window: int = 30
    # At least this many eligible windows are needed to make any comparison.
    min_eligible_windows: int = 3
    # Itinerary count in the recommended window at/above which it earns an
    # evidence point for sample size.
    strong_sample_size: int = 100

    # Bootstrap confidence interval for each window's median.
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42
    ci_level: float = 0.95

    # Sanity cap used to reject obviously corrupt fares (in dataset currency).
    max_valid_fare: float = 2_000_000.0

    def __post_init__(self):
        if not self.windows:
            raise ValueError("At least one booking window is required.")
        prev_hi = None
        for lo, hi in self.windows:
            if lo < 0 or hi < lo:
                raise ValueError(f"Invalid window ({lo}, {hi}).")
            if prev_hi is not None and lo != prev_hi + 1:
                raise ValueError("Booking windows must be sorted and contiguous.")
            prev_hi = hi
        if self.min_samples_per_window < 1:
            raise ValueError("min_samples_per_window must be >= 1.")
        if not 0 < self.ci_level < 1:
            raise ValueError("ci_level must be between 0 and 1.")


DEFAULT_CONFIG = AnalysisConfig()
