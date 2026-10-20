"""Validation, normalisation and cleaning of raw fare observations.

Input contract (CSV columns)
----------------------------
Required: obs_id, origin, destination, observation_date, departure_date, fare, currency
Optional: carrier, trip_type, itinerary_id, source

Dates must be ISO-8601 (YYYY-MM-DD, optionally with a time). Fares must be plain
numbers (no thousands separators). Rows that fail a rule are excluded - never
silently repaired - and counted in the CleaningReport shown in the app.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, AnalysisConfig

REQUIRED_COLUMNS = (
    "obs_id",
    "origin",
    "destination",
    "observation_date",
    "departure_date",
    "fare",
    "currency",
)
OPTIONAL_COLUMNS = ("carrier", "trip_type", "itinerary_id", "source")

# Reason code -> human-readable rule. Order = order of evaluation: each row is
# attributed to the first rule it fails.
EXCLUSION_RULES = {
    "missing_route": "Origin or destination missing",
    "same_origin_destination": "Origin equals destination",
    "missing_or_invalid_date": "Observation or departure date missing / not ISO-8601",
    "observation_after_departure": "Fare observed after the departure date",
    "missing_currency": "Currency missing",
    "other_currency": "Currency differs from the dataset currency (no conversion is applied)",
    "invalid_fare": "Fare missing, non-numeric, zero, negative or above the sanity cap",
    "duplicate_record": "Duplicate observation (repeated obs_id or identical itinerary/date/fare)",
}

CLEAN_COLUMNS = [
    "obs_id",
    "origin",
    "destination",
    "carrier",
    "trip_type",
    "itinerary_id",
    "observation_date",
    "departure_date",
    "fare",
    "currency",
    "lead_days",
]


class SchemaError(Exception):
    """Raised when required columns are absent."""


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    excluded: dict = field(default_factory=dict)

    @property
    def rows_excluded(self) -> int:
        return self.rows_in - self.rows_out

    def as_table(self) -> pd.DataFrame:
        rows = [
            {"Rule": EXCLUSION_RULES[code], "Rows excluded": int(self.excluded.get(code, 0))}
            for code in EXCLUSION_RULES
        ]
        return pd.DataFrame(rows)


def _flag(mask: pd.Series) -> pd.Series:
    """Boolean mask with missing values treated as False."""
    return mask.fillna(False).astype(bool)


def _text(series: pd.Series, upper: bool = False) -> pd.Series:
    out = series.astype("string").str.strip()
    if upper:
        out = out.str.upper()
    return out.replace("", pd.NA)


def _date(series: pd.Series) -> pd.Series:
    out = pd.to_datetime(series, errors="coerce", format="ISO8601")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.dt.normalize()


def clean_fares(
    raw: pd.DataFrame,
    currency: str,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and normalise raw observations.

    Returns (clean_dataframe, report). lead_days = departure_date -
    observation_date in whole days, and is guaranteed >= 0 in the output.
    """
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise SchemaError(
            "Dataset is missing required column(s): " + ", ".join(missing_cols)
        )

    currency = currency.strip().upper()
    df = raw.copy()
    n_in = len(df)

    df["obs_id"] = _text(df["obs_id"])
    df["origin"] = _text(df["origin"], upper=True)
    df["destination"] = _text(df["destination"], upper=True)
    df["currency"] = _text(df["currency"], upper=True)
    for col, default in (("carrier", "UNKNOWN"), ("trip_type", "UNSPECIFIED")):
        df[col] = _text(df[col]) if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
        df[col] = df[col].fillna(default)
    df["observation_date"] = _date(df["observation_date"])
    df["departure_date"] = _date(df["departure_date"])
    df["fare"] = pd.to_numeric(df["fare"], errors="coerce").astype(float)

    reason = pd.Series([None] * n_in, index=df.index, dtype=object)

    def mark(mask: pd.Series, code: str) -> None:
        mask = _flag(mask)
        reason.loc[reason.isna() & mask] = code

    mark(df["origin"].isna() | df["destination"].isna(), "missing_route")
    mark(df["origin"] == df["destination"], "same_origin_destination")
    mark(df["observation_date"].isna() | df["departure_date"].isna(), "missing_or_invalid_date")
    mark(df["departure_date"] < df["observation_date"], "observation_after_departure")
    mark(df["currency"].isna(), "missing_currency")
    mark(df["currency"] != currency, "other_currency")
    fare = df["fare"]
    mark(
        fare.isna() | ~np.isfinite(fare) | (fare <= 0) | (fare > config.max_valid_fare),
        "invalid_fare",
    )

    valid = df[reason.isna()].copy()

    # Duplicates: a repeated obs_id, or the same itinerary observed on the same
    # day at the same fare. Keep the first occurrence.
    dup_key = ["origin", "destination", "carrier", "departure_date", "observation_date", "fare"]
    dup = valid.duplicated(subset=dup_key, keep="first")
    dup |= valid["obs_id"].notna() & valid["obs_id"].duplicated(keep="first")
    valid = valid[~dup].copy()
    duplicates = int(dup.sum())

    if "itinerary_id" in valid.columns and valid["itinerary_id"].notna().all():
        valid["itinerary_id"] = valid["itinerary_id"].astype("string")
    else:
        valid["itinerary_id"] = (
            valid["origin"]
            + "|"
            + valid["destination"]
            + "|"
            + valid["carrier"]
            + "|"
            + valid["departure_date"].dt.strftime("%Y-%m-%d")
        )
    valid["lead_days"] = (valid["departure_date"] - valid["observation_date"]).dt.days.astype(int)

    excluded = {code: int((reason == code).sum()) for code in EXCLUSION_RULES}
    excluded["duplicate_record"] = duplicates

    clean = valid[CLEAN_COLUMNS].reset_index(drop=True)
    report = CleaningReport(rows_in=n_in, rows_out=len(clean), excluded=excluded)
    return clean, report
