"""Dataset loading: CSV + provenance metadata -> clean, analysis-ready data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaning import CleaningReport, SchemaError, clean_fares
from .config import DEFAULT_CONFIG, AnalysisConfig, resolve_data_path
from .metadata import DatasetMeta, MetadataError


class DataLoadError(Exception):
    """User-presentable failure while loading or processing the dataset."""


@dataclass
class Dataset:
    df: pd.DataFrame
    meta: DatasetMeta
    report: CleaningReport
    path: str  # file name only; never an absolute local path


def metadata_path_for(csv_path: Path) -> Path:
    return csv_path.with_name(csv_path.stem + ".metadata.json")


def load_dataset(csv_path: Path | str | None = None, config: AnalysisConfig = DEFAULT_CONFIG) -> Dataset:
    path = Path(csv_path) if csv_path else resolve_data_path()
    try:
        meta = DatasetMeta.from_json(metadata_path_for(path))
    except MetadataError as exc:
        raise DataLoadError(str(exc)) from exc

    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except FileNotFoundError as exc:
        raise DataLoadError(f"Fare data file not found: {path.name}") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataLoadError(f"Fare data file {path.name} could not be read: {exc}") from exc

    try:
        df, report = clean_fares(raw, meta.currency, config)
    except SchemaError as exc:
        raise DataLoadError(str(exc)) from exc

    return Dataset(df=df, meta=meta, report=report, path=path.name)


def dataset_coverage(df: pd.DataFrame) -> dict:
    """Facts about what the cleaned dataset actually contains."""
    if df.empty:
        return {"routes": pd.DataFrame(), "obs_start": None, "obs_end": None,
                "dep_start": None, "dep_end": None, "airports": [], "years": []}
    routes = (
        df.groupby(["origin", "destination"])
        .agg(
            observations=("fare", "size"),
            itineraries=("itinerary_id", "nunique"),
            first_departure=("departure_date", "min"),
            last_departure=("departure_date", "max"),
        )
        .reset_index()
    )
    airports = sorted(set(df["origin"]) | set(df["destination"]))
    return {
        "routes": routes,
        "obs_start": df["observation_date"].min(),
        "obs_end": df["observation_date"].max(),
        "dep_start": df["departure_date"].min(),
        "dep_end": df["departure_date"].max(),
        "airports": airports,
        "years": sorted(int(y) for y in df["departure_date"].dt.year.unique()),
    }
