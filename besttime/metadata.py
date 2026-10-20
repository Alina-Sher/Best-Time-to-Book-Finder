"""Dataset provenance metadata.

Every dataset ships with a "<name>.metadata.json" file that states plainly
whether it is real or simulated, where it came from and how to read it. The app
refuses to load a dataset without this file so provenance can never be implied.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEMO_LABEL = "DEMO / SIMULATED DATA"


class MetadataError(Exception):
    """Raised when dataset metadata is missing or malformed."""


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    is_synthetic: bool
    currency: str
    source_url: str | None = None
    source_description: str = ""
    retrieved_date: str | None = None
    geographic_coverage: str = ""
    observation_semantics: str = ""
    field_definitions: dict = field(default_factory=dict)
    provenance: str = ""
    limitations: tuple = ()
    license: str | None = None

    @property
    def data_type_label(self) -> str:
        return DEMO_LABEL if self.is_synthetic else "Historical fare observations"

    @property
    def disclosure(self) -> str:
        """Banner text shown whenever the data is simulated."""
        if not self.is_synthetic:
            return ""
        return (
            f"{DEMO_LABEL}. The fares in this app are randomly generated to "
            "demonstrate the methodology. They are not real airline fares, and "
            "the booking-window patterns shown do not describe real-world "
            "booking behaviour."
        )

    @classmethod
    def from_json(cls, path: Path) -> "DatasetMeta":
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise MetadataError(
                f"Metadata file not found: {path.name}. Every dataset needs a "
                "metadata file stating whether it is real or simulated."
            ) from exc
        except json.JSONDecodeError as exc:
            raise MetadataError(f"Metadata file {path.name} is not valid JSON: {exc}") from exc

        missing = [k for k in ("name", "is_synthetic", "currency") if k not in raw]
        if missing:
            raise MetadataError(f"Metadata file {path.name} is missing: {', '.join(missing)}")
        if not isinstance(raw["is_synthetic"], bool):
            raise MetadataError("'is_synthetic' must be true or false.")

        return cls(
            name=str(raw["name"]),
            is_synthetic=raw["is_synthetic"],
            currency=str(raw["currency"]).strip().upper(),
            source_url=raw.get("source_url"),
            source_description=str(raw.get("source_description", "")),
            retrieved_date=raw.get("retrieved_date"),
            geographic_coverage=str(raw.get("geographic_coverage", "")),
            observation_semantics=str(raw.get("observation_semantics", "")),
            field_definitions=dict(raw.get("field_definitions", {})),
            provenance=str(raw.get("provenance", "")),
            limitations=tuple(raw.get("limitations", [])),
            license=raw.get("license"),
        )
