"""Small display-formatting helpers (pure functions)."""
from __future__ import annotations

import calendar
import math


def is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def format_fare(value, currency: str) -> str:
    """Whole-unit fare, e.g. 'PKR 21,340'. Missing values render as an em dash."""
    if is_missing(value):
        return "—"
    return f"{currency} {value:,.0f}"


def format_pct(value) -> str:
    """Whole-percent display; avoids implying false precision."""
    if is_missing(value):
        return "—"
    return f"{value * 100:+.0f}%"


def month_name(month: int) -> str:
    return calendar.month_name[int(month)]


def month_year(year: int, month: int) -> str:
    return f"{month_name(month)} {int(year)}"
