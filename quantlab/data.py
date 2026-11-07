"""Convenience loaders that turn settings into provider calls."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

import pandas as pd

from .providers import MarketDataProvider
from .settings import Settings


def lookback_window(settings: Settings, end: date | None = None) -> tuple[date, date]:
    end = end or date.today()
    return end - timedelta(days=settings.data.lookback_days), end


def load_daily(provider: MarketDataProvider, symbols: Sequence[str], settings: Settings, end: date | None = None) -> dict[str, pd.DataFrame]:
    start, end = lookback_window(settings, end)
    bars = provider.daily_bars(sorted(set(symbols)), start, end)
    return {s: df for s, df in bars.items() if not df.empty}


def closes(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: df["close"] for s, df in bars.items()}).sort_index()
