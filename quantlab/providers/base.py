"""Market-data provider interface.

Bar frames: DatetimeIndex, columns ``open, high, low, close, volume``.
  - daily bars are indexed by tz-naive session date (midnight)
  - intraday bars are indexed by tz-aware timestamps in the market timezone
Option chains: one row per contract with columns
  ``symbol, underlying, expiry (Timestamp), strike, right ('call'|'put'), bid, ask, mid, provider_iv``.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, Sequence, runtime_checkable

import pandas as pd

BAR_COLUMNS = ("open", "high", "low", "close", "volume")
CHAIN_COLUMNS = ("symbol", "underlying", "expiry", "strike", "right", "bid", "ask", "mid", "provider_iv")


class ProviderUnavailable(RuntimeError):
    """Raised when a provider cannot serve data (missing keys, network policy, auth failure)."""


@runtime_checkable
class MarketDataProvider(Protocol):
    @property
    def name(self) -> str: ...

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, pd.DataFrame]: ...

    def intraday_bars(self, symbols: Sequence[str], session: date, bar_minutes: int) -> dict[str, pd.DataFrame]: ...

    def option_chain(self, underlying: str, spot: float, as_of: date) -> pd.DataFrame: ...


def normalize_bars(df: pd.DataFrame, daily: bool, tz: str) -> pd.DataFrame:
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bars missing columns {missing}")
    out = df.loc[:, list(BAR_COLUMNS)].astype(float).sort_index()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(tz)
    out.index = idx.normalize().tz_localize(None) if daily else idx
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "timestamp"
    return out
