"""CSV cache decorator for any MarketDataProvider.

Cache keys include the provider name (which encodes feed/adjustment), timeframe and
range, so switching feeds or adjustments never serves stale data from another config.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from .base import MarketDataProvider

log = logging.getLogger(__name__)


class CachedProvider:
    def __init__(self, inner: MarketDataProvider, cache_dir: Path, max_age_hours: float, tz: str, refresh: bool = False):
        self._inner = inner
        self._dir = cache_dir
        self._max_age_s = max_age_hours * 3600.0
        self._tz = tz
        self._refresh = refresh
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def inner(self) -> MarketDataProvider:
        return self._inner

    def _path(self, kind: str, symbol: str, *key: object) -> Path:
        digest = hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:10]
        safe = "".join(c if c.isalnum() else "_" for c in symbol)
        return self._dir / f"{self._inner.name}__{kind}__{safe}__{digest}.csv"

    def _fresh(self, path: Path) -> bool:
        return (not self._refresh) and path.is_file() and (time.time() - path.stat().st_mtime) <= self._max_age_s

    def _read(self, path: Path, daily: bool) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0)
        idx = pd.to_datetime(df.index, utc=not daily)
        df.index = idx if daily else idx.tz_convert(self._tz)
        df.index.name = "timestamp"
        return df

    def _cached(self, kind: str, symbols: Sequence[str], key: tuple, fetch, daily: bool) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        paths = {s: self._path(kind, s, *key) for s in symbols}
        missing = [s for s, p in paths.items() if not self._fresh(p)]
        for s in symbols:
            if s not in missing:
                out[s] = self._read(paths[s], daily)
        if missing:
            log.info("fetching %s %s from %s", kind, missing, self._inner.name)
            fetched = fetch(missing)
            for s, df in fetched.items():
                df.to_csv(paths[s])
                out[s] = df
        return out

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        return self._cached("daily", symbols, (start, end), lambda syms: self._inner.daily_bars(syms, start, end), daily=True)

    def intraday_bars(self, symbols: Sequence[str], session: date, bar_minutes: int) -> dict[str, pd.DataFrame]:
        return self._cached(
            f"intraday{bar_minutes}m",
            symbols,
            (session, bar_minutes),
            lambda syms: self._inner.intraday_bars(syms, session, bar_minutes),
            daily=False,
        )

    def option_chain(self, underlying: str, spot: float, as_of: date) -> pd.DataFrame:
        # Quotes go stale quickly; never cache chains.
        return self._inner.option_chain(underlying, spot, as_of)
