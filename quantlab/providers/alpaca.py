"""Alpaca market data (read-only). Feed, adjustment and timezone come from settings."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, time, timedelta
from typing import Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from ..settings import Credentials, DataSettings, MarketSettings
from .base import CHAIN_COLUMNS, ProviderUnavailable, normalize_bars

log = logging.getLogger(__name__)

# OCC option symbol: ROOT + YYMMDD + C/P + strike*1000 (8 digits)
_OCC = re.compile(r"^(?P<root>[A-Z.]+)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


def parse_occ(symbol: str) -> tuple[str, pd.Timestamp, str, float]:
    m = _OCC.match(symbol)
    if not m:
        raise ValueError(f"not an OCC option symbol: {symbol!r}")
    return (
        m["root"],
        pd.Timestamp(datetime.strptime(m["ymd"], "%y%m%d")),
        "call" if m["cp"] == "C" else "put",
        int(m["strike"]) / 1000.0,
    )


class AlpacaProvider:
    def __init__(self, creds: Credentials, data: DataSettings, market: MarketSettings):
        from alpaca.data.enums import Adjustment, DataFeed, OptionsFeed
        from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient

        self._data, self._mkt = data, market
        self._stocks = StockHistoricalDataClient(creds.api_key, creds.secret_key)
        self._options = OptionHistoricalDataClient(creds.api_key, creds.secret_key)
        try:
            self._feed = DataFeed(data.feed)
            self._options_feed = OptionsFeed(data.options_feed)
            self._adjustment = Adjustment(data.adjustment)
        except ValueError as e:
            raise ValueError(f"invalid Alpaca data setting: {e}") from e

    @property
    def name(self) -> str:
        return f"alpaca-{self._data.feed}-{self._data.adjustment}"

    def _bars(self, symbols: Sequence[str], start: datetime, end: datetime, timeframe, daily: bool) -> dict[str, pd.DataFrame]:
        from alpaca.data.requests import StockBarsRequest

        req = StockBarsRequest(
            symbol_or_symbols=list(symbols),
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=self._adjustment,
            feed=self._feed,
        )
        try:
            df = self._stocks.get_stock_bars(req).df
        except Exception as e:  # network policy, auth, rate limits
            raise ProviderUnavailable(f"Alpaca bars request failed: {e}") from e
        out: dict[str, pd.DataFrame] = {}
        if df.empty:
            return out
        for sym, sub in df.groupby(level="symbol"):
            out[str(sym)] = normalize_bars(sub.droplevel("symbol"), daily=daily, tz=self._mkt.timezone)
        return out

    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        from alpaca.data.timeframe import TimeFrame

        tz = ZoneInfo(self._mkt.timezone)
        return self._bars(
            symbols,
            datetime.combine(start, time.min, tz),
            datetime.combine(end, time.max, tz),
            TimeFrame.Day,
            daily=True,
        )

    def intraday_bars(self, symbols: Sequence[str], session: date, bar_minutes: int) -> dict[str, pd.DataFrame]:
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tz = ZoneInfo(self._mkt.timezone)
        start = datetime.combine(session, time(9, 30), tz)
        end = start + timedelta(minutes=self._mkt.minutes_per_session)
        bars = self._bars(symbols, start, end, TimeFrame(bar_minutes, TimeFrameUnit.Minute), daily=False)
        return {s: b[(b.index >= start) & (b.index < end)] for s, b in bars.items()}

    def option_chain(self, underlying: str, spot: float, as_of: date) -> pd.DataFrame:
        from alpaca.data.requests import OptionChainRequest

        try:
            snaps = self._options.get_option_chain(
                OptionChainRequest(underlying_symbol=underlying, feed=self._options_feed)
            )
        except Exception as e:
            raise ProviderUnavailable(f"Alpaca option chain request failed: {e}") from e
        rows = []
        for sym, snap in snaps.items():
            try:
                root, expiry, right, strike = parse_occ(sym)
            except ValueError:
                log.debug("skipping unparsable option symbol %s", sym)
                continue
            q = snap.latest_quote
            bid = float(q.bid_price) if q and q.bid_price else float("nan")
            ask = float(q.ask_price) if q and q.ask_price else float("nan")
            rows.append(
                {
                    "symbol": sym,
                    "underlying": underlying,
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "provider_iv": float(snap.implied_volatility) if snap.implied_volatility else float("nan"),
                }
            )
        return pd.DataFrame(rows, columns=list(CHAIN_COLUMNS))
