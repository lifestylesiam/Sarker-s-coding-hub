"""Deterministic synthetic market: one-factor model with an injected crash day.

Lets the apps and tests run with no keys or network. Every number comes from
``SyntheticSettings``; randomness is seeded per symbol so results are reproducible
and independent of which other symbols were requested.
"""

from __future__ import annotations

import zlib
from datetime import date
from typing import Sequence

import numpy as np
import pandas as pd

from ..settings import MarketSettings, SyntheticSettings
from .base import CHAIN_COLUMNS, normalize_bars


class SyntheticProvider:
    def __init__(self, cfg: SyntheticSettings, market: MarketSettings):
        self._cfg = cfg
        self._mkt = market

    @property
    def name(self) -> str:
        return "synthetic"

    # ------------------------------------------------------------------ helpers
    def _rng(self, *parts: object) -> np.random.Generator:
        key = "|".join(str(p) for p in (self._cfg.seed, *parts))
        return np.random.default_rng(zlib.crc32(key.encode()))

    def _sessions(self, start: date, end: date) -> pd.DatetimeIndex:
        return pd.bdate_range(start, end)

    def _crash_day(self, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
        k = self._cfg.crash_days_before_end
        return sessions[-k] if len(sessions) > k else None

    def _market_returns(self, sessions: pd.DatetimeIndex) -> np.ndarray:
        """Common factor; seeded by the date range so every symbol in a request shares it."""
        d = 1.0 / self._mkt.trading_days_per_year
        rng = self._rng("market", sessions[0].date(), sessions[-1].date())
        r = rng.standard_normal(len(sessions)) * self._cfg.market_vol * np.sqrt(d)
        crash = self._crash_day(sessions)
        if crash is not None:
            i = sessions.get_loc(crash)
            r[i] = np.log1p(self._cfg.crash_return)
            # vol clustering after the shock
            tail = slice(i + 1, min(i + 15, len(r)))
            r[tail] *= 2.0
        return r

    # ------------------------------------------------------------------ API
    def daily_bars(self, symbols: Sequence[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        sessions = self._sessions(start, end)
        if len(sessions) < 2:
            return {s: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for s in symbols}
        mkt = self._market_returns(sessions)
        d = 1.0 / self._mkt.trading_days_per_year
        out = {}
        for sym in symbols:
            p = self._cfg.params_for(sym)
            rng = self._rng("daily", sym)
            idio = rng.standard_normal(len(sessions)) * p.idio_vol * np.sqrt(d)
            r = p.beta * mkt + idio
            close = self._cfg.start_price * np.exp(np.cumsum(r))
            prev = np.concatenate([[self._cfg.start_price], close[:-1]])
            day_sigma = np.sqrt((p.beta * self._cfg.market_vol) ** 2 + p.idio_vol ** 2) * np.sqrt(d)
            open_ = prev * np.exp(rng.standard_normal(len(r)) * day_sigma * 0.3)
            span = np.abs(rng.standard_normal(len(r))) * day_sigma * 0.6
            high = np.maximum(open_, close) * np.exp(span)
            low = np.minimum(open_, close) * np.exp(-span - np.maximum(0, -r) * 0.5)
            vol = rng.lognormal(mean=15, sigma=0.3, size=len(r)) * (1 + 3 * np.abs(r) / day_sigma / 10)
            df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=sessions)
            out[sym] = normalize_bars(df.tz_localize(self._mkt.timezone), daily=True, tz=self._mkt.timezone)
        return out

    def intraday_bars(self, symbols: Sequence[str], session: date, bar_minutes: int) -> dict[str, pd.DataFrame]:
        n = self._mkt.minutes_per_session // bar_minutes
        open_ts = pd.Timestamp(session).tz_localize(self._mkt.timezone) + pd.Timedelta(hours=9, minutes=30)
        idx = pd.date_range(open_ts, periods=n, freq=f"{bar_minutes}min")
        trough = min(max(1, self._cfg.intraday_trough_minute // bar_minutes), n - 1)
        out = {}
        for sym in symbols:
            p = self._cfg.params_for(sym)
            rng = self._rng("intraday", sym, session)
            step_sigma = np.sqrt((p.beta * self._cfg.market_vol) ** 2 + p.idio_vol ** 2) / np.sqrt(
                self._mkt.trading_days_per_year * n
            )
            drop = p.beta * np.log1p(self._cfg.crash_return) * 1.6  # intraday trough deeper than the close
            shape = np.concatenate([np.linspace(0, drop, trough + 1)[1:], np.linspace(drop, drop * 0.55, n - trough)])
            noise = np.cumsum(rng.standard_normal(n) * step_sigma)
            px = self._cfg.start_price * np.exp(shape + noise)
            opn = np.concatenate([[self._cfg.start_price], px[:-1]])
            wick = np.abs(rng.standard_normal(n)) * step_sigma
            df = pd.DataFrame(
                {
                    "open": opn,
                    "high": np.maximum(opn, px) * np.exp(wick),
                    "low": np.minimum(opn, px) * np.exp(-wick),
                    "close": px,
                    "volume": rng.lognormal(10, 0.5, n),
                },
                index=idx,
            )
            out[sym] = normalize_bars(df, daily=False, tz=self._mkt.timezone)
        return out

    def option_chain(self, underlying: str, spot: float, as_of: date) -> pd.DataFrame:
        from .. import bsm

        sk = self._cfg.options_skew
        rng = self._rng("chain", underlying, as_of)
        expiries = [pd.Timestamp(as_of) + pd.Timedelta(days=d) for d in (14, 30, 60, 90)]
        strikes = spot * np.exp(np.linspace(-0.35, 0.35, 15))
        rows = []
        r = self._mkt.risk_free_rate
        for exp in expiries:
            t = (exp - pd.Timestamp(as_of)).days / self._mkt.calendar_days_per_year
            fwd = spot * np.exp(r * t)
            for k in strikes:
                lm = np.log(k / fwd)
                iv = max(0.05, sk.atm_vol + sk.slope * lm + sk.curvature * lm * lm)
                for right in ("call", "put"):
                    mid = float(bsm.price(spot, k, t, r, 0.0, iv, right == "call"))
                    half = max(0.01, 0.02 * mid) * (1 + 0.2 * rng.random())
                    rows.append(
                        {
                            "symbol": f"{underlying}{exp:%y%m%d}{right[0].upper()}{int(round(k * 1000)):08d}",
                            "underlying": underlying,
                            "expiry": exp,
                            "strike": float(k),
                            "right": right,
                            "bid": max(0.0, mid - half),
                            "ask": mid + half,
                            "mid": mid,
                            "provider_iv": iv,
                        }
                    )
        return pd.DataFrame(rows, columns=list(CHAIN_COLUMNS))
