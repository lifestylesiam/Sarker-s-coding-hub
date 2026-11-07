"""Realized-volatility estimators. Pure functions: every parameter is an argument.

All estimators return *annualized* volatility given ``periods_per_year`` (e.g. 252 for daily bars).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(close: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """ln(P_t / P_{t-1}); first row dropped."""
    return np.log(close / close.shift(1)).iloc[1:]


def close_to_close_vol(returns: pd.Series | pd.DataFrame, window: int, periods_per_year: int) -> pd.Series | pd.DataFrame:
    """Rolling sample std of log returns, annualized."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods_per_year)


def parkinson_vol(high: pd.Series, low: pd.Series, window: int, periods_per_year: int) -> pd.Series:
    """Parkinson (1980) range estimator: sigma^2 = E[ln(H/L)^2] / (4 ln 2).

    Uses intraday range, so it is ~5x more efficient than close-to-close for a driftless
    diffusion, but it misses overnight gaps and is biased low with discrete sampling.
    """
    hl2 = np.log(high / low) ** 2
    var = hl2.rolling(window, min_periods=window).mean() / (4.0 * np.log(2.0))
    return np.sqrt(var * periods_per_year)


def ewma_vol(returns: pd.Series | pd.DataFrame, lam: float, periods_per_year: int) -> pd.Series | pd.DataFrame:
    """RiskMetrics EWMA: s2_t = lam * s2_{t-1} + (1 - lam) * r_t^2 (zero-mean assumption)."""
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    var = (returns ** 2).ewm(alpha=1.0 - lam, adjust=False).mean()
    return np.sqrt(var * periods_per_year)


def percentile_rank(series: pd.Series, window: int) -> float:
    """Where the latest value sits in its trailing ``window`` history (0-100)."""
    s = series.dropna().iloc[-window:]
    if s.empty:
        return float("nan")
    return float((s <= s.iloc[-1]).mean() * 100.0)


def rolling_beta(asset: pd.Series, benchmark: pd.Series, window: int) -> pd.Series:
    """OLS beta of asset returns on benchmark returns over a rolling window."""
    joined = pd.concat([asset, benchmark], axis=1, join="inner").dropna()
    a, b = joined.iloc[:, 0], joined.iloc[:, 1]
    cov = a.rolling(window, min_periods=window).cov(b)
    var = b.rolling(window, min_periods=window).var()
    return cov / var


def vol_summary(vol_series: pd.Series, percentile_window: int) -> dict[str, float]:
    s = vol_series.dropna()
    if s.empty:
        return {"current": float("nan"), "mean": float("nan"), "min": float("nan"), "max": float("nan"), "percentile": float("nan")}
    return {
        "current": float(s.iloc[-1]),
        "mean": float(s.mean()),
        "min": float(s.min()),
        "max": float(s.max()),
        "percentile": percentile_rank(s, percentile_window),
    }
