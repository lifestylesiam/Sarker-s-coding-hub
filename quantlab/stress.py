"""Stress-day analysis: find the event, then compare how each asset moved in its own units of risk."""

from __future__ import annotations

from typing import Literal, Mapping

import numpy as np
import pandas as pd

from . import vol as V

DetectionMethod = Literal["worst_benchmark_return", "largest_benchmark_range"]


def detect_stress_day(benchmark_bars: pd.DataFrame, method: DetectionMethod) -> pd.Timestamp:
    """Pick the most stressful session for the benchmark over the available history."""
    close = benchmark_bars["close"]
    prev = close.shift(1)
    if method == "worst_benchmark_return":
        score = -(close / prev - 1.0)
    elif method == "largest_benchmark_range":
        score = (benchmark_bars["high"] - benchmark_bars["low"]) / prev
    else:  # pragma: no cover - guarded by settings validation
        raise ValueError(f"unknown detection method {method!r}")
    score = score.dropna()
    if score.empty:
        raise ValueError("not enough benchmark history to detect a stress day")
    return pd.Timestamp(score.idxmax())


def stress_table(
    bars: Mapping[str, pd.DataFrame],
    day: pd.Timestamp,
    benchmark: str,
    zscore_window: int,
    beta_window: int,
    periods_per_year: int,
) -> pd.DataFrame:
    """Per-symbol stats for ``day``.

    - ``return``: close-to-close return on the day
    - ``range``: (high - low) / previous close
    - ``trailing_vol``: annualized vol over ``zscore_window`` days *ending the day before* (no look-ahead)
    - ``z``: log return / trailing daily vol, i.e. how many "normal days" the move was
    - ``beta``: trailing beta to the benchmark (also ending the day before)
    - ``beta_expected`` / ``residual``: the move beta predicts from the benchmark, and what is left
    """
    if benchmark not in bars:
        raise KeyError(f"benchmark {benchmark!r} missing from bars")
    day = pd.Timestamp(day)
    bench_r = V.log_returns(bars[benchmark]["close"])
    if day not in bench_r.index:
        raise KeyError(f"{day.date()} is not a session in the benchmark history")
    bench_day = float(np.expm1(bench_r.loc[day]))

    rows = []
    for sym, df in bars.items():
        r = V.log_returns(df["close"])
        if day not in r.index:
            continue
        hist = r.loc[: day].iloc[:-1]
        trailing = hist.iloc[-zscore_window:]
        daily_sigma = trailing.std(ddof=1) if len(trailing) >= 2 else np.nan
        joined = pd.concat([hist, bench_r.loc[: day].iloc[:-1]], axis=1, join="inner").dropna().iloc[-beta_window:]
        if len(joined) >= 2 and joined.iloc[:, 1].var() > 0:
            beta = float(np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)[0, 1] / joined.iloc[:, 1].var(ddof=1))
        else:
            beta = np.nan
        prev_close = df["close"].shift(1).loc[day]
        ret = float(np.expm1(r.loc[day]))
        rows.append(
            {
                "symbol": sym,
                "return": ret,
                "range": float((df.loc[day, "high"] - df.loc[day, "low"]) / prev_close),
                "trailing_vol": float(daily_sigma * np.sqrt(periods_per_year)),
                "z": float(r.loc[day] / daily_sigma) if daily_sigma and np.isfinite(daily_sigma) else np.nan,
                "beta": beta,
                "beta_expected": beta * bench_day if np.isfinite(beta) else np.nan,
                "residual": ret - beta * bench_day if np.isfinite(beta) else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("symbol").sort_values("z")


def intraday_profile(minute_bars: pd.DataFrame, prev_close: float) -> dict[str, float | pd.Timestamp]:
    """Shape of an intraday selloff relative to the previous close."""
    if minute_bars.empty:
        raise ValueError("no intraday bars")
    path = minute_bars["close"] / prev_close - 1.0
    low_path = minute_bars["low"] / prev_close - 1.0
    trough_ts = low_path.idxmin()
    trough = float(low_path.min())
    close_ret = float(path.iloc[-1])
    open_ret = float(minute_bars["open"].iloc[0] / prev_close - 1.0)
    minutes_to_trough = (trough_ts - minute_bars.index[0]).total_seconds() / 60.0
    recovered = (close_ret - trough) / -trough if trough < 0 else np.nan
    return {
        "open_gap": open_ret,
        "trough": trough,
        "trough_time": trough_ts,
        "minutes_to_trough": float(minutes_to_trough),
        "close": close_ret,
        "recovered_frac": float(recovered),
    }
