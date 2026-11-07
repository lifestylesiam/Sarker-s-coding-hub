import numpy as np
import pandas as pd
import pytest

from quantlab import stress, vol
from quantlab.data import load_daily
from quantlab.providers import build_provider


def _gbm(sigma, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(n) * sigma / np.sqrt(252)
    idx = pd.bdate_range("2000-01-03", periods=n + 1)
    return pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(r)])), index=idx)


def test_close_to_close_recovers_sigma():
    r = vol.log_returns(_gbm(0.3))
    est = vol.close_to_close_vol(r, len(r), 252).iloc[-1]
    assert est == pytest.approx(0.3, rel=0.03)


def test_ewma_tracks_sigma():
    r = vol.log_returns(_gbm(0.25))
    assert vol.ewma_vol(r, 0.97, 252).iloc[-500:].mean() == pytest.approx(0.25, rel=0.1)


def test_parkinson_on_constant_range():
    idx = pd.bdate_range("2020-01-01", periods=30)
    hl = 0.02
    high = pd.Series(np.exp(hl), index=idx)
    low = pd.Series(1.0, index=idx)
    expected = np.sqrt(hl ** 2 / (4 * np.log(2)) * 252)
    assert vol.parkinson_vol(high, low, 20, 252).iloc[-1] == pytest.approx(expected)


def test_rolling_beta_of_scaled_series():
    r = vol.log_returns(_gbm(0.2, n=500))
    assert vol.rolling_beta(2 * r, r, 100).dropna().iloc[-1] == pytest.approx(2.0)


def test_percentile_rank_bounds():
    s = pd.Series(np.arange(100.0))
    assert vol.percentile_rank(s, 100) == 100.0
    assert vol.percentile_rank(s[::-1].reset_index(drop=True), 100) == 1.0


def test_stress_detection_and_table(settings):
    p = build_provider(settings)
    bars = load_daily(p, ["SPY", "MU", "KO"], settings)
    day = stress.detect_stress_day(bars["SPY"], "worst_benchmark_return")
    spy_r = bars["SPY"]["close"].pct_change()
    assert spy_r.loc[day] == spy_r.min()
    tbl = stress.stress_table(bars, day, "SPY", settings.vol.zscore_window, settings.vol.beta_window, 252)
    assert set(tbl.index) == {"SPY", "MU", "KO"}
    assert tbl.loc["SPY", "beta"] == pytest.approx(1.0)
    assert tbl.loc["SPY", "residual"] == pytest.approx(0.0, abs=1e-12)
    # Synthetic: the injected crash is a multi-sigma event for everyone
    assert (tbl["z"] < -2).all()


def test_stress_table_has_no_lookahead(settings):
    """Changing the stress day's own return must not change its trailing vol."""
    p = build_provider(settings)
    bars = load_daily(p, ["SPY", "MU"], settings)
    day = stress.detect_stress_day(bars["SPY"], "worst_benchmark_return")
    base = stress.stress_table(bars, day, "SPY", 60, 120, 252)
    bumped = {k: v.copy() for k, v in bars.items()}
    bumped["MU"].loc[day, "close"] *= 0.5
    after = stress.stress_table(bumped, day, "SPY", 60, 120, 252)
    assert after.loc["MU", "trailing_vol"] == pytest.approx(base.loc["MU", "trailing_vol"])


def test_intraday_profile(settings):
    p = build_provider(settings)
    session = pd.Timestamp("2026-03-10").date()
    mins = p.intraday_bars(["SPY"], session, 1)["SPY"]
    prof = stress.intraday_profile(mins, prev_close=settings.synthetic.start_price)
    assert prof["trough"] < prof["close"]
    assert 0 <= prof["recovered_frac"] <= 1.5
