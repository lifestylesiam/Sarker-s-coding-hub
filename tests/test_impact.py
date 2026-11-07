import numpy as np
import pytest

from quantlab import impact as I


def _mkt(**kw):
    base = dict(price=20.0, annual_vol=0.6, adv_shares=500_000, spread_bps=30.0, eta=1.0, alpha=0.5, gamma=0.1,
                half_life_min=10.0, minutes_per_day=390, trading_days_per_year=252)
    base.update(kw)
    return I.MarketParams(**base)


Q = 100_000


def _cost(m, strategy, n=12, horizon=120, side="sell"):
    t, q = I.schedule(strategy, Q, n, horizon, 3.0)
    return I.deterministic_execution(m, t, q, side)


def test_schedules_sum_to_order():
    for s in ("all_at_once", "twap", "front_loaded"):
        t, q = I.schedule(s, Q, 10, 60, 3.0)
        assert q.sum() == pytest.approx(Q)
        assert (np.diff(t) >= 0).all()
    _, q = I.schedule("front_loaded", Q, 10, 60, 3.0)
    assert (np.diff(q) < 0).all()


def test_zero_frictions_zero_cost():
    m = _mkt(eta=0.0, gamma=0.0, spread_bps=0.0)
    for s in ("all_at_once", "twap"):
        assert _cost(m, s).impact_cost == pytest.approx(0.0)


def test_slicing_beats_dumping_with_resilience():
    m = _mkt(gamma=0.0)
    assert _cost(m, "twap").impact_cost < _cost(m, "all_at_once").impact_cost


def test_no_resilience_means_no_transient_saving():
    """Walking the same book in pieces with no refill costs the same as walking it at once."""
    m = _mkt(gamma=0.0, half_life_min=1e12)
    assert _cost(m, "twap").transient_cost == pytest.approx(_cost(m, "all_at_once").transient_cost, rel=1e-9)


def test_cost_increases_with_order_size():
    m = _mkt()
    costs = []
    for q in (10_000, 50_000, 200_000):
        t, s = I.schedule("twap", q, 12, 120, 0)
        costs.append(I.to_bps(I.deterministic_execution(m, t, s, "sell").impact_cost, m, q))
    assert costs[0] < costs[1] < costs[2]


def test_timing_risk_zero_for_dump_positive_for_twap():
    m = _mkt()
    assert _cost(m, "all_at_once").timing_std == 0.0
    assert _cost(m, "twap").timing_std > 0.0


def test_monte_carlo_matches_closed_form():
    m = _mkt()
    b = _cost(m, "twap")
    costs = I.simulate_costs(m, b, "sell", 20_000, np.random.default_rng(1))
    assert costs.mean() == pytest.approx(b.impact_cost, rel=0.02)
    assert costs.std() == pytest.approx(b.timing_std, rel=0.03)


def test_buy_and_sell_cost_symmetric():
    m = _mkt()
    assert _cost(m, "twap", side="buy").impact_cost == pytest.approx(_cost(m, "twap", side="sell").impact_cost)
    sell_fills = _cost(m, "twap", side="sell").fill_prices
    assert (sell_fills < m.price).all()


def test_mid_path_recovers_to_permanent_level():
    m = _mkt()
    t, q = I.schedule("all_at_once", Q, 1, 0, 0)
    grid = np.arange(0, 400.0)
    path = I.mid_path(m, t, q, "sell", grid)
    perm = m.permanent_per_share * Q
    assert path[0] < path[-1]  # recovers after the dump
    assert path[-1] == pytest.approx(m.price - perm, abs=1e-6)


def test_sweep_has_interior_optimum_with_risk_aversion():
    m = _mkt()
    sw = I.slice_sweep(m, Q, 120, 60, "sell", risk_aversion=1.0)
    assert sw["mean_bps"][0] > sw["mean_bps"].min()   # dumping is not the cheapest
    assert sw["std_bps"][0] == 0.0
