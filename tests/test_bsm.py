import numpy as np
import pytest

from quantlab import bsm

S, K, T, R, Q, SIG = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2


def test_textbook_call_value():
    assert float(bsm.price(S, K, T, R, Q, SIG, True)) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("q", [0.0, 0.03])
@pytest.mark.parametrize("k", [80.0, 100.0, 125.0])
def test_put_call_parity(k, q):
    c = float(bsm.price(S, k, T, R, q, SIG, True))
    p = float(bsm.price(S, k, T, R, q, SIG, False))
    assert c - p == pytest.approx(S * np.exp(-q * T) - k * np.exp(-R * T), abs=1e-10)


@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(is_call):
    q = 0.02
    g = {k: float(v) for k, v in bsm.greeks(S, 105.0, 0.5, R, q, 0.3, is_call).items()}
    f = lambda **kw: float(bsm.price(kw.get("S", S), 105.0, kw.get("T", 0.5), kw.get("r", R), q, kw.get("sig", 0.3), is_call))
    h = 1e-4
    assert g["delta"] == pytest.approx((f(S=S + h) - f(S=S - h)) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((f(S=S + 1e-2) - 2 * f() + f(S=S - 1e-2)) / 1e-4, rel=1e-4)
    assert g["vega"] == pytest.approx((f(sig=0.3 + h) - f(sig=0.3 - h)) / (2 * h), rel=1e-5)
    assert g["rho"] == pytest.approx((f(r=R + h) - f(r=R - h)) / (2 * h), rel=1e-5)
    assert g["theta"] == pytest.approx(-(f(T=0.5 + h) - f(T=0.5 - h)) / (2 * h), rel=1e-5)


@pytest.mark.parametrize("sigma", [0.05, 0.3, 1.5])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(sigma, is_call):
    px = float(bsm.price(S, 110.0, 0.25, R, 0.01, sigma, is_call))
    iv = bsm.implied_vol(px, S, 110.0, 0.25, R, 0.01, is_call, 1e-4, 5.0, 1e-10)
    assert iv == pytest.approx(sigma, abs=1e-6)


def test_implied_vol_rejects_arbitrage_prices():
    assert np.isnan(bsm.implied_vol(150.0, S, K, T, R, Q, True, 1e-4, 5.0, 1e-8))  # above S
    assert np.isnan(bsm.implied_vol(0.0, S, 50.0, T, R, Q, True, 1e-4, 5.0, 1e-8))  # below intrinsic


def test_expiry_is_intrinsic():
    assert float(bsm.price(110.0, 100.0, 0.0, R, Q, SIG, True)) == pytest.approx(10.0)
    assert float(bsm.price(90.0, 100.0, 0.0, R, Q, SIG, True)) == 0.0


def test_european_tree_converges_to_bsm():
    tree = bsm.binomial_price(S, K, T, R, 0.01, SIG, False, 2000, american=False)
    assert tree == pytest.approx(float(bsm.price(S, K, T, R, 0.01, SIG, False)), abs=5e-3)


def test_american_call_no_dividend_equals_european():
    am = bsm.binomial_price(S, K, T, R, 0.0, SIG, True, 1000, american=True)
    eu = bsm.binomial_price(S, K, T, R, 0.0, SIG, True, 1000, american=False)
    assert am == pytest.approx(eu, abs=1e-9)


def test_american_put_has_early_exercise_premium():
    am = bsm.binomial_price(S, K, T, R, 0.0, SIG, False, 1000, american=True)
    assert am > float(bsm.price(S, K, T, R, 0.0, SIG, False)) + 0.3


def test_pnl_attribution_small_move_is_explained():
    s0 = bsm.OptionState(S, K, 0.25, R, Q, 0.3, True)
    s1 = s0.bump(dS=1.0, dT=-1 / 365, dsigma=0.01)
    att = bsm.pnl_attribution(s0, s1)
    explained = sum(v for k, v in att.items() if k not in ("actual", "unexplained"))
    assert explained + att["unexplained"] == pytest.approx(att["actual"])
    assert abs(att["unexplained"]) < 0.01 * abs(att["actual"])


def test_scale_greeks_units():
    g = bsm.greeks(S, K, T, R, Q, SIG, True)
    sc = bsm.scale_greeks(g, 1, 1, 1, 365)
    assert float(sc["vega"]) == pytest.approx(float(g["vega"]) / 100)
    assert float(sc["theta"]) == pytest.approx(float(g["theta"]) / 365)
