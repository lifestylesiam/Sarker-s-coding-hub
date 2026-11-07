import numpy as np
import pandas as pd
import pytest

from quantlab.brokers import TradingModeError, authorize_mode, verify_account_mode
from quantlab.brokers.base import AccountInfo
from quantlab.portfolio import SizingError, build_orders, check_gross, portfolio_risk, target_notionals
from quantlab.settings import ConstraintSettings, SizingSettings, TargetSpec, TradingMode, TradingSettings

TARGETS = (TargetSpec(symbol="aaa", side="long", risk_weight=0.5), TargetSpec(symbol="BBB", side="short", risk_weight=0.5))
SIZING = SizingSettings(method="inverse_vol", vol_window=60, gross_notional=10_000, max_gross_to_equity=0.5)
CONS = ConstraintSettings(max_order_notional=100_000, min_order_notional=50, on_cap_breach="clip", allow_fractional=False, close_unlisted=False)
PRICES = {"AAA": 50.0, "BBB": 20.0, "OLD": 10.0}
VOLS = {"AAA": 0.2, "BBB": 0.4}


def test_inverse_vol_equalizes_risk():
    n = target_notionals(TARGETS, VOLS, SIZING)
    assert abs(n["AAA"]) * 0.2 == pytest.approx(abs(n["BBB"]) * 0.4)
    assert n["BBB"] < 0 < n["AAA"]
    assert sum(abs(v) for v in n.values()) == pytest.approx(10_000)


def test_build_orders_from_flat():
    orders = {o.symbol: o for o in build_orders(TARGETS, {}, PRICES, VOLS, SIZING, CONS)}
    assert orders["AAA"].side == "buy" and orders["BBB"].side == "sell"
    assert float(orders["AAA"].qty).is_integer()


def test_rebalance_skips_small_and_respects_existing():
    first = build_orders(TARGETS, {}, PRICES, VOLS, SIZING, CONS)
    held = {o.symbol: (o.qty if o.side == "buy" else -o.qty) for o in first}
    assert build_orders(TARGETS, held, PRICES, VOLS, SIZING, CONS) == []


def test_cap_clip_and_error():
    tight = CONS.model_copy(update={"max_order_notional": 1000})
    orders = build_orders(TARGETS, {}, PRICES, VOLS, SIZING, tight)
    assert all(o.notional <= 1000 and o.clipped for o in orders)
    with pytest.raises(SizingError):
        build_orders(TARGETS, {}, PRICES, VOLS, SIZING, tight.model_copy(update={"on_cap_breach": "error"}))


def test_close_unlisted():
    cons = CONS.model_copy(update={"close_unlisted": True})
    orders = {o.symbol: o for o in build_orders(TARGETS, {"OLD": 30}, PRICES, VOLS, SIZING, cons)}
    assert orders["OLD"].side == "sell" and orders["OLD"].qty == 30


def test_gross_check():
    check_gross(SIZING, 50_000)
    with pytest.raises(SizingError):
        check_gross(SIZING, 10_000)


def test_risk_contributions_sum_to_total():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.standard_normal((500, 3)) * 0.01, columns=["A", "B", "C"])
    rets["B"] += rets["A"]
    sigma, tbl = portfolio_risk({"A": 1000, "B": -500, "C": 2000}, rets, 252)
    assert tbl["risk_contrib"].sum() == pytest.approx(sigma)
    assert tbl["pct_of_risk"].sum() == pytest.approx(1.0)


# --------------------------------------------------------------------------- trading-mode guard

def _trading(**kw):
    base = dict(mode="paper", allow_live=False, live_env_flag="FLAG", live_confirm_value="CONFIRM-123", paper_account_prefix="PA")
    base.update(kw)
    return TradingSettings(**base)


def test_paper_needs_no_opt_in():
    assert authorize_mode(_trading(), cli_live_ack=False, environ={}) is TradingMode.PAPER


@pytest.mark.parametrize(
    "allow,env,ack",
    [(False, "CONFIRM-123", True), (True, "wrong", True), (True, "CONFIRM-123", False)],
)
def test_live_requires_all_three_switches(allow, env, ack):
    with pytest.raises(TradingModeError):
        authorize_mode(_trading(mode="live", allow_live=allow), cli_live_ack=ack, environ={"FLAG": env})


def test_live_with_all_switches():
    t = _trading(mode="live", allow_live=True)
    assert authorize_mode(t, cli_live_ack=True, environ={"FLAG": "CONFIRM-123"}) is TradingMode.LIVE


def test_account_mode_mismatch_detected():
    paper_acct = AccountInfo("PA123", 1, 1, True, None)
    live_acct = AccountInfo("9XY123", 1, 1, True, None)
    verify_account_mode(paper_acct, TradingMode.PAPER, _trading())
    with pytest.raises(TradingModeError):
        verify_account_mode(live_acct, TradingMode.PAPER, _trading())
    with pytest.raises(TradingModeError):
        verify_account_mode(paper_acct, TradingMode.LIVE, _trading())
