import json
from datetime import date

import pandas as pd
import pytest

from quantlab.brokers.fake import FakeBroker
from quantlab.cli import paper_positions as cli
from quantlab.portfolio import SizingError, select_contract
from quantlab.providers.synthetic import SyntheticProvider
from quantlab.settings import OptionLegSpec, load_config

AS_OF = date(2026, 11, 6)


@pytest.fixture
def pcfg(config_dir):
    return load_config("portfolio", config_dir)


@pytest.fixture
def provider(settings):
    return SyntheticProvider(settings.synthetic, settings.market)


def test_plan_on_flat_account(settings, pcfg, provider):
    broker = FakeBroker(equity=100_000)
    plan = cli.plan_rebalance(settings, pcfg, broker, provider, AS_OF, is_synthetic=True)
    syms = {o.symbol for o in plan.orders}
    assert syms == {t.symbol for t in pcfg.targets}
    assert all(o.notional <= pcfg.constraints.max_order_notional + 1e-9 for o in plan.orders)
    assert any("SYNTHETIC" in w for w in plan.warnings)
    assert "PA" in plan.account and "FAKE" not in plan.account  # masked


def test_execute_refuses_synthetic(settings, pcfg, provider):
    broker = FakeBroker(equity=100_000)
    plan = cli.plan_rebalance(settings, pcfg, broker, provider, AS_OF, is_synthetic=True)
    with pytest.raises(SizingError):
        cli.execute(plan, broker, pcfg)
    assert broker.submitted == []


def test_execute_then_replan_is_noop(settings, pcfg, provider):
    broker = FakeBroker(equity=100_000)
    plan = cli.plan_rebalance(settings, pcfg, broker, provider, AS_OF, is_synthetic=False)
    cli.execute(plan, broker, pcfg)
    replan = cli.plan_rebalance(settings, pcfg, broker, provider, AS_OF, is_synthetic=False)
    # Only clipped legs may still have residual orders
    assert all(o.symbol in {p.symbol for p in plan.orders if p.clipped} for o in replan.orders)


def test_gross_vs_equity_guard(settings, pcfg, provider):
    with pytest.raises(SizingError):
        cli.plan_rebalance(settings, pcfg, FakeBroker(equity=1_000), provider, AS_OF, is_synthetic=False)


def test_option_leg_requires_level(settings, pcfg, provider):
    on = pcfg.model_copy(update={"option_legs": pcfg.option_legs.model_copy(update={"enabled": True, "min_options_level": 3})})
    with pytest.raises(SizingError):
        cli.plan_rebalance(settings, on, FakeBroker(equity=100_000), provider, AS_OF, is_synthetic=False)


def test_option_leg_selected(settings, pcfg, provider):
    on = pcfg.model_copy(update={"option_legs": pcfg.option_legs.model_copy(update={"enabled": True})})
    plan = cli.plan_rebalance(settings, on, FakeBroker(equity=100_000), provider, AS_OF, is_synthetic=False)
    assert len(plan.option_orders) == len(on.option_legs.legs)
    assert plan.option_orders[0].limit_price > 0


def test_select_contract_nearest():
    chain = pd.DataFrame(
        {
            "symbol": ["A", "B", "C", "D"],
            "expiry": pd.to_datetime(["2026-12-01", "2026-12-01", "2027-01-15", "2027-01-15"]),
            "strike": [95.0, 105.0, 100.0, 110.0],
            "right": ["call"] * 4,
            "bid": [1, 1, 1, 1.0],
            "ask": [1.2, 1.2, 1.2, 1.2],
            "mid": [1.1] * 4,
        }
    )
    leg = OptionLegSpec(underlying="X", right="call", side="buy", contracts=1, target_dte=60, moneyness=1.0)
    assert select_contract(chain, 101.0, pd.Timestamp("2026-11-06"), leg)["symbol"] == "C"


def test_run_record_written_without_secrets(tmp_path, monkeypatch, settings, pcfg, provider):
    fake_secret = "sk-test-DO-NOT-LEAK-0123456789"
    monkeypatch.setenv(settings.credentials.key_env, "PKTESTKEY")
    monkeypatch.setenv(settings.credentials.secret_env, fake_secret)
    plan = cli.plan_rebalance(settings, pcfg, FakeBroker(equity=100_000), provider, AS_OF, is_synthetic=True)
    out = cli.write_run_record(tmp_path, settings, pcfg, plan, None)
    data = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert data["synthetic_data"] is True
    for f in out.iterdir():
        text = f.read_text(encoding="utf-8")
        assert fake_secret not in text and "PKTESTKEY" not in text


def test_cli_without_keys_fails_cleanly(config_dir, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no .env here
    rc = cli.main(["--config-dir", str(config_dir), "--set", "credentials.dotenv_path=null"])
    assert rc == 1
