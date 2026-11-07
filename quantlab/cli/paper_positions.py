"""Rebalance a (paper) account toward config/portfolio.yaml.

    quantlab-paper                       # dry run: print the plan, submit nothing
    quantlab-paper --execute             # submit to the broker in the configured mode (paper by default)
    quantlab-paper --set data.feed=sip --portfolio-set sizing.gross_notional=10000

Safety: dry-run by default; live mode needs three independent opt-ins (see brokers.authorize_mode);
the connected account must match the mode; orders are never sized from synthetic data.
Every run writes its resolved config and plan to ``trading.runs_dir/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from .. import vol as V
from ..brokers import Broker, OptionOrderIntent, connect_broker
from ..data import load_daily
from ..portfolio import OrderIntent, SizingError, build_orders, check_gross, select_contract
from ..providers import MarketDataProvider, build_provider
from ..settings import PortfolioConfig, Settings, find_config_dir, load_config, parse_override, resolve_path

log = logging.getLogger("quantlab.paper")


@dataclass
class RebalancePlan:
    as_of: str
    mode: str
    account: str
    equity: float
    data_source: str
    synthetic_data: bool
    prices: dict[str, float]
    vols: dict[str, float]
    orders: list[OrderIntent]
    option_orders: list[OptionOrderIntent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _mask(account_number: str) -> str:
    return account_number[:2] + "…" + account_number[-3:] if len(account_number) > 5 else "…"


def plan_rebalance(settings: Settings, pcfg: PortfolioConfig, broker: Broker, provider: MarketDataProvider, as_of: date, is_synthetic: bool) -> RebalancePlan:
    acct = broker.account()
    check_gross(pcfg.sizing, acct.equity)
    positions = {p.symbol: p.qty for p in broker.positions() if p.asset_class == "us_equity"}
    held_options = {p.symbol for p in broker.positions() if p.asset_class != "us_equity"}

    symbols = [t.symbol for t in pcfg.targets] + ([s for s in positions if pcfg.constraints.close_unlisted])
    bars = load_daily(provider, symbols, settings, as_of)
    missing = sorted(set(symbols) - set(bars))
    if missing:
        raise SizingError(f"no market data for {missing}")
    prices = {s: float(b["close"].iloc[-1]) for s, b in bars.items()}
    ppy = settings.market.trading_days_per_year
    vols = {s: float(V.close_to_close_vol(V.log_returns(b["close"]), pcfg.sizing.vol_window, ppy).iloc[-1]) for s, b in bars.items()}

    orders = build_orders(pcfg.targets, positions, prices, vols, pcfg.sizing, pcfg.constraints)
    warnings = []
    if not acct.shorting_enabled:
        for o in orders:
            if o.side == "sell" and positions.get(o.symbol, 0.0) - o.qty < 0:
                raise SizingError(f"{o.symbol}: order would open a short but the account has shorting disabled")

    option_orders: list[OptionOrderIntent] = []
    if pcfg.option_legs.enabled:
        level = acct.options_level or 0
        if level < pcfg.option_legs.min_options_level:
            raise SizingError(f"option legs need options level >= {pcfg.option_legs.min_options_level}, account has {level}")
        for leg in pcfg.option_legs.legs:
            if any(sym.startswith(leg.underlying) for sym in held_options):
                warnings.append(f"skipping {leg.underlying} {leg.right}: an option on it is already held")
                continue
            spot = prices.get(leg.underlying) or float(load_daily(provider, [leg.underlying], settings, as_of)[leg.underlying]["close"].iloc[-1])
            row = select_contract(provider.option_chain(leg.underlying, spot, as_of), spot, pd.Timestamp(as_of), leg)
            if pcfg.option_legs.limit_at == "mid":
                limit = float(row["mid"])
            else:
                limit = float(row["ask"] if leg.side == "buy" else row["bid"])
            option_orders.append(
                OptionOrderIntent(str(row["symbol"]), leg.side, leg.contracts, limit, f"{leg.right} K={row['strike']:g} exp={pd.Timestamp(row['expiry']).date()}")
            )

    if is_synthetic:
        warnings.append("market data is SYNTHETIC: prices and sizes are not real; execution is refused")
    return RebalancePlan(
        as_of=str(as_of), mode=broker.mode.value, account=_mask(acct.account_number), equity=acct.equity,
        data_source=provider.name, synthetic_data=is_synthetic, prices=prices, vols=vols,
        orders=orders, option_orders=option_orders, warnings=warnings,
    )


def execute(plan: RebalancePlan, broker: Broker, pcfg: PortfolioConfig) -> list[dict]:
    if plan.synthetic_data:
        raise SizingError("refusing to execute: orders were sized from synthetic data")
    results = []
    for o in plan.orders:
        oid = broker.submit(o, pcfg.order)
        log.info("submitted %s %g %s -> %s", o.side, o.qty, o.symbol, oid)
        results.append({"symbol": o.symbol, "side": o.side, "qty": o.qty, "order_id": oid})
    for o in plan.option_orders:
        oid = broker.submit_option(o, pcfg.order)
        log.info("submitted %s %d %s @ %.2f -> %s", o.side, o.contracts, o.contract, o.limit_price, oid)
        results.append({"symbol": o.contract, "side": o.side, "qty": o.contracts, "order_id": oid})
    return results


def write_run_record(runs_dir: Path, settings: Settings, pcfg: PortfolioConfig, plan: RebalancePlan, executed: list[dict] | None) -> Path:
    out = runs_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    (out / "settings.resolved.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    (out / "portfolio.resolved.json").write_text(pcfg.model_dump_json(indent=2), encoding="utf-8")
    (out / "plan.json").write_text(json.dumps(asdict(plan), indent=2, default=str), encoding="utf-8")
    (out / "executed.json").write_text(json.dumps(executed, indent=2) if executed is not None else "null", encoding="utf-8")
    return out


def render(plan: RebalancePlan) -> str:
    lines = [
        f"as of {plan.as_of} | mode={plan.mode} | account {plan.account} | equity ${plan.equity:,.0f} | data {plan.data_source}",
    ]
    if plan.orders:
        df = pd.DataFrame(
            [{"symbol": o.symbol, "side": o.side, "qty": o.qty, "ref_price": o.ref_price, "notional": o.notional,
              "vol": plan.vols.get(o.symbol), "clipped": o.clipped, "reason": o.reason} for o in plan.orders]
        )
        lines.append(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    else:
        lines.append("no equity orders: portfolio already at target (within min_order_notional)")
    for o in plan.option_orders:
        lines.append(f"option: {o.side} {o.contracts} x {o.contract} limit {o.limit_price:.2f} ({o.reason})")
    lines += [f"WARNING: {w}" for w in plan.warnings]
    return "\n".join(lines)


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="quantlab-paper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="submit orders (default: dry run)")
    ap.add_argument("--i-understand-live", action="store_true", help="one of three opt-ins required for live mode")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override settings.yaml (repeatable)")
    ap.add_argument("--portfolio-set", action="append", default=[], metavar="KEY=VALUE", help="override portfolio.yaml (repeatable)")
    ap.add_argument("--refresh", action="store_true", help="bypass the market-data cache")
    ap.add_argument("--config-dir", type=Path, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        cfg_dir = args.config_dir or find_config_dir()
        settings = load_config("settings", cfg_dir, dict(map(parse_override, args.set)))
        pcfg = load_config("portfolio", cfg_dir, dict(map(parse_override, args.portfolio_set)))
        broker = connect_broker(settings, cli_live_ack=args.i_understand_live, config_dir=cfg_dir)
        provider = build_provider(settings, cfg_dir, refresh=args.refresh)
        plan = plan_rebalance(settings, pcfg, broker, provider, date.today(), provider.is_fallback)
        log.info("plan:\n%s", render(plan))
        executed = execute(plan, broker, pcfg) if args.execute else None
        if not args.execute:
            log.info("dry run: nothing submitted. Re-run with --execute to send these orders (%s mode).", plan.mode)
        record = write_run_record(resolve_path(settings.trading.runs_dir, cfg_dir), settings, pcfg, plan, executed)
        log.info("run record: %s", record)
        return 0
    except Exception as e:  # surface a clean message; full trace with -v
        log.error("%s: %s", type(e).__name__, e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
