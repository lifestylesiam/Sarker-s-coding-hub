"""Pure portfolio logic: target sizing, order generation, and risk analytics. No I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .settings import ConstraintSettings, OptionLegSpec, SizingSettings, TargetSpec


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    ref_price: float
    reason: str
    clipped: bool = False

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


class SizingError(ValueError):
    pass


def target_notionals(targets: Sequence[TargetSpec], vols: Mapping[str, float], sizing: SizingSettings) -> dict[str, float]:
    """Signed $ targets. inverse_vol: each leg's risk ~ its risk_weight; equal_notional ignores vol."""
    raw: dict[str, float] = {}
    for t in targets:
        if sizing.method == "inverse_vol":
            v = vols.get(t.symbol)
            if v is None or not np.isfinite(v) or v <= 0:
                raise SizingError(f"no usable volatility for {t.symbol}")
            raw[t.symbol] = t.risk_weight / v
        else:
            raw[t.symbol] = t.risk_weight
    total = sum(raw.values())
    return {t.symbol: sizing.gross_notional * raw[t.symbol] / total * (1 if t.side == "long" else -1) for t in targets}


def build_orders(
    targets: Sequence[TargetSpec],
    positions: Mapping[str, float],
    prices: Mapping[str, float],
    vols: Mapping[str, float],
    sizing: SizingSettings,
    constraints: ConstraintSettings,
) -> list[OrderIntent]:
    """Orders that move current ``positions`` (signed shares) toward the configured targets."""
    goal_notional = target_notionals(targets, vols, sizing)
    goal_qty: dict[str, float] = {}
    for sym, notional in goal_notional.items():
        px = prices.get(sym)
        if px is None or px <= 0:
            raise SizingError(f"no usable price for {sym}")
        q = notional / px
        if not constraints.allow_fractional:
            q = math.trunc(q)
        goal_qty[sym] = q
    if constraints.close_unlisted:
        for sym, held in positions.items():
            goal_qty.setdefault(sym, 0.0)

    orders: list[OrderIntent] = []
    for sym, goal in goal_qty.items():
        delta = goal - positions.get(sym, 0.0)
        px = prices.get(sym)
        if px is None:
            raise SizingError(f"no usable price for {sym}")
        notional = abs(delta) * px
        if delta == 0 or notional < constraints.min_order_notional:
            continue
        clipped = False
        if notional > constraints.max_order_notional:
            if constraints.on_cap_breach == "error":
                raise SizingError(f"{sym}: order ${notional:,.0f} exceeds max_order_notional ${constraints.max_order_notional:,.0f}")
            capped = constraints.max_order_notional / px
            delta = math.copysign(capped if constraints.allow_fractional else math.floor(capped), delta)
            clipped = True
            if delta == 0:
                continue
        reason = "close (not in targets)" if sym not in goal_notional else f"target {goal:+g} sh, held {positions.get(sym, 0.0):+g}"
        orders.append(OrderIntent(sym, "buy" if delta > 0 else "sell", abs(delta), px, reason, clipped))
    return orders


def select_contract(chain: pd.DataFrame, spot: float, as_of: pd.Timestamp, leg: OptionLegSpec) -> pd.Series:
    """Pick the contract closest to the leg's target DTE, then to strike = moneyness * spot.

    Only contracts with a two-sided quote are eligible.
    """
    c = chain[(chain["right"] == leg.right) & (chain["bid"] > 0) & (chain["ask"] > chain["bid"])].copy()
    if c.empty:
        raise SizingError(f"no quoted {leg.right}s for {leg.underlying}")
    c["dte"] = (pd.to_datetime(c["expiry"]) - pd.Timestamp(as_of)).dt.days
    c = c[c["dte"] > 0]
    if c.empty:
        raise SizingError(f"no unexpired {leg.right}s for {leg.underlying}")
    best_dte = c.loc[(c["dte"] - leg.target_dte).abs().idxmin(), "dte"]
    c = c[c["dte"] == best_dte]
    return c.loc[(c["strike"] - leg.moneyness * spot).abs().idxmin()]


def check_gross(sizing: SizingSettings, equity: float) -> None:
    if equity <= 0:
        raise SizingError("account equity must be positive")
    if sizing.gross_notional > sizing.max_gross_to_equity * equity:
        raise SizingError(
            f"gross_notional ${sizing.gross_notional:,.0f} exceeds {sizing.max_gross_to_equity:.0%} of equity ${equity:,.0f}"
        )


def portfolio_risk(values: Mapping[str, float], returns: pd.DataFrame, periods_per_year: int) -> tuple[float, pd.DataFrame]:
    """Annualized $ vol of a position set and each position's contribution (Euler decomposition).

    ``values`` are signed market values; ``returns`` are aligned simple/log returns per symbol.
    Contributions sum to the total: c_i = w_i (Sigma w)_i / sigma_p.
    """
    syms = [s for s in values if s in returns.columns]
    if not syms:
        return 0.0, pd.DataFrame(columns=["value", "standalone_vol", "risk_contrib", "pct_of_risk"])
    w = np.array([values[s] for s in syms], dtype=float)
    cov = returns[syms].dropna().cov().to_numpy() * periods_per_year
    var = float(w @ cov @ w)
    sigma = np.sqrt(max(var, 0.0))
    contrib = w * (cov @ w) / sigma if sigma > 0 else np.zeros_like(w)
    table = pd.DataFrame(
        {
            "value": w,
            "standalone_vol": np.abs(w) * np.sqrt(np.diag(cov)),
            "risk_contrib": contrib,
            "pct_of_risk": contrib / sigma if sigma > 0 else 0.0,
        },
        index=syms,
    )
    return sigma, table
