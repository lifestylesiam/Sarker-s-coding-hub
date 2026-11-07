"""Toy market-impact simulator. ILLUSTRATIVE, not calibrated.

Model (sell order; a buy is the mirror image):
  * Unaffected mid follows arithmetic Brownian motion with daily vol sigma_d * S0.
  * Limit-order book with power-law shape. Selling ``X`` shares of consumed depth displaces the
    mid by f(X) = eta * sigma_d * S0 * (X / ADV)^alpha  (alpha = 0.5 ~ the empirical square-root law).
    A child order of size q starting from consumed depth X walks the book, paying on average
    [F(X + q) - F(X)] / q below mid, where F = integral of f.
  * The book refills: consumed depth decays exponentially with a half-life (resilience).
    This is the Obizhaeva-Wang / Alfonsi-Fruth-Schied family of models.
  * Permanent impact is linear in shares (gamma * sigma_d * S0 / ADV per share), as
    no-arbitrage requires (Huberman-Stanzl). It applies after each fill.
  * Each fill also pays half the quoted spread.

Why slicing helps: with resilience, the book refills between child orders, so later
slices walk a fresher book. Why not slice forever: the unexecuted inventory is exposed to
price moves (timing risk), which grows with the horizon. Without resilience
(half-life -> infinity) slicing saves nothing on transient impact in this model.

Because impact here does not depend on the noise, cost = deterministic impact cost + timing
cost, where timing cost = -side * sum_k q_k W(t_k). The Monte Carlo distribution is reported
alongside the closed-form std as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Strategy = Literal["all_at_once", "twap", "front_loaded"]
Side = Literal["sell", "buy"]

_LN2 = np.log(2.0)


@dataclass(frozen=True)
class MarketParams:
    price: float
    annual_vol: float
    adv_shares: float
    spread_bps: float
    eta: float
    alpha: float
    gamma: float
    half_life_min: float
    minutes_per_day: float
    trading_days_per_year: float

    def __post_init__(self):
        if self.price <= 0 or self.adv_shares <= 0 or self.minutes_per_day <= 0:
            raise ValueError("price, adv_shares and minutes_per_day must be positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.half_life_min <= 0:
            raise ValueError("half_life_min must be positive (use a large value for no resilience)")

    @property
    def sigma_daily_px(self) -> float:
        """Daily price vol in $ per share."""
        return self.annual_vol / np.sqrt(self.trading_days_per_year) * self.price

    def displacement(self, depth: np.ndarray | float) -> np.ndarray | float:
        return self.eta * self.sigma_daily_px * (np.asarray(depth) / self.adv_shares) ** self.alpha

    def book_cost(self, depth: np.ndarray | float) -> np.ndarray | float:
        """F(X): $ paid (vs mid) to consume X shares of a fresh book."""
        x = np.asarray(depth) / self.adv_shares
        return self.eta * self.sigma_daily_px * self.adv_shares * x ** (1.0 + self.alpha) / (1.0 + self.alpha)

    @property
    def permanent_per_share(self) -> float:
        return self.gamma * self.sigma_daily_px / self.adv_shares

    @property
    def half_spread(self) -> float:
        return 0.5 * self.spread_bps * 1e-4 * self.price

    def decay(self, minutes: np.ndarray | float) -> np.ndarray | float:
        return np.exp(-_LN2 * np.asarray(minutes) / self.half_life_min)


def schedule(strategy: Strategy, order_shares: float, n_slices: int, horizon_min: float, front_load_decay: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (fill times in minutes, shares per fill)."""
    if order_shares <= 0:
        raise ValueError("order_shares must be positive")
    if strategy == "all_at_once" or n_slices <= 1:
        return np.array([0.0]), np.array([float(order_shares)])
    if horizon_min <= 0:
        raise ValueError("horizon_min must be positive for sliced strategies")
    times = np.arange(n_slices) * horizon_min / n_slices
    if strategy == "twap":
        w = np.ones(n_slices)
    elif strategy == "front_loaded":
        w = np.exp(-front_load_decay * times / horizon_min)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return times, order_shares * w / w.sum()


@dataclass(frozen=True)
class ImpactBreakdown:
    times: np.ndarray
    shares: np.ndarray
    fill_prices: np.ndarray        # deterministic (no-noise) average fill price per child
    spread_cost: float
    transient_cost: float
    permanent_cost: float
    timing_std: float              # closed-form std of timing cost ($)
    final_permanent_shift: float   # $ the mid ends up moved by permanent impact
    max_displacement: float        # $ largest mid displacement during execution

    @property
    def impact_cost(self) -> float:
        return self.spread_cost + self.transient_cost + self.permanent_cost


def deterministic_execution(m: MarketParams, times: np.ndarray, shares: np.ndarray, side: Side) -> ImpactBreakdown:
    """Expected execution of a schedule (noise-free), plus closed-form timing-risk std."""
    if side not in ("sell", "buy"):
        raise ValueError("side must be 'sell' or 'buy'")
    sgn = -1.0 if side == "sell" else 1.0
    order = np.argsort(times, kind="stable")
    times, shares = times[order], shares[order]

    depth, perm, last_t = 0.0, 0.0, None
    fills, spread_c, trans_c, perm_c, max_disp = [], 0.0, 0.0, 0.0, 0.0
    for t, q in zip(times, shares):
        if last_t is not None:
            depth *= float(m.decay(t - last_t))
        walk = float(m.book_cost(depth + q) - m.book_cost(depth))
        spread_c += q * m.half_spread
        trans_c += walk
        perm_c += q * perm
        avg_px = m.price + sgn * (perm + m.half_spread + walk / q)
        fills.append(avg_px)
        depth += q
        perm += m.permanent_per_share * q
        max_disp = max(max_disp, perm + float(m.displacement(depth)))
        last_t = t

    # Var(sum q_i W(t_i)) = s^2 * sum_ij q_i q_j min(t_i, t_j), with t in days
    t_days = times / m.minutes_per_day
    cov = np.minimum.outer(t_days, t_days)
    timing_std = float(m.sigma_daily_px * np.sqrt(max(shares @ cov @ shares, 0.0)))
    return ImpactBreakdown(times, shares, np.array(fills), spread_c, trans_c, perm_c, timing_std, perm, max_disp)


def mid_path(m: MarketParams, times: np.ndarray, shares: np.ndarray, side: Side, grid_min: np.ndarray) -> np.ndarray:
    """Noise-free mid price on a time grid (fills applied at the first grid point >= fill time)."""
    sgn = -1.0 if side == "sell" else 1.0
    out = np.empty_like(grid_min, dtype=float)
    depth, perm, k, last = 0.0, 0.0, 0, grid_min[0]
    order = np.argsort(times, kind="stable")
    times, shares = times[order], shares[order]
    for i, t in enumerate(grid_min):
        depth *= float(m.decay(t - last))
        last = t
        while k < len(times) and times[k] <= t + 1e-9:
            depth += shares[k]
            perm += m.permanent_per_share * shares[k]
            k += 1
        out[i] = m.price + sgn * (perm + float(m.displacement(depth)))
    return out


def simulate_costs(m: MarketParams, b: ImpactBreakdown, side: Side, n_paths: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo total cost ($) per path = impact cost + timing cost."""
    sgn = -1.0 if side == "sell" else 1.0
    t_days = b.times / m.minutes_per_day
    dt = np.diff(np.concatenate([[0.0], t_days]))
    increments = rng.standard_normal((n_paths, len(dt))) * np.sqrt(dt) * m.sigma_daily_px
    w_at_fills = np.cumsum(increments, axis=1)
    # Sell: cost = sum q (S0 - fill) and fill contains +W, so noise adds -sum q W (sgn = -1).
    # Buy:  cost = sum q (fill - S0), so noise adds +sum q W (sgn = +1).
    timing = (w_at_fills @ b.shares) * sgn
    return b.impact_cost + timing


def to_bps(cost: np.ndarray | float, m: MarketParams, order_shares: float) -> np.ndarray | float:
    return np.asarray(cost) / (m.price * order_shares) * 1e4


def slice_sweep(m: MarketParams, order_shares: float, horizon_min: float, max_slices: int, side: Side, risk_aversion: float) -> dict[str, np.ndarray]:
    """TWAP over a fixed horizon with N = 1..max_slices: expected cost, risk, and risk-adjusted score (bps)."""
    ns = np.arange(1, max_slices + 1)
    mean, std = [], []
    for n in ns:
        t, q = schedule("twap" if n > 1 else "all_at_once", order_shares, int(n), horizon_min, 0.0)
        b = deterministic_execution(m, t, q, side)
        mean.append(to_bps(b.impact_cost, m, order_shares))
        std.append(to_bps(b.timing_std, m, order_shares))
    mean, std = np.array(mean, dtype=float), np.array(std, dtype=float)
    return {"n_slices": ns, "mean_bps": mean, "std_bps": std, "score_bps": mean + risk_aversion * std}
