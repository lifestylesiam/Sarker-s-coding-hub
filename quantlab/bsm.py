"""Black-Scholes-Merton pricing, Greeks, implied vol, an American binomial tree, and
Taylor P&L attribution.

Conventions (core functions use raw units; scale for display with ``scale_greeks``):
  - T in years, r and q continuously compounded, sigma annualized
  - vega: dV/dsigma per 1.00 of vol; rho: dV/dr per 1.00 of rate; theta: dV/dt per year
  - ``is_call``: bool or bool array
All functions broadcast over numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

ArrayLike = float | np.ndarray

_EPS = 1e-12


def _d1d2(S, K, T, r, q, sigma):
    sqrt_t = np.sqrt(np.maximum(T, _EPS))
    vol_t = np.maximum(sigma * sqrt_t, _EPS)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_t
    return d1, d1 - vol_t


def price(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, q: ArrayLike, sigma: ArrayLike, is_call) -> np.ndarray:
    S, K, T, r, q, sigma = map(np.asarray, (S, K, T, r, q, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    d1, d2 = _d1d2(S, K, T, r, q, sigma)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    call = S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    put = K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)
    out = np.where(is_call, call, put)
    # At/after expiry or zero vol: discounted intrinsic on the forward.
    degenerate = (T <= _EPS) | (sigma <= _EPS)
    if np.any(degenerate):
        fwd_intr = np.where(is_call, np.maximum(S * df_q - K * df_r, 0.0), np.maximum(K * df_r - S * df_q, 0.0))
        out = np.where(degenerate, fwd_intr, out)
    return out


def greeks(S, K, T, r, q, sigma, is_call) -> dict[str, np.ndarray]:
    S, K, T, r, q, sigma = map(np.asarray, (S, K, T, r, q, sigma))
    is_call = np.asarray(is_call, dtype=bool)
    T_ = np.maximum(T, _EPS)
    d1, d2 = _d1d2(S, K, T_, r, q, sigma)
    df_q, df_r = np.exp(-q * T_), np.exp(-r * T_)
    pdf = norm.pdf(d1)
    sqrt_t = np.sqrt(T_)

    delta = np.where(is_call, df_q * norm.cdf(d1), -df_q * norm.cdf(-d1))
    gamma = df_q * pdf / (S * np.maximum(sigma, _EPS) * sqrt_t)
    vega = S * df_q * pdf * sqrt_t
    common = -S * df_q * pdf * sigma / (2.0 * sqrt_t)
    theta_call = common - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)
    theta_put = common + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)
    # theta here is dV/dt (calendar time passing), i.e. -dV/dT
    theta = np.where(is_call, theta_call, theta_put)
    rho = np.where(is_call, K * T_ * df_r * norm.cdf(d2), -K * T_ * df_r * norm.cdf(-d2))
    vanna = -df_q * pdf * d2 / np.maximum(sigma, _EPS)
    volga = vega * d1 * d2 / np.maximum(sigma, _EPS)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho, "vanna": vanna, "volga": volga}


def scale_greeks(g: dict[str, np.ndarray], vega_per_vol_points: float, rho_per_rate_points: float, theta_per_days: float, days_per_year: float) -> dict[str, np.ndarray]:
    """Convert raw Greeks to trader units (per N vol points / rate points / days)."""
    out = dict(g)
    out["vega"] = g["vega"] * vega_per_vol_points / 100.0
    out["rho"] = g["rho"] * rho_per_rate_points / 100.0
    out["theta"] = g["theta"] * theta_per_days / days_per_year
    return out


def no_arb_bounds(S, K, T, r, q, is_call) -> tuple[float, float]:
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if is_call:
        return max(S * df_q - K * df_r, 0.0), S * df_q
    return max(K * df_r - S * df_q, 0.0), K * df_r


def implied_vol(target: float, S: float, K: float, T: float, r: float, q: float, is_call: bool, lower: float, upper: float, tol: float) -> float:
    """Brent root-find of BSM price = target. NaN if the price violates no-arbitrage bounds."""
    lo_px, hi_px = no_arb_bounds(S, K, T, r, q, is_call)
    if not np.isfinite(target) or T <= 0 or target <= lo_px + tol or target >= hi_px:
        return float("nan")
    f = lambda s: float(price(S, K, T, r, q, s, is_call)) - target  # noqa: E731
    f_lo, f_hi = f(lower), f(upper)
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(f, lower, upper, xtol=tol, maxiter=200))


def binomial_price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool, steps: int, american: bool) -> float:
    """Cox-Ross-Rubinstein tree. American exercise checked at every node."""
    if T <= 0 or sigma <= 0:
        return float(np.maximum((S - K) if is_call else (K - S), 0.0))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("binomial: risk-neutral probability outside (0,1); increase steps")
    j = np.arange(steps + 1)
    spots = S * u ** (steps - 2 * j)
    payoff = lambda s: np.maximum(s - K, 0.0) if is_call else np.maximum(K - s, 0.0)  # noqa: E731
    values = payoff(spots)
    for n in range(steps - 1, -1, -1):
        spots = spots[: n + 1] * d  # S*u^(n-2j)
        values = disc * (p * values[: n + 1] + (1.0 - p) * values[1 : n + 2])
        if american:
            values = np.maximum(values, payoff(spots))
    return float(values[0])


# --------------------------------------------------------------------------- P&L attribution


@dataclass(frozen=True)
class OptionState:
    S: float
    K: float
    T: float
    r: float
    q: float
    sigma: float
    is_call: bool

    def value(self) -> float:
        return float(price(self.S, self.K, self.T, self.r, self.q, self.sigma, self.is_call))

    def bump(self, dS: float = 0.0, dT: float = 0.0, dr: float = 0.0, dsigma: float = 0.0) -> "OptionState":
        return replace(self, S=self.S + dS, T=max(self.T + dT, 0.0), r=self.r + dr, sigma=max(self.sigma + dsigma, 1e-6))


def pnl_attribution(start: OptionState, end: OptionState) -> dict[str, float]:
    """Second-order Taylor decomposition of the price change from ``start`` to ``end``.

    delta*dS + 1/2 gamma dS^2 + vega*dsig + theta*dt + rho*dr + vanna*dS*dsig + 1/2 volga*dsig^2,
    with the remainder reported as ``unexplained`` (higher-order terms).
    """
    g = {k: float(v) for k, v in greeks(start.S, start.K, start.T, start.r, start.q, start.sigma, start.is_call).items()}
    dS, dsig, dr = end.S - start.S, end.sigma - start.sigma, end.r - start.r
    dt = start.T - end.T  # calendar time elapsed (years)
    parts = {
        "delta": g["delta"] * dS,
        "gamma": 0.5 * g["gamma"] * dS * dS,
        "vega": g["vega"] * dsig,
        "theta": g["theta"] * dt,
        "rho": g["rho"] * dr,
        "vanna": g["vanna"] * dS * dsig,
        "volga": 0.5 * g["volga"] * dsig * dsig,
    }
    actual = end.value() - start.value()
    parts["unexplained"] = actual - sum(parts.values())
    parts["actual"] = actual
    return parts
