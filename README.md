# Quant Lab

Three linked projects on one shared, tested Python package (`quantlab`), with Streamlit front ends:

| # | Project | The one insight it shows |
|---|---|---|
| 1 | **Volatility dashboard** | A -8% day means different things for SPY and MU. Measured in units of each name's *own* trailing vol (z-score), with the beta-explained part split from the idiosyncratic part, moves become comparable. |
| 2 | **Options lab** | Price and Greeks from Black-Scholes-Merton, plus where BSM breaks down: early exercise (binomial tree), the vol smile from a real chain, and a Greek-by-Greek P&L attribution. |
| 3 | **Market-impact simulator** *(toy model)* | Selling fast into a thin book is expensive because the book has no time to refill. Slicing cuts that cost but adds timing risk; the cost/risk curve shows the trade-off. |

A fourth piece, `quantlab-paper`, rebalances an **Alpaca paper account** toward a config-defined book, and the dashboard then measures that book's risk.

## Quick start (on your machine)

You need Python 3.10 or newer (`python3 --version`).

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

```bash
cp .env.example .env    # then put your Alpaca paper key ID and secret in .env
```

```bash
pytest
```

```bash
streamlit run apps/Home.py
```

With no keys the apps still run on a **deterministic synthetic market** (one-factor model with an injected crash day). Every page says so in a warning banner, so synthetic numbers can't be mistaken for real ones.

## Configuration: nothing is hard-coded

All behaviour lives in `config/`:

| File | Controls |
|---|---|
| `settings.yaml` | data provider and fallback, Alpaca feed/adjustment, lookback, cache, vol windows, stress-day detection, synthetic market, credential env-var names, trading mode and guards |
| `universe.yaml` | the dashboard's symbols and groups |
| `options.yaml` | Greek display units, IV solver bounds, tree steps, chart ranges, every sidebar input's default/min/max/step |
| `impact.yaml` | simulation size/seed, order side, every model parameter's default/min/max/step |
| `portfolio.yaml` | the paper book: targets, risk weights, sizing method, caps, order type, option legs |

Precedence: **CLI `--set a.b=c` > env `QUANTLAB__A__B=c` > YAML > model defaults**. Env vars can target the other files too, for example `QUANTLAB__IMPACT__SIMULATION__SEED=1`. Unknown keys and out-of-range values are rejected at startup with a clear error (pydantic). The Streamlit controls are generated from the same specs, so UI and config cannot drift.

## Architecture

```
config/*.yaml ──► quantlab/settings.py (typed, validated, precedence)
                      │
   ┌──────────────────┼─────────────────────────┐
   ▼                  ▼                         ▼
providers/          pure core                 brokers/
 base.py (Protocol)  vol.py     stress.py      base.py (Protocol + mode guard)
 alpaca.py           bsm.py     impact.py      alpaca.py
 cache.py (CSV)      portfolio.py              fake.py (tests)
 synthetic.py
 __init__.py (ResilientProvider: configured source, flagged fallback)
   │                  │                         │
   └──────────► apps/ (Streamlit)    quantlab/cli/paper_positions.py
```

- **Pure core.** The math modules take every parameter as an argument: no globals, no I/O, seedable RNG.
- **Interfaces at the edges.** Data and brokerage sit behind `typing.Protocol`s. Tests use the synthetic provider and `FakeBroker`, so the 70-test suite needs no network or keys.
- **Cache keys** include provider, feed and adjustment, so switching feeds never serves stale data.

## Project notes

### 1 · Volatility dashboard
- Estimators: close-to-close, Parkinson (high-low range) and EWMA (RiskMetrics λ), all annualized.
- **Stress day:** auto-detected as the benchmark's worst return or widest range in the lookback, with a date-picker override (`stress.manual_date` pins one in config). The table shows return, range, trailing vol, z-score, beta, beta-implied move and residual. Trailing stats use only data *before* the day (tested for no look-ahead).
- **Intraday:** 1-minute bars for the stress day: time to the low, depth of the low, share of the drop recovered by the close. On the free IEX feed, volume is only IEX's slice of the market.
- **My paper portfolio:** live positions, $ vol, Euler risk contributions (they sum to the total), and what the book would have made or lost on the stress day.

### 2 · Options lab
- BSM with continuous dividend yield; Greeks including vanna/volga, validated against finite differences; Brent IV solver that rejects prices outside no-arbitrage bounds.
- CRR binomial tree: the American-minus-European premium across strikes (≈0 for calls without dividends, real for deep in-the-money puts).
- P&L attribution waterfall: delta, gamma, vega, theta, rho, vanna, volga, unexplained.
- Vol smile: implied vols from chain mid prices (OTM options only) against realized vol, giving the vol risk premium. The free `indicative` options feed is not the real OPRA tape, so treat its prices as approximate.
- **Limits:** BSM assumes constant vol and no jumps. The smile exists precisely because markets disagree.

### 3 · Market-impact simulator (toy)
- Limit-order book with power-law shape (α = 0.5 ≈ the square-root law) and exponential resilience (Obizhaeva-Wang / Alfonsi-Fruth-Schied), linear permanent impact (as no-arbitrage requires), half-spread, and Brownian price noise.
- Strategies: all at once, TWAP, front-loaded. Outputs: expected mid path with fills, cost breakdown (spread / book walk / permanent), Monte Carlo cost distribution (checked against the closed form in tests), and a cost-vs-risk curve over the number of slices.
- **Not calibrated.** The parameters are illustrative. The model leaves out other traders reacting to your flow, volume curves, and limit orders.

## Paper trading: `quantlab-paper`

```bash
quantlab-paper                 # dry run: prints the plan, submits nothing, writes runs/<timestamp>/
```

```bash
quantlab-paper --execute       # submits in the configured mode (paper by default)
```

The starter book in `config/portfolio.yaml` is **illustrative, not advice**: MU (pitch thesis, 40% of risk), NVDA and AMD (20% each), and a short SMH sector hedge (20%). Legs are sized by inverse volatility, so each contributes risk in proportion to its weight. An optional MU call leg is disabled by default.

Safety, by design:
- **Dry run by default**; `--execute` is required to send anything.
- **Paper by default.** Live mode needs three independent opt-ins: `trading.allow_live: true`, `$QUANTLAB_ALLOW_LIVE` set to the configured confirmation phrase, and the `--i-understand-live` flag. After connecting, the account number must match the mode (paper accounts start with `trading.paper_account_prefix`).
- **Never sized from synthetic data.** If the real data provider is unavailable, execution is refused.
- Checks gross notional against equity, enforces per-order caps (clip or error), refuses accidental shorts when shorting is disabled, and requires a minimum options level for option legs.
- Each run writes its resolved config and plan to `runs/` (gitignored). Secrets are never written; that is covered by a test.

Note that paper fills carry no market impact or slippage, which is exactly the cost Project 3 models. Paper P&L flatters real-world results.

## Security
- Keys live only in `.env` (gitignored) or your shell environment. Never commit them or paste them into chats or issues.
- If a key has been exposed anywhere, regenerate it in the Alpaca dashboard.
