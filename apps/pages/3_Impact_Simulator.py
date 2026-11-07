import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab import impact as I
from quantlab import ui

st.set_page_config(page_title="Impact Simulator", layout="wide")
s = ui.settings()
cfg = ui.config("impact")
P, sim = cfg.params, cfg.simulation
side = sim.side

LABELS = {"all_at_once": "All at once", "twap": "TWAP (even slices)", "front_loaded": "Front-loaded"}
colors = ui.color_map([LABELS[x] for x in cfg.strategies], [LABELS[x] for x in cfg.strategies])

# --------------------------------------------------------------------------- sidebar
sb = st.sidebar
sb.markdown("**Market**")
price = ui.number_from_spec(P["price"], "price")
annual_vol = ui.slider_from_spec(P["annual_vol"], "vol")
adv = ui.number_from_spec(P["adv_shares"], "adv")
spread = ui.slider_from_spec(P["spread_bps"], "spread")
sb.markdown("**Order book**")
eta = ui.slider_from_spec(P["eta"], "eta")
alpha = ui.slider_from_spec(P["alpha"], "alpha")
gamma = ui.slider_from_spec(P["gamma"], "gamma")
half_life = ui.slider_from_spec(P["half_life_min"], "hl")
sb.markdown("**Order & schedule**")
pct_adv = ui.slider_from_spec(P["order_pct_adv"], "pct")
horizon = ui.slider_from_spec(P["horizon_min"], "horizon", default=min(P["horizon_min"].default, s.market.minutes_per_session))
n_slices = ui.slider_from_spec(P["n_slices"], "n")
decay = ui.slider_from_spec(P["front_load_decay"], "decay")
lam = ui.slider_from_spec(P["risk_aversion"], "lam")

order = pct_adv * adv
m = I.MarketParams(price, annual_vol, adv, spread, eta, alpha, gamma, half_life, s.market.minutes_per_session, s.market.trading_days_per_year)

st.title("Market-Impact Simulator")
st.warning(
    "**Toy model.** Parameters are illustrative and not calibrated to any market. It shows the *mechanics* of impact "
    "and the cost-vs-risk trade-off, not a forecast of real trading costs.",
    icon="🧪",
)
st.markdown(
    f"{side.title()} **{order:,.0f} shares** (${order * price:,.0f}, {pct_adv:.1%} of ADV) at ${price:,.2f}. "
    f"Sliced strategies run over {horizon} min in {n_slices} child orders."
)

results = {}
for strat in cfg.strategies:
    t, q = I.schedule(strat, order, n_slices, horizon, decay)
    b = I.deterministic_execution(m, t, q, side)
    costs = I.simulate_costs(m, b, side, sim.n_paths, np.random.default_rng(sim.seed))  # common random numbers
    results[LABELS[strat]] = (b, costs)

tab_path, tab_cost, tab_sweep, tab_model = st.tabs(["Price impact", "Cost & risk", "How many slices?", "Model & talking points"])

# --------------------------------------------------------------------------- price path
with tab_path:
    grid = np.arange(0, horizon + sim.post_trade_minutes + 1, 1.0 / sim.steps_per_minute)
    fig = go.Figure()
    for name, (b, _) in results.items():
        fig.add_trace(go.Scatter(x=grid, y=I.mid_path(m, b.times, b.shares, side, grid), name=name, line=dict(color=colors[name], shape="hv"), hovertemplate="$%{y:,.3f}"))
        fig.add_trace(
            go.Scatter(
                x=b.times, y=b.fill_prices, name=f"{name} fills", mode="markers", showlegend=False,
                marker=dict(color=colors[name], size=8, line=dict(color="white", width=2)),
                hovertemplate="fill $%{y:,.3f}",
            )
        )
    fig.add_hline(y=price, line=dict(color=ui.NEUTRAL, dash="dot"), annotation_text="arrival price")
    ui.show(ui.style(fig, title="Expected mid price (no noise) and average fill of each child order", x_title="Minutes from arrival", y_title="Price ($)", y_format=",.2f", height=460))
    st.caption(
        "Dumping walks deep into the book at once; the price then partly recovers as the book refills, but your shares were already sold at the bottom. "
        "Slicing gives the book time to refill between child orders. The level the price recovers to is the permanent impact."
    )

# --------------------------------------------------------------------------- cost table + distribution
with tab_cost:
    rows = {}
    for name, (b, costs) in results.items():
        bps = I.to_bps(costs, m, order)
        rows[name] = {
            "expected_bps": I.to_bps(b.impact_cost, m, order),
            "std_bps": I.to_bps(b.timing_std, m, order),
            "p95_bps": float(np.percentile(bps, 95)),
            "spread_bps": I.to_bps(b.spread_cost, m, order),
            "transient_bps": I.to_bps(b.transient_cost, m, order),
            "permanent_bps": I.to_bps(b.permanent_cost, m, order),
            "expected_usd": b.impact_cost,
            "score_bps": I.to_bps(b.impact_cost + lam * b.timing_std, m, order),
        }
    tbl = pd.DataFrame(rows).T
    best = tbl["score_bps"].idxmin()
    c = st.columns(len(tbl))
    for col, (name, r) in zip(c, tbl.iterrows()):
        col.metric(name, f"{r['expected_bps']:.1f} bps", help=f"${r['expected_usd']:,.0f} expected cost")
        col.caption(f"risk ±{r['std_bps']:.1f} bps, 95th pct {r['p95_bps']:.1f} bps")
    st.success(f"Best risk-adjusted (mean + {lam:g} x std): **{best}**", icon="✅")
    bps_fmt = st.column_config.NumberColumn(format="%.1f")
    st.dataframe(
        tbl,
        column_config={
            "expected_bps": st.column_config.NumberColumn("Expected cost (bps)", format="%.1f"),
            "std_bps": st.column_config.NumberColumn("Timing risk, std (bps)", format="%.1f"),
            "p95_bps": st.column_config.NumberColumn("95th pct cost (bps)", format="%.1f"),
            "spread_bps": st.column_config.NumberColumn("of which spread", format="%.1f"),
            "transient_bps": st.column_config.NumberColumn("of which book walk", format="%.1f"),
            "permanent_bps": st.column_config.NumberColumn("of which permanent", format="%.1f"),
            "expected_usd": st.column_config.NumberColumn("Expected cost ($)", format="dollar"),
            "score_bps": st.column_config.NumberColumn(f"Score (mean + {lam:g} std)", format="%.1f"),
        },
    )
    fig = go.Figure()
    for name, (b, costs) in results.items():
        fig.add_trace(go.Histogram(x=I.to_bps(costs, m, order), name=name, marker=dict(color=colors[name]), opacity=0.6, nbinsx=60))
    fig = ui.style(fig, title=f"Distribution of implementation shortfall over {sim.n_paths:,} simulated paths", x_title="Cost (bps of arrival notional)", y_title="Paths", height=380)
    fig.update_layout(barmode="overlay", hovermode="closest")
    ui.show(fig)
    st.caption("Dumping has a certain (but large) cost. Slicing lowers the expected cost but makes it uncertain, because the price can drift while you wait.")

# --------------------------------------------------------------------------- slice sweep
with tab_sweep:
    sw = I.slice_sweep(m, order, horizon, sim.sweep_max_slices, side, lam)
    n_best = int(sw["n_slices"][np.argmin(sw["score_bps"])])
    fig = go.Figure()
    for key, name, i in (("mean_bps", "Expected cost", 0), ("std_bps", "Timing risk (std)", 1), ("score_bps", f"Mean + {lam:g} x std", 2)):
        fig.add_trace(go.Scatter(x=sw["n_slices"], y=sw[key], name=name, line=dict(color=ui.CATEGORICAL[i]), hovertemplate="%{y:.1f} bps"))
    fig.add_vline(x=n_best, line=dict(color=ui.NEUTRAL, dash="dot"), annotation_text=f"best N = {n_best}")
    ui.show(ui.style(fig, title=f"TWAP over {horizon} min: cost and risk vs number of slices", x_title="Number of slices (1 = all at once)", y_title="bps", y_format=".1f"))
    st.caption(
        "More slices means more refill time between child orders, so book-walk cost falls. Beyond the point where slices are "
        "far apart compared with the resilience half-life, extra slices barely help. Note the horizon is fixed here: "
        "stretching the horizon is the lever that trades lower impact for more timing risk."
    )

# --------------------------------------------------------------------------- explainer
with tab_model:
    st.markdown(
        rf"""
### What the model assumes
- **Book shape.** Consuming $X$ shares of depth moves the price by
  $f(X) = \eta\,\sigma_d S_0\,(X/\mathrm{{ADV}})^\alpha$. With $\alpha = 0.5$ this is the empirical *square-root law*.
- **Walking the book.** A child order of $q$ shares starting from consumed depth $X$ pays on average
  $\frac{{F(X+q) - F(X)}}{{q}}$ below mid, where $F$ is the integral of $f$.
- **Resilience.** Consumed depth decays with a half-life of **{half_life:g} min** as liquidity providers refill the book
  (Obizhaeva-Wang; Alfonsi-Fruth-Schied).
- **Permanent impact** is linear in shares, $\gamma\,\sigma_d S_0 / \mathrm{{ADV}}$ per share, as no-arbitrage requires.
- **Spread.** Every share pays half the quoted spread.
- **Noise.** The unaffected price is arithmetic Brownian motion with daily vol $\sigma_d = \sigma/\sqrt{{{s.market.trading_days_per_year}}}$.
  The Monte Carlo spread matches the closed-form timing-risk formula (unit-tested).

### What it leaves out
Other traders reacting to your flow, information leakage, intraday volume curves, queue position and limit orders,
dark pools, and any calibration to real data. Paper-trading fills have none of these costs either, which is why paper
results usually flatter a strategy.

### Talking points
1. **Why slice:** impact is about *speed*. Selling into a book faster than it refills means each share is sold lower.
2. **Why not slice forever:** waiting exposes the unsold inventory to price moves; that is the Almgren-Chriss cost/risk trade-off.
3. **Thin markets** (low ADV, high $\eta$, wide spread, slow resilience) make every term bigger, and slow resilience is
   what makes dumping especially expensive.
4. **Permanent impact cannot be avoided** by any schedule; it is the information the market takes from your trade.
"""
    )
    ui.resolved_config_expander(cfg)
