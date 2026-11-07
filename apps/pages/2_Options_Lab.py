from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab import bsm, ui
from quantlab import vol as V

st.set_page_config(page_title="Options Lab", layout="wide")
s = ui.settings()
oc = ui.config("options")
uni = ui.config("universe")
P, A, U = oc.params, oc.attribution, oc.greek_units
DPY = s.market.calendar_days_per_year
st.session_state.setdefault("refresh_token", 0)

# --------------------------------------------------------------------------- sidebar
sb = st.sidebar
is_call = sb.radio("Option", ("Call", "Put"), horizontal=True) == "Call"
link = sb.checkbox("Link to a ticker", help="Seed spot and vol from market data; enables the vol-smile tab")
seed: dict[str, float] = {}
linked_sym, realized = None, None
if link:
    linked_sym = sb.selectbox("Ticker", uni.all_symbols)
    bars, src, fb = ui.daily_bars(s.model_dump_json(), (linked_sym,), date.today(), st.session_state.refresh_token)
    if linked_sym in bars:
        df = bars[linked_sym]
        w = s.vol.windows[0]
        realized = float(V.close_to_close_vol(V.log_returns(df["close"]), w, s.market.trading_days_per_year).iloc[-1])
        spot0 = float(df["close"].iloc[-1])
        seed = {"spot": spot0, "strike": float(round(spot0)), "vol": round(realized, 2)}
        sb.caption(f"{linked_sym}: last close {spot0:,.2f}, {w}d realized vol {realized:.1%}" + (" (synthetic)" if fb else ""))

spot = ui.number_from_spec(P["spot"], f"spot_{linked_sym}", default=seed.get("spot"))
strike = ui.number_from_spec(P["strike"], f"strike_{linked_sym}", default=seed.get("strike"))
days = ui.number_from_spec(P["days"], "days")
sigma = ui.number_from_spec(P["vol"], f"vol_{linked_sym}", default=seed.get("vol"))
rate = ui.number_from_spec(P["rate"], "rate", default=P["rate"].default if P["rate"].default is not None else s.market.risk_free_rate)
q = ui.number_from_spec(P["div_yield"], "q")
T = days / DPY
kind = "call" if is_call else "put"


def scaled(S=spot, K=strike, T_=T, sig=sigma):
    g = bsm.greeks(S, K, T_, rate, q, sig, is_call)
    return bsm.scale_greeks(g, U.vega_per_vol_points, U.rho_per_rate_points, U.theta_per_days, DPY)


st.title("Options Lab")
st.caption("Black-Scholes-Merton (European, continuous dividend yield) with a CRR binomial tree for American exercise.")

value = float(bsm.price(spot, strike, T, rate, q, sigma, is_call))
g = {k: float(v) for k, v in scaled().items()}
american = bsm.binomial_price(spot, strike, T, rate, q, sigma, is_call, oc.binomial.steps, american=True)
cols = st.columns(7)
cols[0].metric(f"{kind.title()} value", f"{value:,.4f}")
cols[1].metric("Delta", f"{g['delta']:.4f}")
cols[2].metric("Gamma", f"{g['gamma']:.5f}")
cols[3].metric(f"Vega / {U.vega_per_vol_points:g} vol pt", f"{g['vega']:.4f}")
cols[4].metric(f"Theta / {U.theta_per_days:g} day", f"{g['theta']:.4f}")
cols[5].metric(f"Rho / {U.rho_per_rate_points:g} rate pt", f"{g['rho']:.4f}")
cols[6].metric("American (tree)", f"{american:,.4f}", delta=f"{american - value:+.4f} vs BSM", delta_color="off", help="Early-exercise premium plus tree discretization error")

span = oc.chart.spot_range_pct
S_grid = np.linspace(spot * (1 - span), spot * (1 + span), oc.chart.points)

tabs = st.tabs(["Price vs spot", "Greeks vs spot", "Time decay", "Volatility", "American vs European", "P&L attribution", "Vol smile"])
SER = ui.CATEGORICAL

# --------------------------------------------------------------------------- price vs spot
with tabs[0]:
    fig = go.Figure()
    horizons = [(days, f"{days}d to expiry"), (max(days // 2, 1), f"{max(days // 2, 1)}d"), (max(days // 6, 1), f"{max(days // 6, 1)}d")]
    for i, (d, name) in enumerate(dict.fromkeys(horizons)):
        fig.add_trace(go.Scatter(x=S_grid, y=bsm.price(S_grid, strike, d / DPY, rate, q, sigma, is_call), name=name, line=dict(color=SER[i])))
    payoff = np.maximum(S_grid - strike, 0) if is_call else np.maximum(strike - S_grid, 0)
    fig.add_trace(go.Scatter(x=S_grid, y=payoff, name="Payoff at expiry", line=dict(color=ui.NEUTRAL, dash="dot")))
    fig.add_vline(x=spot, line=dict(color=ui.NEUTRAL, width=1), annotation_text="spot")
    ui.show(ui.style(fig, title="Option value as the stock moves", x_title="Spot", y_title="Value", y_format=",.2f"))
    st.caption("The gap between each curve and the payoff is time value. It is largest at the strike and shrinks as expiry approaches.")

# --------------------------------------------------------------------------- greeks vs spot
with tabs[1]:
    G = scaled(S=S_grid)
    names = {
        "delta": "Delta",
        "gamma": "Gamma",
        "vega": f"Vega per {U.vega_per_vol_points:g} vol pt",
        "theta": f"Theta per {U.theta_per_days:g} day",
    }
    c = st.columns(2)
    for i, (k, title) in enumerate(names.items()):
        fig = go.Figure(go.Scatter(x=S_grid, y=G[k], name=title, line=dict(color=SER[0])))
        fig.add_vline(x=strike, line=dict(color=ui.NEUTRAL, width=1, dash="dot"), annotation_text="K")
        with c[i % 2]:
            ui.show(ui.style(fig, title=title, x_title="Spot", height=300))
    st.caption("Gamma and vega peak near the strike; theta is most negative there too: the cost of owning convexity.")

# --------------------------------------------------------------------------- time decay
with tabs[2]:
    d_grid = np.arange(max(days, 2), 0, -1)
    moneyness = {"At strike (S=K)": strike, f"Current spot ({spot:,.2f})": spot}
    fig_v, fig_t = go.Figure(), go.Figure()
    for i, (name, S_) in enumerate(moneyness.items()):
        fig_v.add_trace(go.Scatter(x=d_grid, y=bsm.price(S_, strike, d_grid / DPY, rate, q, sigma, is_call), name=name, line=dict(color=SER[i])))
        fig_t.add_trace(go.Scatter(x=d_grid, y=scaled(S=S_, T_=d_grid / DPY)["theta"], name=name, line=dict(color=SER[i])))
    for f in (fig_v, fig_t):
        f.update_xaxes(autorange="reversed")
    c1, c2 = st.columns(2)
    with c1:
        ui.show(ui.style(fig_v, title="Value as time passes", x_title="Days to expiry", y_format=",.2f"))
    with c2:
        ui.show(ui.style(fig_t, title=f"Theta per {U.theta_per_days:g} day", x_title="Days to expiry"))
    st.caption("Decay is not linear: an at-the-money option loses value fastest in its final weeks (theta ~ 1/sqrt(T)).")

# --------------------------------------------------------------------------- volatility
with tabs[3]:
    v_grid = np.linspace(P["vol"].min, min(P["vol"].max, max(1.0, sigma * 3)), oc.chart.points)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Scatter(x=v_grid, y=bsm.price(spot, strike, T, rate, q, v_grid, is_call), name="Value", line=dict(color=SER[0])))
        fig.add_vline(x=sigma, line=dict(color=ui.NEUTRAL, width=1), annotation_text="sigma")
        if realized is not None:
            fig.add_vline(x=realized, line=dict(color=SER[1], width=1, dash="dot"), annotation_text="realized")
        f = ui.style(fig, title="Value vs volatility", x_title="Volatility")
        f.update_xaxes(tickformat=".0%")
        ui.show(f)
    with c2:
        fig = go.Figure(go.Scatter(x=v_grid, y=scaled(sig=v_grid)["vega"], name="Vega", line=dict(color=SER[0])))
        f = ui.style(fig, title=f"Vega per {U.vega_per_vol_points:g} vol pt vs volatility", x_title="Volatility")
        f.update_xaxes(tickformat=".0%")
        ui.show(f)
    st.caption("ATM value is almost linear in vol (so vega is near-constant); OTM options are convex in vol (volga > 0).")

# --------------------------------------------------------------------------- american vs european
with tabs[4]:
    K_grid = np.linspace(spot * (1 - span), spot * (1 + span), 41)
    eu = bsm.price(spot, K_grid, T, rate, q, sigma, is_call)
    am = np.array([bsm.binomial_price(spot, k, T, rate, q, sigma, is_call, oc.binomial.steps, american=True) for k in K_grid])
    fig = go.Figure(go.Bar(x=K_grid, y=am - eu, name="Early-exercise premium", marker=dict(color=SER[0], cornerradius=4)))
    fig = ui.style(fig, title=f"Early-exercise premium of an American {kind} (tree minus BSM)", x_title="Strike", y_format=",.3f", height=380)
    fig.update_layout(hovermode="closest", bargap=0.3)
    ui.show(fig)
    st.caption(
        "Without dividends an American call is never exercised early, so its premium is ~0 (only tree discretization). "
        "Deep in-the-money puts carry real premium: once interest on the strike exceeds the remaining time value, exercising early wins. "
        f"Tree: {oc.binomial.steps} steps."
    )

# --------------------------------------------------------------------------- P&L attribution
with tabs[5]:
    st.markdown("Decompose a scenario's P&L into Greek contributions (second-order Taylor expansion).")
    c = st.columns(4)
    dS_pct = ui.slider_from_spec(A["spot_move_pct"], "att_ds", container=c[0])
    dsig = ui.slider_from_spec(A["vol_change"], "att_dv", container=c[1])
    dd = ui.slider_from_spec(A["days_passed"], "att_dd", container=c[2])
    dr = ui.slider_from_spec(A["rate_change"], "att_dr", container=c[3])
    s0 = bsm.OptionState(spot, strike, T, rate, q, sigma, is_call)
    s1 = s0.bump(dS=spot * dS_pct, dT=-dd / DPY, dr=dr, dsigma=dsig)
    att = bsm.pnl_attribution(s0, s1)
    parts = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga", "unexplained"]
    fig = go.Figure(
        go.Waterfall(
            x=[p.title() for p in parts] + ["Actual"],
            y=[att[p] for p in parts] + [att["actual"]],
            measure=["relative"] * len(parts) + ["total"],
            increasing=dict(marker=dict(color=ui.POSITIVE)),
            decreasing=dict(marker=dict(color=ui.NEGATIVE)),
            totals=dict(marker=dict(color=ui.NEUTRAL)),
            connector=dict(line=dict(color=ui.NEUTRAL, width=1)),
            hovertemplate="%{x}: %{y:,.4f}<extra></extra>",
        )
    )
    fig = ui.style(fig, title=f"P&L per option: {att['actual']:+,.4f} ({att['actual'] / value:+.1%} of premium)" if value else "P&L", y_format=",.3f", height=400)
    fig.update_layout(hovermode="closest", showlegend=False)
    ui.show(fig)
    st.caption(
        "Big moves make 'unexplained' grow: the expansion is local. A long option in a crash typically shows delta loss "
        "partly offset by gamma and by vega when implied vol spikes."
    )

# --------------------------------------------------------------------------- vol smile
with tabs[6]:
    if not linked_sym or not seed:
        st.info("Tick **Link to a ticker** in the sidebar to load an option chain.")
    else:

        @st.cache_data(ttl=300, show_spinner="Loading option chain…")
        def chain(settings_json: str, sym: str, spot_: float, token: int):
            p = ui.provider(s, token)
            return p.option_chain(sym, spot_, date.today()), p.name, p.fallback_reason

        ch, csrc, cfb = chain(s.model_dump_json(), linked_sym, seed["spot"], st.session_state.refresh_token)
        ui.source_banner(csrc, cfb)
        S0 = seed["spot"]
        today = pd.Timestamp(date.today())
        ch = ch.dropna(subset=["mid"]).copy()
        ch["dte"] = (pd.to_datetime(ch["expiry"]) - today).dt.days
        ch = ch[(ch["dte"] >= oc.chain.min_days_to_expiry) & (ch["dte"] <= oc.chain.max_days_to_expiry) & (ch["mid"] > 0)]
        ch["T"] = ch["dte"] / DPY
        ch["fwd"] = S0 * np.exp((rate - q) * ch["T"])
        ch["log_m"] = np.log(ch["strike"] / ch["fwd"])
        # OTM options only: puts below the forward, calls above (more liquid, no early-exercise noise)
        ch = ch[(ch["log_m"].abs() <= oc.chain.max_abs_log_moneyness) & (((ch["right"] == "put") & (ch["log_m"] < 0)) | ((ch["right"] == "call") & (ch["log_m"] >= 0)))]
        if ch.empty:
            st.info("No contracts in the configured expiry/moneyness window.")
        else:
            iv = oc.implied_vol
            ch["iv"] = [
                bsm.implied_vol(r.mid, S0, r.strike, r.T, rate, q, r.right == "call", iv.lower, iv.upper, iv.tolerance)
                for r in ch.itertuples()
            ]
            ch = ch.dropna(subset=["iv"])
            expiries = sorted(ch["expiry"].unique())[: len(SER)]
            fig = go.Figure()
            for i, e in enumerate(expiries):
                sub = ch[ch["expiry"] == e].sort_values("strike")
                fig.add_trace(go.Scatter(x=sub["strike"], y=sub["iv"], name=f"{pd.Timestamp(e):%Y-%m-%d} ({int(sub['dte'].iloc[0])}d)", mode="lines+markers", line=dict(color=SER[i]), marker=dict(size=8)))
            if realized is not None:
                fig.add_hline(y=realized, line=dict(color=ui.NEUTRAL, dash="dot"), annotation_text=f"{s.vol.windows[0]}d realized {realized:.0%}")
            fig.add_vline(x=S0, line=dict(color=ui.NEUTRAL, width=1), annotation_text="spot")
            ui.show(ui.style(fig, title=f"{linked_sym} implied-vol smile (from mid prices, OTM options)", x_title="Strike", y_format=".0%"))
            atm = ch.loc[ch.groupby("expiry")["log_m"].apply(lambda x: x.abs().idxmin())]
            if realized is not None and not atm.empty:
                prem = float(atm["iv"].iloc[0] - realized)
                st.info(f"Nearest-expiry ATM implied vol is {atm['iv'].iloc[0]:.1%} vs {realized:.1%} realized: a vol risk premium of {prem * 100:+.1f} vol points.")
            st.caption(
                "Equity skew: downside puts trade at higher implied vol than upside calls because markets price crash risk "
                "that Black-Scholes's constant-vol lognormal world leaves out."
            )
            with st.expander("Chain data"):
                st.dataframe(ch[["symbol", "expiry", "dte", "strike", "right", "bid", "ask", "mid", "iv", "provider_iv"]].sort_values(["expiry", "strike"]), hide_index=True)
