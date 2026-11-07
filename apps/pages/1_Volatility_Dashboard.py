from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quantlab import stress, ui
from quantlab import vol as V

st.set_page_config(page_title="Volatility Dashboard", layout="wide")
s = ui.settings()
uni = ui.config("universe")
ppy = s.market.trading_days_per_year
bench = s.market.benchmark
colors = ui.color_map(uni.all_symbols, uni.all_symbols)
st.session_state.setdefault("refresh_token", 0)

# --------------------------------------------------------------------------- sidebar
sb = st.sidebar
symbols = sb.multiselect("Symbols", uni.all_symbols, default=list(uni.default_selection))
for g in uni.groups:
    if g.note:
        sb.caption(f"**{g.name}:** {g.note}")
ESTIMATORS = ("Close-to-close", "Parkinson (high-low)", "EWMA")
estimator = sb.radio("Vol estimator", ESTIMATORS, help="EWMA ignores the window and uses vol.ewma_lambda")
window = sb.selectbox("Window (trading days)", s.vol.windows, index=0)
if sb.button("Refresh data", help="Bypass the CSV cache and refetch"):
    st.session_state.refresh_token += 1
    st.cache_data.clear()

st.title("Volatility Dashboard")
if not symbols:
    st.info("Pick at least one symbol.")
    st.stop()

need = tuple(sorted(set(symbols) | {bench}))
bars, src, fallback = ui.daily_bars(s.model_dump_json(), need, date.today(), st.session_state.refresh_token)
ui.source_banner(src, fallback)
missing = [x for x in need if x not in bars]
if missing:
    st.warning(f"No data for {missing}.")
if bench not in bars:
    st.error(f"Benchmark {bench} has no data; stress views need it.")
    st.stop()
shown = [x for x in symbols if x in bars]


def estimate(df: pd.DataFrame, which: str) -> pd.Series:
    r = V.log_returns(df["close"])
    if which == ESTIMATORS[0]:
        return V.close_to_close_vol(r, window, ppy)
    if which == ESTIMATORS[1]:
        return V.parkinson_vol(df["high"], df["low"], window, ppy)
    return V.ewma_vol(r, s.vol.ewma_lambda, ppy)


vols = pd.DataFrame({x: estimate(bars[x], estimator) for x in shown})

tab_vol, tab_stress, tab_intra, tab_px, tab_pf = st.tabs(["Realized vol", "Stress day", "Intraday", "Prices", "My paper portfolio"])

# --------------------------------------------------------------------------- realized vol
with tab_vol:
    label = f"{estimator}" + ("" if estimator == "EWMA" else f", {window}d")
    ui.show(ui.line_fig(vols.dropna(how="all"), colors, title=f"Annualized realized volatility ({label})", y_format=".0%"))
    summary = pd.DataFrame({x: V.vol_summary(vols[x], s.vol.percentile_window) for x in shown}).T
    summary = summary.sort_values("current", ascending=False)
    st.dataframe(
        summary,
        column_config={
            "current": st.column_config.NumberColumn("Current", format="percent"),
            "mean": st.column_config.NumberColumn("Mean", format="percent"),
            "min": st.column_config.NumberColumn("Min", format="percent"),
            "max": st.column_config.NumberColumn("Max", format="percent"),
            "percentile": st.column_config.ProgressColumn(f"Percentile (last {s.vol.percentile_window}d)", min_value=0, max_value=100, format="%.0f"),
        },
    )
    st.markdown("**Estimator comparison**: do range-based and EWMA estimates tell the same story?")
    pick = st.selectbox("Symbol", shown, key="cmp_sym")
    cmp = pd.DataFrame({e: estimate(bars[pick], e) for e in ESTIMATORS}).dropna(how="all")
    ui.show(ui.line_fig(cmp, ui.color_map(ESTIMATORS, ESTIMATORS), title=f"{pick}: estimator comparison", y_format=".0%"))
    st.caption(
        "Close-to-close uses only closes. Parkinson uses the daily high-low range: more efficient, but blind to overnight gaps. "
        "EWMA weights recent days more, so it reacts faster to shocks and decays faster after them."
    )

# --------------------------------------------------------------------------- stress day
with tab_stress:
    methods = ("worst_benchmark_return", "largest_benchmark_range")
    c1, c2 = st.columns(2)
    method = c1.selectbox("Auto-detect by", methods, index=methods.index(s.stress.detection), format_func=lambda m: m.replace("_", " "))
    detected = stress.detect_stress_day(bars[bench], method)
    default_day = pd.Timestamp(s.stress.manual_date) if s.stress.manual_date else detected
    sessions = bars[bench].index
    picked = c2.date_input("Stress day (override)", default_day.date(), min_value=sessions[1].date(), max_value=sessions[-1].date())
    day = sessions[sessions <= pd.Timestamp(picked)][-1]  # snap to the last session on/before the pick
    st.session_state["stress_day"] = day
    if day.date() != picked:
        st.caption(f"{picked} is not a session; using {day.date()}.")
    st.caption(f"Auto-detected ({method.replace('_', ' ')} for {bench}): **{detected.date()}**")

    tbl = stress.stress_table({x: bars[x] for x in set(shown) | {bench}}, day, bench, s.vol.zscore_window, s.vol.beta_window, ppy)
    order = list(tbl.index)

    def bar(col: str, title: str, fmt: str) -> go.Figure:
        fig = go.Figure(go.Bar(x=order, y=tbl[col], marker=dict(color=[colors[x] for x in order], cornerradius=4), hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra></extra>", name=title))
        fig = ui.style(fig, title=title, y_format=fmt, height=340)
        fig.update_layout(hovermode="closest", showlegend=False, bargap=0.35)
        return fig

    c1, c2 = st.columns(2)
    with c1:
        ui.show(bar("return", f"Return on {day.date()}", ".1%"))
    with c2:
        ui.show(bar("z", f"Move in units of own trailing daily vol ({s.vol.zscore_window}d)", ".1f"))
    ui.show(bar("residual", f"Idiosyncratic move (actual minus beta x {bench})", ".1%"))

    worst = tbl["z"].idxmin()
    b = tbl.loc[bench]
    st.info(
        f"**Read-out:** {bench} moved {b['return']:.1%} ({b['z']:.1f} sigma). "
        f"The most extreme move *relative to its own risk* was {worst}: {tbl.loc[worst, 'return']:.1%}, "
        f"or {tbl.loc[worst, 'z']:.1f} sigma. A raw % move overstates how unusual the day was for high-vol names "
        "and understates it for low-vol ones; z-scores make them comparable."
    )
    st.dataframe(
        tbl,
        column_config={
            "return": st.column_config.NumberColumn("Return", format="percent"),
            "range": st.column_config.NumberColumn("High-low range", format="percent"),
            "trailing_vol": st.column_config.NumberColumn("Trailing vol", format="percent"),
            "z": st.column_config.NumberColumn("z (sigma)", format="%.2f"),
            "beta": st.column_config.NumberColumn(f"Beta to {bench}", format="%.2f"),
            "beta_expected": st.column_config.NumberColumn("Beta-implied", format="percent"),
            "residual": st.column_config.NumberColumn("Residual", format="percent"),
        },
    )
    st.caption("Trailing vol and beta use only data before the stress day (no look-ahead).")

# --------------------------------------------------------------------------- intraday
with tab_intra:
    day = st.session_state.get("stress_day", stress.detect_stress_day(bars[bench], s.stress.detection))
    st.markdown(f"**{day.date()}**: {s.data.intraday_bar_minutes}-minute bars, % change vs previous close.")
    intra, isrc, ifb = ui.intraday_bars(s.model_dump_json(), tuple(shown), day.date(), st.session_state.refresh_token)
    if ifb and not fallback:
        st.warning(f"Intraday data fell back to `{isrc}`: {ifb}")
    rows, paths = {}, {}
    for x in shown:
        m = intra.get(x)
        if m is None or m.empty:
            continue
        prev_close = bars[x]["close"].shift(1).loc[day]
        paths[x] = m["close"] / prev_close - 1.0
        rows[x] = stress.intraday_profile(m, prev_close)
    if not paths:
        st.info("No intraday bars for that session (holiday, or outside the feed's history).")
    else:
        ui.show(ui.line_fig(pd.DataFrame(paths), colors, title="Intraday path vs previous close", y_format=".1%"))
        prof = pd.DataFrame(rows).T
        prof["trough_time"] = pd.to_datetime(prof["trough_time"]).dt.strftime("%H:%M")
        st.dataframe(
            prof,
            column_config={
                "open_gap": st.column_config.NumberColumn("Open gap", format="percent"),
                "trough": st.column_config.NumberColumn("Intraday low", format="percent"),
                "trough_time": "Low at",
                "minutes_to_trough": st.column_config.NumberColumn("Minutes to low", format="%.0f"),
                "close": st.column_config.NumberColumn("Close", format="percent"),
                "recovered_frac": st.column_config.NumberColumn("Share of drop recovered", format="percent"),
            },
        )
        if s.data.feed == "iex":
            st.caption("IEX feed: prices are representative, but volume is only IEX's share of the market.")

# --------------------------------------------------------------------------- prices
with tab_px:
    px = pd.DataFrame({x: bars[x]["close"] for x in shown}).dropna()
    ui.show(ui.line_fig(px / px.iloc[0] * 100, colors, title="Growth of 100", y_format=".0f"))

# --------------------------------------------------------------------------- paper portfolio
with tab_pf:
    st.markdown("Live (paper) positions from your broker, with portfolio risk measured on the same data.")
    if st.button("Load positions"):
        from quantlab.brokers import connect_broker

        try:
            broker = connect_broker(s, config_dir=ui.config_dir())
            st.session_state["positions"] = broker.positions()
            st.session_state["account"] = broker.account()
        except Exception as e:  # missing keys, network policy, mode guard
            st.session_state["positions"] = None
            st.error(f"Could not load positions: {e}")
    positions = st.session_state.get("positions")
    if positions:
        acct = st.session_state["account"]
        eq = [p for p in positions if p.asset_class == "us_equity"]
        pf_syms = tuple(sorted({p.symbol for p in eq} | {bench}))
        pf_bars, _, pf_fb = ui.daily_bars(s.model_dump_json(), pf_syms, date.today(), st.session_state.refresh_token)
        if pf_fb:
            st.warning("Positions are real, but returns used for risk are synthetic. Risk figures are illustrative only.")
        rets = pd.DataFrame({x: V.log_returns(b["close"]) for x, b in pf_bars.items()}).iloc[-s.vol.percentile_window:]
        from quantlab.portfolio import portfolio_risk

        sigma, risk = portfolio_risk({p.symbol: p.market_value for p in eq}, rets, ppy)
        c1, c2, c3 = st.columns(3)
        c1.metric("Equity", f"${acct.equity:,.0f}")
        c2.metric("Gross exposure", f"${sum(abs(p.market_value) for p in eq):,.0f}")
        c3.metric("Annualized $ vol", f"${sigma:,.0f}", help=f"{sigma / acct.equity:.1%} of equity" if acct.equity else None)
        day = st.session_state.get("stress_day")
        if day is not None:
            qty = {p.symbol: p.qty for p in eq}

            def day_pnl(sym: str) -> float:
                c = pf_bars.get(sym, pd.DataFrame()).get("close")
                if c is None or day not in c.index:
                    return np.nan
                return qty[sym] * (c.loc[day] - c.shift(1).loc[day])

            risk["stress_day_pnl"] = risk.index.map(day_pnl)
        st.dataframe(
            risk,
            column_config={
                "value": st.column_config.NumberColumn("Market value", format="dollar"),
                "standalone_vol": st.column_config.NumberColumn("Standalone $ vol", format="dollar"),
                "risk_contrib": st.column_config.NumberColumn("Risk contribution", format="dollar"),
                "pct_of_risk": st.column_config.NumberColumn("% of portfolio risk", format="percent"),
                "stress_day_pnl": st.column_config.NumberColumn("P&L if held on stress day", format="dollar"),
            },
        )
        st.caption("Risk contributions sum to portfolio vol (Euler decomposition). Hedges show up as negative contributions.")
        other = [p for p in positions if p.asset_class != "us_equity"]
        if other:
            st.caption(f"Non-equity positions not included in risk: {', '.join(p.symbol for p in other)}")
    elif positions is not None:
        st.info("No open positions.")
