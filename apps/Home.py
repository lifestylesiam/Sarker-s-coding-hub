import streamlit as st

from quantlab import ui

st.set_page_config(page_title="Quant Lab", layout="wide")
s = ui.settings()

st.title("Quant Lab")
st.markdown(
    """
Three linked projects, all driven by the YAML files in `config/`:

| Page | Question it answers |
|---|---|
| **1 · Volatility Dashboard** | How volatile are my names, and how did each move on the stress day *in units of its own risk*? |
| **2 · Options Lab** | What is an option worth, what drives its P&L, and how does real implied vol compare with realized? |
| **3 · Impact Simulator** | What does it cost to sell a large order in a thin market: all at once vs sliced? *(toy model)* |

Use the sidebar to switch pages.
"""
)
st.subheader("Configuration")
c1, c2, c3 = st.columns(3)
c1.metric("Data provider", s.data.provider, help=f"fallback: {s.data.fallback_provider}")
c2.metric("Feed / adjustment", f"{s.data.feed} / {s.data.adjustment}")
c3.metric("Trading mode", s.trading.mode.value)
st.caption(f"Config directory: `{ui.config_dir()}`. Override any value with `QUANTLAB__SECTION__KEY` env vars.")
