"""Shared Streamlit helpers: config-driven controls, cached loaders, chart styling.

Design tokens live here (one place). Categorical colors are assigned by an entity's
fixed position in the configured universe, never by selection order, so filtering
never repaints the survivors.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .settings import ParamSpec, Settings, find_config_dir, load_config, load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Validated categorical order (dataviz reference palette, light-mode steps).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
NEUTRAL = "#898781"
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"  # diverging poles (blue <-> red)


# --------------------------------------------------------------------------- config


@st.cache_resource
def _config_dir() -> Path:
    return find_config_dir(Path(__file__).resolve().parent)


def settings() -> Settings:
    return load_settings(_config_dir())


def config(name: str):
    return load_config(name, _config_dir())


def config_dir() -> Path:
    return _config_dir()


# --------------------------------------------------------------------------- controls


def number_from_spec(spec: ParamSpec, key: str, container=None, default=None, fmt: str | None = None):
    """A number input whose default/bounds/step/label all come from config."""
    c = container or st.sidebar
    value = spec.clamp(spec.default if default is None else default)
    if spec.is_int:
        return int(c.number_input(spec.label, int(spec.min), int(spec.max), int(value), int(spec.step), key=key, help=spec.help))
    return float(
        c.number_input(spec.label, float(spec.min), float(spec.max), float(value), float(spec.step), key=key, help=spec.help, format=fmt or "%.4g")
    )


def slider_from_spec(spec: ParamSpec, key: str, container=None, default=None):
    c = container or st.sidebar
    value = spec.clamp(spec.default if default is None else default)
    if spec.is_int:
        return int(c.slider(spec.label, int(spec.min), int(spec.max), int(value), int(spec.step), key=key, help=spec.help))
    return float(c.slider(spec.label, float(spec.min), float(spec.max), float(value), float(spec.step), key=key, help=spec.help))


# --------------------------------------------------------------------------- data


@st.cache_resource(show_spinner=False)
def _provider(settings_json: str, refresh_token: int):
    from .providers import build_provider

    s = Settings.model_validate_json(settings_json)
    return build_provider(s, _config_dir(), refresh=refresh_token > 0)


def provider(s: Settings, refresh_token: int = 0):
    return _provider(s.model_dump_json(), refresh_token)


@st.cache_data(show_spinner="Loading daily bars…", ttl=3600)
def daily_bars(settings_json: str, symbols: tuple[str, ...], end: date, refresh_token: int = 0) -> tuple[dict[str, pd.DataFrame], str, str | None]:
    from .data import load_daily

    s = Settings.model_validate_json(settings_json)
    p = _provider(settings_json, refresh_token)
    bars = load_daily(p, symbols, s, end)
    return bars, p.name, p.fallback_reason


@st.cache_data(show_spinner="Loading intraday bars…", ttl=3600)
def intraday_bars(settings_json: str, symbols: tuple[str, ...], session: date, refresh_token: int = 0):
    s = Settings.model_validate_json(settings_json)
    p = _provider(settings_json, refresh_token)
    return p.intraday_bars(list(symbols), session, s.data.intraday_bar_minutes), p.name, p.fallback_reason


def source_banner(name: str, fallback_reason: str | None) -> None:
    if fallback_reason:
        st.warning(
            f"**Synthetic data.** Showing `{name}` because the configured provider is unavailable: {fallback_reason}. "
            "Numbers illustrate the method only. Add Alpaca keys to `.env` on your machine for real data.",
            icon="⚠️",
        )
    else:
        st.caption(f"Data source: `{name}`")


# --------------------------------------------------------------------------- charts


def color_map(entities: Sequence[str], universe_order: Sequence[str]) -> dict[str, str]:
    """Stable colors: by position in ``universe_order``; unknown entities appended after."""
    order = list(universe_order) + [e for e in entities if e not in universe_order]
    out = {}
    for e in entities:
        i = order.index(e)
        out[e] = CATEGORICAL[i] if i < len(CATEGORICAL) else NEUTRAL  # never cycle hues
    return out


def style(fig: go.Figure, title: str | None = None, y_title: str | None = None, x_title: str | None = None, y_format: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
    )
    fig.update_xaxes(title=x_title, showgrid=False)
    fig.update_yaxes(title=y_title, tickformat=y_format, gridwidth=1, zeroline=False)
    fig.update_traces(selector=dict(type="scatter"), line=dict(width=2))
    return fig


def line_fig(df: pd.DataFrame, colors: Mapping[str, str], **style_kw) -> go.Figure:
    fig = go.Figure()
    fmt = style_kw.get("y_format")
    hover = f"%{{y:{fmt}}}" if fmt else "%{y:.4g}"
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines", line=dict(color=colors.get(col, NEUTRAL)), hovertemplate=hover))
    return style(fig, **style_kw)


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def resolved_config_expander(*models) -> None:
    with st.expander("Resolved configuration"):
        for m in models:
            st.code(json.dumps(json.loads(m.model_dump_json()), indent=2), language="json")
