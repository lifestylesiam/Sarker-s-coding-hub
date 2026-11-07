"""Typed, validated configuration.

Sources, lowest to highest precedence:
  1. Model defaults below (documented fallbacks only).
  2. YAML files in the config directory.
  3. Environment variables ``QUANTLAB__<FILE?>__<SECTION>__<KEY>`` (see ``_env_overrides``).
  4. Explicit overrides passed by the caller (CLI ``--set a.b=c``).

The config directory is ``$QUANTLAB_CONFIG_DIR`` if set, otherwise the nearest ``config/``
directory found walking up from the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CONFIG_DIR_ENV = "QUANTLAB_CONFIG_DIR"
ENV_PREFIX = "QUANTLAB__"
SETTINGS_FILE = "settings.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- settings.yaml


class DataSettings(_Strict):
    provider: Literal["alpaca", "synthetic"] = "alpaca"
    fallback_provider: Literal["alpaca", "synthetic"] | None = "synthetic"
    feed: str = "iex"
    options_feed: str = "indicative"
    adjustment: Literal["raw", "split", "dividend", "all"] = "all"
    lookback_days: int = Field(730, gt=0)
    cache_dir: Path = Path("data/cache")
    cache_max_age_hours: float = Field(12, ge=0)
    intraday_bar_minutes: int = Field(1, gt=0)


class MarketSettings(_Strict):
    benchmark: str = "SPY"
    timezone: str = "America/New_York"
    trading_days_per_year: int = Field(252, gt=0)
    calendar_days_per_year: int = Field(365, gt=0)
    minutes_per_session: int = Field(390, gt=0)
    risk_free_rate: float = 0.04


class VolSettings(_Strict):
    windows: tuple[int, ...] = (20, 60)
    ewma_lambda: float = Field(0.94, gt=0, lt=1)
    zscore_window: int = Field(60, gt=1)
    beta_window: int = Field(120, gt=1)
    percentile_window: int = Field(252, gt=1)

    @field_validator("windows")
    @classmethod
    def _positive_windows(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v or any(w < 2 for w in v):
            raise ValueError("vol.windows must be a non-empty list of integers >= 2")
        return tuple(sorted(set(v)))


class StressSettings(_Strict):
    detection: Literal["worst_benchmark_return", "largest_benchmark_range"] = "worst_benchmark_return"
    manual_date: str | None = None


class SyntheticSymbol(_Strict):
    beta: float = 1.0
    idio_vol: float = Field(0.3, ge=0)


class SkewSettings(_Strict):
    atm_vol: float = Field(0.4, gt=0)
    slope: float = 0.0
    curvature: float = 0.0


class SyntheticSettings(_Strict):
    seed: int = 7
    start_price: float = Field(100.0, gt=0)
    market_vol: float = Field(0.18, ge=0)
    default_idio_vol: float = Field(0.3, ge=0)
    crash_return: float = Field(-0.06, gt=-1)
    crash_days_before_end: int = Field(45, ge=1)
    intraday_trough_minute: int = Field(75, ge=1)
    symbols: dict[str, SyntheticSymbol] = Field(default_factory=dict)
    options_skew: SkewSettings = SkewSettings()

    def params_for(self, symbol: str) -> SyntheticSymbol:
        return self.symbols.get(symbol, SyntheticSymbol(idio_vol=self.default_idio_vol))


class CredentialSettings(_Strict):
    key_env: str = "ALPACA_API_KEY"
    secret_env: str = "ALPACA_SECRET_KEY"
    dotenv_path: Path | None = Path(".env")


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class TradingSettings(_Strict):
    mode: TradingMode = TradingMode.PAPER
    allow_live: bool = False
    live_env_flag: str = "QUANTLAB_ALLOW_LIVE"
    live_confirm_value: str = Field(..., min_length=8)
    paper_account_prefix: str = "PA"
    runs_dir: Path = Path("runs")


class Settings(_Strict):
    data: DataSettings = DataSettings()
    market: MarketSettings = MarketSettings()
    vol: VolSettings = VolSettings()
    stress: StressSettings = StressSettings()
    synthetic: SyntheticSettings = SyntheticSettings()
    credentials: CredentialSettings = CredentialSettings()
    trading: TradingSettings

    @model_validator(mode="after")
    def _windows_fit_history(self) -> "Settings":
        approx_trading_days = self.data.lookback_days * self.market.trading_days_per_year / self.market.calendar_days_per_year
        needed = max(*self.vol.windows, self.vol.zscore_window, self.vol.beta_window)
        if needed >= approx_trading_days:
            raise ValueError(
                f"data.lookback_days={self.data.lookback_days} gives ~{approx_trading_days:.0f} trading days, "
                f"fewer than the largest vol window ({needed})."
            )
        return self


# --------------------------------------------------------------------------- UI parameter specs


class ParamSpec(_Strict):
    """A numeric parameter with default and bounds; used to generate UI controls and validate input."""

    default: float | int | None
    min: float | int
    max: float | int
    step: float | int
    label: str
    help: str | None = None

    @model_validator(mode="after")
    def _bounds(self) -> "ParamSpec":
        if self.min > self.max:
            raise ValueError(f"{self.label}: min > max")
        if self.default is not None and not (self.min <= self.default <= self.max):
            raise ValueError(f"{self.label}: default {self.default} outside [{self.min}, {self.max}]")
        return self

    def clamp(self, value: float | int) -> float | int:
        return min(max(value, self.min), self.max)

    @property
    def is_int(self) -> bool:
        return all(isinstance(v, int) and not isinstance(v, bool) for v in (self.min, self.max, self.step))


# --------------------------------------------------------------------------- universe.yaml


class SymbolGroup(_Strict):
    name: str
    symbols: tuple[str, ...]
    note: str | None = None


class Universe(_Strict):
    groups: tuple[SymbolGroup, ...]
    default_selection: tuple[str, ...]

    @property
    def all_symbols(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for g in self.groups:
            for s in g.symbols:
                seen.setdefault(s.upper(), None)
        return tuple(seen)

    @model_validator(mode="after")
    def _defaults_in_universe(self) -> "Universe":
        missing = set(s.upper() for s in self.default_selection) - set(self.all_symbols)
        if missing:
            raise ValueError(f"default_selection not in any group: {sorted(missing)}")
        return self


# --------------------------------------------------------------------------- options.yaml


class GreekUnits(_Strict):
    vega_per_vol_points: float = 1
    rho_per_rate_points: float = 1
    theta_per_days: float = 1


class ImpliedVolSettings(_Strict):
    lower: float = Field(1e-4, gt=0)
    upper: float = Field(5.0, gt=0)
    tolerance: float = Field(1e-8, gt=0)


class BinomialSettings(_Strict):
    steps: int = Field(400, ge=10)


class OptionChartSettings(_Strict):
    spot_range_pct: float = Field(0.5, gt=0, lt=1)
    points: int = Field(201, ge=11)


class ChainSettings(_Strict):
    min_days_to_expiry: int = Field(7, ge=0)
    max_days_to_expiry: int = Field(120, gt=0)
    max_abs_log_moneyness: float = Field(0.4, gt=0)


class OptionsConfig(_Strict):
    greek_units: GreekUnits = GreekUnits()
    implied_vol: ImpliedVolSettings = ImpliedVolSettings()
    binomial: BinomialSettings = BinomialSettings()
    chart: OptionChartSettings = OptionChartSettings()
    chain: ChainSettings = ChainSettings()
    params: dict[str, ParamSpec]
    attribution: dict[str, ParamSpec]


# --------------------------------------------------------------------------- impact.yaml


class ImpactSimulation(_Strict):
    n_paths: int = Field(1000, ge=1)
    seed: int = 42
    post_trade_minutes: int = Field(120, ge=0)
    steps_per_minute: int = Field(1, ge=1)
    sweep_max_slices: int = Field(60, ge=2)
    side: Literal["sell", "buy"] = "sell"


class ImpactConfig(_Strict):
    simulation: ImpactSimulation = ImpactSimulation()
    params: dict[str, ParamSpec]
    strategies: tuple[Literal["all_at_once", "twap", "front_loaded"], ...]


# --------------------------------------------------------------------------- portfolio.yaml


class SizingSettings(_Strict):
    method: Literal["inverse_vol", "equal_notional"] = "inverse_vol"
    vol_window: int = Field(60, ge=2)
    gross_notional: float = Field(..., gt=0)
    max_gross_to_equity: float = Field(0.5, gt=0)


class ConstraintSettings(_Strict):
    max_order_notional: float = Field(..., gt=0)
    min_order_notional: float = Field(0, ge=0)
    on_cap_breach: Literal["clip", "error"] = "clip"
    allow_fractional: bool = False
    close_unlisted: bool = False


class OrderSettings(_Strict):
    type: Literal["market", "limit"] = "market"
    limit_offset_bps: float = Field(10, ge=0)
    time_in_force: Literal["day", "gtc"] = "day"


class TargetSpec(_Strict):
    symbol: str
    side: Literal["long", "short"]
    risk_weight: float = Field(..., gt=0)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()


class OptionLegSpec(_Strict):
    underlying: str
    right: Literal["call", "put"]
    side: Literal["buy", "sell"]
    contracts: int = Field(..., ge=1)
    target_dte: int = Field(..., ge=1)
    moneyness: float = Field(1.0, gt=0)  # strike / spot


class OptionLegs(_Strict):
    enabled: bool = False
    min_options_level: int = Field(2, ge=1)  # Alpaca: level 2 = buy calls/puts
    limit_at: Literal["mid", "ask"] = "mid"
    legs: tuple[OptionLegSpec, ...] = ()


class PortfolioConfig(_Strict):
    sizing: SizingSettings
    constraints: ConstraintSettings
    order: OrderSettings = OrderSettings()
    targets: tuple[TargetSpec, ...]
    option_legs: OptionLegs = OptionLegs()

    @model_validator(mode="after")
    def _unique_targets(self) -> "PortfolioConfig":
        syms = [t.symbol for t in self.targets]
        if len(syms) != len(set(syms)):
            raise ValueError("portfolio targets contain duplicate symbols")
        if self.constraints.min_order_notional > self.constraints.max_order_notional:
            raise ValueError("min_order_notional > max_order_notional")
        return self


# --------------------------------------------------------------------------- loading


CONFIG_FILES: dict[str, type[BaseModel]] = {
    "settings": Settings,
    "universe": Universe,
    "options": OptionsConfig,
    "impact": ImpactConfig,
    "portfolio": PortfolioConfig,
}


def find_config_dir(start: Path | None = None) -> Path:
    env = os.environ.get(CONFIG_DIR_ENV)
    if env:
        path = Path(env).expanduser().resolve()
        if not (path / SETTINGS_FILE).is_file():
            raise FileNotFoundError(f"${CONFIG_DIR_ENV}={env} does not contain {SETTINGS_FILE}")
        return path
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "config" / SETTINGS_FILE).is_file():
            return candidate / "config"
    raise FileNotFoundError(
        f"No config/{SETTINGS_FILE} found above {here}. Set ${CONFIG_DIR_ENV} or run from the project root."
    )


def _set_path(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = [k for k in dotted.split(".") if k]
    if not keys:
        raise ValueError(f"empty override key: {dotted!r}")
    node = tree
    for k in keys[:-1]:
        nxt = node.setdefault(k, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"override {dotted!r} descends into non-mapping key {k!r}")
        node = nxt
    node[keys[-1]] = value


def parse_override(expr: str) -> tuple[str, Any]:
    """Parse ``a.b.c=value``; the value is YAML-decoded so numbers/bools/lists keep their types."""
    if "=" not in expr:
        raise ValueError(f"override must look like key.path=value, got {expr!r}")
    key, raw = expr.split("=", 1)
    return key.strip(), yaml.safe_load(raw)


def _env_overrides(file_key: str, environ: Mapping[str, str]) -> dict[str, Any]:
    """``QUANTLAB__DATA__FEED=sip`` targets settings.yaml; ``QUANTLAB__IMPACT__SIMULATION__SEED=1``
    targets impact.yaml (first segment equal to a config file name selects that file)."""
    out: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in name[len(ENV_PREFIX):].split("__") if p]
        if not parts:
            continue
        if parts[0] in CONFIG_FILES and parts[0] != "settings" and len(parts) > 1:
            target, path = parts[0], parts[1:]
        else:
            target, path = "settings", parts
        if target == file_key:
            _set_path(out, ".".join(path), yaml.safe_load(raw))
    return out


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(
    name: str,
    config_dir: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Load and validate one config file by name (see ``CONFIG_FILES``)."""
    if name not in CONFIG_FILES:
        raise KeyError(f"unknown config {name!r}; expected one of {sorted(CONFIG_FILES)}")
    directory = config_dir or find_config_dir()
    path = directory / f"{name}.yaml"
    raw: dict[str, Any] = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{path} must contain a mapping at the top level")
    raw = _deep_merge(raw, _env_overrides(name, os.environ if environ is None else environ))
    for dotted, value in (overrides or {}).items():
        _set_path(raw, dotted, value)
    return CONFIG_FILES[name].model_validate(raw)


def load_settings(config_dir: Path | None = None, overrides: Mapping[str, Any] | None = None) -> Settings:
    return load_config("settings", config_dir, overrides)


def project_root(config_dir: Path | None = None) -> Path:
    return (config_dir or find_config_dir()).parent


def resolve_path(path: Path, config_dir: Path | None = None) -> Path:
    """Relative paths in config are relative to the project root, not the CWD."""
    return path if path.is_absolute() else project_root(config_dir) / path


# --------------------------------------------------------------------------- secrets


@dataclass(frozen=True)
class Credentials:
    api_key: str
    secret_key: str

    def __repr__(self) -> str:  # never leak secrets into logs or tracebacks
        return f"Credentials(api_key='{self.api_key[:4]}…', secret_key='***')"


def load_credentials(settings: Settings, config_dir: Path | None = None) -> Credentials | None:
    """Read credentials from the environment (optionally seeded from a .env file). None if absent."""
    if settings.credentials.dotenv_path is not None:
        from dotenv import load_dotenv

        dotenv = resolve_path(settings.credentials.dotenv_path, config_dir)
        if dotenv.is_file():
            load_dotenv(dotenv, override=False)
    key = os.environ.get(settings.credentials.key_env, "").strip()
    secret = os.environ.get(settings.credentials.secret_env, "").strip()
    if not key or not secret:
        return None
    return Credentials(key, secret)
