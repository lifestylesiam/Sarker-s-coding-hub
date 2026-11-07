"""Provider factory: the only place that decides which data source to use."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ..settings import Settings, load_credentials, resolve_path
from .base import MarketDataProvider, ProviderUnavailable
from .cache import CachedProvider
from .synthetic import SyntheticProvider

log = logging.getLogger(__name__)

__all__ = ["MarketDataProvider", "ProviderUnavailable", "ResilientProvider", "build_provider"]


class ResilientProvider:
    """Wraps the configured provider; on ``ProviderUnavailable`` switches once to the fallback.

    ``fallback_reason`` is set whenever data did not come from the primary source, so UIs
    can flag it loudly instead of silently showing synthetic numbers.
    """

    def __init__(self, primary_factory: Callable[[], MarketDataProvider], fallback_factory: Callable[[], MarketDataProvider] | None):
        self._fallback_factory = fallback_factory
        self.fallback_reason: str | None = None
        try:
            self._active = primary_factory()
        except ProviderUnavailable as e:
            self._switch(e)

    def _switch(self, err: Exception) -> None:
        if self._fallback_factory is None or self.fallback_reason is not None:
            raise err
        log.warning("primary data provider unavailable (%s); using fallback", err)
        self.fallback_reason = str(err)
        self._active = self._fallback_factory()

    @property
    def name(self) -> str:
        return self._active.name

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason is not None

    def _call(self, method: str, *args, **kwargs):
        try:
            return getattr(self._active, method)(*args, **kwargs)
        except ProviderUnavailable as e:
            self._switch(e)
            return getattr(self._active, method)(*args, **kwargs)

    def daily_bars(self, *a, **k):
        return self._call("daily_bars", *a, **k)

    def intraday_bars(self, *a, **k):
        return self._call("intraday_bars", *a, **k)

    def option_chain(self, *a, **k):
        return self._call("option_chain", *a, **k)


def _make(kind: str, settings: Settings, config_dir: Path | None, refresh: bool) -> MarketDataProvider:
    if kind == "synthetic":
        return SyntheticProvider(settings.synthetic, settings.market)  # deterministic; no cache needed
    if kind == "alpaca":
        creds = load_credentials(settings, config_dir)
        if creds is None:
            raise ProviderUnavailable(
                f"Alpaca credentials not found in ${settings.credentials.key_env} / ${settings.credentials.secret_env}"
            )
        from .alpaca import AlpacaProvider

        return CachedProvider(
            AlpacaProvider(creds, settings.data, settings.market),
            resolve_path(settings.data.cache_dir, config_dir),
            settings.data.cache_max_age_hours,
            settings.market.timezone,
            refresh=refresh,
        )
    raise ValueError(f"unknown provider {kind!r}")


def build_provider(settings: Settings, config_dir: Path | None = None, refresh: bool = False) -> ResilientProvider:
    primary = settings.data.provider
    fb = settings.data.fallback_provider
    fallback = (lambda: _make(fb, settings, config_dir, refresh)) if fb and fb != primary else None
    return ResilientProvider(lambda: _make(primary, settings, config_dir, refresh), fallback)
