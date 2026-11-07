from __future__ import annotations

from pathlib import Path

from ..settings import Settings, load_credentials
from .base import (
    AccountInfo,
    Broker,
    OptionOrderIntent,
    PositionInfo,
    TradingModeError,
    authorize_mode,
    verify_account_mode,
)

__all__ = [
    "AccountInfo",
    "Broker",
    "OptionOrderIntent",
    "PositionInfo",
    "TradingModeError",
    "authorize_mode",
    "verify_account_mode",
    "connect_broker",
]


def connect_broker(settings: Settings, cli_live_ack: bool = False, config_dir: Path | None = None) -> Broker:
    """Authorize the mode, connect, and verify the account matches the mode."""
    mode = authorize_mode(settings.trading, cli_live_ack)
    creds = load_credentials(settings, config_dir)
    if creds is None:
        raise RuntimeError(
            f"broker credentials missing: set ${settings.credentials.key_env} and ${settings.credentials.secret_env} (e.g. in .env)"
        )
    from .alpaca import AlpacaBroker

    broker = AlpacaBroker(creds, mode)
    verify_account_mode(broker.account(), mode, settings.trading)
    return broker
