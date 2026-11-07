"""Broker interface and the trading-mode guard."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, Protocol, runtime_checkable

from ..portfolio import OrderIntent
from ..settings import OrderSettings, TradingMode, TradingSettings


class TradingModeError(PermissionError):
    pass


@dataclass(frozen=True)
class AccountInfo:
    account_number: str
    equity: float
    buying_power: float
    shorting_enabled: bool
    options_level: int | None


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    qty: float           # signed
    market_value: float  # signed
    avg_entry_price: float
    current_price: float
    asset_class: str


@dataclass(frozen=True)
class OptionOrderIntent:
    contract: str
    side: Literal["buy", "sell"]
    contracts: int
    limit_price: float
    reason: str


@runtime_checkable
class Broker(Protocol):
    @property
    def mode(self) -> TradingMode: ...

    def account(self) -> AccountInfo: ...

    def positions(self) -> list[PositionInfo]: ...

    def submit(self, order: OrderIntent, cfg: OrderSettings) -> str: ...

    def submit_option(self, order: OptionOrderIntent, cfg: OrderSettings) -> str: ...


def authorize_mode(trading: TradingSettings, cli_live_ack: bool, environ: Mapping[str, str] | None = None) -> TradingMode:
    """Resolve the trading mode. Live needs three independent opt-ins; paper needs none."""
    env = os.environ if environ is None else environ
    if trading.mode is TradingMode.PAPER:
        return TradingMode.PAPER
    problems = []
    if not trading.allow_live:
        problems.append("trading.allow_live is false")
    if env.get(trading.live_env_flag) != trading.live_confirm_value:
        problems.append(f"${trading.live_env_flag} is not set to the confirmation value")
    if not cli_live_ack:
        problems.append("--i-understand-live was not passed")
    if problems:
        raise TradingModeError("live trading refused: " + "; ".join(problems))
    return TradingMode.LIVE


def verify_account_mode(account: AccountInfo, mode: TradingMode, trading: TradingSettings) -> None:
    """Defence in depth: the connected account must match the requested mode."""
    is_paper_acct = account.account_number.startswith(trading.paper_account_prefix)
    if mode is TradingMode.PAPER and not is_paper_acct:
        raise TradingModeError(
            f"mode is paper but account {account.account_number[:4]}… does not look like a paper account "
            f"(expected prefix {trading.paper_account_prefix!r})"
        )
    if mode is TradingMode.LIVE and is_paper_acct:
        raise TradingModeError("mode is live but connected account is a paper account")
