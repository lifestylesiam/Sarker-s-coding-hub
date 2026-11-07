"""Alpaca trading adapter. Mode must already be authorized via ``authorize_mode``."""

from __future__ import annotations

from ..portfolio import OrderIntent
from ..settings import Credentials, OrderSettings, TradingMode
from .base import AccountInfo, OptionOrderIntent, PositionInfo


class AlpacaBroker:
    def __init__(self, creds: Credentials, mode: TradingMode):
        from alpaca.trading.client import TradingClient

        self._mode = mode
        self._client = TradingClient(creds.api_key, creds.secret_key, paper=mode is TradingMode.PAPER)

    @property
    def mode(self) -> TradingMode:
        return self._mode

    def account(self) -> AccountInfo:
        a = self._client.get_account()
        level = getattr(a, "options_trading_level", None)
        return AccountInfo(
            account_number=str(a.account_number),
            equity=float(a.equity),
            buying_power=float(a.buying_power),
            shorting_enabled=bool(a.shorting_enabled),
            options_level=int(level) if level is not None else None,
        )

    def positions(self) -> list[PositionInfo]:
        out = []
        for p in self._client.get_all_positions():
            qty = float(p.qty)
            out.append(
                PositionInfo(
                    symbol=p.symbol,
                    qty=qty,
                    market_value=float(p.market_value),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price) if p.current_price is not None else float("nan"),
                    asset_class=str(getattr(p.asset_class, "value", p.asset_class)),
                )
            )
        return out

    def _tif(self, cfg: OrderSettings):
        from alpaca.trading.enums import TimeInForce

        return TimeInForce(cfg.time_in_force)

    def submit(self, order: OrderIntent, cfg: OrderSettings) -> str:
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        side = OrderSide(order.side)
        if cfg.type == "market":
            req = MarketOrderRequest(symbol=order.symbol, qty=order.qty, side=side, time_in_force=self._tif(cfg))
        else:
            sign = 1 if order.side == "buy" else -1
            limit = round(order.ref_price * (1 + sign * cfg.limit_offset_bps * 1e-4), 2)
            req = LimitOrderRequest(symbol=order.symbol, qty=order.qty, side=side, time_in_force=self._tif(cfg), limit_price=limit)
        return str(self._client.submit_order(req).id)

    def submit_option(self, order: OptionOrderIntent, cfg: OrderSettings) -> str:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest

        req = LimitOrderRequest(
            symbol=order.contract,
            qty=order.contracts,
            side=OrderSide(order.side),
            time_in_force=TimeInForce.DAY,  # Alpaca options orders must be DAY
            limit_price=round(order.limit_price, 2),
        )
        return str(self._client.submit_order(req).id)
