"""In-memory broker for tests and offline dry runs."""

from __future__ import annotations

import itertools

from ..portfolio import OrderIntent
from ..settings import OrderSettings, TradingMode
from .base import AccountInfo, OptionOrderIntent, PositionInfo


class FakeBroker:
    def __init__(self, equity: float, positions: dict[str, float] | None = None, prices: dict[str, float] | None = None, account_number: str = "PA-FAKE"):
        self._equity = equity
        self._pos = dict(positions or {})
        self._px = dict(prices or {})
        self._acct = account_number
        self._ids = itertools.count(1)
        self.submitted: list[object] = []

    @property
    def mode(self) -> TradingMode:
        return TradingMode.PAPER

    def account(self) -> AccountInfo:
        return AccountInfo(self._acct, self._equity, self._equity * 2, True, 2)

    def positions(self) -> list[PositionInfo]:
        return [
            PositionInfo(s, q, q * self._px.get(s, 0.0), self._px.get(s, 0.0), self._px.get(s, 0.0), "us_equity")
            for s, q in self._pos.items()
            if q
        ]

    def submit(self, order: OrderIntent, cfg: OrderSettings) -> str:
        self.submitted.append(order)
        self._pos[order.symbol] = self._pos.get(order.symbol, 0.0) + (order.qty if order.side == "buy" else -order.qty)
        return f"fake-{next(self._ids)}"

    def submit_option(self, order: OptionOrderIntent, cfg: OrderSettings) -> str:
        self.submitted.append(order)
        return f"fake-{next(self._ids)}"
