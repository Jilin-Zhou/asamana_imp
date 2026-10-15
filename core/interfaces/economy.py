"""A world's money: what it is called and what things cost there, fixed at build.

Money itself is a divisible thing in the world (``WorldEntityType.MONEY`` with a ``quantity``);
this says how an amount reads in narrative text, and holds the price references every prompt that
weighs a deal shares. Both are world constants, so they belong in a prompt's system part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

MAX_PRICES = 6


@dataclass(frozen=True)
class PriceReference:
    item: str
    price: int


@dataclass(frozen=True)
class Economy:
    """Empty ``currency``: the world has no money (built before money existed, or the theme
    gave none); nobody is handed cash and no prompt mentions prices."""

    currency: str = ""   # what the money is called: 银子 / 铜钱 / 现金
    unit: str = ""       # what it is counted in: 两 / 文 / 元
    prices: tuple[PriceReference, ...] = ()

    @property
    def has_money(self) -> bool:
        return bool(self.currency)

    def amount_text(self, quantity: int) -> str:
        """``37两银子``: how an amount reads in a prompt, a memory or a line of narration."""
        return f"{quantity}{self.unit}{self.currency or '钱'}"

    def price_reference(self) -> str:
        """``一斗米约30文；一匹马约20两`` for a prompt; empty when there are none."""
        return "；".join(f"{p.item}约{p.price}{self.unit or self.currency}" for p in self.prices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "unit": self.unit,
            "prices": [{"item": p.item, "price": p.price} for p in self.prices],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "Economy":
        """Never raises; anything unreadable is a world without money."""
        if not isinstance(payload, Mapping):
            return cls()
        prices = []
        raw_prices = payload.get("prices")
        for raw in raw_prices if isinstance(raw_prices, list) else []:
            if not isinstance(raw, Mapping):
                continue
            item = str(raw.get("item") or "").strip()
            try:
                price = int(raw.get("price"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if item and price > 0:
                prices.append(PriceReference(item, price))
        return cls(
            currency=str(payload.get("currency") or "").strip(),
            unit=str(payload.get("unit") or "").strip(),
            prices=tuple(prices[:MAX_PRICES]),
        )
