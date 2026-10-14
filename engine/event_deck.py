"""The event deck: happenings this world is prone to, written at build and drawn by rule.

The deck is the world's data (Rule 7: theme content stays out of generic code). Whether to draw
at a check and which card is a seeded weighted draw over the cards whose conditions hold, so a
world's draws are reproducible; who the drawn happening lands on, where and in what words is
still the event editor's LLM call. A drawn card stands in for the editor's "should anything
happen now" gate: the dice already said yes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from core.coerce import coerce_enum
from core.interfaces.severity import Severity
from engine.weather import Sky

_SKIES = {s.value for s in Sky}

MIN_CARDS = 12
MAX_CARDS = 20
WEIGHT_MIN = 1
WEIGHT_MAX = 3

# Chance a check draws at all when some card's conditions hold. Not every check: the editor's
# own judgment still decides the other half.
DRAW_CHANCE = 0.5


class Scope(str, Enum):
    LOCAL  = "local"   # happens at one place
    GLOBAL = "global"  # the whole world hears of it


class Daypart(str, Enum):
    ANY   = "any"
    DAY   = "day"
    NIGHT = "night"

    def holds_at(self, hour: int) -> bool:
        if self is Daypart.ANY:
            return True
        daytime = 6 <= hour < 18
        return daytime if self is Daypart.DAY else not daytime


@dataclass(frozen=True)
class EventCard:
    title: str
    synopsis: str
    scope: Scope = Scope.LOCAL
    severity: Severity = Severity.MEDIUM
    weight: int = 1
    # Conditions, each "any" when absent: the sky it needs, the time of day it needs.
    sky: Sky | None = None
    daypart: Daypart = Daypart.ANY

    def holds(self, *, sky: Sky | None, hour: int) -> bool:
        return (self.sky is None or self.sky is sky) and self.daypart.holds_at(hour)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "synopsis": self.synopsis,
            "scope": self.scope.value,
            "severity": self.severity.value,
            "weight": self.weight,
            "sky": self.sky.value if self.sky is not None else "any",
            "daypart": self.daypart.value,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "EventCard | None":
        """Never raises; a card without a title or synopsis is no card."""
        if not isinstance(payload, Mapping):
            return None
        title = str(payload.get("title") or "").strip()
        synopsis = str(payload.get("synopsis") or "").strip()
        if not title or not synopsis:
            return None
        try:
            weight = max(WEIGHT_MIN, min(WEIGHT_MAX, int(payload.get("weight") or WEIGHT_MIN)))
        except (TypeError, ValueError):
            weight = WEIGHT_MIN
        raw_sky = payload.get("sky")
        return cls(
            title=title,
            synopsis=synopsis,
            scope=coerce_enum(Scope, payload.get("scope"), default=Scope.LOCAL),
            severity=coerce_enum(Severity, payload.get("severity"), default=Severity.MEDIUM),
            weight=weight,
            sky=Sky(raw_sky) if raw_sky in _SKIES else None,
            daypart=coerce_enum(Daypart, payload.get("daypart"), default=Daypart.ANY),
        )


def deck_from_dicts(payloads: Iterable[object]) -> tuple[EventCard, ...]:
    """Readable cards in order, the first of any repeated title kept (the cooldown keys on
    titles), at most ``MAX_CARDS``."""
    cards: list[EventCard] = []
    for raw in payloads:
        card = EventCard.from_dict(raw)
        if card is not None and all(c.title != card.title for c in cards):
            cards.append(card)
    return tuple(cards[:MAX_CARDS])


def draw(
    deck: Sequence[EventCard],
    *,
    seed: str,
    step: int,
    sky: Sky | None,
    hour: int,
    cooling: set[str],
) -> EventCard | None:
    """At most one card for this check, or None. Seeded by world and step, so a restored world
    draws exactly as the original would have."""
    candidates = [c for c in deck if c.title not in cooling and c.holds(sky=sky, hour=hour)]
    if not candidates:
        return None
    rng = random.Random(f"{seed}:deck:{step}")
    if rng.random() >= DRAW_CHANCE:
        return None
    roll = rng.random() * sum(c.weight for c in candidates)
    for card in candidates:
        roll -= card.weight
        if roll < 0:
            return card
    return candidates[-1]
