"""Checks — the dice under an action whose outcome is genuinely in doubt.

A check is settled in three parts, split by who may decide what (CLAUDE.md Design Principle 5):

1. A functional judge reads the scene and names the act's ``Difficulty`` and the actor's ``Edge``
   over it. Subjective, so LLM (``engine.check.CheckAssessor``).
2. Code rolls 3d6 + the edge's modifier against the difficulty and reads off a ``CheckTier``.
   An objective threshold over structured signals, so rule.
3. The executor's narration writes the tier it is handed; it never decides success itself.

The same rule holds for every agent: narrative tier never switches it.

Numbers are code-layer. Dice, DC and modifier reach the API and the renderer through
``CheckRecord``; a prompt, a memory or a line of dialogue only ever gets the tier as words.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from core.coerce import coerce_enum


class Difficulty(str, Enum):
    """How hard the act is for an ordinary person under the present conditions.

    Judged without regard to who is acting: who he is goes into ``Edge``. For a game of pure
    chance it is the odds of winning instead.
    """

    EASY      = "easy"
    NORMAL    = "normal"
    HARD      = "hard"
    VERY_HARD = "very_hard"

    @property
    def dc(self) -> int:
        return _DC[self]


_DC: dict[Difficulty, int] = {
    Difficulty.EASY:      8,
    Difficulty.NORMAL:    11,
    Difficulty.HARD:      14,
    Difficulty.VERY_HARD: 17,
}


class Edge(str, Enum):
    """The actor's standing on this act against an ordinary person; against a person, the balance
    between the two sides (strength, wounds, numbers, weapons, wariness)."""

    MUCH_WORSE  = "much_worse"
    WORSE       = "worse"
    EVEN        = "even"
    BETTER      = "better"
    MUCH_BETTER = "much_better"

    @property
    def modifier(self) -> int:
        return _MODIFIER[self]


_MODIFIER: dict[Edge, int] = {
    Edge.MUCH_WORSE:  -4,
    Edge.WORSE:       -2,
    Edge.EVEN:         0,
    Edge.BETTER:       2,
    Edge.MUCH_BETTER:  4,
}


class Certainty(str, Enum):
    """Whether the act needs a roll at all.

    Rolling only when failure is interesting keeps the dice meaningful: helping a willing man to
    his feet never misses, and a wall is not pushed over by hand however the dice fall.
    """

    SURE_SUCCESS = "sure_success"
    UNCERTAIN    = "uncertain"
    SURE_FAILURE = "sure_failure"


class CheckTier(str, Enum):
    """The graded result of an act. ``COSTLY_SUCCESS`` is the dramatic band: it lands, and the
    narration must say what it cost — the hook for whatever happens next."""

    CRITICAL_SUCCESS = "critical_success"
    SUCCESS          = "success"
    COSTLY_SUCCESS   = "costly_success"
    FAILURE          = "failure"
    CRITICAL_FAILURE = "critical_failure"

    @property
    def succeeded(self) -> bool:
        return self in _SUCCEEDING

    @classmethod
    def for_margin(cls, margin: int) -> "CheckTier":
        """Margin = 3d6 + modifier − DC. The only place a margin becomes a tier."""
        for floor, tier in _TIER_FLOORS:
            if margin >= floor:
                return tier
        return cls.CRITICAL_FAILURE


_SUCCEEDING = frozenset({
    CheckTier.CRITICAL_SUCCESS, CheckTier.SUCCESS, CheckTier.COSTLY_SUCCESS,
})

# (lowest margin, tier), read top-down. With 3d6 an even match at NORMAL lands 5/33/25/33/5 % from
# critical success down to critical failure, and each edge or difficulty step shifts that by about
# one band; tests/unit/test_check_resolution.py enumerates all 216 rolls to hold it there.
# 3d6 rather than d20: on a flat die a capable actor fails as often as a middling one.
_TIER_FLOORS: tuple[tuple[int, CheckTier], ...] = (
    (5,  CheckTier.CRITICAL_SUCCESS),
    (1,  CheckTier.SUCCESS),
    (-1, CheckTier.COSTLY_SUCCESS),
    (-5, CheckTier.FAILURE),
)

DICE_COUNT: int = 3
DIE_SIDES: int = 6


@dataclass(frozen=True)
class CheckRecord:
    """One roll as it happened. Code layer only: the API and the renderer read it; no prompt,
    memory or dialogue ever does (the narration gets the tier as words).

    The tier is derived from the dice, never stored beside them, so the two cannot disagree.
    """

    difficulty: Difficulty
    edge: Edge
    # Luck alone decides (dice, lots, a lottery ticket): the edge doesn't count.
    pure_chance: bool
    dice: tuple[int, ...]
    # The assessor's rationale. Third person and god-view, like ``ActionResult.outcome``: for the
    # observer's display only.
    reason: str = ""

    @property
    def dc(self) -> int:
        return self.difficulty.dc

    @property
    def modifier(self) -> int:
        return 0 if self.pure_chance else self.edge.modifier

    @property
    def total(self) -> int:
        return sum(self.dice) + self.modifier

    @property
    def tier(self) -> CheckTier:
        return CheckTier.for_margin(self.total - self.dc)

    def to_dict(self) -> dict[str, Any]:
        """JSON-native, with the derived numbers spelled out so readers don't recompute them."""
        return {
            "difficulty": self.difficulty.value,
            "edge": self.edge.value,
            "pure_chance": self.pure_chance,
            "dice": list(self.dice),
            "dc": self.dc,
            "modifier": self.modifier,
            "total": self.total,
            "tier": self.tier.value,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "CheckRecord | None":
        """Rebuild from ``to_dict`` output; ``None`` for anything that isn't one.

        Never raises: it runs on snapshot read, where one bad record must not fail the step.
        Derived fields in the payload are ignored and recomputed from the dice.
        """
        if not isinstance(payload, Mapping):
            return None
        raw_dice = payload.get("dice")
        if not isinstance(raw_dice, (list, tuple)) or len(raw_dice) != DICE_COUNT:
            return None
        try:
            dice = tuple(int(d) for d in raw_dice)
        except (TypeError, ValueError):
            return None
        if any(not 1 <= d <= DIE_SIDES for d in dice):
            return None
        return cls(
            difficulty=coerce_enum(Difficulty, payload.get("difficulty"), default=Difficulty.NORMAL),
            edge=coerce_enum(Edge, payload.get("edge"), default=Edge.EVEN),
            pure_chance=bool(payload.get("pure_chance", False)),
            dice=dice,
            reason=str(payload.get("reason") or ""),
        )
