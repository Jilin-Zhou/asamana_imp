"""Checks — the dice under an action whose outcome is genuinely in doubt.

A check is settled in three parts, split by who may decide what (CLAUDE.md Design Principle 5):

1. A functional judge reads the scene and names the act's ``Difficulty``, which of the actor's
   abilities it draws on (and the opponent's, against a person), and the ``Edge`` the
   circumstances give beyond them. Subjective, so LLM (``engine.check.CheckAssessor``).
2. Code adds the abilities' levels and the edge into one modifier, rolls 3d6 + modifier against
   the difficulty and reads off a ``CheckTier``. Arithmetic over structured signals, so rule.
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
from core.interfaces.ability import ATTRIBUTE_BASELINE, ATTRIBUTE_MAX, ATTRIBUTE_MIN, SKILL_MAX, Attribute


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
    """What the circumstances give the actor beyond the abilities a check already counts: wounds,
    being bound, numbers, weapons, preparation, wariness, footing. Narrower than an ability's
    range, so who someone is outweighs where they stand."""

    MUCH_WORSE  = "much_worse"
    WORSE       = "worse"
    EVEN        = "even"
    BETTER      = "better"
    MUCH_BETTER = "much_better"

    @property
    def modifier(self) -> int:
        return _MODIFIER[self]


_MODIFIER: dict[Edge, int] = {
    Edge.MUCH_WORSE:  -2,
    Edge.WORSE:       -1,
    Edge.EVEN:         0,
    Edge.BETTER:       1,
    Edge.MUCH_BETTER:  2,
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
# critical success down to critical failure, and each difficulty step (3 points of modifier) shifts
# that by about one band; tests/unit/test_check_resolution.py enumerates all 216 rolls to hold it.
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
class AbilityBasis:
    """The abilities one roll drew on, as their levels at the time (``core.interfaces.ability``).

    No attribute: the assessor named none that applies. No opposing attribute: nobody resists.
    """

    attribute: Attribute | None = None
    attribute_level: int = ATTRIBUTE_BASELINE
    skill: str = ""
    skill_level: int = 0
    opposing_attribute: Attribute | None = None
    opposing_level: int = ATTRIBUTE_BASELINE

    @property
    def modifier(self) -> int:
        """The actor's attribute and skill, less the opponent's attribute, each from the baseline."""
        own = self.attribute_level - ATTRIBUTE_BASELINE if self.attribute is not None else 0
        against = self.opposing_level - ATTRIBUTE_BASELINE if self.opposing_attribute is not None else 0
        return own + self.skill_level - against

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute.value if self.attribute is not None else None,
            "attribute_level": self.attribute_level,
            "skill": self.skill,
            "skill_level": self.skill_level,
            "opposing_attribute": (
                self.opposing_attribute.value if self.opposing_attribute is not None else None
            ),
            "opposing_level": self.opposing_level,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "AbilityBasis":
        """Never raises; anything unreadable is no ability at all."""
        if not isinstance(payload, Mapping):
            return cls()

        def attribute(key: str) -> Attribute | None:
            value = payload.get(key)
            return Attribute(value) if value in {a.value for a in Attribute} else None

        def level(key: str, low: int, high: int, default: int) -> int:
            try:
                return max(low, min(high, int(payload.get(key, default))))
            except (TypeError, ValueError):
                return default

        skill = str(payload.get("skill") or "")
        return cls(
            attribute=attribute("attribute"),
            attribute_level=level("attribute_level", ATTRIBUTE_MIN, ATTRIBUTE_MAX, ATTRIBUTE_BASELINE),
            skill=skill,
            skill_level=level("skill_level", 0, SKILL_MAX, 0) if skill else 0,
            opposing_attribute=attribute("opposing_attribute"),
            opposing_level=level("opposing_level", ATTRIBUTE_MIN, ATTRIBUTE_MAX, ATTRIBUTE_BASELINE),
        )


@dataclass(frozen=True)
class CheckRecord:
    """One roll as it happened. Code layer only: the API and the renderer read it; no prompt,
    memory or dialogue ever does (the narration gets the tier as words).

    The tier is derived from the dice, never stored beside them, so the two cannot disagree.
    """

    difficulty: Difficulty
    edge: Edge
    # Luck alone decides (dice, lots, a lottery ticket): neither abilities nor the edge count.
    pure_chance: bool
    dice: tuple[int, ...]
    basis: AbilityBasis = AbilityBasis()
    # The assessor's rationale. Third person and god-view, like ``ActionResult.outcome``: for the
    # observer's display only.
    reason: str = ""

    @property
    def dc(self) -> int:
        return self.difficulty.dc

    @property
    def modifier(self) -> int:
        return 0 if self.pure_chance else self.basis.modifier + self.edge.modifier

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
            "basis": self.basis.to_dict(),
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
            basis=AbilityBasis.from_dict(payload.get("basis")),
            reason=str(payload.get("reason") or ""),
        )
