"""Abilities: what a character is good and bad at, fixed when the world is built.

Five general attributes, free of any theme (Rule 7), at levels 1–5 where 3 is an ordinary adult;
up to four named skills at levels 1–3. They are identity like a name or an age: the build sets
them once and nothing changes them at runtime.

Numbers are code-layer, like a check's. A check adds them up and the API and renderer show them;
a prompt only ever gets the level words (``attribute_word`` / ``skill_word``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Attribute(str, Enum):
    """Declared in the order every prompt and the renderer list them."""

    PHYSIQUE  = "physique"
    AGILITY   = "agility"
    INTELLECT = "intellect"
    ELOQUENCE = "eloquence"
    NERVE     = "nerve"

    @property
    def label(self) -> str:
        return _LABELS[self][0]

    @property
    def scope(self) -> str:
        """What the attribute covers, for a prompt's legend."""
        return _LABELS[self][1]


_LABELS: dict[Attribute, tuple[str, str]] = {
    Attribute.PHYSIQUE:  ("体魄", "力量、耐力、抗打击"),
    Attribute.AGILITY:   ("身手", "敏捷、潜行、手上功夫"),
    Attribute.INTELLECT: ("心智", "学识、谋算、洞察"),
    Attribute.ELOQUENCE: ("口才", "说服、欺瞒、交涉"),
    Attribute.NERVE:     ("胆识", "意志、镇定、威慑"),
}

ATTRIBUTE_MIN = 1
ATTRIBUTE_MAX = 5
# An ordinary adult; an attribute's modifier is its distance from here.
ATTRIBUTE_BASELINE = 3
SKILL_MIN = 1
SKILL_MAX = 3
MAX_SKILLS = 4

_ATTRIBUTE_WORDS: dict[int, str] = {1: "孱弱", 2: "平平", 3: "寻常", 4: "出众", 5: "超群"}
_SKILL_WORDS: dict[int, str] = {1: "略通", 2: "精熟", 3: "宗师"}


def _clamp(value: object, low: int, high: int, default: int) -> int:
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def attribute_word(level: int) -> str:
    return _ATTRIBUTE_WORDS[_clamp(level, ATTRIBUTE_MIN, ATTRIBUTE_MAX, ATTRIBUTE_BASELINE)]


def skill_word(level: int) -> str:
    return _SKILL_WORDS[_clamp(level, SKILL_MIN, SKILL_MAX, SKILL_MIN)]


@dataclass(frozen=True)
class Skill:
    name: str
    level: int


@dataclass(frozen=True)
class Abilities:
    """One character's attributes and skills. Missing attributes read as the baseline, so a
    character with none set is an ordinary adult with no skills: modifier 0 on every check."""

    attributes: Mapping[Attribute, int] = field(default_factory=dict)
    skills: tuple[Skill, ...] = ()

    def __post_init__(self) -> None:
        levels = {
            attribute: _clamp(self.attributes.get(attribute), ATTRIBUTE_MIN, ATTRIBUTE_MAX, ATTRIBUTE_BASELINE)
            for attribute in Attribute
        }
        object.__setattr__(self, "attributes", levels)
        object.__setattr__(self, "skills", tuple(self.skills)[:MAX_SKILLS])

    def level(self, attribute: Attribute) -> int:
        return self.attributes[attribute]

    def persona_line(self) -> str:
        """``所长：剑术精熟、身手出众；所短：体魄孱弱`` in words only; empty for an ordinary person.

        Every skill is a strength; an attribute is one above the baseline and a weakness below it.
        """
        strengths = [f"{s.name}{skill_word(s.level)}" for s in self.skills]
        strengths += [
            f"{a.label}{attribute_word(level)}"
            for a, level in self.attributes.items() if level > ATTRIBUTE_BASELINE
        ]
        weaknesses = [
            f"{a.label}{attribute_word(level)}"
            for a, level in self.attributes.items() if level < ATTRIBUTE_BASELINE
        ]
        parts = []
        if strengths:
            parts.append("所长：" + "、".join(strengths))
        if weaknesses:
            parts.append("所短：" + "、".join(weaknesses))
        return "；".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attributes": {a.value: level for a, level in self.attributes.items()},
            "skills": [{"name": s.name, "level": s.level} for s in self.skills],
        }

    @classmethod
    def from_dict(cls, payload: object) -> "Abilities":
        """Never raises: anything unreadable reads as the baseline, the same as a world built
        before characters had abilities."""
        if not isinstance(payload, Mapping):
            return cls()
        raw_attributes = payload.get("attributes")
        attributes: dict[Attribute, int] = {}
        if isinstance(raw_attributes, Mapping):
            for attribute in Attribute:
                if attribute.value in raw_attributes:
                    attributes[attribute] = _clamp(
                        raw_attributes[attribute.value], ATTRIBUTE_MIN, ATTRIBUTE_MAX, ATTRIBUTE_BASELINE,
                    )
        skills = []
        raw_skills = payload.get("skills")
        for raw in raw_skills if isinstance(raw_skills, list) else []:
            if isinstance(raw, Mapping) and str(raw.get("name") or "").strip():
                skills.append(Skill(
                    name=str(raw["name"]).strip(),
                    level=_clamp(raw.get("level"), SKILL_MIN, SKILL_MAX, SKILL_MIN),
                ))
        return cls(attributes=attributes, skills=tuple(skills))
