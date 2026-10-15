"""Weather: one sky over the whole world, moved on by a seeded rule every few world hours.

Whether it rains is a world fact like the hour, with nothing subjective in it, so a rule decides
it (CLAUDE.md Design Principle 5), never an LLM. The climate is chosen once at build from a closed
set; the season follows from the calendar month. Everyone, indoors or out, perceives the same sky:
places have no indoor/outdoor distinction.

Narrative text gets the sky as words (``Weather.describe``); the API gets the enum values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from core.coerce import coerce_enum
from core.interfaces.phenomenon import Phenomenon
from core.interfaces.severity import Severity
from core.logging import get_logger
from engine.clock import WorldTime

logger = get_logger(__name__)


class Sky(str, Enum):
    """What the sky is doing. The precipitation kinds match the renderer's ``Phenomenon`` names."""

    CLEAR    = "clear"
    OVERCAST = "overcast"
    RAIN     = "rain"
    SNOW     = "snow"
    WIND     = "wind"


class Climate(str, Enum):
    """Chosen once at world build; an unknown value reads as the default."""

    TEMPERATE_MONSOON = "temperate_monsoon"
    HUMID_SUBTROPICAL = "humid_subtropical"
    ARID              = "arid"
    COLD              = "cold"
    TROPICAL          = "tropical"

    @property
    def label(self) -> str:
        return _CLIMATE_LABELS[self]


_CLIMATE_LABELS: dict[Climate, str] = {
    Climate.TEMPERATE_MONSOON: "温带季风（四季分明，夏雨冬雪）",
    Climate.HUMID_SUBTROPICAL: "湿润亚热带（多雨少雪）",
    Climate.ARID:              "干旱（少雨多风）",
    Climate.COLD:              "寒冷（冬长多雪）",
    Climate.TROPICAL:          "热带（终年炎热，雨季多雨）",
}

DEFAULT_CLIMATE = Climate.TEMPERATE_MONSOON


class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

    @classmethod
    def of_month(cls, month: int) -> "Season":
        if 3 <= month <= 5:
            return cls.SPRING
        if 6 <= month <= 8:
            return cls.SUMMER
        if 9 <= month <= 11:
            return cls.AUTUMN
        return cls.WINTER


# The sky moves on every this many world seconds, however long a step is.
TICK_SECONDS = 3 * 3600

# Chance the sky simply carries on at a tick; otherwise it is drawn afresh from the season's
# weights (which may draw the same sky again). Weather comes in spells, not a fresh coin each tick.
_PERSISTENCE = 0.6

_S = Sky
# The share of a season's hours under each sky, per climate; each row sums to 1. Hours, not
# days: a day that sees rain is still dry most of the time, so rain sits well below a count of
# rainy days. Don't raise it to that count, or the world spends half its hours in the rain.
_SEASON_WEIGHTS: dict[Climate, dict[Season, dict[Sky, float]]] = {
    Climate.TEMPERATE_MONSOON: {
        Season.SPRING: {_S.CLEAR: .50, _S.OVERCAST: .28, _S.RAIN: .08, _S.SNOW: .00, _S.WIND: .14},
        Season.SUMMER: {_S.CLEAR: .48, _S.OVERCAST: .28, _S.RAIN: .18, _S.SNOW: .00, _S.WIND: .06},
        Season.AUTUMN: {_S.CLEAR: .60, _S.OVERCAST: .22, _S.RAIN: .08, _S.SNOW: .00, _S.WIND: .10},
        Season.WINTER: {_S.CLEAR: .52, _S.OVERCAST: .25, _S.RAIN: .00, _S.SNOW: .08, _S.WIND: .15},
    },
    Climate.HUMID_SUBTROPICAL: {
        Season.SPRING: {_S.CLEAR: .35, _S.OVERCAST: .40, _S.RAIN: .20, _S.SNOW: .00, _S.WIND: .05},
        Season.SUMMER: {_S.CLEAR: .48, _S.OVERCAST: .30, _S.RAIN: .17, _S.SNOW: .00, _S.WIND: .05},
        Season.AUTUMN: {_S.CLEAR: .55, _S.OVERCAST: .30, _S.RAIN: .10, _S.SNOW: .00, _S.WIND: .05},
        Season.WINTER: {_S.CLEAR: .40, _S.OVERCAST: .42, _S.RAIN: .12, _S.SNOW: .02, _S.WIND: .04},
    },
    Climate.ARID: {
        Season.SPRING: {_S.CLEAR: .60, _S.OVERCAST: .10, _S.RAIN: .02, _S.SNOW: .00, _S.WIND: .28},
        Season.SUMMER: {_S.CLEAR: .75, _S.OVERCAST: .08, _S.RAIN: .05, _S.SNOW: .00, _S.WIND: .12},
        Season.AUTUMN: {_S.CLEAR: .65, _S.OVERCAST: .10, _S.RAIN: .03, _S.SNOW: .00, _S.WIND: .22},
        Season.WINTER: {_S.CLEAR: .55, _S.OVERCAST: .15, _S.RAIN: .02, _S.SNOW: .08, _S.WIND: .20},
    },
    Climate.COLD: {
        Season.SPRING: {_S.CLEAR: .40, _S.OVERCAST: .35, _S.RAIN: .07, _S.SNOW: .08, _S.WIND: .10},
        Season.SUMMER: {_S.CLEAR: .50, _S.OVERCAST: .28, _S.RAIN: .12, _S.SNOW: .00, _S.WIND: .10},
        Season.AUTUMN: {_S.CLEAR: .40, _S.OVERCAST: .35, _S.RAIN: .07, _S.SNOW: .08, _S.WIND: .10},
        Season.WINTER: {_S.CLEAR: .38, _S.OVERCAST: .30, _S.RAIN: .00, _S.SNOW: .17, _S.WIND: .15},
    },
    Climate.TROPICAL: {
        Season.SPRING: {_S.CLEAR: .52, _S.OVERCAST: .25, _S.RAIN: .18, _S.SNOW: .00, _S.WIND: .05},
        Season.SUMMER: {_S.CLEAR: .40, _S.OVERCAST: .30, _S.RAIN: .25, _S.SNOW: .00, _S.WIND: .05},
        Season.AUTUMN: {_S.CLEAR: .45, _S.OVERCAST: .28, _S.RAIN: .22, _S.SNOW: .00, _S.WIND: .05},
        Season.WINTER: {_S.CLEAR: .62, _S.OVERCAST: .25, _S.RAIN: .08, _S.SNOW: .00, _S.WIND: .05},
    },
}

# How hard it comes down when it does. Clear and overcast skies have no strength.
_STRENGTH_WEIGHTS: dict[Sky, dict[Severity, float]] = {
    Sky.RAIN: {Severity.LOW: .45, Severity.MEDIUM: .40, Severity.HIGH: .15},
    Sky.SNOW: {Severity.LOW: .45, Severity.MEDIUM: .40, Severity.HIGH: .15},
    Sky.WIND: {Severity.LOW: .50, Severity.MEDIUM: .35, Severity.HIGH: .15},
}

_DESCRIPTIONS: dict[tuple[Sky, Severity], str] = {
    (Sky.CLEAR, Severity.LOW):    "天色晴好",
    (Sky.OVERCAST, Severity.LOW): "天色阴沉",
    (Sky.RAIN, Severity.LOW):     "外头飘着细雨",
    (Sky.RAIN, Severity.MEDIUM):  "外头下着雨",
    (Sky.RAIN, Severity.HIGH):    "外头下着大雨",
    (Sky.SNOW, Severity.LOW):     "外头飘着小雪",
    (Sky.SNOW, Severity.MEDIUM):  "外头下着雪",
    (Sky.SNOW, Severity.HIGH):    "外头大雪纷飞",
    (Sky.WIND, Severity.LOW):     "外头有些风",
    (Sky.WIND, Severity.MEDIUM):  "外头刮着大风",
    (Sky.WIND, Severity.HIGH):    "外头狂风大作",
}

# Travel is slower through heavy rain and heavy snow.
_TRAVEL_FACTORS: dict[tuple[Sky, Severity], float] = {
    (Sky.RAIN, Severity.HIGH): 1.25,
    (Sky.SNOW, Severity.HIGH): 1.5,
}


@dataclass(frozen=True)
class Weather:
    sky: Sky = Sky.CLEAR
    # How hard it comes down; always LOW for a sky with no strength (clear, overcast).
    severity: Severity = Severity.LOW

    def __post_init__(self) -> None:
        if self.sky not in _STRENGTH_WEIGHTS:
            object.__setattr__(self, "severity", Severity.LOW)

    def describe(self) -> str:
        """The sky in words, the only form it takes in a prompt or a memory."""
        return _DESCRIPTIONS.get((self.sky, self.severity), _DESCRIPTIONS[(Sky.CLEAR, Severity.LOW)])

    @property
    def phenomenon(self) -> Phenomenon:
        """What the map draws: ``NONE`` for a clear or overcast sky."""
        return {Sky.RAIN: Phenomenon.RAIN, Sky.SNOW: Phenomenon.SNOW, Sky.WIND: Phenomenon.WIND}.get(
            self.sky, Phenomenon.NONE,
        )

    @property
    def travel_factor(self) -> float:
        return _TRAVEL_FACTORS.get((self.sky, self.severity), 1.0)

    def to_dict(self) -> dict[str, str]:
        return {"sky": self.sky.value, "severity": self.severity.value}


def transition_row(sky: Sky, climate: Climate, season: Season) -> dict[Sky, float]:
    """Next tick's sky given this one: carry on, or draw afresh from the season's weights."""
    weights = _SEASON_WEIGHTS[climate][season]
    return {
        nxt: (_PERSISTENCE if nxt == sky else 0.0) + (1 - _PERSISTENCE) * weights[nxt]
        for nxt in Sky
    }


def _draw(weights: Mapping[Any, float], rng: random.Random) -> Any:  # noqa: ANN401
    roll = rng.random() * sum(weights.values())
    for option, weight in weights.items():
        roll -= weight
        if roll < 0:
            return option
    return next(reversed(weights))


def _draw_strength(sky: Sky, rng: random.Random) -> Severity:
    weights = _STRENGTH_WEIGHTS.get(sky)
    return _draw(weights, rng) if weights else Severity.LOW


def change_notice(before: Weather, after: Weather) -> str:
    """What anyone notices when the sky turns; empty when it hasn't noticeably."""
    if before == after:
        return ""
    if before.sky == after.sky:
        stronger = after.severity.level > before.severity.level
        return {
            Sky.RAIN: "雨势大了起来。" if stronger else "雨势小了些。",
            Sky.SNOW: "雪下得更紧了。" if stronger else "雪下得小了些。",
            Sky.WIND: "风势更猛了。" if stronger else "风势弱了些。",
        }.get(after.sky, "")
    ended = {Sky.RAIN: "雨停了。", Sky.SNOW: "雪停了。", Sky.WIND: "风停了。"}.get(before.sky, "")
    began = {
        Sky.CLEAR: "天放晴了。" if not ended else "",
        Sky.OVERCAST: "天色阴了下来。" if not ended else "天仍阴着。",
        Sky.RAIN: "下起雨来了。" if after.severity != Severity.HIGH else "下起大雨来了。",
        Sky.SNOW: "下起雪来了。" if after.severity != Severity.HIGH else "下起大雪来了。",
        Sky.WIND: "起风了。" if after.severity != Severity.HIGH else "刮起了狂风。",
    }[after.sky]
    return f"{ended}{began}"


class WeatherSystem:
    """The world's one sky. Advanced by the runtime at the start of each step.

    Reproducible: each tick's draw is seeded by the world and the tick's number, so the same
    world sees the same weather however the run was paused and restored.
    """

    def __init__(self, *, climate: Climate, seed: str, weather: Weather | None = None, tick: int = 0) -> None:
        self._climate = climate
        self._seed = seed
        self._tick = tick
        self._weather = weather if weather is not None else Weather()

    @classmethod
    def opening(cls, *, climate: Climate, seed: str, month: int) -> "WeatherSystem":
        """The sky the world opens under: drawn from the starting season's weights."""
        rng = random.Random(f"{seed}:weather:0")
        sky = _draw(_SEASON_WEIGHTS[climate][Season.of_month(month)], rng)
        return cls(climate=climate, seed=seed, weather=Weather(sky, _draw_strength(sky, rng)))

    @property
    def weather(self) -> Weather:
        return self._weather

    @property
    def climate(self) -> Climate:
        return self._climate

    def advance(self, world_time: WorldTime) -> Weather:
        """Run every tick due by ``world_time``; return the sky as it stood before."""
        before = self._weather
        season = Season.of_month(world_time.month_of_year)
        due = world_time.elapsed_seconds // TICK_SECONDS
        while self._tick < due:
            self._tick += 1
            rng = random.Random(f"{self._seed}:weather:{self._tick}")
            current = self._weather
            sky = _draw(transition_row(current.sky, self._climate, season), rng)
            # A carried-on sky keeps its strength half the time, so a downpour eases gradually.
            severity = (
                current.severity if sky == current.sky and rng.random() < 0.5
                else _draw_strength(sky, rng)
            )
            self._weather = Weather(sky, severity)
        return before

    def snapshot(self) -> dict[str, Any]:
        return {"climate": self._climate.value, "tick": self._tick, **self._weather.to_dict()}

    @classmethod
    def restore(cls, payload: object, *, seed: str, climate: Climate) -> "WeatherSystem":
        """From ``snapshot``; a world saved without weather opens under the given climate."""
        if not isinstance(payload, Mapping):
            return cls(climate=climate, seed=seed)
        try:
            tick = int(payload.get("tick", 0))
        except (TypeError, ValueError):
            tick = 0
        sky = coerce_enum(Sky, payload.get("sky"), default=Sky.CLEAR)
        return cls(
            climate=coerce_enum(Climate, payload.get("climate"), default=climate),
            seed=seed,
            weather=Weather(sky, coerce_enum(Severity, payload.get("severity"), default=Severity.LOW)),
            tick=tick,
        )
