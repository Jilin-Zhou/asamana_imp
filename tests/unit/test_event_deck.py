"""The event deck: cards parse defensively, draws are seeded and honor conditions and cooldown,
and a drawn card replaces the editor's pacing gate without deciding whom it lands on.
"""

from __future__ import annotations

import json

import pytest

from core.interfaces.llm import LLMResponse
from core.interfaces.severity import Severity
from engine.event_deck import DRAW_CHANCE, MAX_CARDS, Daypart, EventCard, Scope, deck_from_dicts, draw
from engine.weather import Sky
from tests.build_stubs import MOCK_EVENT_DECK_RESPONSE
from tests.unit.test_event_system import (
    _EventRouter, _build_agent, _event_system, _joined, _plan_json, _world_time,
)

_RUMOR = EventCard("坊间传言", "坊间又起一则传言。")
_FLOOD = EventCard("大雨冲垮坊墙", "坊墙塌了。", sky=Sky.RAIN, weight=3)
_CURFEW = EventCard("夜半宵禁", "金吾卫巡街。", scope=Scope.GLOBAL, daypart=Daypart.NIGHT)


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------


def test_a_card_round_trips() -> None:
    for card in (_RUMOR, _FLOOD, _CURFEW):
        assert EventCard.from_dict(card.to_dict()) == card


def test_cards_are_read_defensively() -> None:
    deck = deck_from_dicts([
        {"title": "甲", "synopsis": "s", "weight": 9, "sky": "hail", "scope": "?", "severity": "x"},
        {"title": "甲", "synopsis": "重复的标题"},
        {"title": "", "synopsis": "无题"},
        "not a card",
    ])
    assert deck == (EventCard("甲", "s", scope=Scope.LOCAL, severity=Severity.MEDIUM, weight=3),)


def test_a_deck_is_capped() -> None:
    assert len(deck_from_dicts({"title": f"t{i}", "synopsis": "s"} for i in range(40))) == MAX_CARDS


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _draws(**kw: object) -> list[EventCard | None]:
    args = {"seed": "w", "sky": Sky.CLEAR, "hour": 12, "cooling": set(), **kw}
    return [draw((_RUMOR, _FLOOD, _CURFEW), step=step, **args) for step in range(1, 201)]


def test_draws_are_reproducible_per_world() -> None:
    assert _draws() == _draws()
    assert _draws() != _draws(seed="other")


def test_about_half_the_checks_draw() -> None:
    drawn = sum(card is not None for card in _draws())
    assert abs(drawn / 200 - DRAW_CHANCE) < 0.1


def test_a_card_waits_for_its_sky_and_its_hour() -> None:
    assert _FLOOD not in _draws(sky=Sky.CLEAR) and _CURFEW not in _draws(hour=12)
    assert _FLOOD in _draws(sky=Sky.RAIN) and _CURFEW in _draws(hour=23)


def test_a_cooling_card_is_not_drawn_and_an_empty_field_draws_nothing() -> None:
    assert _RUMOR not in _draws(cooling={_RUMOR.title})
    assert set(_draws(cooling={_RUMOR.title, _FLOOD.title, _CURFEW.title})) == {None}


# ---------------------------------------------------------------------------
# A drawn card in the event system
# ---------------------------------------------------------------------------


def _first_drawing_step(deck: tuple[EventCard, ...], check_interval: int) -> int:
    return next(
        s for s in range(check_interval, 400, check_interval)
        if draw(deck, seed="world-1", step=s, sky=None, hour=_world_time(s).hour_of_day, cooling=set())
    )


async def test_a_drawn_card_skips_the_gate_and_is_written_into_the_plan(container) -> None:
    deck = (_RUMOR,)
    step = _first_drawing_step(deck, 1)
    router = _EventRouter([_plan_json(broadcast={"content": "坊间都在传一件事", "severity": "low", "location_scope": None})])
    system = _event_system(container, router=router, check_interval=1, deck=deck)
    agents = {"a1": _build_agent(container, world_id="world-1", agent_id="a1", name="李世民")}

    await system.poll_event(current_step=step, world_time=_world_time(step), world_id="world-1",
                            all_agents=agents, locations=[], entities=[])
    await system._inflight
    fired = await system.poll_event(current_step=step + 1, world_time=_world_time(step + 1), world_id="world-1",
                                    all_agents=agents, locations=[], entities=[])
    # One call, the plan: the pacing gate never ran.
    assert len(router.calls) == 1
    assert "【抽中的事】" in _joined(router.calls[0][1]) and "坊间传言：坊间又起一则传言。" in _joined(router.calls[0][1])
    assert fired is not None and fired.event.metadata["card"] == "坊间传言"
    # And it cools: the next check can't draw it again.
    assert system._cooling(step + 2) == {"坊间传言"}


async def test_with_no_card_drawn_the_gate_still_decides(container) -> None:
    router = _EventRouter([json.dumps({"reason": "平稳", "inject": False})])
    system = _event_system(container, router=router, check_interval=1)
    agents = {"a1": _build_agent(container, world_id="world-1", agent_id="a1", name="李世民")}
    await system.poll_event(current_step=1, world_time=_world_time(1), world_id="world-1",
                            all_agents=agents, locations=[], entities=[])
    await system._inflight
    assert "【抽中的事】" not in _joined(router.calls[0][1])


# ---------------------------------------------------------------------------
# The build writes the deck
# ---------------------------------------------------------------------------


class _Router:
    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.calls: list[list] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.calls.append(messages)
        return LLMResponse(content=self._replies.pop(0), input_tokens=0, output_tokens=0, model="stub")


@pytest.fixture(autouse=True)
def _no_retry_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _skip(_seconds: float) -> None:
        return None

    monkeypatch.setattr("world.builders.event_deck_designer.asyncio.sleep", _skip)


async def test_the_designer_writes_a_deck_for_this_world() -> None:
    from tests.unit.test_world_builder import _analysis
    from world.builders.event_deck_designer import EventDeckDesigner

    router = _Router(MOCK_EVENT_DECK_RESPONSE)
    analysis = await EventDeckDesigner(router).design(_analysis(), ["太极宫", "玄武门"])
    assert len(analysis.event_deck) == 12
    assert analysis.event_deck[-1].daypart is Daypart.NIGHT
    assert "太极宫、玄武门" in _joined(router.calls[0])
    assert type(analysis).from_dict(analysis.as_dict()).event_deck == analysis.event_deck


async def test_a_thin_deck_is_retried_then_refused() -> None:
    from tests.unit.test_world_builder import _analysis
    from world.builders.event_deck_designer import EventDeckDesigner

    thin = json.dumps({"cards": [_RUMOR.to_dict()]}, ensure_ascii=False)
    router = _Router(thin, thin)
    with pytest.raises(ValueError, match="EventDeckDesigner failed"):
        await EventDeckDesigner(router).design(_analysis(), [])
    assert len(router.calls) == 2
