"""The stage narrator: scene and act rules, quoting by index, continuity, storage and the API."""

from __future__ import annotations

import inspect
import json
import pathlib
from datetime import datetime
from typing import Any

import pytest

from core.event_bus import NarrativeEventBus
from core.interfaces.check import CheckRecord, Difficulty, Edge
from core.interfaces.llm import LLMResponse, LLMScene
from core.interfaces.snapshot import WorldSnapshot
from engine.directory import LiveWorldDirectory
from engine.environment import EnvironmentSystem
from engine.stage_narrator import StageNarrator, StepMaterial, gather_places, opens_act, _Continuity
from providers.snapshot.file import FileSnapshotProvider
from providers.snapshot.in_memory import InMemorySnapshotProvider


def _rec(agent: str, name: str, place: str, phase: str = "settled", **fields: Any) -> dict[str, Any]:
    return {
        "agent_id": agent, "agent_name": name, "initiator_id": fields.pop("by", agent),
        "execution_id": fields.pop("ex", f"{agent}-{place}-{phase}"), "location_id": place,
        "phase": phase, "action_type": fields.pop("type", "talk"), "succeeded": True,
        "action_description": fields.pop("desc", "做事"), "outcome": fields.pop("outcome", "事成"),
        **fields,
    }


def _material(step: int, *actions: dict[str, Any], day: int = 0, **extra: Any) -> StepMaterial:
    return StepMaterial(step=step, time_label=f"第{day}天", day=day, actions=list(actions), **extra)


class _Router:
    """Canned replies in order; records every call. ``None`` raises."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[Any, list]] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.calls.append((scene, messages))
        reply = self.replies.pop(0)
        if reply is None:
            raise RuntimeError("llm down")
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return LLMResponse(content=content, input_tokens=0, output_tokens=0, model="stub")


def _narrator(router: _Router, store=None, bus=None) -> StageNarrator:
    return StageNarrator(
        world_id="w", llm_router=router,
        snapshot_provider=store or InMemorySnapshotProvider(),
        event_bus=bus or NarrativeEventBus(),
        directory=LiveWorldDirectory.from_agents({}, EnvironmentSystem()),
    )


# ---------------------------------------------------------------------------
# Structure decided by code
# ---------------------------------------------------------------------------


def test_places_skip_ticks_non_events_and_fold_joint_acts() -> None:
    places = gather_places(_material(
        1,
        _rec("a", "甲", "hall", ex="t1", dialogue=[{"speaker": "甲", "line": "来了"}]),
        _rec("b", "乙", "hall", ex="t1", by="a"),   # the joint talk's second record
        _rec("c", "丙", "yard", phase="ongoing_tick"),
        _rec("d", "丁", "gate", not_executed=True),
        _rec("e", "戊", "", phase="initialization"),
    ))
    assert [p.location_id for p in places] == ["hall"]
    [hall] = places
    assert [r["agent_id"] for r in hall.records] == ["a"]


def test_at_most_four_scenes_keep_the_busiest_in_record_order() -> None:
    actions = [_rec(f"x{i}", "某", f"p{i}") for i in range(6)]
    actions += [_rec("y", "某", "p5", ex="extra", check={"tier": "success"})]
    places = gather_places(_material(1, *actions))
    assert len(places) == 4
    assert "p5" in [p.location_id for p in places]
    assert [p.location_id for p in places] == sorted(p.location_id for p in places)


def test_an_act_opens_on_the_first_script_a_new_day_or_a_major_event() -> None:
    assert opens_act(_material(1), _Continuity())
    going = _Continuity(act=1, last_day=0)
    assert not opens_act(_material(2, day=0), going)
    assert opens_act(_material(2, day=1), going)
    assert opens_act(_material(2, day=0, broadcasts=[{"severity": "high", "content": "某人已死"}]), going)


# ---------------------------------------------------------------------------
# The narrator end to end
# ---------------------------------------------------------------------------


def _script(*scenes: dict[str, Any], memo: str = "甲乙在殿上说了话") -> dict[str, Any]:
    return {"scenes": list(scenes), "memo": memo}


async def test_lines_are_quoted_verbatim_asides_distilled_and_bad_indices_dropped() -> None:
    router = _Router(_script(
        {"location_index": 1, "stage_direction": "殿内灯暗", "beats": [
            {"kind": "line", "line_index": 1},
            {"kind": "line", "line_index": 2},      # spoken in the yard: not this scene's
            {"kind": "line", "line_index": 9},      # out of range
            {"kind": "aside", "monologue_index": 1, "text": "不能再拖了。"},
            {"kind": "aside", "monologue_index": 1},  # no words: the raw monologue never goes on stage
            {"kind": "narration", "text": "一"}, {"kind": "narration", "text": "二"},
            {"kind": "narration", "text": "三"}, {"kind": "narration", "text": "四"},
        ]},
        {"location_index": 7, "stage_direction": "无此地", "beats": []},
    ))
    store = InMemorySnapshotProvider()
    bus = NarrativeEventBus()
    queue = bus.subscribe("w")
    narrator = _narrator(router, store, bus)
    narrator.submit(_material(
        3,
        _rec("a", "甲", "hall", dialogue=[{"speaker_id": "a", "speaker": "甲", "line": "兵符何在？"}],
             inner_monologue="此事不能再拖。"),
        _rec("b", "乙", "yard", dialogue=[{"speaker_id": "b", "speaker": "乙", "line": "院中无人。"}]),
    ))
    await narrator.drain()

    saved = await store.load_narration("w", 3)
    assert saved is not None and [s["location_id"] for s in saved["scenes"]] == ["hall"]
    beats = saved["scenes"][0]["beats"]
    assert {"kind": "line", "speaker_id": "a", "speaker": "甲", "text": "兵符何在？"} in beats
    assert not any(b.get("text") == "院中无人。" for b in beats)
    # The aside is the narrator's distillation, attributed by index to whose thought it was.
    assert [b for b in beats if b["kind"] == "aside"] == [
        {"kind": "aside", "speaker_id": "a", "speaker": "甲", "text": "不能再拖了。"},
    ]
    assert [b["text"] for b in beats if b["kind"] == "narration"] == ["一", "二", "三"]
    event = queue.get_nowait()
    assert event["type"] == "narration" and event["step"] == 3

    scene, messages = router.calls[0]
    assert scene is LLMScene.STAGE_NARRATION
    prompt = "\n".join(m.content for m in messages)
    assert "兵符何在" in prompt and "[1]" in prompt
    for leak in ("hall", "yard", "a-hall", "第3步"):
        assert leak not in prompt


async def test_the_roll_reaches_the_narrator_as_words() -> None:
    roll = CheckRecord(difficulty=Difficulty.HARD, edge=Edge.EVEN, pure_chance=False, dice=(1, 1, 2))
    router = _Router(_script({"location_index": 1, "stage_direction": "x", "beats": []}))
    narrator = _narrator(router)
    narrator.submit(_material(1, _rec("a", "甲", "hall", type="physical", check=roll.to_dict())))
    await narrator.drain()
    prompt = "\n".join(m.content for m in router.calls[0][1])
    assert "不但没做成，还弄糟了" in prompt
    assert "critical_failure" not in prompt and "14" not in prompt


async def test_a_scene_continues_at_the_same_place_with_the_same_people() -> None:
    reply = _script({"location_index": 1, "stage_direction": "x", "beats": []})
    store = InMemorySnapshotProvider()
    narrator = _narrator(_Router(reply, reply, reply), store)
    narrator.submit(_material(1, _rec("a", "甲", "hall")))
    narrator.submit(_material(2, _rec("a", "甲", "hall", desc="又做事")))
    narrator.submit(_material(3, _rec("a", "甲", "hall", desc="再做事"), day=1))
    await narrator.drain()
    first, second, third = [await store.load_narration("w", s) for s in (1, 2, 3)]
    assert (first["act"], first["scenes"][0]["scene"], first["scenes"][0]["continues"]) == (1, 1, False)
    assert (second["act"], second["scenes"][0]["scene"], second["scenes"][0]["continues"]) == (1, 1, True)
    # A new day turns the act over and opens a new scene.
    assert (third["act"], third["opens_act"], third["scenes"][0]["scene"]) == (2, True, 2)


async def test_a_failed_step_costs_only_its_own_script() -> None:
    reply = _script({"location_index": 1, "stage_direction": "x", "beats": []})
    store = InMemorySnapshotProvider()
    narrator = _narrator(_Router(None, "不是 JSON", reply), store)
    for step in (1, 2, 3):
        narrator.submit(_material(step, _rec("a", "甲", "hall", ex=f"e{step}")))
    await narrator.drain()
    assert await store.load_narration("w", 1) is None
    assert await store.load_narration("w", 2) is None
    assert await store.load_narration("w", 3) is not None


async def test_a_restored_narrator_continues_the_numbering() -> None:
    reply = _script({"location_index": 1, "stage_direction": "x", "beats": []}, memo="前情")
    store = InMemorySnapshotProvider()
    first = _narrator(_Router(reply), store)
    first.submit(_material(4, _rec("a", "甲", "hall")))
    await first.drain()

    router = _Router(reply)
    again = _narrator(router, store)
    again.submit(_material(5, _rec("a", "甲", "hall", ex="next")))
    await again.drain()
    saved = await store.load_narration("w", 5)
    assert (saved["act"], saved["scenes"][0]["scene"], saved["scenes"][0]["continues"]) == (1, 1, True)
    assert "前情" in router.calls[0][1][1].content


async def test_a_step_with_nothing_staged_makes_no_call() -> None:
    router = _Router()
    narrator = _narrator(router)
    narrator.submit(_material(1, _rec("a", "甲", "hall", phase="ongoing_tick")))
    await narrator.drain()
    assert router.calls == []


def test_the_narrator_never_reaches_cognition() -> None:
    """Its prose is for the audience: no agent module may read it back."""
    for path in pathlib.Path("agent").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "stage_narrator" not in source and "load_narration" not in source, path


def test_the_narrator_prompt_carries_the_closed_world_rule() -> None:
    from core.prompts import CLOSED_WORLD_FACT_RULE
    import engine.stage_narrator as mod

    assert CLOSED_WORLD_FACT_RULE in mod._SYSTEM_PROMPT
    assert "CLOSED_WORLD_FACT_RULE" in inspect.getsource(mod)


# ---------------------------------------------------------------------------
# Storage and the read path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["memory", "file"])
async def test_narrations_are_stored_beside_steps_and_reset_with_them(kind: str, tmp_path) -> None:
    store = InMemorySnapshotProvider() if kind == "memory" else FileSnapshotProvider(str(tmp_path))
    for step in (1, 2, 3):
        await store.save("w", step, WorldSnapshot(world_id="w", step=step, timestamp=datetime.now()))
        await store.save_narration("w", step, {"step": step})
    assert await store.list_steps("w") == [1, 2, 3]
    await store.delete_steps_after("w", 1)
    assert await store.load_narration("w", 1) == {"step": 1}
    assert await store.load_narration("w", 2) is None
    await store.delete_world("w")
    assert await store.load_narration("w", 1) is None


async def test_the_api_serves_scripts_in_step_order(container, test_config) -> None:
    from fastapi.testclient import TestClient

    from interaction.api import create_app
    from world import WorldCatalog

    for step in (1, 2, 3):
        await container.snapshot.save("w-narr", step, WorldSnapshot(world_id="w-narr", step=step, timestamp=datetime.now()))
    await container.snapshot.save_narration("w-narr", 1, {
        "step": 1, "act": 1, "opens_act": True, "time_label": "清晨", "memo": "开场",
        "scenes": [{"scene": 1, "continues": False, "location_id": "hall", "location_name": "大殿",
                    "stage_direction": "灯暗", "cast": [{"agent_id": "a", "name": "甲"}],
                    "beats": [{"kind": "line", "speaker_id": "a", "speaker": "甲", "text": "来了"}]}],
    })
    await container.snapshot.save_narration("w-narr", 3, {"step": 3, "act": 1, "scenes": []})
    client = TestClient(create_app(container, test_config, catalog=WorldCatalog(None)))

    scripts = client.get("/api/worlds/w-narr/narration").json()
    assert [s["step"] for s in scripts] == [1, 3]
    assert scripts[0]["scenes"][0]["beats"][0] == {
        "kind": "line", "text": "来了", "speaker_id": "a", "speaker": "甲",
    }
    assert [s["step"] for s in client.get("/api/worlds/w-narr/narration", params={"since": 2}).json()] == [3]
