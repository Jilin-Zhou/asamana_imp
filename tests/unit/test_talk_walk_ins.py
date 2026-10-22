"""Walking into a conversation: TALK toward someone already talking joins that talk.

A conversation takes up to four people. A walk-in spends his own turn on his own decision (not a
conscript), and the conversation then voices, remembers and moves relations for everyone in it.
"""

from __future__ import annotations

import json

import pytest

from core.interfaces.action import ActionTarget, ActionType, AgentAction, Ref
from core.interfaces.llm import LLMScene
from engine.executors.social import SocialExecutor
from engine.executors.trade import TradeExecutor
from tests.unit.test_action_executors import _directory, _joined, _make_agent, _make_environment
from tests.unit.test_runtime_smoke import _arb_runtime, _build_agent, _plan_for, _two_rooms_with

from engine.environment import EnvironmentSystem
from worlds.tiled import TiledWorldConfig


def _talk(actor: str, other: str, desc: str, *, steps: int = 2) -> AgentAction:
    return AgentAction(
        agent_id=actor, step=1, action_type=ActionType.TALK, action_description=desc,
        target=ActionTarget(acts_on=[Ref.agent(other)], claims=[Ref.agent(other)]), estimated_steps=steps,
    )


def _palace(container, world_id: str, names: dict[str, str]):  # noqa: ANN202
    env = EnvironmentSystem(TiledWorldConfig(template="changan_iso"))
    agents = {}
    for aid, name in names.items():
        env.place_agent(agent_id=aid, location_id="palace")
        agents[aid] = _build_agent(container, world_id=world_id, agent_id=aid, name=name, is_main_character=True)
    return env, agents, _arb_runtime(container, world_id, agents, env)


@pytest.mark.asyncio
async def test_talking_to_someone_already_talking_walks_into_that_talk(container) -> None:
    env, agents, rt = _palace(container, "world-walk-in", {"agent-a": "李世民", "agent-b": "常何", "agent-c": "尉迟恭"})
    a, b, c = agents.values()
    planned = [
        ("main", a, _plan_for(a, _talk("agent-a", "agent-b", "商议玄武门值守"), env)),
        ("main", b, _plan_for(b, _talk("agent-b", "agent-a", "回禀值守安排"), env)),
        ("main", c, _plan_for(c, _talk("agent-c", "agent-b", "问常何今夜谁当值"), env)),
    ]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    talk = rt._executor_registry.get_active_for_agent("agent-a")  # noqa: SLF001
    assert talk.participant_ids == ["agent-a", "agent-b", "agent-c"]
    assert rt._executor_registry.get_active_for_agent("agent-c") is talk  # noqa: SLF001
    # His own decision, not a conscription: he keeps his act and aim.
    walk_in = arb["agent-c"]
    assert not walk_in.is_passive_join and walk_in.ongoing_execution_id == talk.execution_id
    assert "尉迟恭凑了过去，加入了李世民与常何的交谈" in walk_in.action_result.outcome
    assert walk_in.action_result.observations
    assert talk.joined["agent-c"].description == "问常何今夜谁当值"
    # Nobody is turned away.
    assert not any(s.extra.get("completed_result") for s in rt._executor_registry.all_active())  # noqa: SLF001


@pytest.mark.asyncio
async def test_a_conversation_under_way_takes_a_walk_in(container) -> None:
    env, agents, rt = _palace(container, "world-walk-in-ongoing", {"agent-a": "李世民", "agent-b": "常何", "agent-c": "尉迟恭"})
    executor = rt._executor_registry.get_executor(ActionType.TALK)  # noqa: SLF001
    ongoing = await executor.start(
        _talk("agent-a", "agent-b", "商议玄武门值守", steps=3), 0, agents=agents, environment=env, message_system=None,
    )
    rt._executor_registry.add_active(ongoing)  # noqa: SLF001
    c = agents["agent-c"]
    planned = [("main", c, _plan_for(c, _talk("agent-c", "agent-a", "向秦王请命"), env))]

    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, {"agent-a", "agent-b"})  # noqa: SLF001

    assert ongoing.participant_ids == ["agent-a", "agent-b", "agent-c"]
    assert arb["agent-c"].ongoing_execution_id == ongoing.execution_id


@pytest.mark.asyncio
async def test_a_full_conversation_turns_the_next_one_away(container) -> None:
    names = {"agent-a": "李世民", "agent-b": "常何", "agent-c": "尉迟恭", "agent-d": "侯君集", "agent-e": "房玄龄"}
    env, agents, rt = _palace(container, "world-walk-in-full", names)
    executor = rt._executor_registry.get_executor(ActionType.TALK)  # noqa: SLF001
    ongoing = await executor.start(
        _talk("agent-a", "agent-b", "商议", steps=3), 0, agents=agents, environment=env, message_system=None,
    )
    for pid in ("agent-c", "agent-d"):
        assert executor.join(ongoing, _talk(pid, "agent-a", "插话"), 0, environment=env) is not None
    rt._executor_registry.add_active(ongoing)  # noqa: SLF001
    e = agents["agent-e"]
    planned = [("main", e, _plan_for(e, _talk("agent-e", "agent-a", "请秦王借一步说话"), env))]

    await rt._arbiter.arbitrate(planned, agents, 1, {"agent-a", "agent-b", "agent-c", "agent-d"})  # noqa: SLF001

    assert len(ongoing.participant_ids) == 4
    rejected = rt._executor_registry.get_active_for_agent("agent-e").extra["completed_result"]  # noqa: SLF001
    assert "李世民正与常何、尉迟恭、侯君集交谈" in rejected.outcome


@pytest.mark.asyncio
async def test_a_carry_off_releases_whoever_was_walking_in(container, caplog) -> None:
    """The host's own talk is revoked when its addressee is dragged off; the walk-in is adjudicated
    again, and meets the man being carried off instead of a conversation."""
    from engine.arbiter import _FOILED_MISS_PRIORITY_THRESHOLD as N

    env, agents, rt = _two_rooms_with(
        container, "world-walk-in-revoked",
        {"agent-a": "李世民", "agent-b": "尉迟恭", "agent-c": "常何", "agent-d": "侯君集"},
    )
    a, b, c, d = agents.values()
    carry = AgentAction(agent_id="agent-d", step=1, action_type=ActionType.MOVE, action_description="拽着常何走",
                        estimated_steps=1, target=ActionTarget(acts_on=[Ref.place("garden")], claims=[Ref.agent("agent-c")]))
    # Starvation puts the talk and the walk-in ahead of the carry, so the carry revokes them.
    planned = [
        ("main", a, _plan_for(a, _talk("agent-a", "agent-c", "盘问常何"), env, foiled_misses=N + 2)),
        ("main", b, _plan_for(b, _talk("agent-b", "agent-c", "也来问常何"), env, foiled_misses=N)),
        ("main", c, _plan_for(c, AgentAction(agent_id="agent-c", step=1, action_type=ActionType.WORK,
                                             action_description="点卯", estimated_steps=1), env)),
        ("main", d, _plan_for(d, carry, env)),
    ]

    import logging

    caplog.set_level(logging.INFO, logger="engine.arbiter")
    arb, _ = await rt._arbiter.arbitrate(planned, agents, 1, set())  # noqa: SLF001

    events = [(r.message, getattr(r, "agent_id", None)) for r in caplog.records]
    assert events.index(("execution_join_booked", "agent-b")) < events.index(("admission_revoked", "agent-a"))
    registry = rt._executor_registry  # noqa: SLF001
    assert registry.get_active_for_agent("agent-c").initiator_id == "agent-d"
    walk_in = registry.get_active_for_agent("agent-b")
    assert walk_in.participant_ids == ["agent-b"] and walk_in.extra["completed_result"].not_executed
    assert "常何正要动身离开" in walk_in.extra["completed_result"].outcome
    assert not arb["agent-b"].is_passive_join


def _three_way_reply() -> str:
    return json.dumps({
        "reason": "三人各有所图。",
        "dialogue": [
            {"speaker": 1, "line": "今夜谁当值？"},
            {"speaker": 3, "line": "末将也想知道。"},
            {"speaker": 2, "line": "二位稍候。"},
        ],
        "observation": "",
        "fact": "我们三人谈了值守。",
        "success": True,
        "why": "",
        "relations": [{"index": 1, "relation": "positive"}, {"index": 2, "relation": "negative"}],
    }, ensure_ascii=False)


async def _three_way(container):  # noqa: ANN202
    provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
    provider.fixed_response = _three_way_reply()
    names = {"agent-a": "甲", "agent-b": "乙", "agent-c": "丙"}
    agents = {aid: _make_agent(container, world_id="w", agent_id=aid, name=n, is_main=True) for aid, n in names.items()}
    env = _make_environment()
    env.place_agent(agent_id="agent-c", location_id="hall")
    executor = SocialExecutor(container.llm_router, _directory(agents), seconds_per_step=3600)
    state = await executor.start(_talk("agent-a", "agent-b", "商议值守"), 1, agents=agents, environment=env, message_system=None)
    opening = executor.join(state, _talk("agent-c", "agent-b", "问乙今夜谁当值"), 1, environment=env)
    assert opening is not None
    return provider, agents, env, executor, state


@pytest.mark.asyncio
async def test_a_three_way_conversation_gives_everyone_a_voice_a_memory_and_relations(container) -> None:
    provider, agents, env, executor, state = await _three_way(container)

    results = await executor.complete(state, 2, agents=agents, environment=env, message_system=None)

    dialogue_prompt = _joined(provider.call_history[0])
    assert "【1：甲】" in dialogue_prompt and "【3：丙】" in dialogue_prompt
    assert "【中途加入】" in dialogue_prompt and "丙走过来找乙，想「问乙今夜谁当值」" in dialogue_prompt
    assert "至多 10 轮" in dialogue_prompt
    by_agent = {r.action.agent_id: r for r in results}
    assert set(by_agent) == {"agent-a", "agent-b", "agent-c"}
    assert "丙：末将也想知道。" in by_agent["agent-a"].outcome
    assert by_agent["agent-a"].gist.endswith("甲、乙与丙交谈。")
    # Each speaker's relations reach both others; the walk-in keeps his own act and aim.
    assert {u[0] for u in by_agent["agent-c"].relation_updates} == {"agent-a", "agent-b"}
    assert by_agent["agent-c"].action.action_description == "问乙今夜谁当值"
    assert by_agent["agent-c"].action.target.acted_on_agents == ["agent-b"]
    summaries = [_joined(m) for m in provider.call_history[1:]]
    joiner_summary = next(s for s in summaries if "我走过去找乙，加了进来" in s)
    assert "#1 乙" in joiner_summary and "#2 甲" in joiner_summary
    assert any("是我主动找乙谈的，丙也凑过来一起谈" in s for s in summaries)


@pytest.mark.asyncio
async def test_anyone_breaking_off_ends_it_for_all_three(container) -> None:
    provider, agents, env, executor, state = await _three_way(container)
    provider.fixed_response = '{"fact": "话没说完。"}'

    results = await executor.interrupt(
        state, 2, agents=agents, environment=env, interrupted_agent_id="agent-c", thought="我另有急事",
    )

    assert {r.action.agent_id for r in results} == {"agent-a", "agent-b", "agent-c"}
    assert all(not r.succeeded and r.relation_updates == [] for r in results)
    assert all("甲、乙与丙的交谈被丙打断" in r.gist for r in results)


def test_a_deal_takes_no_walk_ins(container) -> None:
    trade = TradeExecutor(container.llm_router, _directory())
    assert not trade.admits_joiner(2)
    assert SocialExecutor(container.llm_router, _directory()).admits_joiner(3)
