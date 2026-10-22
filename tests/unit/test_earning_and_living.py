"""Earning and spending: paid work brings money in from outside, each day's living takes it out.

Earnings are the WORK narrator's ruling, capped by the world's price list; living costs are code
arithmetic, charged per midnight crossed, and whoever can't pay goes short.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.need import NeedEngine, NeedType
from core.interfaces.action import ActionResult, ActionType, AgentAction
from core.interfaces.economy import Economy, PriceReference
from core.interfaces.llm import LLMScene
from engine.clock import WorldTimeConfig
from engine.environment import EnvironmentSystem
from engine.execution_processor import ExecutionProcessor
from engine.executors.registry import ActionExecutorRegistry
from engine.executors.work import WorkExecutor
from engine.living_costs import LivingCosts, days_crossed
from tests.unit.check_fixtures import fixed_check
from tests.unit.test_action_executors import _directory, _joined, _make_action, _make_agent, _make_environment

_ECONOMY = Economy(
    currency="钱", unit="文",
    prices=(PriceReference("一斗米", 30), PriceReference("一匹绢", 600)),
    daily_cost=40,
)


def test_the_daily_cost_belongs_to_the_economy() -> None:
    assert Economy.from_dict(_ECONOMY.to_dict()) == _ECONOMY
    assert _ECONOMY.charges_living and not Economy(currency="钱", unit="文").charges_living
    assert _ECONOMY.most_one_job_earns() == 600
    assert Economy(currency="钱", unit="文", daily_cost=40).most_one_job_earns() == 120


def test_midnights_are_counted_between_steps() -> None:
    six_hours = WorldTimeConfig(start_hour=6, seconds_per_step=6 * 3600)
    assert [days_crossed(s, six_hours) for s in range(1, 6)] == [0, 0, 1, 0, 0]
    two_days = WorldTimeConfig(start_hour=6, seconds_per_step=48 * 3600)
    assert days_crossed(1, two_days) == 2
    assert days_crossed(0, six_hours) == 0


def test_taking_money_takes_what_there_is() -> None:
    env = EnvironmentSystem(economy=_ECONOMY)
    env.add_money(50, holder_id="a")
    assert env.take_money("a", 40) == 40 and env.money_of("a") == 10
    assert env.take_money("a", 40) == 10 and env.money_of("a") == 0
    assert env.purse_of("a") is None
    assert env.take_money("a", 40) == 0


def _body(agent_id: str):  # noqa: ANN202
    calls: list[dict] = []

    async def go_short(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)

    return SimpleNamespace(agent_id=agent_id, is_active=True, go_short=go_short, calls=calls)


@pytest.mark.asyncio
async def test_a_day_is_paid_from_the_purse_and_a_short_one_is_felt() -> None:
    env = EnvironmentSystem(economy=_ECONOMY)
    env.add_money(100, holder_id="rich")
    env.add_money(10, holder_id="poor")
    rich, poor, broke = _body("rich"), _body("poor"), _body("broke")
    living = LivingCosts(env, WorldTimeConfig(start_hour=6, seconds_per_step=6 * 3600))

    await living.settle([rich, poor, broke], step=2)       # no midnight yet
    assert env.money_of("rich") == 100 and not poor.calls

    await living.settle([rich, poor, broke], step=3)
    assert env.money_of("rich") == 60 and not rich.calls
    assert env.money_of("poor") == 0
    (short,) = poor.calls
    assert short["shortfall"] == pytest.approx(0.75)
    assert "40文钱" in short["fact"] and "10文钱" in short["fact"]
    assert broke.calls[0]["shortfall"] == 1.0 and "分文没有" in broke.calls[0]["fact"]


@pytest.mark.asyncio
async def test_a_world_without_living_costs_charges_nothing() -> None:
    env = EnvironmentSystem(economy=Economy(currency="钱", unit="文"))
    env.add_money(10, holder_id="a")
    body = _body("a")
    await LivingCosts(env, WorldTimeConfig(start_hour=6, seconds_per_step=24 * 3600)).settle([body], step=1)
    assert env.money_of("a") == 10 and not body.calls


def test_going_short_presses_on_the_physiological_need_alone() -> None:
    engine = NeedEngine()
    before = {NeedType.PHYSIOLOGICAL.value: 0.3, NeedType.SAFETY.value: 0.4}
    after = engine.go_short(before, shortfall=1.0)
    assert after[NeedType.PHYSIOLOGICAL.value] == pytest.approx(0.65)
    assert after[NeedType.SAFETY.value] == 0.4
    assert engine.go_short(before, shortfall=0.5)[NeedType.PHYSIOLOGICAL.value] == pytest.approx(0.475)


@pytest.mark.asyncio
async def test_a_character_who_goes_short_remembers_it(container, monkeypatch) -> None:
    agent = _make_agent(container, world_id="w", agent_id="agent-a", name="甲")
    recorded: list[dict] = []

    async def _record(**kwargs):  # noqa: ANN003, ANN202
        recorded.append(kwargs)
        return []

    monkeypatch.setattr(agent.memory_system, "record_event", _record)
    before = agent.personality.state.need_intensities.get(NeedType.PHYSIOLOGICAL.value, 0.3)

    await agent.go_short(shortfall=1.0, fact="该付这一日的吃住了，我身上分文没有，只能饿着肚子将就。", step=4)

    assert agent.personality.state.need_intensities[NeedType.PHYSIOLOGICAL.value] > before
    assert recorded[0]["raw_content"].startswith("该付这一日的吃住了")


async def _work(container, economy: Economy, earned: object) -> tuple[ActionResult, str]:
    provider = container.llm_router.get(LLMScene.AGENT_ACTION_NARRATION)
    provider.fixed_response = json.dumps({
        "fact": "我在铺里卖了一上午的胡饼，收了些钱。", "earned": earned, "updated_index": 0,
        "outcome": "甲在铺里卖了一上午的胡饼。", "observation": "甲在铺里忙着卖饼。", "why": "",
    }, ensure_ascii=False)
    agents = {"agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲")}
    executor = WorkExecutor(container.llm_router, _directory(agents), check_assessor=fixed_check(), economy=economy)
    env = _make_environment()
    state = await executor.start(
        _make_action(agent_id="agent-a", action_type=ActionType.WORK, description="开铺卖饼"), 1,
        agents=agents, environment=env, message_system=None,
    )
    (result,) = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
    return result, _joined(provider.call_history[-1])


@pytest.mark.asyncio
async def test_paid_work_brings_in_what_the_narrator_rules(container) -> None:
    result, prompt = await _work(container, _ECONOMY, 80)
    assert result.earned == 80
    assert '"earned": 整数（按文计' in prompt and "一匹绢约600文" in prompt


@pytest.mark.asyncio
async def test_one_job_never_brings_in_more_than_the_dearest_thing_on_the_list(container) -> None:
    result, _ = await _work(container, _ECONOMY, 99999)
    assert result.earned == 600


@pytest.mark.asyncio
async def test_work_in_a_world_without_money_earns_nothing_and_asks_nothing(container) -> None:
    result, prompt = await _work(container, Economy(), 80)
    assert result.earned == 0 and "earned" not in prompt


@pytest.mark.asyncio
async def test_earnings_land_in_the_workers_purse() -> None:
    env = EnvironmentSystem(economy=_ECONOMY)
    env.add_money(5, holder_id="agent-a")
    processor = ExecutionProcessor(
        executor_registry=ActionExecutorRegistry(), environment=env, message_system=None,  # type: ignore[arg-type]
        directory=_directory(),
    )
    result = ActionResult(
        action=AgentAction(agent_id="agent-a", step=1, action_type=ActionType.WORK, action_description="卖饼"),
        expected_outcome="", outcome="", succeeded=True, earned=80,
    )
    await processor.land_result_feedback(result, {}, 1)
    assert env.money_of("agent-a") == 85


def test_a_character_knows_where_money_comes_from_and_where_it_goes() -> None:
    from agent.decision import _ACTION_SPACE, DecisionEngine
    from tests.unit.test_decision import _make_packet, _make_personality

    system, _user, _facts = DecisionEngine(None, economy=_ECONOMY)._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )
    assert "外头的主顾会付钱" in system and "一日吃住约40文钱" in system and "一斗米约30文" in system
    plain, _u, _f = DecisionEngine(None)._build_decision_prompt(  # noqa: SLF001
        _make_personality(), _make_packet(), _ACTION_SPACE,
    )
    assert "钱的来路" not in plain
