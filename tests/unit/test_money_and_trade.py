"""Money as a divisible thing, and TRADE: piles split and merge, purses are private, a decision
binds a trade's terms, and a struck deal moves everything at once or nothing.
"""

from __future__ import annotations

import json

import pytest

from agent.decision import DecisionEngine
from core.interfaces.action import (
    KIND_MONEY, ActionTarget, ActionType, AgentAction, EntityStateChange, EntitySpawn, Ref, TradeTerms,
    Transfer, errand_from_dict, errand_to_dict, ErrandOrder,
)
from core.interfaces.economy import Economy, PriceReference
from core.interfaces.llm import LLMResponse
from core.interfaces.perception import PerceivedIdentity, PerceivedPresence
from core.interfaces.place import Place
from engine.directory import LiveWorldDirectory
from engine.environment import OTHERS_PURSE, EnvironmentSystem
from engine.executors.trade import TradeExecutor
from tests.unit.test_action_executors import _make_agent
from world.models import EntityPresence, WorldEntity, WorldEntityType

_SILVER = Economy(currency="银子", unit="两", prices=(PriceReference("一斗米", 1),))


def _env() -> EnvironmentSystem:
    env = EnvironmentSystem(economy=_SILVER)
    for pid in ("hall", "yard"):
        env.space.register_place(Place(place_id=pid, name=pid, description="", connections={}, is_public=True, capacity=9))
    env.place_agent(agent_id="a", location_id="hall")
    env.place_agent(agent_id="b", location_id="hall")
    return env


def _piles(env: EnvironmentSystem) -> dict[str, int]:
    return {
        (e.owner_id or f"@{e.location_id}"): e.quantity
        for e in env.all_live_entities() if e.is_money
    }


def _pile_of(env: EnvironmentSystem, holder: str) -> WorldEntity:
    return next(e for e in env.all_live_entities() if e.is_money and e.owner_id == holder)


# ---------------------------------------------------------------------------
# The economy
# ---------------------------------------------------------------------------


def test_amounts_and_prices_read_in_the_world_s_own_words() -> None:
    assert _SILVER.amount_text(37) == "37两银子"
    assert _SILVER.price_reference() == "一斗米约1两"
    assert Economy.from_dict(_SILVER.to_dict()) == _SILVER


@pytest.mark.parametrize("payload", [None, {}, {"currency": "", "prices": "x"}])
def test_an_unreadable_economy_is_a_world_without_money(payload: object) -> None:
    assert not Economy.from_dict(payload).has_money


def test_bad_price_lines_are_dropped() -> None:
    economy = Economy.from_dict({"currency": "钱", "prices": [{"item": "米", "price": "x"}, {"item": "", "price": 3}, {"item": "盐", "price": 2}]})
    assert economy.prices == (PriceReference("盐", 2),)


# ---------------------------------------------------------------------------
# Piles
# ---------------------------------------------------------------------------


def test_each_holder_keeps_one_pile() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    env.add_money(12, holder_id="a")
    assert _piles(env) == {"a": 42}
    assert _pile_of(env, "a").description == "42两银子"
    assert env.money_of("a") == 42 and env.money_of("b") == 0


def test_moving_part_of_a_pile_splits_it_and_merges_at_the_other_end() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    env.add_money(5, holder_id="b")
    env.change_entity_state(EntityStateChange(entity_id=_pile_of(env, "a").entity_id, owner_id="b", quantity=10))
    assert _piles(env) == {"a": 20, "b": 15}


def test_moving_a_whole_pile_onto_the_ground_moves_it() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    pile = _pile_of(env, "a")
    env.change_entity_state(EntityStateChange(entity_id=pile.entity_id, location_id="hall"))
    assert pile.location_id == "hall" and pile.owner_id is None
    assert _piles(env) == {"@hall": 30}


def test_an_emptied_pile_leaves_the_world() -> None:
    env = _env()
    env.add_money(8, holder_id="a")
    env.add_money(2, holder_id="b")
    emptied = _pile_of(env, "a").entity_id
    env.change_entity_state(EntityStateChange(entity_id=emptied, owner_id="b", quantity=50))
    assert _piles(env) == {"b": 10}
    assert env.get_entity(emptied) is None


def test_money_is_never_made_from_nothing() -> None:
    env = _env()
    assert not env.spawn_entity(EntitySpawn(name="银子", entity_type="money"), ground="hall")
    assert WorldEntityType.MONEY.is_takeable


def test_only_the_holder_sees_how_much_is_in_a_purse() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    env.add_money(4, location_id="hall")

    def seen_by(viewer: str) -> dict[str | None, str]:
        return {e.holder_id: e.description for e in env.spatial_for(agent_id=viewer).visible_entities}

    assert seen_by("a")["a"] == "30两银子"
    assert seen_by("b")["a"] == OTHERS_PURSE
    assert seen_by("b")[None] == "4两银子"   # money on the ground can be counted


def test_piles_survive_a_snapshot_round_trip() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    restored = EnvironmentSystem(economy=_SILVER)
    restored.restore_state(env.snapshot_state())
    assert restored.money_of("a") == 30


# ---------------------------------------------------------------------------
# An exchange lands whole or not at all
# ---------------------------------------------------------------------------


def _with_ring(env: EnvironmentSystem, holder: str = "b") -> str:
    env.register_entity(WorldEntity(
        "ring", "玉环", WorldEntityType.ITEM, presence=EntityPresence.HELD, presence_ref=holder, is_takeable=True,
    ))
    return "ring"


def test_an_exchange_moves_both_ways() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    ring = _with_ring(env)
    assert env.transfer_all([Transfer(from_id="a", to_id="b", quantity=20), Transfer(from_id="b", to_id="a", entity_id=ring)])
    assert env.get_entity(ring).owner_id == "a"
    assert _piles(env) == {"a": 10, "b": 20}


def test_an_exchange_whose_goods_moved_lands_nothing() -> None:
    env = _env()
    env.add_money(30, holder_id="a")
    ring = _with_ring(env, holder="a")   # no longer in b's hands
    assert not env.transfer_all([Transfer(from_id="a", to_id="b", quantity=20), Transfer(from_id="b", to_id="a", entity_id=ring)])
    assert _piles(env) == {"a": 30}


def test_an_exchange_beyond_the_purse_lands_nothing() -> None:
    env = _env()
    env.add_money(5, holder_id="a")
    ring = _with_ring(env)
    assert not env.transfer_all([Transfer(from_id="a", to_id="b", quantity=20), Transfer(from_id="b", to_id="a", entity_id=ring)])
    assert env.get_entity(ring).owner_id == "b"


def test_an_errand_s_amount_survives_serialization() -> None:
    order = ErrandOrder(npc_id="n", destination_id="yard", item_id="money-0-1", amount=7)
    assert errand_from_dict(errand_to_dict(order)) == order
    assert errand_from_dict({"npc_id": "n", "destination_id": "y", "amount": "x"}).amount == 0


# ---------------------------------------------------------------------------
# The decision binds a trade's terms
# ---------------------------------------------------------------------------


def _packet(env: EnvironmentSystem) -> object:
    from types import SimpleNamespace

    spatial = env.spatial_for(agent_id="a")
    spatial.visible_agents["b"] = PerceivedPresence(identity=PerceivedIdentity(name="乙", gender="男"))
    return SimpleNamespace(
        agent_id="a", spatial=spatial, inbox=[],
        internal_context=SimpleNamespace(relevant_relations=[]),
    )


def _parse(env: EnvironmentSystem, reply: dict) -> object:
    from agent.decision import _ACTION_SPACE

    trade_index = next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.TRADE)
    payload = {"selected_index": trade_index, "action_description": "拿银子换他的玉环", **reply}
    return DecisionEngine.__new__(DecisionEngine)._parse_llm_selection(payload, _ACTION_SPACE, _packet(env))


def _listed(env: EnvironmentSystem, entity_id: str) -> int:
    return [e.entity_id for e in env.spatial_for(agent_id="a").visible_entities].index(entity_id) + 1


@pytest.fixture()
def market() -> EnvironmentSystem:
    env = _env()
    env.add_money(30, holder_id="a")
    _with_ring(env)
    env.register_entity(WorldEntity(
        "fan", "折扇", WorldEntityType.ITEM, presence=EntityPresence.HELD, presence_ref="a", is_takeable=True,
    ))
    return env


def _person_index(env: EnvironmentSystem) -> int:
    from agent.decision import _message_roster

    return [aid for aid, _ in _message_roster(_packet(env))].index("b") + 1


def test_a_trade_binds_its_counterpart_and_terms(market: EnvironmentSystem) -> None:
    selection = _parse(market, {
        "trade_person_index": _person_index(market),
        "trade_give_money": 20, "trade_ask_item_index": _listed(market, "ring"),
    })
    assert selection is not None
    assert selection.trade == TradeTerms(counterpart_id="b", give_amount=20, ask_entity_id="ring")
    assert selection.target == ActionTarget(acts_on=[Ref.agent("b")], claims=[Ref.agent("b")])


@pytest.mark.parametrize(("reply", "why"), [
    ({}, "nothing on either side"),
    ({"trade_give_item_index": "ring"}, "a thing that isn't mine to give"),
    ({"trade_ask_item_index": "fan"}, "a thing that isn't his to give up"),
])
def test_a_trade_that_names_nothing_tradeable_is_rejected(market: EnvironmentSystem, reply: dict, why: str) -> None:
    resolved = {k: _listed(market, v) if isinstance(v, str) else v for k, v in reply.items()}
    assert _parse(market, {"trade_person_index": _person_index(market), **resolved}) is None, why


def test_a_purse_picked_as_the_thing_reads_as_unfilled(market: EnvironmentSystem) -> None:
    purse = _pile_of(market, "a").entity_id
    selection = _parse(market, {
        "trade_person_index": _person_index(market),
        "trade_give_item_index": _listed(market, purse), "trade_give_money": 5,
    })
    assert selection.trade == TradeTerms(counterpart_id="b", give_amount=5)


def test_physical_on_money_carries_its_amount(market: EnvironmentSystem) -> None:
    from agent.decision import _ACTION_SPACE

    physical = next(i for i, c in enumerate(_ACTION_SPACE) if c.action_type is ActionType.PHYSICAL)
    purse = _pile_of(market, "a").entity_id
    selection = DecisionEngine.__new__(DecisionEngine)._parse_llm_selection({
        "selected_index": physical, "action_description": "把银子丢在地上",
        "physical_entity_index": _listed(market, purse), "physical_amount": 9,
    }, _ACTION_SPACE, _packet(market))
    assert selection.amount == 9
    assert selection.target.acts_on == [Ref.entity(purse, KIND_MONEY)]


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------


class _Router:
    """The haggle gets ``deal``; each side's memory gets a first-person line."""

    def __init__(self, deal: dict | None) -> None:
        self._deal = deal
        self.haggles: list[str] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        system, user = messages[0].content, messages[-1].content
        if "讨价还价" in system:
            self.haggles.append(user)
            if self._deal is None:
                raise RuntimeError("llm down")
            reply = self._deal
        else:
            reply = {"fact": "我和对方谈了一笔买卖。", "success": True, "why": "", "relation": "positive"}
        return LLMResponse(content=json.dumps(reply, ensure_ascii=False), input_tokens=0, output_tokens=0, model="stub")


_STRUCK = {
    "dialogue": [{"speaker": 1, "line": "这玉环二十两卖不卖？"}, {"speaker": 2, "line": "二十五两，少一文不卖。"}],
    "reason": "乙缺钱，价钱也算公道", "agreed": True, "give_money": 25, "ask_money": 0, "observation": "两人拱手作别",
}


async def _trade(container, router: _Router, market: EnvironmentSystem, terms: TradeTerms):  # noqa: ANN001, ANN202
    agents = {
        "a": _make_agent(container, world_id="w", agent_id="a", name="甲"),
        "b": _make_agent(container, world_id="w", agent_id="b", name="乙"),
    }
    executor = TradeExecutor(router, LiveWorldDirectory.from_agents(agents, market), economy=_SILVER)
    action = AgentAction(
        agent_id="a", step=1, action_type=ActionType.TRADE, action_description="拿银子换他的玉环",
        target=ActionTarget(acts_on=[Ref.agent("b")], claims=[Ref.agent("b")]),
    )
    action.intent = type("Intent", (), {"trade": terms})()
    state = await executor.start(action, 1, agents=agents, environment=market, message_system=None)
    return await executor.complete(state, 1, agents=agents, environment=market, message_system=None)


_TERMS = TradeTerms(counterpart_id="b", give_amount=20, ask_entity_id="ring")


async def test_a_struck_deal_moves_both_ways_on_one_result(container, market: EnvironmentSystem) -> None:
    router = _Router(_STRUCK)
    initiator, counterpart = await _trade(container, router, market, _TERMS)
    assert initiator.succeeded and counterpart.succeeded
    assert initiator.transfers == [
        Transfer(from_id="a", to_id="b", quantity=25), Transfer(from_id="b", to_id="a", entity_id="ring"),
    ]
    assert counterpart.transfers == []
    assert initiator.relation_updates[0][0] == "b" and counterpart.relation_updates[0][0] == "a"
    assert "25两银子" in initiator.outcome
    # Bystanders see goods change hands, never the sum.
    assert initiator.observations and all("银子" not in o.text for o in initiator.observations)
    assert "30两银子" in router.haggles[0] and "【行情】一斗米约1两" in router.haggles[0]


async def test_a_price_beyond_the_purse_is_cut_to_what_he_has(container, market: EnvironmentSystem) -> None:
    initiator, _ = await _trade(container, _Router({**_STRUCK, "give_money": 99}), market, _TERMS)
    assert initiator.transfers[0] == Transfer(from_id="a", to_id="b", quantity=30)


async def test_a_refused_deal_moves_nothing(container, market: EnvironmentSystem) -> None:
    initiator, counterpart = await _trade(container, _Router({**_STRUCK, "agreed": False}), market, _TERMS)
    assert not initiator.succeeded and initiator.transfers == []
    assert initiator.failure_reason == "乙缺钱，价钱也算公道"
    assert initiator.factual_memory   # a refusal is still something that happened


_PAYMENT = TradeTerms(counterpart_id="b", give_amount=20)
_DEMAND = TradeTerms(counterpart_id="b", ask_entity_id="ring")


async def test_a_payment_moves_one_way_and_reads_as_handing_over(container, market: EnvironmentSystem) -> None:
    """Repaying, lending or bribing: A only gives, and what it buys is said, not handed over."""
    paid = {**_STRUCK, "give_money": 20}
    initiator, counterpart = await _trade(container, _Router(paid), market, _PAYMENT)
    assert initiator.transfers == [Transfer(from_id="a", to_id="b", quantity=20)]
    assert initiator.succeeded and counterpart.succeeded
    assert initiator.gist.endswith("甲把20两银子交给了乙。")
    assert "甲要把20两银子交给乙：" in initiator.outcome
    # Onlookers see something handed over, never how much.
    (seen,) = initiator.observations
    assert "交给了乙" in seen.text and "银子" not in seen.text


async def test_a_refused_payment_moves_nothing(container, market: EnvironmentSystem) -> None:
    initiator, _ = await _trade(container, _Router({**_STRUCK, "agreed": False, "reason": ""}), market, _PAYMENT)
    assert initiator.transfers == [] and not initiator.succeeded
    assert initiator.gist.endswith("甲要把20两银子交给乙，乙没有收。")
    assert initiator.failure_reason == "乙没有收。"


@pytest.mark.parametrize(("agreed", "gist"), [(True, "乙把玉环交给了甲。"), (False, "甲向乙讨要玉环，乙没有给。")])
async def test_a_demand_moves_only_what_he_hands_over(
    container, market: EnvironmentSystem, agreed: bool, gist: str,
) -> None:
    """Collecting a debt or demanding: A only asks, and B may pay up or refuse."""
    initiator, _ = await _trade(container, _Router({**_STRUCK, "agreed": agreed, "give_money": 0}), market, _DEMAND)
    assert initiator.transfers == ([Transfer(from_id="b", to_id="a", entity_id="ring")] if agreed else [])
    assert initiator.gist.endswith(gist)


def test_a_talk_never_hands_money_or_things_over() -> None:
    from engine.executors.social import _DIALOGUE_SYSTEM

    assert "钱物不换手" in _DIALOGUE_SYSTEM


async def test_haggling_that_never_happened_is_a_null_step(container, market: EnvironmentSystem) -> None:
    results = await _trade(container, _Router(None), market, _TERMS)
    assert [r.adjudication_failed for r in results] == [True, True]
    assert all(not r.transfers for r in results)


async def test_a_trade_for_goods_no_longer_there_never_opens(container, market: EnvironmentSystem) -> None:
    market.change_entity_state(EntityStateChange(entity_id="ring", owner_id="a"))
    (result,) = await _trade(container, _Router(_STRUCK), market, _TERMS)
    assert not result.succeeded and "玉环已不在乙手上" in result.failure_reason
    assert result.action.action_type is ActionType.TRADE


# ---------------------------------------------------------------------------
# Money in a physical act
# ---------------------------------------------------------------------------


class _JudgeRouter:
    def __init__(self, amount_moved: object) -> None:
        self._amount = amount_moved
        self.prompts: list[str] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.prompts.append("\n".join(m.content for m in messages))
        reply = {
            "reason": "趁他不备", "deed": "seize", "outcome": "甲从乙怀里摸走了些银子", "fact": "我摸走了乙的银子",
            "why": "", "relation": "negative", "new_entity_state": "", "amount_moved": self._amount,
            "actor_damage": 0.0,
        }
        return LLMResponse(content=json.dumps(reply, ensure_ascii=False), input_tokens=0, output_tokens=0, model="stub")


async def _snatch(container, amount_moved: object, *, asked: int | None, tier=None, victim_aware: bool = False):  # noqa: ANN001, ANN202
    from core.interfaces.check import CheckTier
    from engine.executors.physical import PhysicalExecutor
    from tests.unit.check_fixtures import fixed_check

    env = _env()
    env.add_money(30, holder_id="b")
    purse = env.purse_of("b").entity_id
    agents = {"a": _make_agent(container, world_id="w", agent_id="a", name="甲")}
    if victim_aware:
        agents["b"] = _make_agent(container, world_id="w", agent_id="b", name="乙")
    router = _JudgeRouter(amount_moved)
    executor = PhysicalExecutor(
        router, LiveWorldDirectory.from_agents(agents, env), check_assessor=fixed_check(tier or CheckTier.SUCCESS),
    )
    action = AgentAction(
        agent_id="a", step=1, action_type=ActionType.PHYSICAL, action_description="摸走他的银子",
        target=ActionTarget(acts_on=[Ref.entity(purse, KIND_MONEY)]),
    )
    action.intent = type("Intent", (), {"amount": asked})()
    state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
    (result,) = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
    return result, router


async def test_the_judge_rules_how_much_was_taken_within_the_stake(container) -> None:
    result, router = await _snatch(container, 7, asked=10)
    assert [c.quantity for c in result.entity_state_changes] == [7]
    assert "涉及钱数：10两银子" in router.prompts[-1]


@pytest.mark.parametrize(("ruled", "taken"), [(99, 10), ("??", 10), (0, 10)])
async def test_a_sum_beyond_or_missing_from_the_ruling_is_the_stake(container, ruled: object, taken: int) -> None:
    result, _ = await _snatch(container, ruled, asked=10)
    assert [c.quantity for c in result.entity_state_changes] == [taken]


async def test_an_unnamed_sum_puts_the_whole_purse_at_stake(container) -> None:
    result, router = await _snatch(container, 30, asked=None)
    assert [c.quantity for c in result.entity_state_changes] == [30]
    assert "涉及钱数：30两银子" in router.prompts[-1]


async def test_a_failed_snatch_moves_no_money(container) -> None:
    from core.interfaces.check import CheckTier

    result, _ = await _snatch(container, 7, asked=10, tier=CheckTier.FAILURE)
    assert result.entity_state_changes == []


# ---------------------------------------------------------------------------
# The build hands out cash
# ---------------------------------------------------------------------------


async def test_a_built_world_hands_each_character_his_cash(mock_build_container, test_config) -> None:
    from engine.application import NarrativeApplication
    from world.initializer import WorldInitializer

    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("玄武门之变", template="changan_iso")
    assert world.analysis.economy.currency == "铜钱"
    # The stub rates three people at 500, 800 and nothing: nobody holds an empty pile.
    assert sorted(world.environment.money_of(aid) for aid in world.agents) == [0, 500, 800]
    assert sum(e.is_money for e in world.environment.all_live_entities()) == 2
    restored = await WorldInitializer(mock_build_container).restore(world.world_id)
    assert sorted(restored.environment.money_of(aid) for aid in world.agents) == [0, 500, 800]


async def test_onlookers_see_money_change_hands_but_the_victim_knows_how_much(container) -> None:
    result, router = await _snatch(container, 7, asked=10, victim_aware=True)
    assert "甲拿起了一些银子" in result.entity_state_changes[0].perception
    # The victim's reaction is told the sum he lost, not just the currency.
    assert any("夺走7两银子" in prompt for prompt in router.prompts)
