"""Abilities: the contract, how a check counts them, how the build rates them, and where they show.

Numbers stay in the code layer: a check adds up levels, a prompt only ever reads level words.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from agent.personality import EmotionState, PersonalityLayer, SoulLayer
from core.interfaces.ability import Abilities, Attribute, Skill
from core.interfaces.action import ActionType
from core.interfaces.check import AbilityBasis, CheckRecord, Difficulty, Edge
from core.interfaces.economy import Economy, PriceReference
from core.interfaces.llm import LLMResponse
from engine.check import CheckAssessor
from world.builders.ability_designer import AbilityDesigner
from world.models import AgentDefinition, AgentTier, EntityPresence, WorldEntitySeed

_NIMBLE = Abilities(
    attributes={Attribute.AGILITY: 4, Attribute.PHYSIQUE: 2},
    skills=(Skill("潜行", 2), Skill("开锁", 1)),
)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_an_unset_character_is_an_ordinary_adult() -> None:
    plain = Abilities()
    assert all(plain.level(a) == 3 for a in Attribute)
    assert plain.skills == ()
    assert plain.persona_line() == ""


def test_levels_are_clamped_and_skills_capped() -> None:
    abilities = Abilities(
        attributes={Attribute.NERVE: 9, Attribute.INTELLECT: 0},
        skills=tuple(Skill(f"技{i}", 2) for i in range(6)),
    )
    assert abilities.level(Attribute.NERVE) == 5
    assert abilities.level(Attribute.INTELLECT) == 1
    assert len(abilities.skills) == 4


def test_abilities_round_trip_through_their_dict() -> None:
    assert Abilities.from_dict(_NIMBLE.to_dict()) == _NIMBLE


@pytest.mark.parametrize("payload", [None, "x", {}, {"attributes": "?", "skills": "?"}])
def test_anything_unreadable_reads_as_the_baseline(payload: object) -> None:
    assert Abilities.from_dict(payload) == Abilities()


def test_the_persona_line_is_words_only() -> None:
    line = _NIMBLE.persona_line()
    assert line == "所长：潜行精熟、开锁略通、身手出众；所短：体魄平平"
    assert not re.search(r"\d", line)


def test_the_persona_carries_the_bearers_own_abilities() -> None:
    soul = SoulLayer(name="燕七", gender="男", abilities=_NIMBLE)
    assert "所长：潜行精熟" in PersonalityLayer(soul).to_prompt_context()
    assert "所长" not in PersonalityLayer(SoulLayer(name="路人")).to_prompt_context()


# ---------------------------------------------------------------------------
# What a check counts
# ---------------------------------------------------------------------------


def test_the_modifier_adds_attribute_and_skill_less_the_opponent_plus_the_edge() -> None:
    basis = AbilityBasis(
        attribute=Attribute.AGILITY, attribute_level=4, skill="潜行", skill_level=2,
        opposing_attribute=Attribute.INTELLECT, opposing_level=5,
    )
    assert basis.modifier == (4 - 3) + 2 - (5 - 3)
    record = CheckRecord(
        difficulty=Difficulty.NORMAL, edge=Edge.BETTER, pure_chance=False, dice=(3, 3, 3), basis=basis,
    )
    assert record.modifier == 2
    assert CheckRecord.from_dict(record.to_dict()) == record


def test_unnamed_abilities_count_nothing_whatever_their_level() -> None:
    assert AbilityBasis(attribute_level=5, opposing_level=1).modifier == 0


def test_pure_chance_ignores_the_abilities() -> None:
    record = CheckRecord(
        difficulty=Difficulty.NORMAL, edge=Edge.EVEN, pure_chance=True, dice=(3, 3, 3),
        basis=AbilityBasis(attribute=Attribute.NERVE, attribute_level=5),
    )
    assert record.modifier == 0


class _Router:
    def __init__(self, reply: dict) -> None:
        self._reply = json.dumps(reply, ensure_ascii=False)
        self.calls: list[list] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.calls.append(messages)
        return LLMResponse(content=self._reply, input_tokens=0, output_tokens=0, model="stub")

    def prompt(self) -> str:
        return "\n".join(m.content for m in self.calls[-1])


def _person(name: str, abilities: Abilities, agent_id: str = "a") -> SimpleNamespace:
    soul = SoulLayer(name=name, agent_id=agent_id, gender="男", abilities=abilities)
    return SimpleNamespace(agent_id=agent_id, personality=PersonalityLayer(soul))


_GUARD = Abilities(attributes={Attribute.INTELLECT: 5})


async def _assess(reply: dict, *, opponent: SimpleNamespace | None = None) -> tuple:
    router = _Router({"reason": "r", "certainty": "uncertain", "difficulty": "normal", **reply})
    verdict = await CheckAssessor(router, seed="w").assess(
        _person("燕七", _NIMBLE), action_type=ActionType.PHYSICAL, intent="摸走他腰间的钥匙",
        scene="- 在场的其他人：守卫", step=1, opponent=opponent,
    )
    assert verdict is not None and verdict.record is not None
    return verdict.record, router


async def test_the_assessors_indices_become_the_levels_behind_the_roll() -> None:
    record, _ = await _assess(
        {"attribute_index": 2, "skill_index": 1, "opposing_attribute_index": 3, "edge": "better"},
        opponent=_person("守卫", _GUARD, "g"),
    )
    assert record.basis == AbilityBasis(
        attribute=Attribute.AGILITY, attribute_level=4, skill="潜行", skill_level=2,
        opposing_attribute=Attribute.INTELLECT, opposing_level=5,
    )
    assert record.modifier == 1 + 2 - 2 + 1


@pytest.mark.parametrize("bad", [0, 9, "x", None])
async def test_an_unfilled_or_impossible_index_counts_nothing(bad: object) -> None:
    record, _ = await _assess(
        {"attribute_index": bad, "skill_index": bad, "opposing_attribute_index": bad, "edge": "even"},
        opponent=_person("守卫", _GUARD, "g"),
    )
    assert record.basis == AbilityBasis()


async def test_without_an_opponent_nobody_resists() -> None:
    record, router = await _assess({"attribute_index": 2, "opposing_attribute_index": 3, "edge": "even"})
    assert record.basis.opposing_attribute is None
    assert "【对手的能力】" not in router.prompt()


async def test_the_assessor_reads_level_words_never_level_numbers() -> None:
    _, router = await _assess({"attribute_index": 2}, opponent=_person("守卫", _GUARD, "g"))
    prompt = router.prompt()
    assert "2. 身手：出众" in prompt and "1. 潜行：精熟" in prompt
    assert "【对手的能力】" in prompt and "3. 心智：超群" in prompt
    assert '"attribute_level"' not in prompt and "level" not in prompt


# ---------------------------------------------------------------------------
# The build rates the whole cast at once
# ---------------------------------------------------------------------------


def _definition(agent_id: str, name: str) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id, name=name, tier=AgentTier.MAIN,
        soul=SoulLayer(name=name, agent_id=agent_id, gender="男", role="镖师", background="走镖十年"),
        initial_location="yard", initial_emotion=EmotionState(),
    )


def _analysis(economy: Economy | None = None, seeds: tuple[WorldEntitySeed, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        world_name="镖局", era_description="乱世", economy=economy or Economy(), world_entity_seeds=list(seeds),
    )


class _SeqRouter:
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

    monkeypatch.setattr("world.builders.ability_designer.asyncio.sleep", _skip)


def _cast_reply(*entries: dict) -> str:
    return json.dumps({"reason": "r", "cast": list(entries)}, ensure_ascii=False)


async def test_the_designer_rates_everyone_on_the_prompts_labels() -> None:
    reply = _cast_reply(
        {"index": 2, "basis": "b", "attributes": {"体魄": 5, "口才": 1, "未知": 4}, "skills": [{"name": "刀法", "level": 3}]},
        {"index": 1, "basis": "b", "attributes": {"心智": 4}, "skills": []},
    )
    rated = await AbilityDesigner(_SeqRouter(reply)).design(
        _analysis(), [_definition("a1", "甲"), _definition("a2", "乙")],
    )
    assert [d.agent_id for d in rated] == ["a1", "a2"]
    assert rated[0].soul.abilities == Abilities(attributes={Attribute.INTELLECT: 4})
    assert rated[1].soul.abilities == Abilities(
        attributes={Attribute.PHYSIQUE: 5, Attribute.ELOQUENCE: 1}, skills=(Skill("刀法", 3),),
    )
    # Everything else on the soul survives.
    assert rated[1].soul.background == "走镖十年"


async def test_leaving_someone_unrated_retries_then_raises() -> None:
    partial = _cast_reply({"index": 1, "attributes": {}, "skills": []})
    router = _SeqRouter(partial, partial)
    with pytest.raises(ValueError, match="AbilityDesigner failed"):
        await AbilityDesigner(router).design(_analysis(), [_definition("a1", "甲"), _definition("a2", "乙")])
    assert len(router.calls) == 2


async def test_a_retry_that_answers_is_taken() -> None:
    whole = _cast_reply({"index": 1, "attributes": {"胆识": 2}, "skills": []})
    rated = await AbilityDesigner(_SeqRouter("不是 JSON", whole)).design(_analysis(), [_definition("a1", "甲")])
    assert rated[0].soul.abilities.level(Attribute.NERVE) == 2


async def test_the_designer_prompt_names_people_not_ids() -> None:
    router = _SeqRouter(_cast_reply({"index": 1, "attributes": {}, "skills": []}))
    await AbilityDesigner(router).design(_analysis(), [_definition("agent-zz9", "甲")])
    prompt = "\n".join(m.content for m in router.calls[0])
    assert "agent-zz9" not in prompt
    assert "#1 甲，男" in prompt


# ---------------------------------------------------------------------------
# The build puts each character's own things in his hands
# ---------------------------------------------------------------------------


def _seed(name: str, entity_type: str = "item") -> WorldEntitySeed:
    return WorldEntitySeed(name=name, entity_type=entity_type, description=f"{name}的样子", location_name="西市")


async def test_the_designer_puts_a_characters_own_things_in_his_hands() -> None:
    qin, wall = _seed("古琴"), _seed("夹墙", "landmark")
    router = _SeqRouter(_cast_reply(
        {"index": 1, "attributes": {}, "skills": [], "carries": [1]},
        {"index": 2, "attributes": {}, "skills": [], "carries": [{"name": "古琴"}]},
    ))
    owner = _definition("a1", "甲")
    owner.metadata["brief"] = "抱着古琴来西市寻买主"
    rated = await AbilityDesigner(router).design(_analysis(seeds=(qin, wall)), [owner, _definition("a2", "乙")])
    # Only indices bind: a thing named instead of numbered binds nothing.
    assert [d.carried for d in rated] == [(qin.entity_id,), ()]
    prompt = router.calls[0][1].content
    assert "#1 古琴：古琴的样子（在西市）" in prompt
    # The opening sketch is how the designer knows what someone has on him.
    assert "开场：抱着古琴来西市寻买主" in prompt
    # A landmark never changes hands, so it is never offered.
    assert "夹墙" not in prompt


async def test_a_thing_claimed_twice_stays_with_the_first_in_cast_order() -> None:
    qin = _seed("古琴")
    rated = await AbilityDesigner(_SeqRouter(_cast_reply(
        {"index": 2, "attributes": {}, "skills": [], "carries": [1, 7, "x"]},
        {"index": 1, "attributes": {}, "skills": [], "carries": [1]},
    ))).design(_analysis(seeds=(qin,)), [_definition("a1", "甲"), _definition("a2", "乙")])
    assert [d.carried for d in rated] == [(qin.entity_id,), ()]


async def test_a_world_without_things_to_carry_says_so() -> None:
    router = _SeqRouter(_cast_reply({"index": 1, "attributes": {}, "skills": []}))
    rated = await AbilityDesigner(router).design(_analysis(), [_definition("a1", "甲")])
    assert rated[0].carried == ()
    assert "没有可随身的物件" in router.calls[0][1].content


async def test_a_built_world_starts_each_thing_in_its_owners_hands(mock_build_container, test_config) -> None:
    from engine.application import NarrativeApplication
    from world.initializer import WorldInitializer

    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("玄武门之变", template="changan_iso")
    seal = WorldEntitySeed(name="传国玉玺", entity_type="item").entity_id
    # The stub gives the first character the seal; the landmark stays where it stands.
    held = world.environment.get_entity(seal)
    assert held.presence is EntityPresence.HELD and held.presence_ref in world.agents
    restored = await WorldInitializer(mock_build_container).restore(world.world_id)
    assert restored.environment.get_entity(seal).presence_ref == held.presence_ref


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_definition_keeps_its_abilities_through_a_round_trip() -> None:
    definition = _definition("a1", "甲")
    definition.soul = SoulLayer(name="甲", agent_id="a1", abilities=_NIMBLE)
    assert AgentDefinition.from_dict(definition.as_dict()).soul.abilities == _NIMBLE


def test_a_definition_keeps_what_it_carries_through_a_round_trip() -> None:
    definition = _definition("a1", "甲")
    definition.carried = ("seed_qin",)
    assert AgentDefinition.from_dict(definition.as_dict()).carried == ("seed_qin",)


def test_a_world_saved_without_abilities_loads_as_ordinary_people() -> None:
    payload = _definition("a1", "甲").as_dict()
    del payload["soul_data"]["abilities"]
    assert AgentDefinition.from_dict(payload).soul.abilities == Abilities()


# ---------------------------------------------------------------------------
# Who resists a physical act
# ---------------------------------------------------------------------------


def _subject(**overrides: object) -> object:
    from engine.executors.physical import _Subject

    fields: dict[str, object] = dict(
        actor_id="thief", item_type="item", target_id="key", target_block="",
        item=None, target_agent=None, recipient_id="", recipient_agent=None,
    )
    fields.update(overrides)
    return _Subject(**fields)


def _opponent(subject: object, *, holder_here: bool = True) -> object:
    from engine.executors.physical import PhysicalExecutor

    guard = SimpleNamespace(agent_id="guard", is_active=True)
    places = {"thief": "gate", "guard": "gate" if holder_here else "hall"}
    environment = SimpleNamespace(get_body_location=places.get)
    return PhysicalExecutor._opponent(subject, agents={"guard": guard}, environment=environment)


def _key(holder: str | None) -> object:
    from world.models import EntityPresence, WorldEntity, WorldEntityType

    if holder is None:
        return WorldEntity("key", "钥匙", WorldEntityType.ITEM, presence_ref="gate", is_takeable=True)
    return WorldEntity(
        "key", "钥匙", WorldEntityType.ITEM, presence=EntityPresence.HELD, presence_ref=holder, is_takeable=True,
    )


def test_the_person_acted_on_resists() -> None:
    guard = SimpleNamespace(agent_id="guard")
    assert _opponent(_subject(item_type="agent", target_id="guard", target_agent=guard)) is guard


def test_whoever_holds_the_thing_resists_when_present() -> None:
    assert _opponent(_subject(item=_key("guard"))).agent_id == "guard"
    assert _opponent(_subject(item=_key("guard")), holder_here=False) is None


def test_nobody_resists_a_thing_on_the_ground_or_in_his_own_hands() -> None:
    assert _opponent(_subject(item=_key(None))) is None
    assert _opponent(_subject(item=_key("thief"))) is None


async def test_the_designer_hands_out_cash_on_the_world_s_scale() -> None:
    silver = Economy(currency="银子", unit="两", prices=(PriceReference("一匹马", 20),))
    router = _SeqRouter(_cast_reply({"index": 1, "attributes": {}, "skills": [], "cash": 35}, {"index": 2, "cash": -5}))
    rated = await AbilityDesigner(router).design(_analysis(silver), [_definition("a1", "甲"), _definition("a2", "乙")])
    assert [d.initial_cash for d in rated] == [35, 0]
    prompt = "\n".join(m.content for m in router.calls[0])
    assert "银子，按两计；参考物价：一匹马约20两" in prompt


async def test_a_world_without_money_hands_out_none() -> None:
    rated = await AbilityDesigner(_SeqRouter(_cast_reply({"index": 1, "cash": 99}))).design(_analysis(), [_definition("a1", "甲")])
    assert rated[0].initial_cash == 0


def test_cash_survives_a_round_trip() -> None:
    definition = _definition("a1", "甲")
    definition.initial_cash = 42
    assert AgentDefinition.from_dict(definition.as_dict()).initial_cash == 42
