"""Unseen hands: a PHYSICAL act on a thing done behind people's backs (stealing, planting).

The thing changes hands as in the open; who learns of it follows the roll. Unseen, nobody is told
and the victim finds out only when he looks; seen, it is an act in the open after all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.interfaces.action import ActionTarget, ActionType, AgentAction, Deed, Ref
from core.interfaces.check import CheckTier
from engine.executors.base import ActionExecutionState
from engine.executors.physical import PhysicalExecutor, _Verdict
from tests.unit.check_fixtures import fixed_check
from tests.unit.test_action_executors import _directory, _make_agent, _make_environment
from world.models import EntityPresence, WorldEntity, WorldEntityType


def _verdict(deed: Deed, success: bool = True) -> _Verdict:
    return _Verdict(
        failure_reason="" if success else "手一滑", success=success, outcome="甲从乙腰间顺走了虎符",
        fact="我从乙腰间顺走了虎符", relation_dir="negative", actor_damage=0.0, target_damage=0.0,
        target_relief=0.0, new_entity_state="", deed=deed,
    )


async def _act(
    container, monkeypatch, *, tier: CheckTier, holder: str = "agent-b", recipient: str | None = None,
    deed: Deed = Deed.SEIZE, covert: bool = True, b_elsewhere: bool = False,
):  # noqa: ANN001, ANN202
    seen: dict = {}

    async def _judge(self, agent, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.update(kwargs)
        return _verdict(deed, success=tier.succeeded)

    monkeypatch.setattr(PhysicalExecutor, "_judge", _judge)
    agents = {
        "agent-a": _make_agent(container, world_id="w", agent_id="agent-a", name="甲"),
        "agent-b": _make_agent(container, world_id="w", agent_id="agent-b", name="乙"),
    }
    assessor = fixed_check(tier)
    executor = PhysicalExecutor(container.llm_router, _directory(agents), check_assessor=assessor)
    env = _make_environment()
    if b_elsewhere:
        env.place_agent(agent_id="agent-b", location_id="garden")
    env.register_entity(WorldEntity(
        entity_id="tally", name="虎符", entity_type=WorldEntityType.ITEM,
        presence=EntityPresence.HELD if holder else EntityPresence.AT_LOCATION,
        presence_ref=holder or "hall", is_takeable=True, state="intact",
    ))
    action = AgentAction(
        agent_id="agent-a", step=1, action_type=ActionType.PHYSICAL, action_description="趁乱顺走他的虎符",
        target=ActionTarget(
            acts_on=[Ref.entity("tally", "item")], reaches=[Ref.agent(recipient)] if recipient else [],
        ),
    )
    action.intent = SimpleNamespace(covert=covert)
    state = await executor.start(action, 1, agents=agents, environment=env, message_system=None)
    assert isinstance(state, ActionExecutionState)
    (result,) = await executor.complete(state, 1, agents=agents, environment=env, message_system=None)
    return result, seen, assessor


@pytest.mark.asyncio
async def test_an_unseen_theft_moves_the_thing_and_tells_nobody(container, monkeypatch) -> None:
    result, seen, assessor = await _act(container, monkeypatch, tier=CheckTier.SUCCESS)
    (change,) = result.entity_state_changes
    assert change.owner_id == "agent-a" and change.perception == ""
    assert result.observations == [] and result.target_effects == []
    assert not result.detected
    # The victim resists, and both the assessor and the judge know it is meant to go unseen.
    assert assessor.calls[0]["opponent"].agent_id == "agent-b"
    assert "暗中下手" in assessor.calls[0]["details"]
    assert "在场没有人察觉" in seen["covert_note"]


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", [CheckTier.COSTLY_SUCCESS, CheckTier.CRITICAL_FAILURE])
async def test_a_costly_or_botched_theft_in_company_is_seen(container, monkeypatch, tier: CheckTier) -> None:
    result, seen, _ = await _act(container, monkeypatch, tier=tier)
    assert result.detected
    assert result.observations
    assert [e.agent_id for e in result.target_effects] == ["agent-b"]
    assert "看破" in seen["covert_note"]


@pytest.mark.asyncio
async def test_with_nobody_else_here_nothing_is_seen(container, monkeypatch) -> None:
    result, _, _ = await _act(container, monkeypatch, tier=CheckTier.COSTLY_SUCCESS, holder="", b_elsewhere=True)
    assert not result.detected and result.observations == []
    assert result.entity_state_changes[0].owner_id == "agent-a"


@pytest.mark.asyncio
async def test_a_failed_attempt_unseen_moves_nothing(container, monkeypatch) -> None:
    result, _, _ = await _act(container, monkeypatch, tier=CheckTier.FAILURE)
    assert result.entity_state_changes == [] and not result.detected and result.observations == []


@pytest.mark.asyncio
async def test_a_plant_is_resisted_by_whoever_it_is_slipped_onto(container, monkeypatch) -> None:
    result, _, assessor = await _act(
        container, monkeypatch, tier=CheckTier.SUCCESS, holder="agent-a", recipient="agent-b", deed=Deed.RELINQUISH,
    )
    assert assessor.calls[0]["opponent"].agent_id == "agent-b"
    (change,) = result.entity_state_changes
    assert change.owner_id == "agent-b" and change.perception == ""
    # He doesn't know it was slipped onto him.
    assert result.target_effects == []


@pytest.mark.asyncio
async def test_an_act_in_the_open_is_told_as_before(container, monkeypatch) -> None:
    result, seen, assessor = await _act(container, monkeypatch, tier=CheckTier.SUCCESS, covert=False)
    assert result.observations and [e.agent_id for e in result.target_effects] == ["agent-b"]
    assert result.entity_state_changes[0].perception
    assert seen["covert_note"] == "" and "暗中下手" not in assessor.calls[0]["details"]
