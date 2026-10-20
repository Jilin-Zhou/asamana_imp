"""The run statistics count each act once and read its result, not its beats."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.interfaces.check import CheckRecord, Difficulty, Edge
from core.interfaces.snapshot import WorldSnapshot
from tuning.action_stats import action_stats

_ROLL = CheckRecord(
    difficulty=Difficulty.NORMAL, edge=Edge.EVEN, pure_chance=False, dice=(1, 1, 2),
).to_dict()  # 4 against 11: a critical failure


def _snap(step: int, *records: dict[str, Any]) -> WorldSnapshot:
    return WorldSnapshot(
        world_id="w", step=step, timestamp=datetime.now(timezone.utc), world_time={},
        agent_states={}, agent_relations={}, actions_this_step=list(records),
    )


def _rec(execution_id: str, agent_id: str, phase: str, **fields: Any) -> dict[str, Any]:
    return {
        "execution_id": execution_id, "agent_id": agent_id, "initiator_id": fields.pop("by", agent_id),
        "phase": phase, "action_type": fields.pop("type", "physical"), "succeeded": True, **fields,
    }


def test_each_act_counts_once_at_its_result() -> None:
    stats = action_stats([
        _snap(1, _rec("w1", "a", "begin", type="work")),
        _snap(2, _rec("w1", "a", "ongoing_tick", type="work")),
        _snap(3, _rec("w1", "a", "ongoing_complete", type="work", succeeded=False, check=_ROLL)),
        # A joint talk: one record per participant, the initiator's speaks for it.
        _snap(3,
              _rec("t1", "b", "settled", type="talk", by="a", succeeded=False),
              _rec("t1", "a", "settled", type="talk", succeeded=True)),
    ])
    assert stats["acts"] == 2
    assert stats["by_type"] == {"work": 1, "talk": 1}
    assert stats["succeeded"] == {"talk": 1}
    assert stats["tiers"]["critical_failure"] == 1
    assert stats["unrolled"] == {"talk": 1}


def test_non_events_and_interrupts_are_counted_apart() -> None:
    stats = action_stats([_snap(
        1,
        _rec("t1", "a", "settled", type="talk", succeeded=False, not_executed=True),
        _rec("c1", "b", "interrupt", type="covert", succeeded=False),
        _rec("x", "c", "initialization", type=""),
    )])
    assert stats["acts"] == 2
    assert stats["not_executed"] == {"talk": 1}
    assert stats["interrupted"] == {"covert": 1}
    assert stats["succeeded"] == {}
    assert stats["unrolled"] == {}
    assert list(stats["tiers"]) == [
        "critical_success", "success", "costly_success", "failure", "critical_failure",
    ]


def _world(step: int, purses: dict[str, int], ring_with: str) -> WorldSnapshot:
    states: dict[str, Any] = {
        f"money_{holder}": {"entity_type": "money", "presence": "held", "presence_ref": holder, "quantity": n}
        for holder, n in purses.items()
    }
    states["ring"] = {"entity_type": "item", "presence": "held", "presence_ref": ring_with}
    snap = _snap(step)
    snap.metadata = {"environment": {"entity_states": states}}
    return snap


def test_what_changed_hands_is_read_off_the_world_between_snapshots() -> None:
    stats = action_stats([
        _world(2, {"a": 10, "b": 20}, ring_with="a"),
        _world(0, {"a": 30}, ring_with="b"),
        _snap(1),  # saved no world state: skipped, not read as everything vanishing
    ])
    assert stats["holdings"] == {"money_moved": 20, "things_moved": 1}


def test_a_run_without_world_state_moved_nothing() -> None:
    assert action_stats([_snap(1), _snap(2)])["holdings"] == {"money_moved": 0, "things_moved": 0}
