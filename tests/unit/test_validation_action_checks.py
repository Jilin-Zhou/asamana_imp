"""The action validator's field checks accept the shapes the executors legitimately produce."""

from __future__ import annotations

from tuning.validation_action import _deterministic_checks


def _phase(change: dict) -> dict:
    result = {"outcome": "甲拿起了兵符", "entity_state_changes": [change]}
    return {"outputs": {"kind": "lifecycle", "phases": {"complete": {"results": [result]}}}}


def test_a_seize_that_only_changes_hands_is_not_missing_a_state() -> None:
    checks = _deterministic_checks(
        _phase({"entity_id": "token", "new_state": "", "owner_id": "agent-a", "location_id": None}),
        set(),
    )
    assert "entity_state_change new_state 为空" not in checks["field_violations"]


def test_a_use_with_no_state_is_still_flagged() -> None:
    checks = _deterministic_checks(
        _phase({"entity_id": "door", "new_state": "", "owner_id": None, "location_id": None}),
        set(),
    )
    assert "entity_state_change new_state 为空" in checks["field_violations"]
