"""Objective action statistics for a run: what kinds of acts happened, how often they worked, how
the dice fell, and how much changed hands. Counted from snapshots, no LLM.

One act counts once: a joint act writes one record per participant and a multi-step act one per
beat, so only terminal beats count, folded by execution and read from the initiator's record.
What changed hands is read off the world's own state between snapshots, whichever act moved it.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from core.interfaces.check import CheckTier
from core.interfaces.snapshot import WorldSnapshot

# Beats that carry a result. An interrupt is terminal too, but counted apart: it was cut short.
_RESULT_PHASES = frozenset({"settled", "ongoing_complete"})
_INTERRUPT_PHASE = "interrupt"


def _terminal_acts(snapshots: Iterable[WorldSnapshot]) -> list[Mapping[str, Any]]:
    by_execution: dict[str, Mapping[str, Any]] = {}
    for snapshot in snapshots:
        for record in snapshot.actions_this_step:
            if record.get("phase") not in _RESULT_PHASES | {_INTERRUPT_PHASE}:
                continue
            key = str(record.get("execution_id") or id(record))
            # The initiator's record speaks for the act; any record stands in until it shows up.
            if key not in by_execution or record.get("agent_id") == record.get("initiator_id"):
                by_execution[key] = record
    return list(by_execution.values())


def _holdings(snapshot: WorldSnapshot) -> "tuple[Counter[str], dict[str, tuple[str, str]]] | None":
    """Money per holder or place, and where each other thing is; None when the snapshot saved no
    world state."""
    states = (snapshot.metadata.get("environment") or {}).get("entity_states")
    if not isinstance(states, Mapping):
        return None
    money: Counter[str] = Counter()
    things: dict[str, tuple[str, str]] = {}
    for entity_id, entity in states.items():
        if not isinstance(entity, Mapping) or entity.get("presence") == "destroyed":
            continue
        where = (str(entity.get("presence")), str(entity.get("presence_ref") or ""))
        if entity.get("entity_type") == "money":
            money[where[1]] += int(entity.get("quantity") or 0)
        else:
            things[str(entity_id)] = where
    return money, things


def holdings_moved(snapshots: Iterable[WorldSnapshot]) -> dict[str, int]:
    """Money that reached new hands (or the ground) and things that changed place or holder,
    between consecutive snapshots that saved world state. Money only splits and merges, so the
    sum of what each holder gained is what moved."""
    money_moved = things_moved = 0
    previous = None
    for snapshot in sorted(snapshots, key=lambda s: s.step):
        current = _holdings(snapshot)
        if current is None:
            continue
        if previous is not None:
            money, things = current
            money_moved += sum(max(0, amount - previous[0].get(holder, 0)) for holder, amount in money.items())
            things_moved += sum(1 for eid, where in things.items() if eid in previous[1] and previous[1][eid] != where)
        previous = current
    return {"money_moved": money_moved, "things_moved": things_moved}


def action_stats(snapshots: Iterable[WorldSnapshot]) -> dict[str, Any]:
    """Counts per action type, per outcome and per check tier, over every terminal act, plus what
    changed hands."""
    snapshots = list(snapshots)
    acts = _terminal_acts(snapshots)
    by_type: Counter[str] = Counter()
    succeeded: Counter[str] = Counter()
    not_executed: Counter[str] = Counter()
    interrupted: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    unrolled: Counter[str] = Counter()
    for act in acts:
        kind = str(act.get("action_type") or "")
        by_type[kind] += 1
        if act.get("phase") == _INTERRUPT_PHASE:
            interrupted[kind] += 1
        elif act.get("not_executed"):
            not_executed[kind] += 1
        elif act.get("succeeded"):
            succeeded[kind] += 1
        check = act.get("check")
        if isinstance(check, Mapping) and check.get("tier"):
            tiers[str(check["tier"])] += 1
        elif act.get("phase") != _INTERRUPT_PHASE and not act.get("not_executed"):
            unrolled[kind] += 1
    return {
        "acts": len(acts),
        "by_type": dict(by_type.most_common()),
        "succeeded": dict(succeeded),
        "not_executed": dict(not_executed),
        "interrupted": dict(interrupted),
        # Ordered from best to worst, every tier present so two runs line up column for column.
        "tiers": {tier.value: tiers.get(tier.value, 0) for tier in CheckTier},
        # Results with no roll behind them: types that never roll, and acts judged beyond doubt.
        "unrolled": dict(unrolled),
        "holdings": holdings_moved(snapshots),
    }
