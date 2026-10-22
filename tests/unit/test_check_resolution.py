"""Checks: the 3d6 resolution rule, the record contract, and the assessor's parse paths.

The probability tables enumerate all 216 rolls, so the tier bands asserted here are exact, not
sampled: changing a threshold or a DC shows up as a precise diff.
"""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace

import pytest

from agent.personality import PersonalityLayer, SoulLayer
from core.interfaces.action import ActionType
from core.interfaces.check import (
    CheckRecord, CheckTier, Difficulty, Edge,
)
from core.interfaces.llm import LLMResponse
from core.prompts import SituationVoice
from engine.check import CheckAssessor, CheckVerdict, roll_check, tier_phrase

_TIERS = (
    CheckTier.CRITICAL_SUCCESS, CheckTier.SUCCESS, CheckTier.COSTLY_SUCCESS,
    CheckTier.FAILURE, CheckTier.CRITICAL_FAILURE,
)


def _distribution(difficulty: Difficulty, edge: Edge) -> list[int]:
    """Rounded % per tier (critical success → critical failure) over every 3d6 roll."""
    counts = dict.fromkeys(_TIERS, 0)
    for dice in itertools.product(range(1, 7), repeat=3):
        record = CheckRecord(difficulty=difficulty, edge=edge, pure_chance=False, dice=dice)
        counts[record.tier] += 1
    return [round(counts[t] * 100 / 216) for t in _TIERS]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("margin", "tier"),
    [
        (5, CheckTier.CRITICAL_SUCCESS), (4, CheckTier.SUCCESS), (1, CheckTier.SUCCESS),
        (0, CheckTier.COSTLY_SUCCESS), (-1, CheckTier.COSTLY_SUCCESS),
        (-2, CheckTier.FAILURE), (-5, CheckTier.FAILURE), (-6, CheckTier.CRITICAL_FAILURE),
    ],
)
def test_margin_bands(margin: int, tier: CheckTier) -> None:
    assert CheckTier.for_margin(margin) is tier


@pytest.mark.parametrize(
    ("difficulty", "expected"),
    [
        (Difficulty.EASY,      [26, 48, 17, 9, 0]),
        (Difficulty.NORMAL,    [5, 33, 25, 33, 5]),
        (Difficulty.HARD,      [0, 9, 17, 48, 26]),
        (Difficulty.VERY_HARD, [0, 0, 4, 33, 62]),
    ],
)
def test_even_match_distribution_per_difficulty(difficulty: Difficulty, expected: list[int]) -> None:
    """The table in docs/plans/theater-roadmap.zh-CN.md §3.1, for an even match."""
    assert _distribution(difficulty, Edge.EVEN) == expected


def test_an_even_match_at_normal_is_symmetric() -> None:
    """Success and failure mirror each other: the rule has no built-in lean."""
    dist = _distribution(Difficulty.NORMAL, Edge.EVEN)
    assert dist == dist[::-1]


def test_a_better_edge_never_lowers_the_odds_of_success() -> None:
    """Monotone in edge at every difficulty: a stronger actor is never likelier to fail."""
    edges = list(Edge)  # declared weakest → strongest
    for difficulty in Difficulty:
        success_share = [sum(_distribution(difficulty, e)[:3]) for e in edges]
        assert success_share == sorted(success_share), difficulty


def test_tier_success_split() -> None:
    assert [t.succeeded for t in _TIERS] == [True, True, True, False, False]


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_record_derives_its_numbers_from_the_dice() -> None:
    record = CheckRecord(
        difficulty=Difficulty.HARD, edge=Edge.BETTER, pure_chance=False, dice=(4, 5, 3),
    )
    assert (record.dc, record.modifier, record.total) == (14, 1, 13)
    assert record.tier is CheckTier.COSTLY_SUCCESS


def test_pure_chance_ignores_the_edge() -> None:
    record = CheckRecord(
        difficulty=Difficulty.NORMAL, edge=Edge.MUCH_BETTER, pure_chance=True, dice=(2, 3, 4),
    )
    assert record.modifier == 0
    assert record.total == 9


def test_record_round_trips_through_its_dict() -> None:
    record = CheckRecord(
        difficulty=Difficulty.EASY, edge=Edge.WORSE, pure_chance=False, dice=(1, 6, 2),
        reason="对方早有防备",
    )
    payload = record.to_dict()
    assert payload["tier"] == record.tier.value
    assert payload["total"] == 8
    assert CheckRecord.from_dict(payload) == record


@pytest.mark.parametrize(
    "payload",
    [None, "x", {}, {"dice": [1, 2]}, {"dice": [0, 2, 3]}, {"dice": [1, 2, 7]}, {"dice": ["a", 1, 1]}],
)
def test_from_dict_rejects_what_is_not_a_record(payload: object) -> None:
    assert CheckRecord.from_dict(payload) is None


def test_from_dict_recomputes_rather_than_trusting_derived_fields() -> None:
    """A stale or tampered tier in the payload can't contradict the dice."""
    payload = CheckRecord(
        difficulty=Difficulty.NORMAL, edge=Edge.EVEN, pure_chance=False, dice=(1, 1, 1),
    ).to_dict()
    payload["tier"] = "critical_success"
    restored = CheckRecord.from_dict(payload)
    assert restored is not None and restored.tier is CheckTier.CRITICAL_FAILURE


# ---------------------------------------------------------------------------
# Rolling
# ---------------------------------------------------------------------------


def test_roll_is_reproducible_from_its_rng() -> None:
    a = roll_check(Difficulty.NORMAL, Edge.EVEN, pure_chance=False, rng=random.Random("s"))
    b = roll_check(Difficulty.NORMAL, Edge.EVEN, pure_chance=False, rng=random.Random("s"))
    assert a == b
    assert all(1 <= d <= 6 for d in a.dice) and len(a.dice) == 3


def test_pure_chance_roll_records_an_even_edge() -> None:
    record = roll_check(Difficulty.HARD, Edge.MUCH_BETTER, pure_chance=True, rng=random.Random(1))
    assert record.edge is Edge.EVEN


def test_every_tier_has_words_in_both_voices_and_no_numbers() -> None:
    """The tier is the only form a check takes in a prompt: words, never dice or a DC."""
    for tier in CheckTier:
        for voice in (SituationVoice.THIRD, SituationVoice.FIRST):
            phrase = tier_phrase(tier, voice=voice)
            assert phrase and not any(ch.isdigit() for ch in phrase)
        assert "我" in tier_phrase(tier, voice=SituationVoice.FIRST)
        assert "我" not in tier_phrase(tier, voice=SituationVoice.THIRD)


# ---------------------------------------------------------------------------
# The assessor
# ---------------------------------------------------------------------------


class _Router:
    """Router stub: canned replies in order, every call recorded."""

    def __init__(self, *replies: str, raises: bool = False) -> None:
        self._replies = list(replies)
        self._raises = raises
        self.calls: list[list] = []

    async def complete(self, scene, messages, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.calls.append(messages)
        if self._raises:
            raise RuntimeError("llm down")
        return LLMResponse(content=self._replies.pop(0), input_tokens=0, output_tokens=0, model="stub")


def _actor(agent_id: str = "agent-a") -> SimpleNamespace:
    soul = SoulLayer(name="李四", agent_id=agent_id, gender="男", core_traits=("谨慎",))
    return SimpleNamespace(agent_id=agent_id, personality=PersonalityLayer(soul))


async def _assess(router: _Router, *, seed: str = "w1", step: int = 3, agent_id: str = "agent-a") -> CheckVerdict | None:
    return await CheckAssessor(router, seed=seed).assess(
        _actor(agent_id), action_type=ActionType.PHYSICAL, intent="推开那扇门",
        scene="- 在场的其他人：无", step=step,
    )


async def test_sure_success_skips_the_roll() -> None:
    verdict = await _assess(_Router('{"reason": "门没锁", "certainty": "sure_success"}'))
    assert verdict == CheckVerdict(tier=CheckTier.SUCCESS)


async def test_sure_failure_skips_the_roll() -> None:
    verdict = await _assess(_Router('{"reason": "门是铁铸的", "certainty": "sure_failure"}'))
    assert verdict == CheckVerdict(tier=CheckTier.FAILURE)


async def test_uncertain_rolls_and_the_tier_matches_the_record() -> None:
    verdict = await _assess(_Router(
        '{"reason": "门闩生锈", "certainty": "uncertain", "pure_chance": false,'
        ' "difficulty": "hard", "edge": "better"}'
    ))
    assert verdict is not None and verdict.record is not None
    assert verdict.record.difficulty is Difficulty.HARD
    assert verdict.record.edge is Edge.BETTER
    assert verdict.record.reason == "门闩生锈"
    assert verdict.tier is verdict.record.tier


async def test_garbled_fields_roll_at_normal_and_even() -> None:
    """Doubt is the safe reading of a garbled certainty: a free success is not."""
    verdict = await _assess(_Router('{"certainty": "maybe", "difficulty": "??"}'))
    assert verdict is not None and verdict.record is not None
    assert (verdict.record.difficulty, verdict.record.edge) == (Difficulty.NORMAL, Edge.EVEN)


async def test_llm_failure_means_no_assessment() -> None:
    assert await _assess(_Router(raises=True)) is None


async def test_unparseable_reply_means_no_assessment() -> None:
    assert await _assess(_Router("不是 JSON")) is None


_UNCERTAIN = '{"certainty": "uncertain", "difficulty": "normal", "edge": "even"}'


async def test_dice_are_reproducible_within_a_world() -> None:
    a = await _assess(_Router(_UNCERTAIN))
    b = await _assess(_Router(_UNCERTAIN))
    assert a is not None and b is not None and a.record == b.record


async def test_dice_differ_between_worlds() -> None:
    """Over a run of steps two worlds' dice don't move in lockstep."""
    rolls = {}
    for seed in ("w1", "w2"):
        rolls[seed] = [
            (await _assess(_Router(_UNCERTAIN), seed=seed, step=step)).record.dice
            for step in range(1, 9)
        ]
    assert rolls["w1"] != rolls["w2"]


async def test_the_assessment_prompt_carries_no_ids_or_steps() -> None:
    """Narrative-layer prompt: the actor is named, never addressed by id or step."""
    router = _Router(_UNCERTAIN)
    await _assess(router, agent_id="agent-zz9", step=47)
    prompt = "\n".join(m.content for m in router.calls[0])
    assert "agent-zz9" not in prompt
    assert "47" not in prompt
    assert "李四" in prompt
