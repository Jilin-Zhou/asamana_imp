"""A check assessor for executor tests that settles every check at one tier.

It makes no LLM call, so a test's scripted LLM replies still line up with the executor's own
calls; the assessor's real prompt and parse paths are covered in test_check_resolution.
"""

from __future__ import annotations

from typing import Any

from core.interfaces.check import CheckRecord, CheckTier
from engine.check import CheckAssessor, CheckVerdict


class FixedCheckAssessor(CheckAssessor):
    """Every assessment returns the same verdict; each call's arguments are recorded."""

    def __init__(self, tier: CheckTier = CheckTier.SUCCESS, record: CheckRecord | None = None) -> None:
        # No router: this double never reaches an LLM.
        self.verdict = CheckVerdict(tier=record.tier if record is not None else tier, record=record)
        self.calls: list[dict[str, Any]] = []

    async def assess(self, agent: Any, **kwargs: Any) -> CheckVerdict | None:
        self.calls.append({"agent": agent, **kwargs})
        return self.verdict


def fixed_check(tier: CheckTier = CheckTier.SUCCESS, record: CheckRecord | None = None) -> FixedCheckAssessor:
    return FixedCheckAssessor(tier, record)


class FailedCheckAssessor(CheckAssessor):
    """Every assessment fails, as when the assessor's LLM call raises."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def assess(self, agent: Any, **kwargs: Any) -> CheckVerdict | None:
        self.calls.append({"agent": agent, **kwargs})
        return None
