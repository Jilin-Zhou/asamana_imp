"""Living costs: each world day, every living character pays his food and lodging.

The money leaves the world (paid to people nobody models), the counterpart of what paid work
brings in (``ActionResult.earned``). The sum is the world's ``Economy.daily_cost``, charged per
midnight crossed; this is objective arithmetic, so code does it. Whoever can't pay all of it goes
short (``Agent.go_short``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from core.logging import get_logger
from engine.clock import WorldTime, WorldTimeConfig

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem

logger = get_logger(__name__)


def days_crossed(step: int, config: WorldTimeConfig) -> int:
    """Midnights between the previous step and this one."""
    if step <= 0:
        return 0
    return (
        WorldTime.from_step(step, config).total_days - WorldTime.from_step(step - 1, config).total_days
    )


class LivingCosts:
    def __init__(self, environment: "EnvironmentSystem", time_config: WorldTimeConfig) -> None:
        self._environment = environment
        self._time_config = time_config

    async def settle(self, agents: List["Agent"], step: int) -> None:
        economy = self._environment.economy
        days = days_crossed(step, self._time_config)
        if not economy.charges_living or days <= 0:
            return
        owed = economy.daily_cost * days
        for agent in agents:
            if not agent.is_active:
                continue
            paid = self._environment.take_money(agent.agent_id, owed)
            if paid >= owed:
                continue
            short = (owed - paid) / owed
            fact = (
                f"该付这{'一' if days == 1 else days}日的吃住了，要{economy.amount_text(owed)}，"
                + (f"我只凑得出{economy.amount_text(paid)}" if paid else "我身上分文没有")
                + "，只能饿着肚子将就。"
            )
            logger.info(
                "living_costs_short",
                extra={"agent_id": agent.agent_id, "step": step, "owed": owed, "paid": paid},
            )
            try:
                await agent.go_short(shortfall=short, fact=fact, step=step)
            except Exception as exc:  # noqa: BLE001 — Rule 1: one agent's writeback must not stop the rest
                logger.warning(
                    "living_costs_writeback_failed",
                    extra={"agent_id": agent.agent_id, "step": step, "error": str(exc)},
                )
