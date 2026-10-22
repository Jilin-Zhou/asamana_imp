"""Event deck design: the LLM call that writes the happenings this world is prone to."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Sequence

from core.coerce import coerce_mapping_list
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json_object, output_budget
from core.interfaces.severity import Severity
from core.logging import get_logger
from engine.event_deck import MAX_CARDS, MIN_CARDS, WEIGHT_MAX, WEIGHT_MIN, Daypart, Scope, deck_from_dicts
from engine.weather import Sky

from world.builders import BUILD_ATTEMPTS
from world.models import ThemeAnalysis

logger = get_logger(__name__)

_TITLE_MAX_CHARS = 12
_SYNOPSIS_MAX_CHARS = 40
_REASON_MAX_CHARS = 80

# reason ≤80 chars ≈120 + ≤20 cards × (title ≤12 chars 18 + synopsis ≤40 chars 60
# + scope/severity/weight/sky/daypart ≈15 + 7 fields of structure 35 ≈128) ≈2560 → ~2680 tok.
_MAX_TOKENS = output_budget(2680)

_SKY_CHOICES = "any或" + "或".join(s.value for s in Sky)

_SYSTEM_PROMPT = f"""\
你负责为一个故事世界写一副「事件牌」：这个世界里时常会自己发生的事。只返回 JSON，不包含任何其他内容。

【任务】
先在 reason 里想清楚这个世界的时代、地方、行当与人情里，会自己冒出哪些事；再写出 {MIN_CARDS}–{MAX_CARDS} 张牌。
一张牌是一类会发生的事，不是某个人的遭遇：日后抽中它时，另有人决定它落在谁身上、在哪里、怎么说出来。

【避免】
- 换个世界照样成立的套话（「发生了意外」「出了一件怪事」）：每张牌都要只配得上这个世界。
- 直接了结核心张力的天降转机，或替人物做决定的事。
- 写死某个具体人物的生死伤病、去向或心意。
- 一副牌里同类的事扎堆：天时、营生、官府与规矩、流言、往来的人、器物、意外、节令都可以有，不必样样都有。
- 每张都惊天动地：多数该是小事，少数才是大事。

【字段】
- scope：{Scope.LOCAL.value} 发生在某一处；{Scope.GLOBAL.value} 全世界都会知道。
- severity：{"/".join(s.value for s in Severity)}，这件事有多大。
- weight：{WEIGHT_MIN}–{WEIGHT_MAX}，越常见越大。
- sky：只在某种天色下才会发生时填（{_SKY_CHOICES}），否则 any。
- daypart：只在白天或夜里才会发生时填 {Daypart.DAY.value} 或 {Daypart.NIGHT.value}，否则 {Daypart.ANY.value}。

【输出】
只返回符合下面格式的紧凑 JSON：
- reason：这个世界会自己冒出哪些事（≤{_REASON_MAX_CHARS}字）。
- cards：{MIN_CARDS}–{MAX_CARDS} 张；每张先写 title（≤{_TITLE_MAX_CHARS}字），再写 synopsis（≤{_SYNOPSIS_MAX_CHARS}字，写这类事怎么发生、会带来什么变化），再写其余字段。
格式示意（内容不代表任何世界）：
{{"reason": "≤{_REASON_MAX_CHARS}字", "cards": [{{"title": "≤{_TITLE_MAX_CHARS}字", "synopsis": "≤{_SYNOPSIS_MAX_CHARS}字", "scope": "local或global", "severity": "low或medium或high", "weight": 1, "sky": "any", "daypart": "any"}}]}}
"""


class EventDeckDesigner:
    """Single LLM call that writes this world's event deck."""

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def design(self, analysis: ThemeAnalysis, places: Sequence[str]) -> ThemeAnalysis:
        """Return the analysis with its ``event_deck`` filled in. ``places`` are the map's place
        names, so the cards can belong to where this world actually is."""
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=self._user_prompt(analysis, places)),
        ]
        last_exc: BaseException | None = None
        for attempt in range(BUILD_ATTEMPTS):
            try:
                response = await self._llm_router.complete(
                    LLMScene.EVENT_DECK_DESIGN, messages, max_tokens=_MAX_TOKENS, json_mode=True,
                )
                payload = extract_json_object(response.content)
                deck = deck_from_dicts(coerce_mapping_list(payload.get("cards"))) if payload else ()
                if len(deck) >= MIN_CARDS:
                    return dataclasses.replace(analysis, event_deck=deck)
                last_exc = ValueError(
                    f"EventDeckDesigner wrote {len(deck)} usable cards for world "
                    f"'{analysis.world_name}' (attempt {attempt + 1})"
                )
            except Exception as exc:
                last_exc = exc
            logger.warning(
                "event_deck_attempt_failed",
                extra={"world_name": analysis.world_name, "attempt": attempt + 1, "reason": str(last_exc)},
            )
            if attempt < BUILD_ATTEMPTS - 1:
                await asyncio.sleep(1.0)
        raise ValueError(
            f"EventDeckDesigner failed for world '{analysis.world_name}' after {BUILD_ATTEMPTS} attempts"
        ) from last_exc

    @staticmethod
    def _user_prompt(analysis: ThemeAnalysis, places: Sequence[str]) -> str:
        return f"""\
【输入】
- 世界：{analysis.world_name}
- 时代与环境：{analysis.era_description}
- 气候：{analysis.climate.label}
- 核心张力：{analysis.core_tension}
- 叙事主题：{analysis.narrative_theme}
- 地点：{"、".join(places) or "（未列出）"}

依上面说定的 JSON 格式写出这副牌（reason 在前），只输出 JSON、不写任何多余内容。"""

