"""Ability design: the LLM call that rates the whole cast's attributes, skills and cash on one scale."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Mapping, Sequence

from core.coerce import coerce_int, coerce_mapping_list, coerce_str
from core.interfaces.ability import (
    ATTRIBUTE_BASELINE, ATTRIBUTE_MAX, ATTRIBUTE_MIN, MAX_SKILLS, SKILL_MAX, SKILL_MIN, Abilities, Attribute,
    attribute_word, skill_word,
)
from core.interfaces.llm import IndexedRef, LLMMessage, LLMRouter, LLMScene, extract_json_object, output_budget
from core.logging import get_logger

from world.builders import BUILD_ATTEMPTS
from world.models import AgentDefinition, ThemeAnalysis

logger = get_logger(__name__)

_BASIS_MAX_CHARS = 30
_REASON_MAX_CHARS = 120
_SKILL_NAME_MAX_CHARS = 6

# reason ≤120 chars ≈ 180 tok. Per character: index ≈3 + basis ≤30 chars ≈45 + attributes
# (5 × ~6 + braces) ≈ 40 + skills (≤4 × (name ≤6 chars ≈9 + level 3 + structure 10)) ≈ 88
# + cash ≈3 + structure 25 ≈ 204 tok.
_REASON_TOKENS = 180
_PER_CHARACTER_TOKENS = 204

_LEVEL_LEGEND = " / ".join(
    f"{level} {attribute_word(level)}" for level in range(ATTRIBUTE_MIN, ATTRIBUTE_MAX + 1)
)
_SKILL_LEGEND = " / ".join(f"{level} {skill_word(level)}" for level in range(SKILL_MIN, SKILL_MAX + 1))
_ATTRIBUTE_LEGEND = "\n".join(f"- {a.label}：{a.scope}" for a in Attribute)
_SCHEMA_ATTRIBUTES = ", ".join(f'"{a.label}": 1到5' for a in Attribute)

_SYSTEM_PROMPT = f"""\
你负责为一个故事世界里的全体人物评定能力。只返回 JSON，不包含任何其他内容。

【任务】
下面给出全体人物的人设。先在 reason 里把全体人物放在一起比较，再为每个人评定五项属性、至多 {MAX_SKILLS} 项专长，
以及他此刻随身带着多少钱。所有人用同一把尺子量：谁力气最大、谁最有心计、谁最会说话、谁最有钱，要能横向比出来。

【属性】每项 {ATTRIBUTE_MIN}–{ATTRIBUTE_MAX} 级：{_LEVEL_LEGEND}
{_ATTRIBUTE_LEGEND}

【专长】
- 专长是这个人练过、做惯的一门具体本事：某种武艺、手艺、学问、营生、技艺。
- 等级：{_SKILL_LEGEND}。
- 人设里看不出来的本事不写；一项也没有就给空数组。

【约束】
- {ATTRIBUTE_BASELINE} 级是寻常成年人；{ATTRIBUTE_MAX} 级是当世顶尖，满世界寥寥无几。不要人人出众。
- 只凭人设里写到的经历、营生、年龄、体貌来定：
  - 身份显赫不等于本事大，性格果决不等于胆识过人；
  - 主角不因为是主角就分高；
  - 老人、孩童、病弱者的体魄与身手不会高。
- 每人至少一项属性低于 {ATTRIBUTE_BASELINE} 级：人都有短处。
- 「{skill_word(SKILL_MAX)}」级专长要有人设依据，全世界至多一两人。
- 专长名称是一门本事的名字，不是性格或身份（「镇定」「掌柜」都不是专长）。
- 带多少钱按身份、家境与此刻的处境比出多寡，参照下面给的物价：穷人可以分文没有，富人也不会把家底都揣在身上。
  这个世界没有钱时一律填 0。

【输出】
只返回符合下面格式的紧凑 JSON：
- reason：通盘比较全体人物，谁强在哪、弱在哪（≤{_REASON_MAX_CHARS}字）。
- cast：每个人物一条，不得遗漏；每条依次写：
  - index：人物编号（整数）；
  - basis：此人擅长什么、短在哪（≤{_BASIS_MAX_CHARS}字）；
  - attributes：五项属性的等级；
  - skills：至多 {MAX_SKILLS} 项，name ≤{_SKILL_NAME_MAX_CHARS}字，level {SKILL_MIN}–{SKILL_MAX}；
  - cash：随身带着的钱，按下面给的单位计的整数。
格式示意（数值不代表任何人）：
{{"reason": "≤{_REASON_MAX_CHARS}字", "cast": [{{"index": 1, "basis": "≤{_BASIS_MAX_CHARS}字", "attributes": {{{_SCHEMA_ATTRIBUTES}}}, "skills": [{{"name": "≤{_SKILL_NAME_MAX_CHARS}字", "level": 1}}], "cash": 0}}]}}
"""

_BY_LABEL: dict[str, Attribute] = {a.label: a for a in Attribute}


class AbilityDesigner:
    """Single LLM call that gives every character attributes and skills on one shared scale.

    Cast-wide rather than per persona: personas are generated in parallel and can't see each
    other, so only a call that reads them all can say who is strongest or cleverest.
    """

    def __init__(self, llm_router: LLMRouter) -> None:
        self._llm_router = llm_router

    async def design(
        self, analysis: ThemeAnalysis, definitions: Sequence[AgentDefinition],
    ) -> list[AgentDefinition]:
        """Return the definitions with ``soul.abilities`` filled in, in the same order."""
        if not definitions:
            return list(definitions)
        messages = [
            LLMMessage(role="system", content=_SYSTEM_PROMPT),
            LLMMessage(role="user", content=self._user_prompt(analysis, definitions)),
        ]
        last_exc: BaseException | None = None
        for attempt in range(BUILD_ATTEMPTS):
            try:
                response = await self._llm_router.complete(
                    LLMScene.ABILITY_DESIGN, messages,
                    max_tokens=output_budget(_REASON_TOKENS + _PER_CHARACTER_TOKENS * len(definitions)),
                    json_mode=True,
                )
                rated = self._parse(response.content, definitions)
                if rated is not None:
                    return [
                        dataclasses.replace(
                            d,
                            soul=dataclasses.replace(d.soul, abilities=rated[d.agent_id][0]),
                            initial_cash=rated[d.agent_id][1] if analysis.economy.has_money else 0,
                        )
                        for d in definitions
                    ]
                last_exc = ValueError(
                    f"AbilityDesigner left characters unrated for world '{analysis.world_name}' "
                    f"(attempt {attempt + 1})"
                )
            except Exception as exc:
                last_exc = exc
            logger.warning(
                "ability_design_attempt_failed",
                extra={"world_name": analysis.world_name, "attempt": attempt + 1, "reason": str(last_exc)},
            )
            if attempt < BUILD_ATTEMPTS - 1:
                await asyncio.sleep(1.0)
        raise ValueError(
            f"AbilityDesigner failed for world '{analysis.world_name}' after {BUILD_ATTEMPTS} attempts"
        ) from last_exc

    @staticmethod
    def _user_prompt(analysis: ThemeAnalysis, definitions: Sequence[AgentDefinition]) -> str:
        people = []
        for i, d in enumerate(definitions, 1):
            soul = d.soul
            parts = [soul.identity_text()]
            if soul.role:
                parts.append(f"身份：{soul.role}")
            if soul.background:
                parts.append(f"背景：{soul.background}")
            if soul.core_traits:
                parts.append(f"性格：{soul.traits_text()}")
            if soul.appearance:
                parts.append(f"外貌：{soul.appearance}")
            if soul.secret:
                parts.append(f"秘密：{soul.secret}")
            people.append(f"#{i} " + "｜".join(parts))
        roster = "\n".join(people)
        economy = analysis.economy
        money = (
            f"{economy.currency}，按{economy.unit or economy.currency}计"
            + (f"；参考物价：{economy.price_reference()}" if economy.prices else "")
            if economy.has_money else "这个世界没有钱"
        )
        return f"""\
【输入】
- 世界：{analysis.world_name}
- 时代与环境：{analysis.era_description}
- 钱：{money}
- 人物：
{roster}

依上面说定的 JSON 格式输出（reason 在前），覆盖全部 {len(definitions)} 个人物，不写多余内容。"""

    @staticmethod
    def _parse(raw: str, definitions: Sequence[AgentDefinition]) -> dict[str, tuple[Abilities, int]] | None:
        """agent_id → (abilities, cash), or ``None`` when anyone is left unrated (the caller
        retries)."""
        payload = extract_json_object(raw)
        if payload is None:
            return None
        ids = IndexedRef([d.agent_id for d in definitions])
        rated: dict[str, tuple[Abilities, int]] = {}
        for entry in coerce_mapping_list(payload.get("cast")):
            resolved = ids.resolve([entry.get("index")])
            if not resolved:
                continue
            raw_attributes = entry.get("attributes")
            # Keyed by the labels the prompt uses; Abilities.from_dict clamps the levels.
            abilities = Abilities.from_dict({
                "attributes": (
                    {_BY_LABEL[k].value: v for k, v in raw_attributes.items() if k in _BY_LABEL}
                    if isinstance(raw_attributes, Mapping) else {}
                ),
                "skills": [
                    {"name": coerce_str(s.get("name")), "level": s.get("level")}
                    for s in coerce_mapping_list(entry.get("skills"))
                ],
            })
            rated[resolved[0]] = (abilities, coerce_int(entry.get("cash"), default=0, minimum=0))
        if len(rated) < len(definitions):
            return None
        return rated

