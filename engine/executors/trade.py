"""Trade executor for TRADE actions."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List

from core.context import GivenFacts, annotate_call
from core.interfaces.action import (
    ActionResult, ActionType, AgentAction, TradeTerms, Transfer,
)
from core.interfaces.directory import WorldDirectory
from core.interfaces.economy import Economy
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.logging import get_logger
from core.prompts import ABSOLUTE_TIME_RULE, CLOSED_WORLD_FACT_RULE, SituationVoice, condition_line
from engine.executors.base import ActionExecutionState, participant_action_desc
from engine.executors.social import (
    _DIRECTION_TALK_DELTAS, _EMPTY_CTX, SocialExecutor, _format_context, _talk_target,
)
from engine.narration import observed_here, scene_line
from engine.scene import SceneVisibility, assemble_scene_context, observe_location

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

_MAX_LINES = 6

_TRADE_SYSTEM = f"""\
【任务】
你是中立的叙事者。A 想与 B 当面做一笔买卖或交换，条款由 A 提出。基于双方各自的处境，客观写出两人讨价还价的对白，
再据对白如实判定谈没谈成、按什么钱数成交。你不代入任何一方。

【约束】
- 成不成交是 B 自己的决定：看他要不要、舍不舍得、信不信得过 A、价钱公不公道（参照所给行情）。不替任何一方成全，谈不拢是正当结局。
- 钱数可以讨价还价、谈高谈低；但换手的东西只能是条款里的那几样，不添不减。成交的钱数不得超过付钱一方手上有的。
- 每一方只能说他自己那一栏里有的东西，不替对方亮底牌。
- 对白从人设、处境与彼此关系里自然长出，简练口语；不说教、不背景倾倒、不书面腔。
{CLOSED_WORLD_FACT_RULE}

【输出】
严格输出以下 JSON，不要任何多余内容（speaker 用 1 代表 A、2 代表 B）。字段按下面的次序写：先写对白，再据对白写判定的依据，再给结论。
- dialogue：至多 {_MAX_LINES} 句，每句 ≤40字。
- reason：B 为什么答应或不答应、钱数是怎么谈定的（≤60字）。
- agreed：成交为 true，没成交为 false。
- give_money：成交时 A 实际付给 B 的钱数（整数）；条款里 A 不出钱、或没成交，填 0。
- ask_money：成交时 B 实际付给 A 的钱数（整数）；条款里 A 不要钱、或没成交，填 0。
- observation：一个听不见内容的路人看到的样子（≤25字）：只写神色、举止与收场，不写谈的是什么、不写钱数。主语不必写，系统会补上。
{{"dialogue": [{{"speaker": 1或2, "line": "≤40字"}}], "reason": "≤60字", "agreed": true或false, "give_money": 0, "ask_money": 0, "observation": "≤25字"}}

落笔写上面那些对白之前，还有一条：
{ABSOLUTE_TIME_RULE}"""

# dialogue ≤6 × (speaker 3 + line ≤40 chars 60 + structure 10) ≈ 438 + reason ≤60 chars 90
# + agreed 3 + two sums 6 + observation ≤25 chars 38 + 6 fields of structure 30 → ~605 tok.
_MAX_TOKENS = output_budget(605)


class TradeExecutor(SocialExecutor):
    """Executor for TRADE: a face-to-face bargain that moves things both ways at once.

    Shares TALK's frame: the counterpart is invited (his turn is spent), one neutral narrator
    voices both sides, and each side then records the scene in his own words. What it adds is
    the deal: the narrator rules whether it was struck and for how much, and the exchange lands
    as one ``Transfer`` set on the initiator's result, whole or not at all.

    No roll: whether B agrees is B's own will, never a die.
    """

    def __init__(
        self, llm_router: LLMRouter, directory: WorldDirectory, seconds_per_step: int = 3600,
        *, world_start_second_of_day: int = 0, economy: Economy | None = None,
    ) -> None:
        super().__init__(
            llm_router, directory, seconds_per_step, world_start_second_of_day=world_start_second_of_day,
        )
        self._economy = economy if economy is not None else Economy()

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        location = observe_location(environment, action.agent_id)
        terms: TradeTerms | None = getattr(action.intent, "trade", None)
        counterpart = terms.counterpart_id if terms is not None else ""
        name = self._directory.agent_name(counterpart) if counterpart else "对方"
        if terms is None or not counterpart:
            return self._failed(action, step, "却没说定要和谁换些什么。", location)
        feasibility = environment.check_talk_feasibility(action.agent_id, [counterpart], target_label=name)
        if not feasibility.ok:
            return self._failed(action, step, feasibility.reason, location)
        if counterpart not in agents:
            return self._failed(action, step, f"没有找到{name}。", location)
        for entity_id, holder in ((terms.give_entity_id, action.agent_id), (terms.ask_entity_id, counterpart)):
            entity = environment.get_entity(entity_id) if entity_id else None
            if entity_id and (entity is None or entity.owner_id != holder):
                thing = entity.name if entity is not None else "那样东西"
                owner = "自己" if holder == action.agent_id else name
                return self._failed(action, step, f"可{thing}已不在{owner}手上。", location)
        state = ActionExecutionState.create(
            action_type=ActionType.TRADE,
            initiator_id=action.agent_id,
            participant_ids=[action.agent_id, counterpart],
            purpose=action.action_description or "交易",
            started_step=step,
            estimated_steps=1,
            opening_outcome="",
            target=action.target,
            expected_outcome=action.expected_outcome,
        )
        state.extra["target_id"] = counterpart
        state.extra["terms"] = {
            "give_entity_id": terms.give_entity_id, "give_amount": terms.give_amount,
            "ask_entity_id": terms.ask_entity_id, "ask_amount": terms.ask_amount,
        }
        return state

    def _failed(self, action: AgentAction, step: int, outcome: str, location: str) -> ActionExecutionState:
        """A trade that couldn't open: an actor-only, same-step failure (as TALK's)."""
        actor_name = self._directory.agent_name(action.agent_id)
        failure = ActionResult(
            action=AgentAction(
                agent_id=action.agent_id, step=step, action_type=ActionType.TRADE,
                action_description=action.action_description or "交易", target=action.target,
            ),
            expected_outcome=action.expected_outcome,
            outcome=scene_line(location, f"{actor_name}本想做「{action.action_description}」，{outcome}"),
            observations=[],
            succeeded=False,
            failure_reason=outcome,
            factual_memory=f"想要做：{action.action_description}，但是结果是：{outcome}",
        )
        return ActionExecutionState.create_failed(
            action_type=ActionType.TRADE, initiator_id=action.agent_id,
            failure_result=failure, started_step=step, purpose=action.action_description or "交易",
        )

    async def complete(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[ActionResult]:
        stored = self._stored_result(state)
        if stored is not None:
            return [stored]
        a_id = state.initiator_id
        b_id = str(state.extra.get("target_id", ""))
        a, b = agents.get(a_id), agents.get(b_id)
        a_name, b_name = self._directory.agent_name(a_id), self._directory.agent_name(b_id)
        location = observe_location(environment, a_id)
        raw = state.extra.get("terms") or {}
        terms = TradeTerms(
            counterpart_id=b_id,
            give_entity_id=str(raw.get("give_entity_id") or ""), give_amount=int(raw.get("give_amount") or 0),
            ask_entity_id=str(raw.get("ask_entity_id") or ""), ask_amount=int(raw.get("ask_amount") or 0),
        )
        a_purse, b_purse = environment.money_of(a_id), environment.money_of(b_id)

        verdict = None
        if a is not None and b is not None:
            a_ctx_r, b_ctx_r = await asyncio.gather(
                self._gather_context(a, b_id, b_name, state.purpose, step),
                self._gather_context(b, a_id, a_name, state.purpose, step),
                return_exceptions=True,
            )
            verdict = await self._haggle(
                a=a, b=b, a_name=a_name, b_name=b_name, purpose=state.purpose, terms=terms,
                a_ctx=dict(_EMPTY_CTX) if isinstance(a_ctx_r, BaseException) else a_ctx_r,
                b_ctx=dict(_EMPTY_CTX) if isinstance(b_ctx_r, BaseException) else b_ctx_r,
                a_purse=a_purse, b_purse=b_purse, environment=environment, agents=agents, step=step,
            )

        if verdict is None:
            # The haggling never happened: a null step for both, nothing changes hands.
            null = scene_line(location, f"{a_name}与{b_name}谈买卖。")
            return [
                self._trade_result(pid, other, other_name, step=step, state=state, outcome=null,
                                   summary=None, dialogue=[], observations=[])
                for pid, other, other_name in ((a_id, b_id, b_name), (b_id, a_id, a_name))
                if agents.get(pid) is not None
            ]
        dialogue, agreed, give_money, ask_money, reason, bystander = verdict
        give_money = min(give_money, a_purse) if terms.give_amount else 0
        ask_money = min(ask_money, b_purse) if terms.ask_amount else 0
        transfers = [
            *([Transfer(from_id=a_id, to_id=b_id, entity_id=terms.give_entity_id)] if terms.give_entity_id else []),
            *([Transfer(from_id=a_id, to_id=b_id, quantity=give_money)] if give_money else []),
            *([Transfer(from_id=b_id, to_id=a_id, entity_id=terms.ask_entity_id)] if terms.ask_entity_id else []),
            *([Transfer(from_id=b_id, to_id=a_id, quantity=ask_money)] if ask_money else []),
        ] if agreed else []
        agreed = bool(transfers)
        deal = self._deal_line(a_name, b_name, terms, give_money, ask_money, environment) if agreed else "没有谈拢。"
        transcript = "\n".join(f"{t['speaker']}：{t['line']}" for t in dialogue)
        record = f"{transcript}\n（{deal}）"
        outcome = scene_line(location, f"{a_name}与{b_name}谈买卖：\n{transcript}\n——{deal}")
        gist = scene_line(location, f"{a_name}与{b_name}谈买卖，{deal}")
        seen = f"{a_name}与{b_name}谈了一笔买卖" + ("，东西换了手。" if agreed else "，没有谈拢。")
        observations = observed_here(environment, a_id, scene_line(
            location, f"{seen}{bystander}" if bystander else seen,
        ))

        ending = "谈成了" if agreed else "没谈拢"
        summary_a, summary_b = await asyncio.gather(
            self._memory_summary(
                agent=a, other_id=b_id, other_name=b_name, transcript=record, purpose=state.purpose,
                fallback_fact=f"与{b_name}谈了一笔买卖，{ending}。", is_initiator=True, now_step=step,
                environment=environment, expected_outcome=state.expected_outcome,
            ),
            self._memory_summary(
                agent=b, other_id=a_id, other_name=a_name, transcript=record, purpose=state.purpose,
                fallback_fact=f"与{a_name}谈了一笔买卖，{ending}。", is_initiator=False, now_step=step,
                environment=environment,
            ),
            return_exceptions=True,
        )
        # Whether the deal was struck is the narrator's ruling, not either side's feeling about it.
        why = "" if agreed else (reason or "没有谈拢")
        results = [self._trade_result(
            a_id, b_id, b_name, step=step, state=state, outcome=outcome, gist=gist, dialogue=dialogue,
            observations=observations, transfers=transfers, agreed=agreed, why=why,
            summary=None if isinstance(summary_a, BaseException) else summary_a,
        )]
        if b is not None:
            results.append(self._trade_result(
                b_id, a_id, a_name, step=step, state=state, outcome=outcome, gist=gist, dialogue=dialogue,
                observations=observations, agreed=agreed, why=why,
                summary=None if isinstance(summary_b, BaseException) else summary_b,
            ))
        return results

    def _deal_line(
        self, a_name: str, b_name: str, terms: TradeTerms, give_money: int, ask_money: int,
        environment: "EnvironmentSystem",
    ) -> str:
        def side(entity_id: str, money: int) -> str:
            parts = [environment.get_entity(entity_id).name] if entity_id and environment.get_entity(entity_id) else []
            if money:
                parts.append(self._economy.amount_text(money))
            return "与".join(parts) or "分文未出"

        return (
            f"成交：{a_name}付出{side(terms.give_entity_id, give_money)}，"
            f"{b_name}付出{side(terms.ask_entity_id, ask_money)}。"
        )

    def _terms_block(
        self, a_name: str, b_name: str, terms: TradeTerms, a_purse: int, b_purse: int,
        environment: "EnvironmentSystem",
    ) -> str:
        def side(entity_id: str, money: int) -> str:
            entity = environment.get_entity(entity_id) if entity_id else None
            parts = [f"{entity.name}（{entity.description}）" if entity.description else entity.name] if entity else []
            if money:
                parts.append(self._economy.amount_text(money))
            return "、".join(parts) or "无"

        return f"""\
- {a_name}拿出：{side(terms.give_entity_id, terms.give_amount)}
- {a_name}想要{b_name}的：{side(terms.ask_entity_id, terms.ask_amount)}
- {a_name}手上的钱：{self._economy.amount_text(a_purse) if a_purse else "分文没有"}
- {b_name}手上的钱：{self._economy.amount_text(b_purse) if b_purse else "分文没有"}"""

    async def _haggle(
        self, *, a: "Agent", b: "Agent", a_name: str, b_name: str, purpose: str, terms: TradeTerms,
        a_ctx: dict, b_ctx: dict, a_purse: int, b_purse: int,
        environment: "EnvironmentSystem", agents: dict[str, "Agent"], step: int,
    ) -> "tuple[list[dict], bool, int, int, str, str] | None":
        """The haggling and its ruling in one functional call: ``(dialogue, agreed, give_money,
        ask_money, reason, observation)``, or ``None`` when it can't happen (a null step)."""
        lanes = []
        for agent, ctx, name, other in ((a, a_ctx, a_name, b_name), (b, b_ctx, b_name, a_name)):
            ctx_text = _format_context(
                ctx, name, other, now_step=step, seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day,
            )
            condition = condition_line(
                agent.personality.state.condition, voice=SituationVoice.THIRD,
                now_step=step, seconds_per_step=self._seconds_per_step,
            )
            lanes.append(
                f"{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}"
                f"{condition}\n{ctx_text}"
            )
        terms_text = self._terms_block(a_name, b_name, terms, a_purse, b_purse, environment)
        prices = self._economy.price_reference()
        scene = assemble_scene_context(
            a.agent_id, environment=environment, directory=self._directory, agents=agents,
            visibility=SceneVisibility.GOD,
        ).text
        user = f"""\
{f"【行情】{prices}{chr(10)}{chr(10)}" if prices else ""}【A：{a_name}】
{lanes[0]}

【B：{b_name}】
{lanes[1]}

【现场】
{scene}

【这笔买卖】（{a_name}提出的条款与盘算，不是已定的事实）
{a_name}的打算：{purpose}
{terms_text}

依上面说定的 JSON 格式（speaker 用 1 代表{a_name}、2 代表{b_name}）输出，只输出 JSON、不写任何多余内容。"""
        facts = (
            GivenFacts()
            .add("行情", prices)
            .add(f"{a_name} 手上的", lanes[0])
            .add(f"{b_name} 手上的", lanes[1])
            .add("现场", scene)
            .add("这笔买卖", terms_text)
        )
        try:
            with annotate_call(given_facts=facts):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [LLMMessage(role="system", content=_TRADE_SYSTEM), LLMMessage(role="user", content=user)],
                    temperature=0.8,
                    max_tokens=_MAX_TOKENS,
                    json_mode=True,
                )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning("trade_haggle_failed", extra={"agent_id": a.agent_id, "step": step, "error": str(exc)})
            return None
        dialogue = self._parse_dialogue(data, (a.agent_id, a_name), (b.agent_id, b_name))
        if not dialogue:
            logger.warning("trade_dialogue_unattributable", extra={"agent_id": a.agent_id, "step": step})
            return None

        def money(key: str) -> int:
            try:
                return max(0, int(data.get(key) or 0))
            except (TypeError, ValueError):
                return 0

        return (
            dialogue,
            coerce_bool(data.get("agreed"), False),
            money("give_money"),
            money("ask_money"),
            " ".join(str(data.get("reason") or "").split()),
            " ".join(str(data.get("observation") or "").split()),
        )

    def _trade_result(
        self,
        agent_id: str,
        other_id: str,
        other_name: str,
        *,
        step: int,
        state: ActionExecutionState,
        outcome: str,
        summary: "tuple[str, bool, str, str] | None",
        gist: str = "",
        dialogue: list[dict],
        observations: list,
        transfers: "list[Transfer] | None" = None,
        agreed: bool = False,
        why: str = "",
    ) -> ActionResult:
        """One participant's TRADE result. ``summary=None`` is a null step: no memory, no relation
        change, and nothing changes hands."""
        is_initiator = agent_id == state.initiator_id
        stub = AgentAction(
            agent_id=agent_id, step=step, action_type=ActionType.TRADE,
            action_description=participant_action_desc(self._directory, agent_id, state),
            target=_talk_target(other_id),
            estimated_steps=1,
        )
        expected = state.expected_outcome if is_initiator else ""
        if summary is None:
            return ActionResult(
                action=stub, expected_outcome=expected, outcome=outcome, observations=observations,
                succeeded=False, dialogue=dialogue, adjudication_failed=True,
            )
        fact, _felt_success, direction, _felt_why = summary
        td, ad = _DIRECTION_TALK_DELTAS.get(direction, _DIRECTION_TALK_DELTAS["neutral"])
        return ActionResult(
            action=stub,
            expected_outcome=expected,
            outcome=outcome,
            # The deal without the haggling, for readers who want only how it ended.
            gist=gist,
            observations=observations,
            # A lurker hears the haggling.
            happening=outcome,
            succeeded=agreed,
            failure_reason=why,
            factual_memory=fact,
            dialogue=dialogue,
            relation_updates=[(other_id, td, ad)],
            transfers=list(transfers or []),
        )
