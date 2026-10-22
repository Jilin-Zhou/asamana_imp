"""Trade executor for TRADE: money and things changing hands face to face."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    _DIRECTION_TALK_DELTAS, SocialExecutor, _empty_ctx, _format_context, _talk_target,
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
你是中立的叙事者。A 想与 B 当面交割一笔钱物，条款由 A 提出：可以是买卖、交换；可以只有 A 出（还债、借出、付酬、
送礼、行贿）；也可以只有 A 要（讨债、索要）。基于双方各自的处境，客观写出两人当面的对白，再据对白如实判定
B 肯不肯、按什么钱数换手。你不代入任何一方。

【约束】
- 肯不肯是 B 自己的决定：看他要不要、舍不舍得、信不信得过 A、价钱公不公道（参照所给行情）、收下有没有风险、
  欠的该不该还、拿不拿得出。不替任何一方成全，谈不拢、不肯收、赖着不给都是正当结局。
- 条款只有一边时，A 出钱物换来的可以是一句话、一个答应、一笔旧账的了结，不必是东西；这些只在对白里说定，不算换手的东西。
- 钱数可以讨价还价、谈高谈低；但换手的东西只能是条款里的那几样，不添不减。成交的钱数不得超过付钱一方手上有的。
- 每一方只能说他自己那一栏里有的东西，不替对方亮底牌。
- 对白从人设、处境与彼此关系里自然长出，简练口语；不说教、不背景倾倒、不书面腔。
{CLOSED_WORLD_FACT_RULE}

【输出】
严格输出以下 JSON，不要任何多余内容（speaker 用 1 代表 A、2 代表 B）。字段按下面的次序写：先写对白，再据对白写判定的依据，再给结论。
- dialogue：至多 {_MAX_LINES} 句，每句 ≤40字。
- reason：B 为什么肯或不肯、钱数是怎么谈定的（≤60字）。
- agreed：B 肯了、钱物当场换手为 true，否则为 false。
- give_money：成交时 A 实际付给 B 的钱数（整数）；条款里 A 不出钱、或没成交，填 0。
- ask_money：成交时 B 实际付给 A 的钱数（整数）；条款里 A 不要钱、或没成交，填 0。
- observation：一个听不见内容的路人看到的样子（≤25字）：只写神色、举止与收场，不写谈的是什么、不写钱数。主语不必写，系统会补上。
{{"dialogue": [{{"speaker": 1或2, "line": "≤40字"}}], "reason": "≤60字", "agreed": true或false, "give_money": 0, "ask_money": 0, "observation": "≤25字"}}

落笔写上面那些对白之前，还有一条：
{ABSOLUTE_TIME_RULE}"""

# dialogue ≤6 × (speaker 3 + line ≤40 chars 60 + structure 10) ≈ 438 + reason ≤60 chars 90
# + agreed 3 + two sums 6 + observation ≤25 chars 38 + 6 fields of structure 30 → ~605 tok.
_MAX_TOKENS = output_budget(605)


@dataclass(frozen=True)
class _Wording:
    """How one handover reads, by the shape of its terms: both ways is a bargain, A only giving
    is a payment (a debt repaid, a loan, a bribe), A only asking is a demand."""
    topic: str    # what A set out to do, before any ruling
    deal: str     # how it ended, amounts as they landed
    gist: str
    seen: str     # an onlooker's view: who handed something to whom, never how much
    a_fact: str   # each side's fallback memory, subject left out
    b_fact: str


class TradeExecutor(SocialExecutor):
    """Executor for TRADE: things and money changing hands face to face, whole or not at all.
    Both ways it is a bargain; one way it is a payment (a debt repaid, a loan, a bribe) or a
    demand. What a one-way payment buys is said, not handed over.

    Shares TALK's frame: the counterpart is invited (his turn is spent), one neutral narrator
    voices both sides, and each side then records the scene in his own words. What it adds is
    the deal: the narrator rules whether it was struck and for how much, and the exchange lands
    as one ``Transfer`` set on the initiator's result, whole or not at all.

    No roll: whether B agrees is B's own will, never a die. A deal is between two people:
    nobody walks into one under way.
    """

    def __init__(
        self, llm_router: LLMRouter, directory: WorldDirectory, seconds_per_step: int = 3600,
        *, world_start_second_of_day: int = 0, economy: Economy | None = None,
    ) -> None:
        super().__init__(
            llm_router, directory, seconds_per_step, world_start_second_of_day=world_start_second_of_day,
        )
        self._economy = economy if economy is not None else Economy()

    def admits_joiner(self, parties: int) -> bool:
        return False

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
                self._gather_context(a, [(b_id, b_name)], state.purpose, step),
                self._gather_context(b, [(a_id, a_name)], state.purpose, step),
                return_exceptions=True,
            )
            verdict = await self._haggle(
                a=a, b=b, a_name=a_name, b_name=b_name, purpose=state.purpose, terms=terms,
                a_ctx=_empty_ctx() if isinstance(a_ctx_r, BaseException) else a_ctx_r,
                b_ctx=_empty_ctx() if isinstance(b_ctx_r, BaseException) else b_ctx_r,
                a_purse=a_purse, b_purse=b_purse, environment=environment, agents=agents, step=step,
            )

        if verdict is None:
            # The haggling never happened: a null step for both, nothing changes hands.
            topic = self._wording(a_name, b_name, terms, 0, 0, False, environment).topic
            null = scene_line(location, f"{topic}。")
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
        words = self._wording(a_name, b_name, terms, give_money, ask_money, agreed, environment)
        transcript = "\n".join(f"{t['speaker']}：{t['line']}" for t in dialogue)
        record = f"{transcript}\n（{words.deal}）"
        outcome = scene_line(location, f"{words.topic}：\n{transcript}\n——{words.deal}")
        gist = scene_line(location, words.gist)
        observations = observed_here(environment, a_id, scene_line(
            location, f"{words.seen}{bystander}" if bystander else words.seen,
        ))

        parties = self._parties(state)
        summary_a, summary_b = await asyncio.gather(
            self._memory_summary(
                agent=a, others=[(b_id, b_name)], transcript=record, purpose=state.purpose,
                role_line=self._role_line(parties, 0)[0], fallback_fact=words.a_fact,
                talk_role="initiator", now_step=step, environment=environment,
                expected_outcome=state.expected_outcome,
            ),
            self._memory_summary(
                agent=b, others=[(a_id, a_name)], transcript=record, purpose=state.purpose,
                role_line=self._role_line(parties, 1)[0], fallback_fact=words.b_fact,
                talk_role="addressee", now_step=step, environment=environment,
            ),
            return_exceptions=True,
        )
        # Whether the deal was struck is the narrator's ruling, not either side's feeling about it.
        why = "" if agreed else (reason or words.deal)
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

    def _wording(
        self, a_name: str, b_name: str, terms: TradeTerms, give_money: int, ask_money: int,
        agreed: bool, environment: "EnvironmentSystem",
    ) -> _Wording:
        def side(entity_id: str, money: int) -> str:
            entity = environment.get_entity(entity_id) if entity_id else None
            parts = [entity.name] if entity is not None else []
            if money:
                parts.append(self._economy.amount_text(money))
            return "与".join(parts)

        a, b = a_name, b_name
        gives = bool(terms.give_entity_id or terms.give_amount)
        asks = bool(terms.ask_entity_id or terms.ask_amount)
        given, taken = side(terms.give_entity_id, give_money), side(terms.ask_entity_id, ask_money)
        if gives and asks:
            deal = (
                f"成交：{a}付出{given or '分文未出'}，{b}付出{taken or '分文未出'}。" if agreed else "没有谈拢。"
            )
            ending = "谈成了" if agreed else "没谈拢"
            return _Wording(
                topic=f"{a}与{b}谈买卖", deal=deal, gist=f"{a}与{b}谈买卖，{deal}",
                seen=f"{a}与{b}谈了一笔买卖" + ("，钱物换了手。" if agreed else "，没有谈拢。"),
                a_fact=f"与{b}谈了一笔买卖，{ending}。", b_fact=f"与{a}谈了一笔买卖，{ending}。",
            )
        if gives:
            offer = side(terms.give_entity_id, terms.give_amount)
            return _Wording(
                topic=f"{a}要把{offer}交给{b}",
                deal=f"{b}收下了{given}。" if agreed else f"{b}没有收。",
                gist=f"{a}把{given}交给了{b}。" if agreed else f"{a}要把{offer}交给{b}，{b}没有收。",
                seen=f"{a}把一些钱物交给了{b}。" if agreed else f"{a}要把一些钱物交给{b}，{b}没有收。",
                a_fact=f"把{given}交给了{b}。" if agreed else f"要把{offer}交给{b}，{b}没有收。",
                b_fact=f"{a}把{given}交给了我，我收下了。" if agreed else f"{a}要把{offer}交给我，我没有收。",
            )
        demand = side(terms.ask_entity_id, terms.ask_amount)
        return _Wording(
            topic=f"{a}向{b}讨要{demand}",
            deal=f"{b}交出了{taken}。" if agreed else f"{b}没有给。",
            gist=f"{b}把{taken}交给了{a}。" if agreed else f"{a}向{b}讨要{demand}，{b}没有给。",
            seen=f"{b}把一些钱物交给了{a}。" if agreed else f"{a}向{b}讨要什么，{b}没有给。",
            a_fact=f"向{b}讨要{demand}，{b}交出了{taken}。" if agreed else f"向{b}讨要{demand}，{b}没有给。",
            b_fact=f"{a}向我讨要{demand}，我交出了{taken}。" if agreed else f"{a}向我讨要{demand}，我没有给。",
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
        for agent, ctx, name in ((a, a_ctx, a_name), (b, b_ctx, b_name)):
            ctx_text = _format_context(
                ctx, name, now_step=step, seconds_per_step=self._seconds_per_step,
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

【这笔交割】（{a_name}提出的条款与盘算，不是已定的事实）
{a_name}的打算：{purpose}
{terms_text}

依上面说定的 JSON 格式（speaker 用 1 代表{a_name}、2 代表{b_name}）输出，只输出 JSON、不写任何多余内容。"""
        facts = (
            GivenFacts()
            .add("行情", prices)
            .add(f"{a_name} 手上的", lanes[0])
            .add(f"{b_name} 手上的", lanes[1])
            .add("现场", scene)
            .add("这笔交割", terms_text)
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
        dialogue = self._parse_dialogue(data, [(a.agent_id, a_name), (b.agent_id, b_name)])
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
        summary: "tuple[str, bool, dict[str, str], str] | None",
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
        fact, _felt_success, directions, _felt_why = summary
        td, ad = _DIRECTION_TALK_DELTAS.get(directions.get(other_id, "neutral"), _DIRECTION_TALK_DELTAS["neutral"])
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
