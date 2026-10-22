"""Social action executor for TALK actions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from agent.relation import format_relation_block, parse_relation_direction, relation_legend
from core.context import GivenFacts, annotate_call, observe_stage
from core.interfaces.action import (
    ActionResult, ActionTarget, ActionType, AgentAction, Observed, Ref, TargetAgentEffect,
)
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import (
    IndexedRef, LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget,
)
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.prompts import (
    ABSOLUTE_TIME_RULE,
    ABSOLUTE_TIME_RULE_FIRST_PERSON,
    condition_line,
    CLOSED_WORLD_FACT_RULE,
    CLOSED_WORLD_FACT_RULE_FIRST_PERSON,
    MEMORY_ORDER_HINT,
    SituationVoice,
    person_referent,
    render_memory_lines,
    render_situation_header,
)
from core.duration import describe_duration
from core.interfaces.execution import TickResult
from engine.executors.base import (
    ActionExecutionState,
    ActionExecutor,
    Conscription,
    Joining,
    JoinOpening,
    participant_action_desc,
)
from engine.narration import (
    format_interrupt_reason_3p, format_interrupt_thought, observed_here, scene_line,
)
from engine.scene import SceneVisibility, assemble_scene_context, observe_location, situation_for

if TYPE_CHECKING:
    from agent.agent import Agent
    from engine.environment import EnvironmentSystem
    from engine.message_system import MessageSystem

logger = get_logger(__name__)

# Each side's LLM-judged direction → (trust_delta, affection_delta): TALK's single source of
# relation change. One notch below PHYSICAL's _REL_DELTA_* (a deed persuades a little more than
# a word). Keep small: relations move by accumulation, and apply_interaction triples negative
# trust.
_DIRECTION_TALK_DELTAS: dict[str, tuple[float, float]] = {
    "positive": (0.03, 0.04),
    "neutral":  (0.01, 0.01),
    "negative": (-0.02, -0.02),
}

# Soft cap on generated turns for two speakers; each walk-in adds a few, so he gets a voice. A
# long TALK means more elapsed time, not more turns.
_MAX_DIALOGUE_TURNS: int = 8
_TURNS_PER_JOINER: int = 2

# The most people one conversation holds. Past this the transcript can't give each a voice; a
# gathering is SEND_MESSAGE's broadcast, or separate talks.
_MAX_TALK_PARTIES: int = 4

# Recent factual memories injected per side, picked as for the goal judge and interrupt evaluation.
_TALK_RECENT_FACTUAL_K = 5
# Memories about each other speaker: the one he came to talk with gets the most.
_ABOUT_ADDRESSED_K = 5
_ABOUT_OTHER_K = 3


def _empty_ctx() -> dict[str, Any]:
    return {"others": [], "about_topic": [], "recent": []}


def _name_list(names: "List[str]") -> str:
    """「甲」/「甲与乙」/「甲、乙与丙」: the speakers as one subject."""
    if len(names) <= 1:
        return "".join(names)
    return "、".join(names[:-1]) + "与" + names[-1]


@dataclass(frozen=True)
class _Party:
    """One speaker in a conversation, as its dialogue and memory calls see him."""

    agent_id: str
    name: str
    # The trace's talk_role: "initiator", "addressee" (invited), or "joiner" (walked in).
    role: str
    # Whom he came to talk with; the addressee's is the initiator.
    addressed_id: str
    # His own act and expectation; the addressee wrote neither.
    description: str
    expected_outcome: str


def _talk_target(other_id: str | None) -> ActionTarget:
    """A talk's target: the other party is acted on and claimed, filed exactly as the decision
    layer does, since this rebuilds the action for feedback."""
    if not other_id:
        return ActionTarget()
    return ActionTarget(acts_on=[Ref.agent(other_id)], claims=[Ref.agent(other_id)])


def _present_listeners(
    candidate_ids: "List[str]",
    *,
    environment: "EnvironmentSystem",
    agents: dict[str, "Agent"],
    anchor_id: str,
    exclude: "set[str]",
) -> List[str]:
    """The named listeners actually standing here. Checked again at complete: listeners don't
    spend their turn and may walk away mid-conversation."""
    here = environment.get_body_location(anchor_id)
    seen: set[str] = set(exclude)
    present: List[str] = []
    for pid in candidate_ids:
        if pid in seen:
            continue
        seen.add(pid)
        agent = agents.get(pid)
        if agent is None or not agent.is_active:
            continue
        if environment.get_body_location(pid) != here:
            continue
        present.append(pid)
    return present


def _known_relation_block(rel, *, include_legend: bool = True) -> str:
    """format_relation_block only if there's a real relationship; strangers get '' so none is
    invented."""
    if rel is None:
        return ""
    labels = list(getattr(rel, "labels", []) or [])
    trust = getattr(rel, "trust", None)
    if trust is None:
        trust = getattr(rel, "trust_objective", 0.5)
    affection = getattr(rel, "affection", None)
    if affection is None:
        affection = getattr(rel, "affection_objective", 0.0)
    if not labels and abs(trust - 0.5) < 0.05 and abs(affection) < 0.05:
        return ""
    return format_relation_block(labels=labels, trust=trust, affection=affection, include_legend=include_legend)


def _format_context(
    ctx: dict, owner_name: str, *, now_step: int, seconds_per_step: int,
    world_start_second_of_day: int = 0,
) -> str:
    """Render gathered context as list items: per other speaker, the relation and what the owner
    knows of him; then the topic and the owner's recent past. Memories go through
    ``render_memory_lines`` for recency prefixes."""
    lines: list[str] = []

    def memory_lines(memories: list) -> list[str]:
        return [
            c for c in render_memory_lines(
                memories, now_step=now_step, seconds_per_step=seconds_per_step,
                world_start_second_of_day=world_start_second_of_day)
            if c
        ]

    for other in ctx.get("others", []):
        other_name = other["name"]
        rel = other.get("relation")
        if rel is not None:
            trust = getattr(rel, "trust", None)
            if trust is None:
                trust = getattr(rel, "trust_objective", 0.5)
            affection = getattr(rel, "affection", None)
            if affection is None:
                affection = getattr(rel, "affection_objective", 0.0)
            labels = list(getattr(rel, "labels", []) or [])
            # The caller adds the legend once for every block.
            block = format_relation_block(labels=labels, trust=trust, affection=affection, include_legend=False)
            lines.append(f"- 与{other_name}的关系：{block}")
            history = getattr(rel, "history_summary", "") or ""
            if history:
                lines.append(f"- 与{other_name}的过往：{history}")
        about = memory_lines(other.get("about", []))
        if about:
            lines.append(f"- {owner_name}对{other_name}的认识（{MEMORY_ORDER_HINT}）：")
            lines.extend(f"  - {c}" for c in about)

    about_topic = memory_lines(ctx.get("about_topic", []))
    if about_topic:
        # The heading doesn't name the purpose: it's the initiator's private calculation and
        # would leak into the other columns. And it only claims "possibly related": these are
        # semantic neighbors, often not about the other people at all.
        lines.append(f"- {owner_name}记得的其他事，或许与这次交谈有关（{MEMORY_ORDER_HINT}）：")
        lines.extend(f"  - {c}" for c in about_topic)

    recent = memory_lines(ctx.get("recent", []))
    if recent:
        lines.append(f"- {owner_name}近来的经历（{MEMORY_ORDER_HINT}）：")
        lines.extend(f"  - {c}" for c in recent)

    return "\n".join(lines)


#: The byte-identical system prefix for dialogue generation (prefix cache).
_DIALOGUE_SYSTEM: str = f"""\
【任务】
你是中立的叙事者。基于给定的每个说话人各自的处境（两人或几人），客观写出他们之间的一段对话。你不代入任何一方，只如实呈现这场交谈会怎样发生。

【约束】
- **谈完之后，有一点东西该不一样了，有实质的变化。** 面对面说话，是他们此刻改变「谁知道什么」的机会：把自己
  那一栏里别人还不知道的说出去一点、把话挑明、把某件事定死、翻脸、松口、死咬着不松口——都算，
  也远不止这几样，怎样都行，只要不是白谈一场。
- **不一定谈成。** 谈崩、被搪塞、话说一半就散，都是正当结局；不正当的是谈完之后，他们之间跟开口
  之前一模一样。同时，对于谈话的各方来说，也不一定一味地对抗，谈不成或者不欢而散，而是要根据实际情况，灵活应对。
- 实在没有新东西可给的时候，让他说不出口、说不清楚、或是不肯说——**别让他把自己那一栏里写着
  「已经说过、已经问过」的话，换个说法再说一遍。**
- 对白要从各自的人设、情绪、彼此之间关系和位置与已知信息自然长出，而非为推进话题服务：上面那
  一点变化不是你安排出来的，是他们各自想要的东西逼出来的。
- **每个人只能说出他自己那一栏里有的东西。** 一栏里写着的事，别人未必知道、更未必认账；绝不可让任
  何人承认、证实或否认一件只有别人那栏才写着的事，也不可替别人把他还没亮的底牌先亮出来。
- **这是一场只动嘴的交谈：钱物不换手。** 不写递钱、塞银子、交出或收下东西；谁想给钱给东西、向人要钱要
  东西，只能说出口、讲条件，此刻东西仍在原主手上。
- **各栏说法相左时，那不是要你调和的错漏**——那正是这场交谈的关节。他们各自都当自己那套是真的，
  把这个演出来，别把它抹平成一套说得通的说法。允许他们误解、话不投机乃至不欢而散。
- **有人中途走过来加入时**（见【中途加入】；没有这一栏就略过这一条）：原先在谈的人怎样待他，由各自的处境、
  彼此关系和正在谈的事定，当着他还谈不谈原来的事、谈到几分，也由他们定。别当他没来过，也别让他一来就
  把整场话头抢走；他插不插得上话、碰不碰钉子，看他自己那一栏和别人怎么看他。
- 不要说教、不要把心事直白说尽、不要背景信息倾倒、不要书面腔。
- 对于人物的生死问题，需要特别关注，如果出现死者复现的问题，比如某人已经死亡了，但是现在感知到他在场，这个我需要关注并思考什么这样。
- 语言简练口语，长短随情境，不必每句等长。
- 如果对话中要提及某人，一定要符合逻辑和常量和地理空间规则，比如不能让谁命令一个不在场的人去攻击别人。
- 谈完即止；轮数上限由输入给出。
- 落笔前先看清【现场】给的当前时间与当前地点，避免在对白里的时间和处所错乱。同时有一点必须要注意时间已经过了并不代表某件之前约定好的事情一定发生，它是否发生需要以给定的上下文为准。
{CLOSED_WORLD_FACT_RULE}

【输出】
严格输出以下 JSON，不要任何多余内容（speaker 填说话人的编号，对应下方【1】【2】……各栏的人）。
三个字段按下面的次序写：先 reason（想清楚这场交谈），再据此写 dialogue，再由 dialogue 推出 observation。
{{"reason": "…", "dialogue": [{{"speaker": 说话人编号, "line": "这句话，≤60字"}}, ...], "observation": "旁人视角看见的这场交谈（≤25字）"}}

reason：动笔写对白之前，先把这场交谈想一遍（≤80字）——每个人此刻要什么、各自知道和不知道什么、
这话谈得下去还是谈不下去、谁会先撑不住。想到什么写什么，不限于这几样。它只为让你落笔时有据可依，
不进对白、不进 observation。

observation 是**一个不在近旁、听不见内容的路人**看到的样子：他只见这几个人的神色、姿态。
- 不必每场都看出戏来：**多数交谈从外面看就是平平无奇的**，看不出什么就把这一项留空。
- 不要为了让这句话有内容，而捏造神色、姿态、收场加戏。
**主语不必你写**：系统会在你这句前面补上「谁与谁的交谈：」，你只接着写他们的举止与收场。
**绝不可写出任何谈话内容、话题或任何一方的意图**（那是他无从得知的）；也不要写他们心里怎么想。

落笔写上面那些对白之前，还有一条：
{ABSOLUTE_TIME_RULE}"""


class SocialExecutor(ActionExecutor):
    """Executor for TALK actions.

    - Started 1v1. Someone deciding to talk with a participant walks into the conversation
      instead of being turned away (``admits_joiner`` / ``join``), up to ``_MAX_TALK_PARTIES``.
      Extra people named become listeners (ActionTarget.reaches): their bodies stay free and
      they only get an overheard TargetAgentEffect. Group announcements belong to SEND_MESSAGE.
    - The whole dialogue is generated once at complete(); ticks are content-free.
    - One neutral narrator voices every side. Accepted cost: one prompt holds every side's
      private memories, so asymmetry can leak at generation (and persist via memory_summary).
    - Each side's memory_summary is its own first-person record.
    - Anyone breaking off ends the conversation for everyone in it.
    """

    # By invitation: if the other person is busy, the talk fails (see Conscription).
    conscription = Conscription.INVITE

    def __init__(
        self, llm_router: LLMRouter, directory: WorldDirectory, seconds_per_step: int = 3600,
        *, world_start_second_of_day: int = 0,
    ) -> None:
        self._llm = llm_router
        self._directory = directory
        self._seconds_per_step = seconds_per_step
        # The memory prefix's "今日/昨日" counts midnights crossed (see render_memory).
        self._world_start_second_of_day = world_start_second_of_day

    def claim_bodies(
        self,
        action: AgentAction,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
    ) -> list[str]:
        """Whose turn this conversation takes, by start()'s criteria. An infeasible talk takes
        nobody's, or "he isn't here" would be reported as "he's busy"."""
        target_ids = action.target.acted_on_agents[:1]
        if not target_ids or target_ids[0] not in agents:
            return []
        if not environment.check_talk_feasibility(
            action.agent_id, target_ids, target_label="对方",
        ).ok:
            return []
        return [target_ids[0]]

    def admits_joiner(self, parties: int) -> bool:
        return parties < _MAX_TALK_PARTIES

    def join(
        self,
        state: ActionExecutionState,
        action: AgentAction,
        step: int,
        *,
        environment: "EnvironmentSystem",
    ) -> JoinOpening | None:
        """Walk ``action``'s actor into this conversation. Arbitration checked he stands with the
        one he went up to; what the others make of him is the dialogue's to tell."""
        joiner_id = action.agent_id
        if joiner_id in state.participant_ids or not self.admits_joiner(len(state.participant_ids)):
            return None
        talkers = _name_list([self._directory.agent_name(pid) for pid in state.participant_ids])
        state.participant_ids.append(joiner_id)
        state.joined[joiner_id] = Joining(
            description=action.action_description or "交谈",
            target=action.target,
            expected_outcome=action.expected_outcome,
        )
        state.listener_ids = [pid for pid in state.listener_ids if pid != joiner_id]
        location = observe_location(environment, joiner_id)
        opening = scene_line(
            location, f"{self._directory.agent_name(joiner_id)}凑了过去，加入了{talkers}的交谈。",
        )
        return JoinOpening(outcome=opening, observations=observed_here(environment, joiner_id, opening))

    async def start(
        self,
        action: AgentAction,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> ActionExecutionState:
        target_ids = action.target.acted_on_agents
        # The dialogue starts 1v1; anyone else the decision layer bound is REACHED, not acted on.
        if len(target_ids) > 1:
            target_ids = target_ids[:1]

        target_label = self._directory.agent_name(target_ids[0]) if target_ids else "对方"

        location = observe_location(environment, action.agent_id)

        feasibility = environment.check_talk_feasibility(
            action.agent_id, target_ids, target_label=target_label,
        )
        if not feasibility.ok:
            return self._failed_result(action, step, feasibility.reason, location)

        if target_ids[0] not in agents:
            return self._failed_result(action, step, f"没有找到{target_label}，无法交谈。", location)

        duration = max(action.estimated_steps, 1)
        participant_ids = [action.agent_id, target_ids[0]]
        initiator_name = self._directory.agent_name(action.agent_id)
        target_name = self._directory.agent_name(target_ids[0])
        # The opening is public; the privileged transcript appears only in complete's outcome.
        opening = scene_line(location, f"{initiator_name}与{target_name}开始聊了起来。")
        state = ActionExecutionState.create(
            action_type=ActionType.TALK,
            initiator_id=action.agent_id,
            participant_ids=participant_ids,
            purpose=action.action_description or "对话",
            started_step=step,
            estimated_steps=duration,
            opening_outcome=opening,
            opening_observations=observed_here(environment, action.agent_id, opening),
            target=action.target,
            expected_outcome=action.expected_outcome,
            listener_ids=_present_listeners(
                action.target.reached_agents, environment=environment, agents=agents,
                anchor_id=action.agent_id, exclude={action.agent_id, target_ids[0]},
            ),
        )
        state.extra["target_id"] = target_ids[0]
        return state

    def _failed_result(self, action: AgentAction, step: int, outcome: str, location: str) -> ActionExecutionState:
        """A TALK feasibility failure as an actor-only ``create_failed`` execution."""
        stub = AgentAction(
            agent_id=action.agent_id,
            step=step,
            action_type=ActionType.TALK,
            action_description=action.action_description or "talk",
            target=action.target,
        )
        actor_name = self._directory.agent_name(action.agent_id)
        failure = ActionResult(
            action=stub,
            expected_outcome=action.expected_outcome,
            # Quoted: unquoted, his first-person words render as "甲本想我去劝他…".
            outcome=scene_line(location, f"{actor_name}本想做「{action.action_description}」，{outcome}"),
            # Nothing happened, and the intent is private.
            observations=[],
            succeeded=False,
            failure_reason=outcome,
            factual_memory=f"想要做：{action.action_description}，但是结果是：{outcome}",
        )
        return ActionExecutionState.create_failed(
            action_type=ActionType.TALK, initiator_id=action.agent_id,
            failure_result=failure, started_step=step,
            purpose=action.action_description or "对话",
        )

    def _parties(self, state: ActionExecutionState) -> list[_Party]:
        """The speakers in the order they came: initiator, addressee, then each walk-in."""
        target_id = state.extra.get("target_id", "")
        parties = [_Party(
            agent_id=state.initiator_id, name=self._directory.agent_name(state.initiator_id),
            role="initiator", addressed_id=target_id, description=state.purpose,
            expected_outcome=state.expected_outcome,
        )]
        if target_id:
            parties.append(_Party(
                agent_id=target_id, name=self._directory.agent_name(target_id), role="addressee",
                addressed_id=state.initiator_id,
                description=participant_action_desc(self._directory, target_id, state),
                expected_outcome="",
            ))
        for pid in state.participant_ids:
            joining = state.joined.get(pid)
            if joining is None:
                continue
            addressed = next(
                (a for a in joining.target.acted_on_agents if a != pid and a in state.participant_ids),
                target_id,
            )
            parties.append(_Party(
                agent_id=pid, name=self._directory.agent_name(pid), role="joiner",
                addressed_id=addressed, description=joining.description,
                expected_outcome=joining.expected_outcome,
            ))
        return parties

    @staticmethod
    def _role_line(parties: list[_Party], index: int) -> tuple[str, str]:
        """How party ``index`` came into this conversation, first person: the line his memory
        prompts open with, and the head of his fallback record."""
        me = parties[index]
        names = {p.agent_id: p.name for p in parties}
        addressed = names.get(me.addressed_id, "对方")
        if me.role == "initiator":
            line, head = f"是我主动找{addressed}谈的", f"我主动找{addressed}"
        elif me.role == "addressee":
            line, head = f"{addressed}来找我谈", f"{addressed}找我"
        else:
            earlier = _name_list([p.name for p in parties[:index]])
            line, head = f"{earlier}正谈着，我走过去找{addressed}，加了进来", f"我加进{earlier}的交谈"
        later = [p.name for p in parties[index + 1:] if p.role == "joiner"]
        if later:
            line += f"，{_name_list(later)}也凑过来一起谈"
        return line, head

    async def tick(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem",
        message_system: "MessageSystem",
    ) -> List[TickResult]:
        elapsed = state.estimated_steps - state.remaining_steps
        talkers = _name_list([p.name for p in self._parties(state)])
        location = observe_location(environment, state.initiator_id)
        if elapsed <= 0:
            body = f"{talkers}的交谈刚开始。"
        else:
            body = f"{talkers}的交谈仍在继续，已持续{describe_duration(elapsed, self._seconds_per_step)}。"
        narrative = scene_line(location, body)
        # Content-free, so public: observation == outcome.
        return [
            TickResult(
                agent_id=pid, outcome=narrative,
                observations=observed_here(environment, pid, narrative),
            )
            for pid in state.participant_ids
        ]

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
            return [stored]  # feasibility failure (no pairing)
        initiator_id = state.initiator_id
        parties = self._parties(state)
        talkers = _name_list([p.name for p in parties])
        location = observe_location(environment, initiator_id)

        def others_of(party: _Party) -> list[tuple[str, str]]:
            """The other speakers, the one he came to talk with first."""
            others = [(p.agent_id, p.name) for p in parties if p.agent_id != party.agent_id]
            return sorted(others, key=lambda o: o[0] != party.addressed_id)

        # Per participant: relation + memories about each other speaker + about the topic.
        gathered_ctx = await asyncio.gather(
            *(
                self._gather_context(agents.get(p.agent_id), others_of(p), state.purpose, step)
                for p in parties
            ),
            return_exceptions=True,
        )
        contexts = [_empty_ctx() if isinstance(c, BaseException) else c for c in gathered_ctx]

        scene = assemble_scene_context(
            initiator_id, environment=environment, directory=self._directory, agents=agents,
            visibility=SceneVisibility.GOD,  # functional judge
        ).text

        # A speaker gone missing drops out; without the first two there is no conversation.
        speakers = [
            (agents[p.agent_id], p, ctx) for p, ctx in zip(parties, contexts) if p.agent_id in agents
        ]
        dialogue: list[dict] = []
        bystander_view = ""
        if len(speakers) >= 2 and all(p.agent_id in agents for p in parties[:2]):
            dialogue, bystander_view = await self._llm_full_dialogue(
                speakers=speakers,
                purpose=state.purpose,
                expected_outcome=state.expected_outcome,
                # Don't use min(estimated_steps, MAX): a 1-step TALK would become a monologue.
                turns=_MAX_DIALOGUE_TURNS + _TURNS_PER_JOINER * (len(speakers) - 2),
                scene=scene,
                now_step=step,
            )

        transcript = (
            "\n".join(f"{t['speaker']}：{t['line']}" for t in dialogue) if dialogue else "无对话"
        )

        # No transcript, no conversation: ask nothing, or an infrastructure failure becomes the
        # narrative fact "we talked" (Rule 1). summary=None is the null-step exit.
        listener_ids: list[str] = []
        listener_facts: list[Any] = []
        summaries: list["tuple[str, bool, dict[str, str], str] | None"] = [None] * len(parties)
        if dialogue:
            # All first-person memories depend only on the transcript: one batch.
            listener_ids = _present_listeners(
                state.listener_ids, environment=environment, agents=agents,
                anchor_id=initiator_id, exclude=set(state.participant_ids),
            )
            results_r = await asyncio.gather(
                *(
                    self._memory_summary(
                        agent=agents.get(p.agent_id), others=others_of(p), transcript=transcript,
                        purpose=state.purpose, role_line=self._role_line(parties, i)[0],
                        fallback_fact=f"与{_name_list([n for _, n in others_of(p)])}就「{state.purpose}」谈了一场。",
                        talk_role=p.role, now_step=step, environment=environment,
                        expected_outcome=p.expected_outcome,
                    )
                    for i, p in enumerate(parties)
                ),
                *(
                    self._listener_memory(
                        agent=agents.get(pid), talkers=talkers, transcript=transcript,
                        now_step=step, environment=environment,
                    )
                    for pid in listener_ids
                ),
                return_exceptions=True,
            )
            summaries = [
                None if isinstance(r, BaseException) else r for r in results_r[:len(parties)]
            ]
            listener_facts = list(results_r[len(parties):])

        # outcome carries the transcript; onlookers only see demeanour and how it ended.
        gist = scene_line(location, f"{talkers}交谈。")
        outcome = scene_line(location, f"{talkers}的交谈：\n{transcript}")
        # Code supplies the subject: left to the LLM it writes "二人", losing "who".
        observations = observed_here(environment, initiator_id, scene_line(
            location,
            f"{talkers}的交谈：{bystander_view}" if bystander_view else f"{talkers}在交谈。",
        ))
        listener_effects: List[TargetAgentEffect] = []
        for pid, fact in zip(listener_ids, listener_facts):
            if isinstance(fact, BaseException) or not fact:
                logger.warning(
                    "talk_listener_memory_skipped",
                    extra={"agent_id": pid, "step": step},
                )
                continue
            listener_effects.append(
                TargetAgentEffect(agent_id=pid, factual_memory=fact, overheard=True)
            )

        results: List[ActionResult] = []
        for i, (party, summary) in enumerate(zip(parties, summaries)):
            # The initiator always gets his result; a missing other has nobody to give it to.
            if i > 0 and party.agent_id not in agents:
                continue
            results.append(self._talk_result(
                agent_id=party.agent_id, step=step,
                action_description=party.description,
                estimated_steps=state.estimated_steps, addressed_id=party.addressed_id,
                outcome=outcome, gist=gist, observations=observations,
                dialogue=dialogue,
                # Don't fall back to purpose: it's intent, not expectation.
                summary=summary, expected_outcome=party.expected_outcome,
                # Attached to the initiator's result only (attaching to all → listeners get it twice).
                target_effects=listener_effects if i == 0 else None,
            ))

        return results

    @staticmethod
    def _talk_result(
        *,
        agent_id: str,
        step: int,
        action_description: str,
        estimated_steps: int,
        addressed_id: str,
        outcome: str,
        gist: str,
        observations: list[Observed],
        dialogue: list[dict],
        summary: "tuple[str, bool, dict[str, str], str] | None",
        expected_outcome: str,
        target_effects: "list[TargetAgentEffect] | None" = None,
    ) -> ActionResult:
        """Build one participant's TALK result.

        ``action_description`` is this person's own act (the conscripted party's differs from
        the shared topic). ``summary=None`` is a null step: no memory, no relation delta.
        Listeners' ``target_effects`` still go out then: what they heard doesn't depend on this
        summary.
        """
        stub = AgentAction(
            agent_id=agent_id, step=step, action_type=ActionType.TALK,
            action_description=action_description,
            target=_talk_target(addressed_id),
            estimated_steps=estimated_steps,
        )
        if summary is None:
            return ActionResult(
                action=stub,
                expected_outcome=expected_outcome,
                outcome=outcome,
                gist=gist,
                observations=observations,
                succeeded=False,
                # No factual_memory: feedback skips null steps before reading it.
                dialogue=dialogue,
                relation_updates=[],
                adjudication_failed=True,
                target_effects=list(target_effects or []),
            )
        fact, succeeded, directions, why = summary
        relation_updates = [
            (other_id, *_DIRECTION_TALK_DELTAS.get(direction, _DIRECTION_TALK_DELTAS["neutral"]))
            for other_id, direction in directions.items()
        ]
        return ActionResult(
            action=stub,
            expected_outcome=expected_outcome,
            outcome=outcome,
            gist=gist,
            observations=observations,
            # A lurker hears the transcript. Not given on interrupt.
            happening=outcome,
            succeeded=succeeded,
            failure_reason=why,     # 3p authoritative "why the talk didn't work"
            factual_memory=fact,
            dialogue=dialogue,
            relation_updates=relation_updates,
            target_effects=list(target_effects or []),
        )

    async def interrupt(
        self,
        state: ActionExecutionState,
        step: int,
        *,
        agents: dict[str, "Agent"],
        environment: "EnvironmentSystem | None" = None,
        interrupted_agent_id: str | None = None,
        thought: str = "",
        cause: str = "",
    ) -> List[ActionResult]:
        initiator_id = state.initiator_id
        parties = self._parties(state)
        talkers = _name_list([p.name for p in parties])
        location = observe_location(environment, initiator_id) if environment is not None else "此处"
        elapsed = state.estimated_steps - state.remaining_steps
        elapsed_label = describe_duration(elapsed, self._seconds_per_step)
        breaker = (
            self._directory.agent_name(interrupted_agent_id)
            if interrupted_agent_id and interrupted_agent_id in state.participant_ids
            else ""
        )
        if cause:
            cut = f"因{cause}而中断"
        elif breaker:
            cut = f"被{breaker}打断"
        else:
            cut = "被打断"
        # The breaker's thought is quoted only on his own result.
        what_happened = f"{talkers}的交谈{cut}。"
        breaker_id = (
            interrupted_agent_id if interrupted_agent_id in state.participant_ids else initiator_id
        )
        breaker_outcome = scene_line(
            location, what_happened + format_interrupt_reason_3p(breaker or parties[0].name, thought),
        )
        others_outcome = scene_line(location, what_happened)
        # Who walked out, as the rest saw it: with one other that is just 对方.
        leaver = "对方" if len(parties) <= 2 else (breaker or "有人")
        results: List[ActionResult] = []
        for i, party in enumerate(parties):
            agent = agents.get(party.agent_id)
            if agent is None:
                continue
            # The breaker knows why; the others only see him break off. Key on the interrupted
            # participant, not the trigger source (rarely a participant). Unknown → all know.
            is_triggered = (interrupted_agent_id is None or party.agent_id == interrupted_agent_id)
            # Own eyes, first person; _interrupt_summary supplies its own header.
            scene = (
                assemble_scene_context(
                    party.agent_id, environment=environment, directory=self._directory, agents=agents,
                    include_header=False, voice=SituationVoice.FIRST,
                    visibility=SceneVisibility.OWN_EYES,
                ).text
                if environment is not None else ""
            )
            relation_block = await self._present_relations_block(agent, environment)
            role_line, head = self._role_line(parties, i)
            factual = await self._interrupt_summary(
                agent=agent, role_line=role_line, fallback_head=head, purpose=state.purpose,
                thought=thought, cause=cause, elapsed_label=elapsed_label,
                is_triggered=is_triggered, leaver=leaver, talk_role=party.role, scene=scene,
                environment=environment, now_step=step, relation_block=relation_block,
            )
            stub = AgentAction(
                agent_id=party.agent_id, step=step, action_type=ActionType.TALK,
                action_description=party.description,
                target=_talk_target(party.addressed_id),
                estimated_steps=state.estimated_steps,
            )
            # An interrupted conversation moves no relation.
            results.append(ActionResult(
                action=stub,
                # Only the initiator and walk-ins wrote an expectation.
                expected_outcome=party.expected_outcome,
                outcome=breaker_outcome if party.agent_id == breaker_id else others_outcome,
                gist=others_outcome,
                succeeded=False,
                factual_memory=factual,     # 1p (_interrupt_summary) → own memory
            ))
        return results

    async def _present_relations_block(self, agent: "Agent", environment: "EnvironmentSystem | None") -> str:
        """The agent's own relation to each co-present person it knows (strangers omitted)."""
        if environment is None:
            return ""
        loc = environment.get_body_location(agent.agent_id)
        # Cognitive tier only: an Npc holds no relation with anyone.
        present = [a for a in environment.agents_at(loc) if a != agent.agent_id]
        lines: list[str] = []
        for pid in present:
            try:
                rel = await agent.relation_system.perceive_existing(
                    pid, emotion=agent.personality.state.emotion)
            except Exception as exc:  # noqa: BLE001 — read failed: skip this relation block
                logger.warning(
                    "relation_perceive_failed",
                    extra={"agent_id": agent.agent_id, "target_id": pid, "error": str(exc)},
                )
                continue
            block = _known_relation_block(rel, include_legend=False)
            if block:
                lines.append(f"- 与在场的{self._directory.agent_name(pid)}：{block}")
        if not lines:
            return ""
        return "\n".join(lines) + f"\n（关系数值含义：{relation_legend()}）"

    async def _interrupt_summary(
        self,
        *,
        agent: "Agent",
        role_line: str,
        fallback_head: str,
        purpose: str,
        thought: str,
        cause: str,
        elapsed_label: str,
        is_triggered: bool,
        leaver: str,
        talk_role: str,
        scene: str,
        environment: "EnvironmentSystem | None" = None,
        now_step: int = 0,
        relation_block: str = "",
    ) -> str:
        """First-person record of an interrupted conversation (one per participant).

        ``role_line`` / ``fallback_head`` say how he came into it (``_role_line``); ``leaver``
        names whoever broke off, as those left behind saw it."""
        thought_part = format_interrupt_thought(thought)
        # The trigger signal stays out (see base.interrupt).
        if cause:
            # Nobody chose to stop, so all share this sentence; the cause must be stated or the
            # model invents one from purpose.
            situation = f"这场谈话因{cause}而中断"
            fallback = f"{fallback_head}就「{purpose}」谈到一半（{elapsed_label}），{situation}"
        elif is_triggered:
            situation = f"我中途撂下了这场谈话{thought_part}"
            fallback = f"{fallback_head}就「{purpose}」谈到一半（{elapsed_label}），我中途撂下了这场谈话{thought_part}"
        else:
            situation = f"{leaver}突然中止了交谈"
            fallback = f"{fallback_head}就「{purpose}」谈到一半，{situation}"

        if agent is None:
            return fallback

        scene_part = f"\n【现场】\n{scene}" if scene else ""
        relation_part = f"\n【我与在场者的关系】\n{relation_block}" if relation_block else ""
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场被打断的对话——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角,这场被打断的对话让我记住了什么、留下什么感受。始终用我自己的口吻。
提到我认识的人时,不妨带上其名字便于日后回想(惯以称谓相称的话,可附成「称谓（名字）」);这只是建议、不必生硬套用,陌生人用描述。
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "一句话（不超过36字）"}
"""
        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        # Without a time/place anchor the model takes place names from purpose.
        situation_header = (
            render_situation_header(situation_for(environment, agent.agent_id),
                                    voice=SituationVoice.FIRST)
            if environment is not None else ""
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【刚刚发生的事】
{role_line}，正就「{purpose}」交谈，已谈了{elapsed_label}，{situation}。{relation_part}{scene_part}

依上面说定的 JSON 格式记下这一场，只输出 JSON、不写任何多余内容。"""
        try:
            # Attribute the trace to this participant. talk_role lets review exempt the addressee
            # from the decision↔action consistency check.
            with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
                given_facts=GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("刚刚发生的事",
                     f"{role_line}，正就「{purpose}」交谈，已谈了{elapsed_label}，{situation}。")
                .add("我与在场者的关系", relation_block)
                .add("现场", scene),
                talk_role=talk_role,
                action_owner=agent.personality.soul.name,
            ):
                resp = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    max_tokens=output_budget(69),  # fact ≤36 chars; est. ~69 tok
                json_mode=True,
                )
            return str(extract_json(resp.content).get("fact", "")).strip() or fallback
        except Exception as exc:
            logger.warning("talk_interrupt_summary_failed", extra={"agent_id": agent.agent_id, "error": str(exc)})
            return fallback

    async def _gather_context(
        self,
        agent: "Agent | None",
        others: list[tuple[str, str]],
        purpose: str,
        step: int,
    ) -> dict:
        """Collect, for each other speaker ``(id, name)`` (the one he came to talk with first),
        the relation and memories about him; then memories about the topic and his recent past.
        Each read failure only empties its own column."""
        if agent is None:
            return _empty_ctx()

        # Fetched first: otherwise the other columns take the latest key event and this timeline
        # loses its key link. Fetch extra (as Agent._recent_factual_lines) to survive dropping
        # entries without factual.
        recent: list = []
        try:
            pairs = agent.memory_system.sample_recent_events(step, top_k=_TALK_RECENT_FACTUAL_K + 8)
        except Exception as exc:  # noqa: BLE001 — read failed: drop recent events, continue
            logger.warning(
                "social_recent_memory_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            pairs = []
        recent = [fact for fact, _exp in pairs if fact is not None][:_TALK_RECENT_FACTUAL_K]
        seen: set[str] = {m.id for m in recent}

        # exclude_ids inside retrieval, so each column still fills its quota; run in order.
        # "What I know of him" filters by his id, then ranks by purpose. Don't fall back to
        # retrieve(name): in a single-theme world that recalls the topic, not the person.
        # Empty means "I don't know him".
        others_ctx: list[dict[str, Any]] = []
        for rank, (other_id, other_name) in enumerate(others):
            relation = None
            try:
                relation = await agent.relation_system.perceive_existing(
                    other_id, emotion=agent.personality.state.emotion,
                )
            except Exception as exc:  # noqa: BLE001 — read failed: relation is None, continue
                logger.warning(
                    "relation_perceive_failed",
                    extra={"agent_id": agent.agent_id, "target_id": other_id, "error": str(exc)},
                )
            about: list = []
            try:
                about = await agent.memory_system.recall_about_agent(
                    other_id, purpose, current_step=step,
                    top_k=_ABOUT_ADDRESSED_K if rank == 0 else _ABOUT_OTHER_K, exclude_ids=seen,
                )
            except Exception as exc:  # noqa: BLE001 — read failed: drop this column, continue
                logger.warning(
                    "social_memory_retrieve_failed",
                    extra={"agent_id": agent.agent_id, "target_id": other_id, "error": str(exc)},
                )
            seen |= {m.id for m in about}
            others_ctx.append({"id": other_id, "name": other_name, "relation": relation, "about": about})

        about_topic: list = []
        try:
            about_topic = await agent.memory_system.retrieve(
                purpose, current_step=step, top_k=5, exclude_ids=seen,
            )
        except Exception as exc:  # noqa: BLE001 — read failed: drop this column, continue
            logger.warning(
                "social_memory_retrieve_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )

        return {"others": others_ctx, "about_topic": about_topic, "recent": recent}

    async def _llm_full_dialogue(
        self,
        *,
        speakers: "list[tuple[Agent, _Party, dict]]",
        purpose: str,
        expected_outcome: str,
        turns: int,
        scene: str,
        now_step: int,
    ) -> tuple[list[dict], str]:
        """Generate the full conversation in one functional call.

        ``speakers`` are ``(agent, party, gathered context)`` in party order, the initiator first
        and the one he addressed second. Returns (turns, observation): turns are
        ``{"speaker_id", "speaker", "line"}``; ``([], "")`` on failure, never a fabricated line.
        Speakers come back as their 1-based lane numbers.
        """
        lanes: list[str] = []
        conditions: list[str] = []
        ctx_texts: list[str] = []
        for agent, party, ctx in speakers:
            ctx_text = _format_context(
                ctx, party.name, now_step=now_step, seconds_per_step=self._seconds_per_step,
                world_start_second_of_day=self._world_start_second_of_day,
            )
            condition = condition_line(
                agent.personality.state.condition, voice=SituationVoice.THIRD,
                now_step=now_step, seconds_per_step=self._seconds_per_step,
            )
            ctx_texts.append(ctx_text)
            conditions.append(condition)
            # Shared with the trace annotation, so the leak review reads exactly what the model saw.
            lanes.append(
                f"{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}"
                f"{condition}\n{ctx_text}"
            )
        initiator = speakers[0][1]
        names = {party.agent_id: party.name for _a, party, _c in speakers}
        addressed_name = names.get(initiator.addressed_id, "对方")
        expected_part = f"\n对话发起者的期望：{expected_outcome}" if expected_outcome else ""
        scene_part = f"\n【现场】\n{scene}" if scene else ""
        joiner_lines = [
            f"- {party.name}走过来找{names.get(party.addressed_id, '他们')}，想「{party.description}」。"
            + (f"期望：{party.expected_outcome}" if party.expected_outcome else "")
            for _a, party, _c in speakers if party.role == "joiner"
        ]
        joiner_part = (
            "\n\n【中途加入】（以下各是走过来那人单方面的说法与盘算，同上）\n" + "\n".join(joiner_lines)
            if joiner_lines else ""
        )
        # One legend for every relation block, only if any has one.
        has_relation = any(
            other.get("relation") for _a, _p, ctx in speakers for other in ctx.get("others", [])
        )
        legend_block = f"关系数值含义：{relation_legend()}\n\n" if has_relation else ""
        lane_blocks = "\n\n".join(
            f"【{i}：{party.name}】\n{lane}"
            for i, ((_a, party, _c), lane) in enumerate(zip(speakers, lanes), start=1)
        )
        numbering = "、".join(f"{i} 是{party.name}" for i, (_a, party, _c) in enumerate(speakers, start=1))
        system = _DIALOGUE_SYSTEM
        user = f"""\
{legend_block}{lane_blocks}

【对话目的】（以下是{initiator.name}单方面的说法与盘算，既非已定的事实，也非这场交谈该有的结局）
{initiator.name}想就「{purpose}」与{addressed_name}交谈。{expected_part}{joiner_part}{scene_part}

【轮数上限】
至多 {turns} 轮，可更少；谈完即止。

依上面说定的 JSON 格式（speaker 填说话人编号：{numbering}）输出，只输出 JSON、不写任何多余内容。"""
        # Same source as user above: every fact this dialogue generation can see.
        facts = GivenFacts()
        for (_a, party, _c), condition, ctx_text in zip(speakers, conditions, ctx_texts):
            facts = facts.add(f"{party.name} 的处境", condition).add(f"{party.name} 手上的", ctx_text)
        facts = (
            facts
            .add("对话目的", f"{initiator.name}想就「{purpose}」与{addressed_name}交谈。")
            .add("发起者的期望", expected_outcome)
            .add("中途加入", "\n".join(joiner_lines))
            .add("现场", scene)
            .add("轮数上限", turns)
        )
        dialogue_lanes = {
            chr(ord("a") + i): {"name": party.name, "text": lane}
            for i, ((_a, party, _c), lane) in enumerate(zip(speakers, lanes))
        }
        # One semantic retry (as DecisionEngine._llm_select) for broken JSON or an
        # unattributable line; same prompt, same bar, never filling in lines.
        for attempt in (1, 2):
            reason = ""
            try:
                with annotate_call(given_facts=facts, dialogue_lanes=dialogue_lanes):
                    response = await self._llm.complete(
                        LLMScene.AGENT_ACTION_NARRATION,
                        [
                            LLMMessage(role="system", content=system),
                            LLMMessage(role="user", content=user),
                        ],
                        temperature=0.8,
                        # Per turn {speaker, line ≤60 chars ≈ 90 tok} ≈ 100 tok, plus reason (≤80 chars
                        # ≈ 120 tok), observation (≤25 chars ≈ 38 tok) and the three fields' structure
                        # (~15 tok).
                        max_tokens=output_budget(turns * 100 + 173),
                        json_mode=True,
                    )
                data = extract_json(response.content)
                parsed = self._parse_dialogue(
                    data, [(party.agent_id, party.name) for _a, party, _c in speakers],
                )
                if parsed:
                    observation = " ".join(str(data.get("observation", "")).split()).strip()
                    return parsed, observation
                reason = "transcript_unattributable"
            except Exception as exc:
                reason = str(exc)
            logger.warning(
                "talk_dialogue_failed",
                extra={"initiator": initiator.agent_id, "attempt": attempt, "reason": reason},
            )
        return [], ""

    @staticmethod
    def _parse_dialogue(data: object, speakers: list[tuple[str, str]]) -> list[dict]:
        """Map ``{"dialogue":[{"speaker":N,"line":...}]}`` to turns tagged with each speaker's
        ``(agent_id, name)``; N is the 1-based lane number.

        One unattributable line voids the whole transcript: a readable stub still looks like a
        finished conversation and gets summarized as one. Empty lines are skipped.
        """
        turns: list[dict] = []
        raw = data.get("dialogue", []) if isinstance(data, dict) else []
        if not isinstance(raw, list):
            return turns
        ref = IndexedRef(str(i) for i in range(len(speakers)))
        for t in raw:
            idx = t.get("speaker") if isinstance(t, dict) else None
            line = str(t.get("line", "")).strip() if isinstance(t, dict) else ""
            if isinstance(t, dict) and not line:
                continue
            resolved = ref.resolve([idx])
            if not resolved:
                logger.warning(
                    "talk_dialogue_unattributable",
                    extra={"speaker": repr(idx), "readable": len(turns), "total": len(raw)},
                )
                return []
            who = speakers[int(resolved[0])]
            turns.append({"speaker_id": who[0], "speaker": who[1], "line": line})
        return turns

    async def _memory_summary(
        self,
        *,
        agent: "Agent | None",
        others: list[tuple[str, str]],
        transcript: str,
        purpose: str,
        role_line: str,
        fallback_fact: str,
        talk_role: str,
        now_step: int = 0,
        environment: "EnvironmentSystem | None" = None,
        expected_outcome: str = "",
    ) -> "tuple[str, bool, dict[str, str], str] | None":
        """First-person record of the conversation from this participant's view, judged in
        character. ``others`` are the other speakers ``(id, name)``; ``role_line`` says how he
        came into it (``_role_line``). Returns (fact, succeeded, {other id: relation direction},
        why), or ``None`` when it can't happen, which the caller turns into a null step.
        """
        if agent is None:
            logger.warning("talk_memory_summary_no_participant", extra={"others": [o for o, _ in others]})
            return None

        rels = await self._present_relations_block(agent, environment)
        relation_part = f"\n【我与在场者的关系】\n{rels}" if rels else ""

        situation_header = (
            render_situation_header(situation_for(environment, agent.agent_id),
                                    voice=SituationVoice.FIRST)
            if environment is not None else ""
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        expected_part = f"\n原本期望：{expected_outcome}" if expected_outcome else ""
        identities = self._directory.agent_identity_map(o for o, _ in others)
        roster = "\n".join(
            f"#{i} {person_referent(name, identities[oid].gender if oid in identities else '')}"
            for i, (oid, name) in enumerate(others, start=1)
        )
        # Names stay out of system (prefix cache); the roster in user numbers them.
        system = """\
我此刻代入这个角色,以第一人称「我」记下这场刚结束的对话——这是我一贯的记事方式,与处境无关。

【我要说的】
从我的视角:这场对话达成我在意的目的了吗?发生了什么实质的事?谈完之后我对同场说话的每个人观感有何变化。始终用我自己的口吻。

【约束】
 - fact 里提到我认识的人时,不妨带上其名字,便于日后回想;若我惯以称谓相称,可把名字附在称谓后(形如「称谓（名字）」)。这只是便于回想的建议,不必生硬套用、更不必每处都加括号;陌生人用描述,不臆造名字。
 - 要区分已经发生、正在发生、将要发生的事情：把“我现在要去超市”写成“我去了超市”、把“我命令他帮我收拾行李”写成“他已经帮我收拾行李”，都是禁止的。
 - 先理解此刻的时间和我此刻所在地点。
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "我视角一句话,我从这场对话得到了什么或发生了什么(≤100字)", "success": true或false, "why": "没能达成目的的缘由（≤20字）；达成了给空字符串", "relations": [{"index": 同场说话的人的序号, "relation": "positive或negative或neutral（我对此人的观感变化）"}]}
其中 why 是**说给旁人听的一句交代**:只写外面看得出的缘由(「对方不肯松口」「话没说到点上」),不写我的懊丧、盘算或对自己的评价——那些只属于 fact。
relations 给下面「同场说话的人」里的每个人各一条。

落笔写上面那句 fact 之前,还有一条:
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON
        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【刚结束的对话】
{role_line}，话题是「{purpose}」。{relation_part}{expected_part}
同场说话的人：
{roster}
对话记录：
{transcript}

依上面说定的 JSON 格式记下这一场，只输出 JSON、不写任何多余内容。"""
        try:
            # Attribute to this participant; the addressee also gets conscripted_input for review.
            call_annotations = {
                "given_facts": GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("刚结束的对话", f"{role_line}，话题是「{purpose}」。")
                .add("我与在场者的关系", rels)
                # Only the one who wrote an expectation sees it; declaring it for anyone else
                # would blind the leak review.
                .add("原本期望", expected_outcome)
                .add("同场说话的人", roster)
                .add("对话记录", transcript),
                "talk_role": talk_role,
                "action_owner": agent.personality.soul.name,
            }
            if talk_role == "addressee":
                call_annotations["conscripted_input"] = f"{role_line}，话题是「{purpose}」"
            with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
                **call_annotations
            ):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    # fact (≤100 chars ≈ 150 tok, including a possible bracketed date) + success
                    # + why (≤20 chars ≈ 30) + structure ≈ 203, plus one relations entry per
                    # other speaker (index + direction + structure ≈ 16).
                    max_tokens=output_budget(203 + 16 * len(others)),
                    json_mode=True,
                )
            data = extract_json(response.content)
            fact = str(data.get("fact", "")).strip() or fallback_fact
            succeeded = coerce_bool(data.get("success"), True)
            ref = IndexedRef(o for o, _ in others)
            directions: dict[str, str] = {}
            raw = data.get("relations")
            for item in raw if isinstance(raw, list) else []:
                if not isinstance(item, dict):
                    continue
                for other_id in ref.resolve([item.get("index")]):
                    directions.setdefault(
                        other_id, parse_relation_direction(str(item.get("relation", "neutral"))).value,
                    )
            # Someone left out of relations still spoke with me: as neutral as the default.
            for other_id, _name in others:
                directions.setdefault(other_id, "neutral")
            # Unsliced: the prompt caps it.
            why = str(data.get("why", "")).strip() if not succeeded else ""
            return fact, succeeded, directions, why
        except Exception as exc:
            logger.warning("talk_memory_summary_failed", extra={"agent_id": agent.agent_id, "error": str(exc)})
            return None

    async def _listener_memory(
        self,
        *,
        agent: "Agent | None",
        talkers: str,
        transcript: str,
        now_step: int = 0,
        environment: "EnvironmentSystem | None" = None,
    ) -> str | None:
        """A listener's first-person record of what he heard: no success verdict or relation,
        since he said nothing. Not the raw transcript, which would embed the whole dialogue.
        ``talkers`` names the speakers as one subject.

        ``None`` on failure, and no memory is written: a content-free line only dilutes recall.
        """
        if agent is None:
            return None

        actor_condition = condition_line(
            agent.personality.state.condition, voice=SituationVoice.FIRST,
            now_step=now_step, seconds_per_step=self._seconds_per_step,
        )
        situation_header = (
            render_situation_header(situation_for(environment, agent.agent_id),
                                    voice=SituationVoice.FIRST)
            if environment is not None else ""
        )
        header_part = f"{situation_header}\n\n" if situation_header else ""
        system = """\
我此刻代入这个角色,以第一人称「我」记下我方才在旁边听到的一场对话——这是我一贯的记事方式,与处境无关。

【我要说的】
我听见了什么,以及这些话对我意味着什么。用我自己的口吻。

【我绝不这样写】
 1.不把自己写进那场对话:我一句话也没说,不能写成我问了、我答了、我劝了。
 2.不复述流水账:不逐句转写他们的对白,只记下落到我心里的那一点。
 3.不添他们没说过的话,也不替他们补上没挑明的意思。
 4.不混淆时态:将要发生的事不能写成已经发生。
 5.提到我认识的人时,不妨带上其名字,便于日后回想;惯以称谓相称的,可把名字附在称谓后
   (形如「称谓（名字）」)。这只是便于回想的建议,不必生硬套用、更不必每处都加括号;
   陌生人用描述,不臆造名字。
""" + CLOSED_WORLD_FACT_RULE_FIRST_PERSON + """

【输出】
严格输出以下 JSON,不要任何多余内容:
{"fact": "我视角一句话,我从旁听到的这场对话里知道了什么(≤60字)"}

落笔写上面那句 fact 之前,还有一条:
""" + ABSOLUTE_TIME_RULE_FIRST_PERSON
        user = f"""\
{header_part}【我是谁】
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}{actor_condition}

【我在旁边听到的对话】
说话的是{talkers}，我在一旁听着。
对话记录：
{transcript}

依上面说定的 JSON 格式记下我听到了什么，只输出 JSON、不写任何多余内容。"""
        try:
            # Attribute to the listener, not the initiator.
            with observe_stage(Stage.ACTION, agent_id=agent.agent_id), annotate_call(
                given_facts=GivenFacts()  # same source as user above
                .add("此刻何时何地", situation_header)
                .add("我的处境", actor_condition)
                .add("我在旁边听到的", f"说话的是{talkers}")
                .add("对话记录", transcript),
                talk_role="listener", action_owner=agent.personality.soul.name,
            ):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=system),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.7,
                    # fact ≤60 chars ≈ 90 tok + JSON structure ~5; est. ~95 tok.
                    max_tokens=output_budget(95),
                    json_mode=True,
                )
            return str(extract_json(response.content).get("fact", "")).strip() or None
        except Exception as exc:
            logger.warning(
                "talk_listener_memory_failed",
                extra={"agent_id": agent.agent_id, "error": str(exc)},
            )
            return None
