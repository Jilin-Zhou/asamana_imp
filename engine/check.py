"""Rolling checks for PHYSICAL, COVERT and WORK (see ``core.interfaces.check``).

``CheckAssessor`` does parts 1 and 2: one functional LLM call names certainty, difficulty and
edge, then code rolls. The executors do part 3, narrating the ``CheckTier`` they are handed in
the words of ``tier_phrase``.

TALK, SEND_MESSAGE, MOVE, REST and ERRAND never roll: a talk's outcome is the other side's own
decision, and the rest carry no doubt worth a die.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.coerce import coerce_enum
from core.context import GivenFacts, annotate_call
from core.interfaces.action import ActionType
from core.interfaces.check import (
    DICE_COUNT, DIE_SIDES, Certainty, CheckRecord, CheckTier, Difficulty, Edge,
)
from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, coerce_bool, extract_json, output_budget
from core.logging import get_logger
from core.prompts import SituationVoice, condition_line, vitality_line

if TYPE_CHECKING:
    from agent.agent import Agent

logger = get_logger(__name__)


@dataclass(frozen=True)
class CheckVerdict:
    """What an executor narrates from: the tier, and the roll behind it when there was one."""

    tier: CheckTier
    # None: the outcome was never in doubt (``Certainty.SURE_*``), so nothing was rolled.
    record: CheckRecord | None = None


def roll_check(
    difficulty: Difficulty,
    edge: Edge,
    *,
    pure_chance: bool,
    rng: random.Random,
    reason: str = "",
) -> CheckRecord:
    """Roll 3d6 for one check. A pure-chance check records an even edge: luck owes nothing to
    who is holding the dice."""
    return CheckRecord(
        difficulty=difficulty,
        edge=Edge.EVEN if pure_chance else edge,
        pure_chance=pure_chance,
        dice=tuple(rng.randint(1, DIE_SIDES) for _ in range(DICE_COUNT)),
        reason=reason,
    )


#: How each tier is told to a third-person functional judge narrating it. A cost or a mess must
#: be written out concretely: "succeeded at some cost" with nothing named is the tier wasted.
_THIRD_PERSON_PHRASES: dict[CheckTier, str] = {
    CheckTier.CRITICAL_SUCCESS: "做成了，而且做得超出预期——比他想要的还要好",
    CheckTier.SUCCESS:          "做成了",
    CheckTier.COSTLY_SUCCESS:   (
        "做成了，但付出了代价或留下了隐患——据材料写出具体是什么"
        "（自身受损、惊动旁人、东西受损、留下把柄等），不得一笔带过"
    ),
    CheckTier.FAILURE:          "没做成",
    CheckTier.CRITICAL_FAILURE: (
        "不但没做成，还把局面弄糟了——据材料写出糟在哪里"
        "（反受其害、暴露了意图、弄坏了东西等），不得一笔带过"
    ),
}

#: The same tiers in the actor's own voice, for in-character narration (WORK).
_FIRST_PERSON_PHRASES: dict[CheckTier, str] = {
    CheckTier.CRITICAL_SUCCESS: "我做成了，而且比我想要的还要好",
    CheckTier.SUCCESS:          "我做成了",
    CheckTier.COSTLY_SUCCESS:   "我做成了，但留下了瑕疵或付出了代价——我要写清是什么，不一笔带过",
    CheckTier.FAILURE:          "我没做成",
    CheckTier.CRITICAL_FAILURE: "我不但没做成，还弄糟了——我要写清糟在哪里（东西弄坏了、白费了材料等）",
}


def tier_phrase(tier: CheckTier, *, voice: SituationVoice) -> str:
    """The tier in words, the only form in which a check reaches a prompt."""
    phrases = _FIRST_PERSON_PHRASES if voice == SituationVoice.FIRST else _THIRD_PERSON_PHRASES
    return phrases[tier]


#: How each kind of act is named to the assessor.
_KIND_LABELS: dict[ActionType, str] = {
    ActionType.PHYSICAL: "当面的肉体行动",
    ActionType.COVERT:   "暗中行动",
    ActionType.WORK:     "独自做事",
}

_REASON_MAX_CHARS = 80

# One system prompt for all three kinds, so every assessment shares one cached prefix.
_SYSTEM_PROMPT = f"""\
【任务】
你是中立的世界裁决者。下面是一次即将落定的行动。你**不写结果**，只做两件事：
1. 判断它的结果有没有悬念；
2. 有悬念时，给出它的难度，以及行动者在这件事上的优势。
结果由之后的掷骰决定，所以你不必、也不能替它定成败。

【判断原则】
- 有没有悬念（certainty）：
  - sure_success：按常理必然做成、没有任何阻力。
  - sure_failure：按所给条件根本做不到，或意图与目标明显冲突又说不出理由。
  - uncertain：其余一切——有人会抵抗、要靠本事或运气、现场有变数。拿不准时选 uncertain。
- 纯运气（pure_chance）：成败只看运气、与本事无关时为 true（掷骰、抽签、赌大小、刮彩票）；此时 edge 不计。
- 难度（difficulty）：按「一个寻常人在当下条件下做这件事」来定，**不看行动者是谁**：
  - easy：寻常人多半能成
  - normal：寻常人成败各半
  - hard：寻常人多半不成
  - very_hard：寻常人几乎不可能
  纯运气时，difficulty 表示中的机会：easy 多半能中，very_hard 几乎中不了。
- 优势（edge）：行动者在这件事上比寻常人强多少、弱多少；对人行动时，就是双方的实力对比：
  - much_worse 明显劣势 / worse 略处劣势 / even 相当 / better 略占优势 / much_better 明显优势
  - 计入本事、体力、伤势、此刻的处境（如被缚）、人数、器械、防备与现场条件。
- 不得因行动者身份显赫或性格果决就判他占优：性格说明他会怎么做，不说明他做得成。
- 在场旁人是证据：结合各自的背景与性情，权衡他们会阻拦、协助还是旁观；标注的「此刻意图」是他们**同时**
  在试图做、尚未落定的事，可以计入，但不能当成已经发生。

【各类行动的要点】
- 当面的肉体行动：
  - 对方愿意配合（搀扶、救治一个不反抗的人），或拿起无主之物、放下自己手里的东西：sure_success。
  - 把东西交到某人手上：对方接不接是他自己的事——按双方关系与性情，明摆着会接为 sure_success，
    明摆着不接为 sure_failure，说不准才是 uncertain。
  - 从别人手上、身上拿东西：持有者醒着还是睡着、留不留神、护不护得住，连同双方实力与旁人立场，都要计入。
- 暗中行动：难处在于既接得住此处发生的事，又不被察觉。在场的人越多、越闲、越留神就越难；
  四下无人时容易。edge 看行动者藏身与留意的本事、对此地的熟悉、在场者是否忙于他事。
- 独自做事：难度看这件事本身的繁难与手边条件（材料、工具、时间够不够）；edge 看行动者的本事、学识、
  此刻的精力与心绪。熟手做寻常小事为 sure_success；缺了必需的东西又无从替代为 sure_failure。

【输出】
严格输出以下 JSON，不要任何多余内容。**先在 reason 里写依据，再给结论**：
- reason：这件事难在哪、行动者凭什么占优或吃亏（≤{_REASON_MAX_CHARS}字）。
- certainty 不是 uncertain 时，difficulty 与 edge 照填，不会被采用。
{{"reason": "≤{_REASON_MAX_CHARS}字", "certainty": "sure_success或uncertain或sure_failure", "pure_chance": true或false, "difficulty": "easy或normal或hard或very_hard", "edge": "much_worse或worse或even或better或much_better"}}
"""

# reason ≤80 chars ≈120 + certainty ≈5 + pure_chance ≈3 + difficulty ≈4 + edge ≈5
# + 5-field JSON structure ≈25 → ≈162 tok.
_MAX_TOKENS = output_budget(162)


class CheckAssessor:
    """Parts 1 and 2 of a check: one functional LLM call, then the roll.

    The roll is seeded by world, step, actor and action type, so a world's dice are reproducible
    without being shared across worlds.
    """

    def __init__(self, llm_router: LLMRouter, *, seconds_per_step: int = 3600, seed: str = "") -> None:
        self._llm = llm_router
        self._seconds_per_step = seconds_per_step
        self._seed = seed

    def _rng(self, step: int, agent_id: str, action_type: ActionType) -> random.Random:
        return random.Random(f"{self._seed}:{step}:{agent_id}:{action_type.value}")

    async def assess(
        self,
        agent: "Agent",
        *,
        action_type: ActionType,
        intent: str,
        scene: str,
        step: int,
        expected_outcome: str = "",
        target_block: str = "",
        details: str = "",
    ) -> CheckVerdict | None:
        """Assess and roll one act. ``None`` when the assessment can't happen (LLM failure): the
        caller returns an ``adjudication_failed`` null step, so nothing is rolled or invented.

        ``target_block`` is what the act is aimed at, as the executor already renders it for its
        own judge; ``details`` holds extra "- 键：值" lines for the 【行动】 block.
        """
        state = agent.personality.state
        actor_vitality = vitality_line(state.vitality, voice=SituationVoice.THIRD, lead="")
        actor_condition = condition_line(
            state.condition, voice=SituationVoice.THIRD,
            now_step=step, seconds_per_step=self._seconds_per_step,
        )
        target_part = f"\n\n【行动对象】\n{target_block}" if target_block else ""
        expected_part = f"\n- 期望结果：{expected_outcome}" if expected_outcome else ""
        user = f"""\
【行动者】（判断依据：你不站在他的立场）
{agent.personality.to_prompt_context(include_emotion=True, include_goals=False)}
{actor_vitality}{actor_condition}

【现场】
{scene or "（现场情况不明）"}{target_part}

【行动】
- 类别：{_KIND_LABELS[action_type]}
- 意图：{intent}{expected_part}{details}

依上面说定的 JSON 格式给出判断（reason 在前），只输出 JSON、不写任何多余内容。"""
        facts = (
            GivenFacts()
            .add("行动者体力", actor_vitality)
            .add("行动者处境", actor_condition)
            .add("现场", scene or "情况不明")
            .add("行动对象", target_block)
            .add("意图", intent)
            .add("期望结果", expected_outcome)
        )
        try:
            with annotate_call(given_facts=facts, check_assessment=True):
                response = await self._llm.complete(
                    LLMScene.AGENT_ACTION_NARRATION,
                    [
                        LLMMessage(role="system", content=_SYSTEM_PROMPT),
                        LLMMessage(role="user", content=user),
                    ],
                    temperature=0.3,
                    max_tokens=_MAX_TOKENS,
                    json_mode=True,
                )
            data = extract_json(response.content)
        except Exception as exc:
            logger.warning(
                "check_assessment_failed",
                extra={"agent_id": agent.agent_id, "action_type": action_type.value, "error": str(exc)},
            )
            return None

        # A garbled certainty rolls: doubt is the safe reading, a free success is not.
        certainty = coerce_enum(Certainty, data.get("certainty"), default=Certainty.UNCERTAIN)
        if certainty == Certainty.SURE_SUCCESS:
            return CheckVerdict(tier=CheckTier.SUCCESS)
        if certainty == Certainty.SURE_FAILURE:
            return CheckVerdict(tier=CheckTier.FAILURE)
        record = roll_check(
            coerce_enum(Difficulty, data.get("difficulty"), default=Difficulty.NORMAL),
            coerce_enum(Edge, data.get("edge"), default=Edge.EVEN),
            pure_chance=coerce_bool(data.get("pure_chance"), False),
            rng=self._rng(step, agent.agent_id, action_type),
            reason=str(data.get("reason") or "").strip(),
        )
        return CheckVerdict(tier=record.tier, record=record)
