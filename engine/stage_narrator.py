"""The stage narrator: writes each step up as scenes of a play, for the audience.

Read-only and off to the side. It reads a finished step's records, never writes simulation state,
and nothing it writes reaches any agent's prompt or memory. That is why it may see everything:
its only reader is the audience. It runs on a background worker, one step at a time and in order
(the running synopsis needs it), so the step loop never waits on it; a failed call costs that
step its script and nothing else.

Code decides the play's structure (which places make a scene, when a scene continues, when an
act turns over); the LLM writes only stage directions and narration. Dialogue lines and inner
monologue are quoted from the record by index, never rewritten, so the script can't put words in
a character's mouth.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from core.context import observe_stage, set_log_context
from core.interfaces.check import CheckRecord
from core.interfaces.directory import WorldDirectory
from core.interfaces.llm import IndexedRef, LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.interfaces.severity import Severity
from core.interfaces.snapshot import SnapshotProvider
from core.interfaces.trace import Stage
from core.logging import get_logger
from core.prompts import CLOSED_WORLD_FACT_RULE, person_referent
from engine.check import tier_gist

if TYPE_CHECKING:
    from core.event_bus import NarrativeEventBus

logger = get_logger(__name__)

# Beats that carry something worth staging. A tick only says "still at it"; an initialization
# record is placement.
_STAGED_PHASES = frozenset({"begin", "settled", "ongoing_complete", "interrupt"})
_MAX_SCENES = 4
_MAX_BEATS = 8
_MAX_NARRATION_BEATS = 3
_STAGE_DIRECTION_CHARS = 60
_NARRATION_CHARS = 80
_MEMO_CHARS = 100
# How far back a restore looks for the last script to continue from.
_RESTORE_LOOKBACK = 50

_SYSTEM_PROMPT = f"""\
【任务】
你是这台戏的说书人。下面是同一时刻、几处地方刚刚发生的事的完整记录。
你把每一处写成一场戏给观众看：舞台提示交代光线、声响与站位，旁白交代动作的经过与结果。

【原则】
- 只演记录里有的事：不新增人物、动作、对白或结果。记录说没做成就是没做成，标明的判定结果要在旁白里演出来。
- 台词与心声不由你写：要用某句台词，引用它的编号（line_index）；要把某人的心声作独白，引用它的编号（monologue_index）。
- 旁白用第三人称，人物称名字，不写任何编号、步数或点数。
- 每场的节奏由你安排：台词、独白与旁白可以穿插，挑最有戏的放进来，不必全用。
- 前情提要只用来接续，不要复述。
{CLOSED_WORLD_FACT_RULE}

【输出】
严格输出以下 JSON，不要任何多余内容。场的顺序与下面给的地点顺序一致，每处一场：
- location_index：这一场在第几处（下面的编号）。
- stage_direction：舞台提示（≤{_STAGE_DIRECTION_CHARS}字）。
- beats：这一场的拍子，最多 {_MAX_BEATS} 拍，其中旁白最多 {_MAX_NARRATION_BEATS} 拍。旁白 {{"kind": "narration", "text": "≤{_NARRATION_CHARS}字"}}；台词 {{"kind": "line", "line_index": 整数}}；独白 {{"kind": "aside", "monologue_index": 整数}}。
- memo：到此为止的前情提要，给下一场接续用（≤{_MEMO_CHARS}字）。
{{"scenes": [{{"location_index": 1, "stage_direction": "≤{_STAGE_DIRECTION_CHARS}字", "beats": [...]}}], "memo": "≤{_MEMO_CHARS}字"}}
"""

# Per scene: location_index ≈3 + stage_direction ≤60 chars ≈90 + 3 narration beats × (≤80 chars
# ≈120 + structure 10) + 5 quoted beats × ≈12 + structure ≈30 ≈ 573; × 4 scenes ≈ 2292
# + memo ≤100 chars ≈150 + wrapper ≈20 → ≈2462 tok.
_MAX_TOKENS = output_budget(2462)


@dataclass(frozen=True)
class StepMaterial:
    """What one finished step hands the narrator: the same JSON-native records the snapshot
    keeps, plus the step's clock."""

    step: int
    time_label: str
    day: int
    actions: Sequence[Mapping[str, Any]]
    events: Sequence[Mapping[str, Any]] = ()
    broadcasts: Sequence[Mapping[str, Any]] = ()


@dataclass
class _Place:
    """One location's material for a scene, gathered by code before the LLM sees it."""

    location_id: str
    records: list[Mapping[str, Any]] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)  # agent ids, first appearance order
    notices: list[str] = field(default_factory=list)

    @property
    def weight(self) -> int:
        """How much happened here: results count double, a roll or a conversation once more."""
        score = len(self.notices) * 2
        for r in self.records:
            score += 2 if r.get("phase") != "begin" else 1
            score += 1 if r.get("check") else 0
            score += 1 if r.get("dialogue") else 0
        return score


@dataclass
class _Continuity:
    """What the next step's script continues from; restored from the last saved script."""

    act: int = 0
    scene: int = 0
    last_step: int = 0
    last_day: int = -1
    memo: str = ""
    # location id → (scene number, cast) of the scene that place had at ``last_step``.
    open_scenes: dict[str, tuple[int, frozenset[str]]] = field(default_factory=dict)


def _staged(actions: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The records worth staging, one per execution (the initiator's speaks for a joint act)."""
    by_execution: dict[str, Mapping[str, Any]] = {}
    for record in actions:
        if record.get("phase") not in _STAGED_PHASES or record.get("not_executed"):
            continue
        key = f"{record.get('execution_id') or id(record)}:{record.get('phase')}"
        if key not in by_execution or record.get("agent_id") == record.get("initiator_id"):
            by_execution[key] = record
    return list(by_execution.values())


def gather_places(material: StepMaterial) -> list[_Place]:
    """Group a step's material by location, busiest first, at most ``_MAX_SCENES``."""
    places: dict[str, _Place] = {}

    def place(location_id: str) -> _Place:
        return places.setdefault(location_id, _Place(location_id=location_id))

    for record in _staged(material.actions):
        location_id = str(record.get("location_id") or "")
        if not location_id:
            continue
        here = place(location_id)
        here.records.append(record)
        for agent_id in (
            str(record.get("agent_id") or ""),
            *(str(a) for a in record.get("participant_ids") or ()),
            *(str(a) for a in record.get("acted_upon") or ()),
        ):
            if agent_id and agent_id not in here.cast:
                here.cast.append(agent_id)
    for notice in (*material.events, *material.broadcasts):
        location_id = str(notice.get("location_scope") or "")
        text = str(notice.get("content") or notice.get("narrative_desc") or "").strip()
        if location_id and text and location_id in places:
            places[location_id].notices.append(text)
    busiest = sorted(places.values(), key=lambda p: -p.weight)[:_MAX_SCENES]
    order = list(places)
    return sorted(busiest, key=lambda p: order.index(p.location_id))


def opens_act(material: StepMaterial, continuity: _Continuity) -> bool:
    """A new act on the first script, a new day, a major injected event or a death."""
    if continuity.act == 0 or material.day != continuity.last_day:
        return True
    return any(
        str(b.get("severity") or "") == Severity.HIGH.value for b in material.broadcasts
    )


class StageNarrator:
    """One world's narrator: a queue of finished steps and one worker writing them up in order."""

    def __init__(
        self,
        *,
        world_id: str,
        llm_router: LLMRouter,
        snapshot_provider: SnapshotProvider,
        event_bus: "NarrativeEventBus",
        directory: WorldDirectory,
    ) -> None:
        self._world_id = world_id
        self._llm = llm_router
        self._snapshots = snapshot_provider
        self._bus = event_bus
        self._directory = directory
        self._queue: asyncio.Queue[StepMaterial] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._continuity: _Continuity | None = None

    def submit(self, material: StepMaterial) -> None:
        """Queue a finished step. Never blocks and never raises into the step loop."""
        self._queue.put_nowait(material)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._work(), name=f"narrator:{self._world_id}")

    async def drain(self) -> None:
        """Wait until every queued step has been written up (tests, shutdown of a finished run)."""
        await self._queue.join()

    async def aclose(self) -> None:
        """Stop writing. Steps still queued go without a script."""
        if self._worker is not None and not self._worker.done():
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    async def _work(self) -> None:
        while True:
            material = await self._queue.get()
            try:
                await self._narrate(material)
            except Exception as exc:  # noqa: BLE001 — one step's script must not stop the rest
                logger.warning(
                    "narration_failed",
                    extra={"world_id": self._world_id, "step": material.step, "error": str(exc)},
                )
            finally:
                self._queue.task_done()

    async def _restore(self, before_step: int) -> _Continuity:
        """Continue numbering and synopsis from the last script saved before this step."""
        for step in range(before_step - 1, max(0, before_step - 1 - _RESTORE_LOOKBACK), -1):
            saved = await self._snapshots.load_narration(self._world_id, step)
            if saved:
                return _Continuity(
                    act=int(saved.get("act") or 0),
                    scene=max((int(s.get("scene") or 0) for s in saved.get("scenes") or ()), default=0),
                    last_step=step,
                    last_day=int(saved.get("day", -1)),
                    memo=str(saved.get("memo") or ""),
                    open_scenes={
                        str(s.get("location_id")): (
                            int(s.get("scene") or 0),
                            frozenset(str(c.get("agent_id")) for c in s.get("cast") or ()),
                        )
                        for s in saved.get("scenes") or ()
                    },
                )
        return _Continuity()

    async def _narrate(self, material: StepMaterial) -> None:
        # The worker task froze the log context of the step that started it; traces must carry
        # the step being written up.
        set_log_context(world_id=self._world_id, step=str(material.step))
        if self._continuity is None:
            self._continuity = await self._restore(material.step)
        continuity = self._continuity
        places = gather_places(material)
        if not places:
            return
        new_act = opens_act(material, continuity)
        act = continuity.act + 1 if new_act else continuity.act
        scenes = self._number_scenes(places, material.step, continuity, new_act)
        lines, monologues = self._quotable(places)
        with observe_stage(Stage.NARRATION):
            response = await self._llm.complete(
                LLMScene.STAGE_NARRATION,
                [
                    LLMMessage(role="system", content=_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=self._user_prompt(
                        material, places, scenes, lines, monologues, continuity.memo,
                    )),
                ],
                temperature=0.8,
                max_tokens=_MAX_TOKENS,
                json_mode=True,
            )
        data = extract_json(response.content)
        script = self._script(data, material, places, scenes, lines, monologues, act, new_act)
        if not script["scenes"]:
            return
        await self._snapshots.save_narration(self._world_id, material.step, script)
        self._bus.publish({"type": "narration", "world_id": self._world_id, "step": material.step, "data": script})
        self._continuity = _Continuity(
            act=act,
            scene=max([continuity.scene, *(s["scene"] for s in script["scenes"])]),
            last_step=material.step,
            last_day=material.day,
            memo=script["memo"] or continuity.memo,
            open_scenes={
                s["location_id"]: (s["scene"], frozenset(c["agent_id"] for c in s["cast"]))
                for s in script["scenes"]
            },
        )

    @staticmethod
    def _number_scenes(
        places: list[_Place], step: int, continuity: _Continuity, new_act: bool,
    ) -> dict[str, tuple[int, bool]]:
        """location id → (scene number, continues). A scene carries on when the same place had
        one on the step just before, in the same act, with someone still on stage."""
        numbered: dict[str, tuple[int, bool]] = {}
        counter = continuity.scene
        for here in places:
            previous = continuity.open_scenes.get(here.location_id)
            if (
                previous is not None and not new_act and continuity.last_step == step - 1
                and previous[1] & set(here.cast)
            ):
                numbered[here.location_id] = (previous[0], True)
            else:
                counter += 1
                numbered[here.location_id] = (counter, False)
        return numbered

    @staticmethod
    def _quotable(places: list[_Place]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Every dialogue line and inner monologue on offer, numbered across the whole step."""
        lines: list[dict[str, str]] = []
        monologues: list[dict[str, str]] = []
        for here in places:
            for record in here.records:
                for turn in record.get("dialogue") or ():
                    text = str(turn.get("line") or "").strip()
                    if text:
                        lines.append({
                            "location_id": here.location_id,
                            "speaker_id": str(turn.get("speaker_id") or ""),
                            "speaker": str(turn.get("speaker") or ""),
                            "text": text,
                        })
                thought = str(record.get("inner_monologue") or "").strip()
                if thought:
                    monologues.append({
                        "location_id": here.location_id,
                        "speaker_id": str(record.get("agent_id") or ""),
                        "speaker": str(record.get("agent_name") or ""),
                        "text": thought,
                    })
        return lines, monologues

    def _person(self, agent_id: str) -> str:
        entry = self._directory.describe(agent_id)
        return person_referent(self._directory.agent_name(agent_id), entry.gender if entry else "")

    def _user_prompt(
        self,
        material: StepMaterial,
        places: list[_Place],
        scenes: dict[str, tuple[int, bool]],
        lines: list[dict[str, str]],
        monologues: list[dict[str, str]],
        memo: str,
    ) -> str:
        blocks: list[str] = []
        for index, here in enumerate(places, 1):
            scene_no, continues = scenes[here.location_id]
            rows = [
                f"#{index} 地点：{self._directory.location_name(here.location_id)}"
                f"（{'接着上一场' if continues else '新的一场'}）",
                "  在场：" + "、".join(self._person(a) for a in here.cast),
            ]
            for record in here.records:
                rows.append("  - " + self._record_line(record))
            rows.extend(f"  - 此处发生：{notice}" for notice in here.notices)
            here_lines = [
                (i, line) for i, line in enumerate(lines, 1) if line["location_id"] == here.location_id
            ]
            if here_lines:
                rows.append("  台词：")
                rows.extend(f"    [{i}] {line['speaker']}：「{line['text']}」" for i, line in here_lines)
            here_thoughts = [
                (i, m) for i, m in enumerate(monologues, 1) if m["location_id"] == here.location_id
            ]
            if here_thoughts:
                rows.append("  心声：")
                rows.extend(f"    [{i}] {m['speaker']}：{m['text']}" for i, m in here_thoughts)
            blocks.append("\n".join(rows))
        everywhere = [
            str(n.get("content") or n.get("narrative_desc") or "").strip()
            for n in (*material.events, *material.broadcasts)
            if not n.get("location_scope")
        ]
        everywhere_part = (
            "\n【此刻各处都知道的事】\n" + "\n".join(f"- {t}" for t in everywhere if t) + "\n"
            if any(everywhere) else ""
        )
        return f"""\
【时间】{material.time_label}
【前情提要】{memo or "（这是开场）"}
{everywhere_part}
【各处发生的事】
{chr(10).join(blocks)}

依上面说定的 JSON 格式写这一刻的戏，只输出 JSON、不写任何多余内容。"""

    @staticmethod
    def _record_line(record: Mapping[str, Any]) -> str:
        """One act in the record, as the narrator reads it: who, what, how it ended."""
        name = str(record.get("agent_name") or "某人")
        intent = str(record.get("action_description") or "").strip()
        outcome = str(record.get("gist") or record.get("outcome") or "").strip()
        if record.get("phase") == "begin":
            return f"{name}开始：{intent}"
        parts = [f"{name}：{intent}" if intent else name, f"结果：{outcome}"]
        check = CheckRecord.from_dict(record.get("check"))
        if check is not None:
            parts.append(f"判定：{tier_gist(check.tier)}")
        elif record.get("succeeded") is False:
            parts.append("没做成")
        if record.get("detected"):
            parts.append("行迹被人察觉")
        return "；".join(parts)

    def _script(
        self,
        data: Mapping[str, Any],
        material: StepMaterial,
        places: list[_Place],
        scenes: dict[str, tuple[int, bool]],
        lines: list[dict[str, str]],
        monologues: list[dict[str, str]],
        act: int,
        new_act: bool,
    ) -> dict[str, Any]:
        """Parse the LLM's script into the stored record, resolving every index back to the
        record it quotes. Out-of-range or repeated indices are dropped, never guessed."""
        place_ref = IndexedRef(p.location_id for p in places)
        line_ref = IndexedRef(str(i) for i in range(len(lines)))
        thought_ref = IndexedRef(str(i) for i in range(len(monologues)))
        by_location = {p.location_id: p for p in places}
        written: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in data.get("scenes") or ():
            if not isinstance(raw, Mapping):
                continue
            located = place_ref.resolve([raw.get("location_index")])
            if not located or located[0] in seen:
                continue
            location_id = located[0]
            seen.add(location_id)
            beats: list[dict[str, str]] = []
            narration_count = 0
            for beat in raw.get("beats") or ():
                if not isinstance(beat, Mapping) or len(beats) >= _MAX_BEATS:
                    continue
                kind = beat.get("kind")
                if kind == "narration":
                    text = str(beat.get("text") or "").strip()
                    if text and narration_count < _MAX_NARRATION_BEATS:
                        beats.append({"kind": "narration", "text": text})
                        narration_count += 1
                elif kind == "line":
                    picked = line_ref.resolve([beat.get("line_index")])
                    if picked and lines[int(picked[0])]["location_id"] == location_id:
                        line = lines[int(picked[0])]
                        beats.append({"kind": "line", "speaker_id": line["speaker_id"],
                                      "speaker": line["speaker"], "text": line["text"]})
                elif kind == "aside":
                    picked = thought_ref.resolve([beat.get("monologue_index")])
                    if picked and monologues[int(picked[0])]["location_id"] == location_id:
                        thought = monologues[int(picked[0])]
                        beats.append({"kind": "aside", "speaker_id": thought["speaker_id"],
                                      "speaker": thought["speaker"], "text": thought["text"]})
            direction = str(raw.get("stage_direction") or "").strip()
            if not beats and not direction:
                continue
            scene_no, continues = scenes[location_id]
            written.append({
                "scene": scene_no,
                "continues": continues,
                "location_id": location_id,
                "location_name": self._directory.location_name(location_id),
                "cast": [
                    {"agent_id": a, "name": self._directory.agent_name(a)}
                    for a in by_location[location_id].cast
                ],
                "stage_direction": direction,
                "beats": beats,
            })
        written.sort(key=lambda s: [p.location_id for p in places].index(s["location_id"]))
        return {
            "step": material.step,
            "act": act,
            "opens_act": new_act,
            "day": material.day,
            "time_label": material.time_label,
            "scenes": written,
            "memo": str(data.get("memo") or "").strip(),
        }
