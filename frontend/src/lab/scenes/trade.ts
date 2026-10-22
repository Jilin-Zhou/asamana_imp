import { ActionType, Deed, Phase } from "../../lib/contract";
import type { LabPlaces } from "../places";
import type { EntityView } from "../../types";
import { Emotion, MO, Need, SHI, aimTalk, did, execId, step, type LabActor, type LabScene, type Room, who } from "./fixtures";

const RING = "seed_seed-ring0001";

/** A purse and a jade ring, where each stands at this beat. */
function goods(ringHolder: string, purses: [string, number][]): Record<string, EntityView> {
  const table: Record<string, EntityView> = {
    [RING]: {
      name: "玉环", entity_type: "item", state: "intact", presence: "held", presence_ref: ringHolder,
      description: "一枚青白玉环", is_public: true, content: "", created_step: 0, quantity: 0,
    },
  };
  for (const [holder, amount] of purses) {
    table[`money-0-${holder.slice(-4)}`] = {
      name: "银子", entity_type: "money", state: "intact", presence: "held", presence_ref: holder,
      description: `${amount}两银子`, is_public: true, content: "", created_step: 0, quantity: amount,
    };
  }
  return table;
}

/**
 * A trade as the engine emits one: settled within the beat, two records sharing one
 * `execution_id`. The haggling is the same dialogue on both; the gist says how the deal ended,
 * and only the initiator's record names the things that changed hands.
 */
function tradeBeat(a: LabActor, b: LabActor, room: Room, lines: [LabActor, string][], deal: string, succeeded: boolean) {
  const eid = execId("trade", a, 1);
  const dialogue = lines.map(([who, line]) => ({ speaker_id: who.id, speaker: who.name, line }));
  const transcript = lines.map(([who, line]) => `${who.name}：${line}`).join("\n");
  return [a, b].map((party) =>
    did(party, {
      action_type: ActionType.trade,
      deed: Deed.trade,
      phase: Phase.settled,
      execution_id: eid,
      initiator_id: a.id,
      succeeded,
      ...(succeeded ? {} : { failure_reason: "乙嫌价低" }),
      target: aimTalk(party === a ? b.id : a.id),
      action_description: `拿二十两银子换${b.name}的玉环。`,
      outcome: `在${room.name}，${a.name}与${b.name}谈买卖：\n${transcript}\n——${deal}`,
      gist: `在${room.name}，${a.name}与${b.name}谈买卖，${deal}`,
      dialogue,
      affected_entity_ids: party === a && succeeded ? [RING] : [],
    }),
  );
}

/** Things changing hands by agreement. */
export function buildTradeScenes({ open: OPEN }: LabPlaces): LabScene[] {
  const lines: [LabActor, string][] = [
    [MO, "这玉环，二十两卖不卖？"],
    [SHI, "二十五两，少一文不卖。"],
    [MO, "也罢，二十五就二十五。"],
  ];
  return [
    {
      id: "trade",
      group: "言语",
      title: "trade · 交易（当拍成交，两样东西同时换手）",
      watch:
        "讨价还价像交谈一样以气泡播出，播完后行动标签显示成交；玉环从乙的口袋移到甲的口袋，" +
        "两人的钱袋金额同时变化（25两从甲到乙）。叙事流里对白之后是一行成交结果，不重复整段对白。",
      steps: [
        step({
          states: [who(MO, OPEN), who(SHI, OPEN)],
          entities: goods(SHI.id, [[MO.id, 30], [SHI.id, 4]]),
        }),
        step({
          states: [
            who(MO, OPEN, { emotion: Emotion.joy, emotion_valence: 0.4, dominant_need: Need.esteem }),
            who(SHI, OPEN, { emotion: Emotion.trust, emotion_valence: 0.2 }),
          ],
          actions: tradeBeat(MO, SHI, OPEN, lines, "成交：阿墨付出25两银子，阿石付出玉环。", true),
          entities: goods(MO.id, [[MO.id, 5], [SHI.id, 29]]),
        }),
      ],
    },
  ];
}
