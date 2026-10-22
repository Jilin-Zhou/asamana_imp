import { describe, expect, it } from "vitest";

import type { CheckView } from "../types";
import { checkArithmetic, checkStakes, tierLabel } from "./check";

const roll = (over: Partial<CheckView> = {}): CheckView => ({
  difficulty: "hard",
  edge: "better",
  pure_chance: false,
  dice: [4, 5, 3],
  basis: {
    attribute: null,
    attribute_level: 3,
    skill: "",
    skill_level: 0,
    opposing_attribute: null,
    opposing_level: 3,
  },
  dc: 14,
  modifier: 2,
  total: 14,
  tier: "costly_success",
  reason: "",
  ...over,
});

describe("check wording", () => {
  it("spells out the roll against its DC", () => {
    expect(checkArithmetic(roll())).toBe("4+5+3 +2 = 14 / 14");
  });

  it("signs a negative modifier and drops a zero one", () => {
    expect(checkArithmetic(roll({ modifier: -4, total: 8 }))).toBe("4+5+3 −4 = 8 / 14");
    expect(checkArithmetic(roll({ modifier: 0, total: 12 }))).toBe("4+5+3 = 12 / 14");
  });

  it("names the stakes, and says when luck alone decided", () => {
    expect(checkStakes(roll())).toBe("困难 · 情境略为有利");
    expect(checkStakes(roll({ pure_chance: true, edge: "even" }))).toBe("困难 · 全凭运气");
  });

  it("itemizes the abilities behind the modifier", () => {
    const basis = {
      attribute: "physique",
      attribute_level: 4,
      skill: "拳脚",
      skill_level: 1,
      opposing_attribute: "agility",
      opposing_level: 5,
    };
    expect(checkStakes(roll({ basis }))).toBe("困难 · 体魄出众 +1 · 拳脚略通 +1 · 对方身手超群 −2 · 情境略为有利");
  });

  it("labels the tier, and shows an unknown one as itself", () => {
    expect(tierLabel(roll())).toBe("有代价的成功");
    expect(tierLabel(roll({ tier: "mystery" }))).toBe("mystery");
  });
});
