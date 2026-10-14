import { describe, expect, it } from "vitest";

import { attributeLabel, attributeRows, attributeWord, skillWord } from "./ability";

describe("ability wording", () => {
  it("lists every attribute in order, a missing one at the baseline", () => {
    expect(attributeRows({ attributes: { nerve: 5 }, skills: [] })).toEqual([
      { key: "physique", level: 3 },
      { key: "agility", level: 3 },
      { key: "intellect", level: 3 },
      { key: "eloquence", level: 3 },
      { key: "nerve", level: 5 },
    ]);
    expect(attributeRows(undefined).every((r) => r.level === 3)).toBe(true);
  });

  it("uses the backend's level words, and shows an unknown value as itself", () => {
    expect([1, 2, 3, 4, 5].map(attributeWord)).toEqual(["孱弱", "平平", "寻常", "出众", "超群"]);
    expect([1, 2, 3].map(skillWord)).toEqual(["略通", "精熟", "宗师"]);
    expect(attributeWord(9)).toBe("9");
    expect(attributeLabel("luck")).toBe("luck");
  });
});
