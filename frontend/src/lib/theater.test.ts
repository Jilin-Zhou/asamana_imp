import { describe, expect, it } from "vitest";

import { actTitle, chineseNumeral, sceneTitle } from "./theater";

describe("theater numbering", () => {
  it("writes act and scene numbers as Chinese numerals", () => {
    expect([1, 9, 10, 12, 20, 21, 99].map(chineseNumeral)).toEqual(["一", "九", "十", "十二", "二十", "二十一", "九十九"]);
    expect([100, 105, 110, 123].map(chineseNumeral)).toEqual(["一百", "一百零五", "一百一十", "一百二十三"]);
    expect(actTitle(2)).toBe("第二幕");
    expect(sceneTitle(14)).toBe("第十四场");
  });

  it("leaves numbers it can't name as digits", () => {
    expect(chineseNumeral(0)).toBe("0");
    expect(chineseNumeral(-3)).toBe("-3");
  });
});
