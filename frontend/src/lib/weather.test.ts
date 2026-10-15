import { describe, expect, it } from "vitest";

import { skyText } from "./weather";

describe("sky wording", () => {
  it("names the sky and how hard it comes down", () => {
    expect(skyText({ sky: "rain", severity: "high" })).toBe("🌧 大雨");
    expect(skyText({ sky: "clear", severity: "low" })).toBe("☀ 晴");
  });

  it("says nothing for a world without weather or an unknown sky", () => {
    expect(skyText(null)).toBe("");
    expect(skyText({ sky: "hail", severity: "low" })).toBe("");
  });
});
