/**
 * The world's sky in words and a glyph, for the observer. The engine sends enum values only
 * (`StepEvent.weather`); the words here are the observer's, not the characters'.
 */

import type { WeatherView } from "../types";

const SKY: Record<string, { glyph: string; words: [string, string, string] }> = {
  clear:    { glyph: "☀", words: ["晴", "晴", "晴"] },
  overcast: { glyph: "☁", words: ["阴", "阴", "阴"] },
  rain:     { glyph: "🌧", words: ["小雨", "雨", "大雨"] },
  snow:     { glyph: "❄", words: ["小雪", "雪", "大雪"] },
  wind:     { glyph: "🌬", words: ["微风", "大风", "狂风"] },
};
const STRENGTH = ["low", "medium", "high"];

/** "🌧 大雨"; empty for a world without weather or a sky this table doesn't know. */
export function skyText(weather: WeatherView | null | undefined): string {
  const entry = weather ? SKY[weather.sky] : undefined;
  if (!entry || !weather) return "";
  const strength = Math.max(0, STRENGTH.indexOf(weather.severity));
  return `${entry.glyph} ${entry.words[strength]}`;
}
