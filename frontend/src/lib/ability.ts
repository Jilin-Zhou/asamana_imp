/**
 * A character's abilities in words: the attribute names and the level words the backend's prompts
 * use (core.interfaces.ability), so the observer reads the same scale the characters were rated on.
 */

import type { AbilitiesView } from "../types";
import { Attribute, type AttributeValue } from "./contract";

export const ATTRIBUTE_LABEL: Record<AttributeValue, string> = {
  [Attribute.physique]:  "体魄",
  [Attribute.agility]:   "身手",
  [Attribute.intellect]: "心智",
  [Attribute.eloquence]: "口才",
  [Attribute.nerve]:     "胆识",
};

export const ATTRIBUTE_ORDER = Object.values(Attribute) as AttributeValue[];
export const ATTRIBUTE_MAX = 5;
export const ATTRIBUTE_BASELINE = 3;

const ATTRIBUTE_WORDS = ["", "孱弱", "平平", "寻常", "出众", "超群"];
const SKILL_WORDS = ["", "略通", "精熟", "宗师"];

/** An unknown key or level reads as itself rather than vanishing. */
export const attributeLabel = (key: string): string => (ATTRIBUTE_LABEL as Record<string, string>)[key] ?? key;
export const attributeWord = (level: number): string => ATTRIBUTE_WORDS[level] ?? String(level);
export const skillWord = (level: number): string => SKILL_WORDS[level] ?? String(level);

/** Every attribute in display order, a missing one at the baseline (a world built without them). */
export function attributeRows(abilities: AbilitiesView | undefined): { key: AttributeValue; level: number }[] {
  return ATTRIBUTE_ORDER.map((key) => ({ key, level: abilities?.attributes[key] ?? ATTRIBUTE_BASELINE }));
}
