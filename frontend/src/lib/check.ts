/**
 * A check in words, for the observer: the three enums' labels and the roll's arithmetic.
 * Total tables, like the contract copies they key on.
 */

import type { CheckView } from "../types";
import { ATTRIBUTE_BASELINE, attributeLabel, attributeWord, skillWord } from "./ability";
import {
  CheckTier, type CheckTierValue, Difficulty, type DifficultyValue, Edge, type EdgeValue,
} from "./contract";

export const TIER_LABEL: Record<CheckTierValue, string> = {
  [CheckTier.critical_success]: "大成功",
  [CheckTier.success]:          "成功",
  [CheckTier.costly_success]:   "有代价的成功",
  [CheckTier.failure]:          "失败",
  [CheckTier.critical_failure]: "大失败",
};

export const DIFFICULTY_LABEL: Record<DifficultyValue, string> = {
  [Difficulty.easy]:      "容易",
  [Difficulty.normal]:    "普通",
  [Difficulty.hard]:      "困难",
  [Difficulty.very_hard]: "极难",
};

// The circumstances beyond the abilities, in the assessor's own words.
export const EDGE_LABEL: Record<EdgeValue, string> = {
  [Edge.much_worse]:  "情境明显不利",
  [Edge.worse]:       "情境略为不利",
  [Edge.even]:        "情境无碍",
  [Edge.better]:      "情境略为有利",
  [Edge.much_better]: "情境明显有利",
};

/** The label for a wire value; an unknown value reads as itself rather than vanishing. */
function label<K extends string>(table: Record<K, string>, value: string): string {
  return (table as Record<string, string>)[value] ?? value;
}

export const tierLabel = (check: CheckView): string => label(TIER_LABEL, check.tier);

const signed = (n: number): string => (n > 0 ? `+${n}` : n < 0 ? `−${-n}` : "±0");

/**
 * What the roll was against and what went into its modifier: difficulty, then each ability drawn
 * on with what it added, then the circumstances (or that luck alone decided).
 */
export function checkStakes(check: CheckView): string {
  const odds = label(DIFFICULTY_LABEL, check.difficulty);
  if (check.pure_chance) return `${odds} · 全凭运气`;
  const parts = [odds];
  const b = check.basis;
  if (b?.attribute) {
    const lift = b.attribute_level - ATTRIBUTE_BASELINE;
    parts.push(`${attributeLabel(b.attribute)}${attributeWord(b.attribute_level)} ${signed(lift)}`);
  }
  if (b?.skill) parts.push(`${b.skill}${skillWord(b.skill_level)} ${signed(b.skill_level)}`);
  if (b?.opposing_attribute) {
    const lift = b.opposing_level - ATTRIBUTE_BASELINE;
    parts.push(`对方${attributeLabel(b.opposing_attribute)}${attributeWord(b.opposing_level)} ${signed(-lift)}`);
  }
  parts.push(label(EDGE_LABEL, check.edge));
  return parts.join(" · ");
}

/** The roll itself: "4+3+3 +2 = 12 / 11". A zero modifier is left out, a negative one signed. */
export function checkArithmetic(check: CheckView): string {
  const dice = check.dice.join("+");
  const mod = check.modifier === 0 ? "" : ` ${check.modifier > 0 ? "+" : "−"}${Math.abs(check.modifier)}`;
  return `${dice}${mod} = ${check.total} / ${check.dc}`;
}
